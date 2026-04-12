import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from probe_only_entry_analysis import PROJECT_ROOT, REPO_ROOT, cosine_similarity, log


def parse_gpu_ids(raw: str) -> List[int]:
    gpu_ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")
    return gpu_ids


def tail_last_line(path: Path) -> str:
    if not path.exists():
        return "log not created yet"
    lines = path.read_text(errors="replace").splitlines()
    return lines[-1] if lines else "log is empty"


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot average an empty embedding list.")
    return np.stack(embeddings, axis=0).mean(axis=0)


def load_video_embeddings(metadata_path: Path, embeddings_path: Path) -> Dict:
    metadata = json.loads(metadata_path.read_text())
    embeddings = np.load(embeddings_path)["embeddings"]
    ok_entries = []
    for entry in metadata["entries"]:
        if entry["status"] != "ok":
            continue
        ok_entries.append(
            {
                "entry_key": entry["entry_key"],
                "video": entry["video"],
                "track_id": entry["track_id"],
                "sequence_name": entry["sequence_name"],
                "frame_count": entry["frame_count"],
                "embedding": embeddings[entry["embedding_index"]].astype(np.float32),
            }
        )
    if not ok_entries:
        raise ValueError(f"No valid embeddings found for {metadata_path}")
    return {
        "video_name": metadata["video_name"],
        "video_path": metadata["video_path"],
        "entries": ok_entries,
    }


def build_identity_groups(video_dir: Path) -> Dict[str, List[Path]]:
    return {
        "pritom": sorted(video_dir.glob("Pritomtrain*.mp4")),
        "coco": sorted(video_dir.glob("Coco*.mp4")),
        "jeevith": sorted(video_dir.glob("Jeevith*.mp4")),
    }


def run_parallel_extraction(args, groups: Dict[str, List[Path]], work_root: Path, gpu_ids: List[int]) -> List[Dict]:
    jobs = []
    for identity, video_paths in groups.items():
        for video_path in video_paths:
            stem = video_path.stem
            jobs.append(
                {
                    "identity": identity,
                    "video_path": video_path,
                    "metadata": work_root / "per_video" / identity / f"{stem}.json",
                    "embeddings": work_root / "per_video" / identity / f"{stem}.npz",
                    "log": work_root / "logs" / identity / f"{stem}.log",
                    "work": work_root / "workers" / identity / stem,
                }
            )

    pending = list(jobs)
    running = []
    completed = []
    failed = []

    log(f"[gallery] extracting {len(jobs)} videos using gpus={gpu_ids} model={args.model}")
    while pending or running:
        while pending and len(running) < len(gpu_ids):
            gpu_id = gpu_ids[len(running)]
            job = pending.pop(0)
            for key in ("metadata", "embeddings", "log", "work"):
                job[key].parent.mkdir(parents=True, exist_ok=True)
            job["work"].mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "extract_video_entries.py"),
                "--video-path",
                str(job["video_path"]),
                "--metadata-json",
                str(job["metadata"]),
                "--embeddings-npz",
                str(job["embeddings"]),
                "--work-root",
                str(job["work"]),
                "--model",
                args.model,
                "--min-frames",
                str(args.min_frames),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log(f"[gallery] launch identity={job['identity']} video={job['video_path'].name} gpu={gpu_id}")
            log_handle = open(job["log"], "w")
            process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            running.append({**job, "gpu": gpu_id, "process": process, "log_handle": log_handle})

        time.sleep(args.poll_seconds)
        still_running = []
        for job in running:
            code = job["process"].poll()
            if code is None:
                still_running.append(job)
                continue
            job["log_handle"].close()
            if code == 0:
                completed.append(job)
                log(f"[gallery] done video={job['video_path'].name} gpu={job['gpu']}")
            else:
                failed.append(job)
                log(f"[gallery] FAILED video={job['video_path'].name} gpu={job['gpu']} code={code}")
                log(f"[gallery] last log: {tail_last_line(job['log'])}")
        running = still_running

        if running:
            heartbeat = [
                {
                    "video": job["video_path"].name,
                    "gpu": job["gpu"],
                    "latest": tail_last_line(job["log"]),
                }
                for job in running
            ]
            log(f"[gallery] heartbeat {json.dumps(heartbeat)}")

    if failed:
        raise RuntimeError(f"{len(failed)} extraction jobs failed. Check logs under {work_root / 'logs'}")
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Build a multi-identity gallery for Pritom/Coco/Jeevith using parallel GPUs."
    )
    parser.add_argument("--video-dir", default=str(PROJECT_ROOT / "newvideos"))
    parser.add_argument("--gallery-out", default=str(REPO_ROOT / "output" / "multi_identity_gallery.npz"))
    parser.add_argument("--metadata-out", default=str(REPO_ROOT / "output" / "multi_identity_gallery.json"))
    parser.add_argument("--work-root", default=str(REPO_ROOT / "demo" / "output" / "multi_identity_gallery"))
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    video_dir = Path(args.video_dir).resolve()
    gallery_out = Path(args.gallery_out).resolve()
    metadata_out = Path(args.metadata_out).resolve()
    work_root = Path(args.work_root).resolve()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    groups = build_identity_groups(video_dir)
    missing = [identity for identity, paths in groups.items() if not paths]
    if missing:
        raise FileNotFoundError(f"Missing gallery videos for: {', '.join(missing)}")

    completed = run_parallel_extraction(args, groups, work_root, gpu_ids)

    all_embeddings = []
    all_labels = []
    all_entry_keys = []
    all_video_names = []
    identity_payload = {}

    for identity in sorted(groups):
        identity_entries = []
        video_payload = {}
        jobs = [job for job in completed if job["identity"] == identity]
        for job in sorted(jobs, key=lambda item: item["video_path"].name):
            video_result = load_video_embeddings(job["metadata"], job["embeddings"])
            video_entries = video_result["entries"]
            video_embeddings = [entry["embedding"] for entry in video_entries]
            video_proto = average_embeddings(video_embeddings)
            scores_to_video_proto = [cosine_similarity(emb, video_proto) for emb in video_embeddings]
            video_payload[video_result["video_name"]] = {
                "entry_count": len(video_entries),
                "entry_keys": [entry["entry_key"] for entry in video_entries],
                "similarity_to_video_prototype": {
                    "min": float(min(scores_to_video_proto)),
                    "max": float(max(scores_to_video_proto)),
                    "mean": float(np.mean(scores_to_video_proto)),
                },
            }
            identity_entries.extend(video_entries)

        identity_embeddings = [entry["embedding"] for entry in identity_entries]
        identity_proto = average_embeddings(identity_embeddings)
        identity_scores = [cosine_similarity(emb, identity_proto) for emb in identity_embeddings]

        identity_payload[identity] = {
            "video_count": len(jobs),
            "entry_count": len(identity_entries),
            "videos": video_payload,
            "similarity_to_identity_prototype": {
                "min": float(min(identity_scores)),
                "max": float(max(identity_scores)),
                "mean": float(np.mean(identity_scores)),
            },
        }

        for entry in identity_entries:
            all_embeddings.append(entry["embedding"])
            all_labels.append(identity)
            all_entry_keys.append(entry["entry_key"])
            all_video_names.append(entry["video"])

    labels = sorted(identity_payload)
    pairwise_proto = {}
    prototypes = {
        identity: average_embeddings(
            [
                all_embeddings[i]
                for i, label in enumerate(all_labels)
                if label == identity
            ]
        )
        for identity in labels
    }
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            pairwise_proto[f"{left}_vs_{right}"] = float(cosine_similarity(prototypes[left], prototypes[right]))

    embedding_matrix = np.stack(all_embeddings, axis=0).astype(np.float32)
    label_array = np.array(all_labels)
    entry_key_array = np.array(all_entry_keys)
    video_name_array = np.array(all_video_names)

    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gallery_out,
        embeddings=embedding_matrix,
        labels=label_array,
        entry_keys=entry_key_array,
        video_names=video_name_array,
    )

    metadata = {
        "config": {
            "video_dir": str(video_dir),
            "gallery_out": str(gallery_out),
            "metadata_out": str(metadata_out),
            "work_root": str(work_root),
            "model_key": args.model,
            "min_frames": args.min_frames,
            "gpu_ids": gpu_ids,
        },
        "summary": {
            "identity_count": len(labels),
            "gallery_embedding_count": int(embedding_matrix.shape[0]),
            "identities": labels,
        },
        "identities": identity_payload,
        "pairwise_identity_prototype_cosine_similarity": pairwise_proto,
        "notes": [
            "This gallery stores all enrollment embeddings, not just one averaged prototype.",
            "Probe matching should compare each probe entry against all gallery embeddings.",
        ],
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2))

    log(json.dumps(metadata["summary"], indent=2))
    log(json.dumps(pairwise_proto, indent=2))
    log(f"[gallery] saved embeddings to {gallery_out}")
    log(f"[gallery] saved metadata to {metadata_out}")


if __name__ == "__main__":
    main()
