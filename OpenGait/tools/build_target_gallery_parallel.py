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
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"unable to read log: {exc}"
    return lines[-1] if lines else "log is empty"


def average(arrays: List[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("Cannot average an empty embedding list.")
    return np.stack(arrays, axis=0).mean(axis=0)


def load_video_result(metadata_path: Path, embeddings_path: Path) -> Dict:
    metadata = json.loads(metadata_path.read_text())
    embeddings = np.load(embeddings_path)["embeddings"]
    ok_entries = []
    for entry in metadata["entries"]:
        if entry["status"] != "ok":
            continue
        ok_entries.append(
            {
                **entry,
                "embedding": embeddings[entry["embedding_index"]],
            }
        )
    if not ok_entries:
        raise ValueError(f"No valid embeddings for {metadata_path}")
    prototype = average([entry["embedding"] for entry in ok_entries])
    scores = [cosine_similarity(entry["embedding"], prototype) for entry in ok_entries]
    return {
        "video_name": metadata["video_name"],
        "video_path": metadata["video_path"],
        "prototype": prototype,
        "entries": ok_entries,
        "entry_count": len(ok_entries),
        "similarity_to_video_prototype": {
            "min": float(min(scores)),
            "max": float(max(scores)),
            "mean": float(np.mean(scores)),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel 4-GPU builder for target-person gallery prototypes."
    )
    parser.add_argument(
        "--video-dir",
        default=str(PROJECT_ROOT / "newvideos"),
        help="Folder containing Pritom/Coco/Jeevith training videos.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "output" / "target_gallery_analysis.json"),
    )
    parser.add_argument(
        "--pritom-prototype-out",
        default=str(REPO_ROOT / "output" / "pritom_prototype.json"),
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "demo" / "output" / "target_gallery_parallel"),
    )
    parser.add_argument("--model", default="grew_gaitbase", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"])
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    video_dir = Path(args.video_dir).resolve()
    output_json = Path(args.output_json).resolve()
    prototype_json = Path(args.pritom_prototype_out).resolve()
    work_root = Path(args.work_root).resolve()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    groups = {
        "pritom": sorted(video_dir.glob("Pritomtrain*.mp4")),
        "coco": sorted(video_dir.glob("Coco*.mp4")),
        "jeevith": sorted(video_dir.glob("Jeevith*.mp4")),
    }
    missing_groups = [name for name, paths in groups.items() if not paths]
    if missing_groups:
        raise FileNotFoundError(f"Missing expected training videos for: {', '.join(missing_groups)}")

    jobs = []
    for group_name, video_paths in groups.items():
        for video_path in video_paths:
            stem = video_path.stem
            jobs.append(
                {
                    "group": group_name,
                    "video_path": video_path,
                    "metadata": work_root / "per_video" / group_name / f"{stem}.json",
                    "embeddings": work_root / "per_video" / group_name / f"{stem}.npz",
                    "log": work_root / "logs" / group_name / f"{stem}.log",
                    "work": work_root / "workers" / group_name / stem,
                }
            )

    log(f"[parallel-gallery] videos={len(jobs)} gpus={gpu_ids} model={args.model}")

    pending = list(jobs)
    running = []
    completed = []
    failed = []

    while pending or running:
        while pending and len(running) < len(gpu_ids):
            gpu_id = gpu_ids[len(running)]
            job = pending.pop(0)
            job["metadata"].parent.mkdir(parents=True, exist_ok=True)
            job["embeddings"].parent.mkdir(parents=True, exist_ok=True)
            job["log"].parent.mkdir(parents=True, exist_ok=True)
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
            log(f"[parallel-gallery] launch group={job['group']} video={job['video_path'].name} gpu={gpu_id}")
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
                log(f"[parallel-gallery] done video={job['video_path'].name} gpu={job['gpu']}")
            else:
                failed.append(job)
                log(f"[parallel-gallery] FAILED video={job['video_path'].name} gpu={job['gpu']} code={code}")
                log(f"[parallel-gallery] last log: {tail_last_line(job['log'])}")
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
            log(f"[parallel-gallery] heartbeat {json.dumps(heartbeat)}")

    if failed:
        raise RuntimeError(f"{len(failed)} video extraction jobs failed. Check logs under {work_root / 'logs'}")

    group_payload = {}
    person_prototypes = {}
    for group_name in groups:
        video_results = [
            load_video_result(job["metadata"], job["embeddings"])
            for job in completed
            if job["group"] == group_name
        ]
        prototype = average([item["prototype"] for item in video_results])
        person_prototypes[group_name] = prototype
        entry_scores = [
            cosine_similarity(entry["embedding"], prototype)
            for item in video_results
            for entry in item["entries"]
        ]
        video_scores = {
            item["video_name"]: float(cosine_similarity(item["prototype"], prototype))
            for item in video_results
        }
        group_payload[group_name] = {
            "video_count": len(video_results),
            "processed_entries": sum(item["entry_count"] for item in video_results),
            "video_prototypes": {
                item["video_name"]: {
                    "entry_count": item["entry_count"],
                    "similarity_to_video_prototype": item["similarity_to_video_prototype"],
                    "similarity_to_group_prototype": video_scores[item["video_name"]],
                }
                for item in video_results
            },
            "similarity_to_group_prototype": {
                "min": float(min(entry_scores)),
                "max": float(max(entry_scores)),
                "mean": float(np.mean(entry_scores)),
            },
            "video_prototype_similarity_to_group_prototype": {
                "min": float(min(video_scores.values())),
                "max": float(max(video_scores.values())),
                "mean": float(np.mean(list(video_scores.values()))),
            },
        }

    pairwise = {
        "pritom_vs_coco": float(cosine_similarity(person_prototypes["pritom"], person_prototypes["coco"])),
        "pritom_vs_jeevith": float(cosine_similarity(person_prototypes["pritom"], person_prototypes["jeevith"])),
        "coco_vs_jeevith": float(cosine_similarity(person_prototypes["coco"], person_prototypes["jeevith"])),
    }
    pritom_self_min = group_payload["pritom"]["similarity_to_group_prototype"]["min"]
    impostor_max = max(pairwise["pritom_vs_coco"], pairwise["pritom_vs_jeevith"])
    suggested_threshold = float((pritom_self_min + impostor_max) / 2.0)

    payload = {
        "config": {
            "video_dir": str(video_dir),
            "work_root": str(work_root),
            "output_json": str(output_json),
            "pritom_prototype_out": str(prototype_json),
            "model_key": args.model,
            "min_frames": args.min_frames,
            "gpu_ids": gpu_ids,
        },
        "groups": group_payload,
        "pairwise_prototype_cosine_similarity": pairwise,
        "threshold_analysis": {
            "pritom_self_min_similarity": float(pritom_self_min),
            "pritom_impostor_max_similarity": float(impostor_max),
            "suggested_midpoint_threshold": suggested_threshold,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    prototype_payload = {
        "person_name": "pritom",
        "model_key": args.model,
        "prototype_embedding": person_prototypes["pritom"].astype(np.float32).tolist(),
        "suggested_threshold": suggested_threshold,
        "pritom_self_min_similarity": float(pritom_self_min),
        "pritom_vs_coco": pairwise["pritom_vs_coco"],
        "pritom_vs_jeevith": pairwise["pritom_vs_jeevith"],
    }
    prototype_json.parent.mkdir(parents=True, exist_ok=True)
    prototype_json.write_text(json.dumps(prototype_payload, indent=2))

    log(json.dumps(payload["threshold_analysis"], indent=2))
    log(f"[parallel-gallery] saved analysis to {output_json}")
    log(f"[parallel-gallery] saved Pritom prototype to {prototype_json}")


if __name__ == "__main__":
    main()
