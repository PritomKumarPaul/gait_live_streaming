import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from probe_only_entry_analysis import PROJECT_ROOT, REPO_ROOT, log


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


def main():
    parser = argparse.ArgumentParser(
        description="Build a clean demo gallery from pritomgallery/cocogallery/jeevithgallery videos."
    )
    parser.add_argument("--clean-root", default=str(PROJECT_ROOT / "clean_demo"))
    parser.add_argument("--gallery-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    clean_root = Path(args.clean_root).resolve()
    gallery_out = Path(args.gallery_out).resolve()
    metadata_out = Path(args.metadata_out).resolve()
    work_root = Path(args.work_root).resolve()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    jobs = [
        ("pritom", clean_root / "gallery" / "pritomgallery.mp4"),
        ("coco", clean_root / "gallery" / "cocogallery.mp4"),
        ("jeevith", clean_root / "gallery" / "jeevithgallery.mp4"),
    ]
    missing = [str(path) for _, path in jobs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing clean gallery videos: " + ", ".join(missing))

    pending = []
    for label, video_path in jobs:
        pending.append(
            {
                "label": label,
                "video_path": video_path,
                "metadata": work_root / "gallery_per_video" / f"{label}.json",
                "embeddings": work_root / "gallery_per_video" / f"{label}.npz",
                "worker_work": work_root / "workers" / label,
                "log": work_root / "logs" / f"{label}.log",
            }
        )

    running = []
    completed = []
    failed = []
    log(f"[clean-gallery] videos={len(pending)} gpus={gpu_ids}")

    while pending or running:
        while pending and len(running) < len(gpu_ids):
            gpu_id = gpu_ids[len(running)]
            job = pending.pop(0)
            for key in ("metadata", "embeddings", "worker_work", "log"):
                job[key].parent.mkdir(parents=True, exist_ok=True)
            job["worker_work"].mkdir(parents=True, exist_ok=True)
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
                str(job["worker_work"]),
                "--model",
                args.model,
                "--min-frames",
                str(args.min_frames),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log(f"[clean-gallery] launch label={job['label']} video={job['video_path'].name} gpu={gpu_id}")
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
                log(f"[clean-gallery] done label={job['label']} gpu={job['gpu']}")
            else:
                failed.append(job)
                log(f"[clean-gallery] FAILED label={job['label']} code={code}")
                log(f"[clean-gallery] last log: {tail_last_line(job['log'])}")
        running = still_running

        if running:
            heartbeat = [
                {"label": job["label"], "gpu": job["gpu"], "latest": tail_last_line(job["log"])}
                for job in running
            ]
            log(f"[clean-gallery] heartbeat {json.dumps(heartbeat)}")

    if failed:
        raise RuntimeError(f"{len(failed)} clean gallery jobs failed. Check {work_root / 'logs'}")

    embeddings = []
    labels = []
    entry_keys = []
    video_names = []
    summary = {}
    for job in sorted(completed, key=lambda item: item["label"]):
        metadata = json.loads(job["metadata"].read_text())
        data = np.load(job["embeddings"])["embeddings"].astype(np.float32)
        if data.shape[0] == 0:
            raise ValueError(f"No valid gallery embeddings for {job['label']}")
        prototype = data.mean(axis=0)
        embeddings.append(prototype)
        labels.append(job["label"])
        entry_keys.append(f"{job['video_path'].stem}:gallery")
        video_names.append(job["video_path"].stem)
        summary[job["label"]] = {
            "source_video": str(job["video_path"]),
            "valid_entries": int(data.shape[0]),
            "raw_entry_metadata": metadata["entries"],
        }

    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gallery_out,
        embeddings=np.stack(embeddings, axis=0).astype(np.float32),
        labels=np.array(labels),
        entry_keys=np.array(entry_keys),
        video_names=np.array(video_names),
    )

    metadata = {
        "config": {
            "clean_root": str(clean_root),
            "gallery_out": str(gallery_out),
            "metadata_out": str(metadata_out),
            "work_root": str(work_root),
            "model_key": args.model,
            "gpu_ids": gpu_ids,
        },
        "summary": summary,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2))
    log(f"[clean-gallery] saved gallery to {gallery_out}")
    log(f"[clean-gallery] saved metadata to {metadata_out}")


if __name__ == "__main__":
    main()
