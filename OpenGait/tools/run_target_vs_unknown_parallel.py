import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

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
        description="Run target-vs-unknown ablation over probe videos in parallel across GPUs."
    )
    parser.add_argument("--gallery-npz", default=str(REPO_ROOT / "output" / "multi_identity_gallery.npz"))
    parser.add_argument("--target-label", default="pritom")
    parser.add_argument("--video-dir", default=str(PROJECT_ROOT / "newvideos"))
    parser.add_argument("--video-paths", default=None, help="Comma-separated probe video paths. Defaults to Test1/Test2.")
    parser.add_argument("--output-json", default=str(REPO_ROOT / "output" / "target_vs_unknown_ablation_parallel.json"))
    parser.add_argument("--summary-txt", default=str(REPO_ROOT / "output" / "target_vs_unknown_ablation_parallel.txt"))
    parser.add_argument("--output-video-dir", default=str(REPO_ROOT / "demo" / "output" / "target_vs_unknown_videos_parallel"))
    parser.add_argument("--work-root", default=str(REPO_ROOT / "demo" / "output" / "target_vs_unknown_parallel"))
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    gallery_npz = Path(args.gallery_npz).resolve()
    video_dir = Path(args.video_dir).resolve()
    output_json = Path(args.output_json).resolve()
    summary_txt = Path(args.summary_txt).resolve()
    output_video_dir = Path(args.output_video_dir).resolve()
    work_root = Path(args.work_root).resolve()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if args.video_paths:
        video_paths = [Path(item.strip()).resolve() for item in args.video_paths.split(",") if item.strip()]
    else:
        video_paths = [video_dir / "Test1.mp4", video_dir / "Test2.mp4"]

    missing = [path for path in video_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing probe videos: " + ", ".join(str(path) for path in missing))

    jobs = []
    for video_path in video_paths:
        stem = video_path.stem
        jobs.append(
            {
                "video_path": video_path,
                "json": work_root / "worker_json" / f"{stem}.json",
                "txt": work_root / "worker_txt" / f"{stem}.txt",
                "video_dir": output_video_dir,
                "work": work_root / "workers" / stem,
                "log": work_root / "logs" / f"{stem}.log",
            }
        )

    pending = list(jobs)
    running = []
    completed = []
    failed = []

    log(f"[parallel-ablation] videos={len(jobs)} gpus={gpu_ids}")
    while pending or running:
        while pending and len(running) < len(gpu_ids):
            gpu_id = gpu_ids[len(running)]
            job = pending.pop(0)
            for key in ("json", "txt", "work", "log"):
                job[key].parent.mkdir(parents=True, exist_ok=True)
            job["video_dir"].mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "target_vs_unknown_ablation.py"),
                "--gallery-npz",
                str(gallery_npz),
                "--target-label",
                args.target_label,
                "--video-path",
                str(job["video_path"]),
                "--model",
                args.model,
                "--threshold",
                str(args.threshold),
                "--margin",
                str(args.margin),
                "--min-frames",
                str(args.min_frames),
                "--work-root",
                str(job["work"]),
                "--output-video-dir",
                str(job["video_dir"]),
                "--output-json",
                str(job["json"]),
                "--summary-txt",
                str(job["txt"]),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log(f"[parallel-ablation] launch video={job['video_path'].name} gpu={gpu_id}")
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
                log(f"[parallel-ablation] done video={job['video_path'].name} gpu={job['gpu']}")
            else:
                failed.append(job)
                log(f"[parallel-ablation] FAILED video={job['video_path'].name} gpu={job['gpu']} code={code}")
                log(f"[parallel-ablation] last log: {tail_last_line(job['log'])}")
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
            log(f"[parallel-ablation] heartbeat {json.dumps(heartbeat)}")

    if failed:
        raise RuntimeError(f"{len(failed)} probe jobs failed. Check logs under {work_root / 'logs'}")

    merged = {
        "config": {
            "gallery_npz": str(gallery_npz),
            "target_label": args.target_label,
            "video_paths": [str(path) for path in video_paths],
            "threshold": args.threshold,
            "margin": args.margin,
            "model_key": args.model,
            "gpu_ids": gpu_ids,
            "work_root": str(work_root),
            "output_video_dir": str(output_video_dir),
        },
        "videos": {},
        "summary": {
            "no_threshold": {},
            "with_threshold": {},
        },
    }
    totals = {
        "no_threshold": Counter(),
        "with_threshold": Counter(),
    }
    for job in completed:
        payload = json.loads(job["json"].read_text())
        for video_name, result in payload["videos"].items():
            merged["videos"][video_name] = result
        for mode in totals:
            totals[mode].update(payload["summary"][mode])

    merged["summary"] = {mode: dict(counter) for mode, counter in totals.items()}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(merged, indent=2))

    lines = [
        "Parallel Target vs Unknown Ablation Summary",
        f"Target label: {args.target_label}",
        f"Threshold: {args.threshold}",
        f"Margin: {args.margin}",
        f"Videos: {[path.name for path in video_paths]}",
        "",
        f"No threshold counts: {merged['summary']['no_threshold']}",
        f"With threshold+margin counts: {merged['summary']['with_threshold']}",
        "",
        f"Annotated videos: {output_video_dir}",
    ]
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(lines))

    log(json.dumps(merged["summary"], indent=2))
    log(f"[parallel-ablation] saved merged JSON to {output_json}")
    log(f"[parallel-ablation] saved summary text to {summary_txt}")
    log(f"[parallel-ablation] saved videos to {output_video_dir}")


if __name__ == "__main__":
    main()
