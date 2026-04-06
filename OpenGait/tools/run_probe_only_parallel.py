import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent


def log(message: str) -> None:
    print(message, flush=True)


def parse_gpu_ids(raw: str):
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    if not ids:
        raise ValueError("At least one GPU id is required.")
    return ids


def read_last_nonempty_line(path: Path):
    if not path.exists():
        return None
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        line = line.strip()
        if line:
            return line
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run probe-only entry analysis in parallel across multiple GPUs."
    )
    parser.add_argument(
        "--video-dir",
        default=str(PROJECT_ROOT / "team2videos"),
        help="Folder containing videos to process.",
    )
    parser.add_argument(
        "--model",
        choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"],
        default="grew_gaitbase",
        help="Which gait model profile to use.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3",
        help="Comma-separated GPU ids to use, for example 0,1,2,3.",
    )
    parser.add_argument(
        "--conda-env",
        default="allinonegait",
        help="Conda environment to use for workers.",
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "demo" / "output" / "probe_only_parallel"),
        help="Root folder for per-video outputs.",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.75,
        help="Cosine threshold passed through to each worker.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum frames passed through to each worker.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=5,
        help="How often to print runner status updates.",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir).resolve()
    work_root = Path(args.work_root).resolve()
    logs_dir = work_root / "logs"
    results_dir = work_root / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder does not exist: {video_dir}")

    video_paths = sorted(
        [path for path in video_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    )
    if not video_paths:
        raise FileNotFoundError(f"No videos found in {video_dir}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    log(f"[parallel] videos={len(video_paths)} gpus={gpu_ids} model={args.model}")

    pending = deque(video_paths)
    active_workers = []
    finished_workers = []
    free_gpus = deque(gpu_ids)

    def launch_worker(video_path: Path, gpu_id: int):
        stem = video_path.stem
        output_json = results_dir / f"{stem}.json"
        worker_root = work_root / f"worker_gpu{gpu_id}" / stem
        log_path = logs_dir / f"{stem}.log"
        cmd = [
            "conda",
            "run",
            "-n",
            args.conda_env,
            "python",
            str(REPO_ROOT / "tools" / "probe_only_entry_analysis.py"),
            "--video-path",
            str(video_path),
            "--output-json",
            str(output_json),
            "--work-root",
            str(worker_root),
            "--cosine-threshold",
            str(args.cosine_threshold),
            "--min-frames",
            str(args.min_frames),
            "--model",
            args.model,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log(f"[parallel] launch video={video_path.name} gpu={gpu_id} log={log_path}")
        log_file = open(log_path, "w")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active_workers.append(
            {
                "video": video_path.name,
                "gpu_id": gpu_id,
                "log_path": log_path,
                "output_json": output_json,
                "process": process,
                "log_file": log_file,
                "last_status_line": None,
            }
        )

    while pending and free_gpus:
        launch_worker(pending.popleft(), free_gpus.popleft())

    while active_workers:
        still_active = []
        for worker in active_workers:
            code = worker["process"].poll()
            if code is None:
                still_active.append(worker)
                continue
            worker["log_file"].close()
            status = "ok" if code == 0 else f"failed({code})"
            log(
                f"[parallel] finished video={worker['video']} gpu={worker['gpu_id']} "
                f"status={status} json={worker['output_json']} log={worker['log_path']}"
            )
            worker["returncode"] = code
            finished_workers.append(worker)
            free_gpus.append(worker["gpu_id"])
            if pending:
                launch_worker(pending.popleft(), free_gpus.popleft())

        active_workers = still_active
        if active_workers:
            status_items = []
            for worker in active_workers:
                last_line = read_last_nonempty_line(worker["log_path"])
                if last_line:
                    worker["last_status_line"] = last_line
                summary = worker["last_status_line"] or "starting..."
                status_items.append(
                    {
                        "video": worker["video"],
                        "gpu": worker["gpu_id"],
                        "latest": summary,
                    }
                )
            log("[parallel] heartbeat " + json.dumps(status_items, ensure_ascii=True))
            time.sleep(args.poll_seconds)

    failed = [worker for worker in finished_workers if worker["returncode"] != 0]
    if failed:
        log(f"[parallel] {len(failed)} worker(s) failed")
        sys.exit(1)

    log("[parallel] all workers completed successfully")


if __name__ == "__main__":
    main()
