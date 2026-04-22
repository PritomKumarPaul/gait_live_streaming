import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "live_demo" / "output"


def find_latest_metrics() -> Path:
    candidates = sorted(OUTPUT_ROOT.glob("buffered_live_*/metrics.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No metrics.json found under {OUTPUT_ROOT}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Summarize buffered live gait timing metrics.")
    parser.add_argument("metrics_json", nargs="?", default=None, help="Path to metrics.json. Defaults to latest run.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json).resolve() if args.metrics_json else find_latest_metrics()
    payload = json.loads(metrics_path.read_text())

    settings = payload.get("settings", {})
    stage_times = payload.get("stage_times", {})
    stage_counts = payload.get("stage_counts", {})
    video_seconds = float(payload.get("video_seconds", 0.0))
    processing_elapsed = float(payload.get("elapsed_seconds", 0.0))
    total_elapsed = float(payload.get("total_elapsed_seconds", processing_elapsed))
    realtime_factor = float(payload.get("realtime_factor", 0.0))

    print(f"metrics: {metrics_path}")
    print(f"model: {settings.get('model_key')}")
    print(f"video seconds: {video_seconds:.2f}")
    print(f"processing wall time: {processing_elapsed:.2f}s")
    print(f"total wall time incl conversion: {total_elapsed:.2f}s")
    print(f"realtime factor: {realtime_factor:.3f}x")
    print(f"counts: {payload.get('counts')}")
    print()
    print("settings:")
    for key in [
        "dense_process_every_n",
        "assigned_process_every_n",
        "detector_input_size",
        "work_frame_max_side",
        "min_detection_score",
        "output_max_side",
        "write_output_video",
        "quiet",
        "identity_buffer_frames",
        "silhouette_every_n_processed",
    ]:
        print(f"  {key}: {settings.get(key)}")
    print()
    print("stage timing:")
    ranked = sorted(stage_times.items(), key=lambda item: float(item[1]), reverse=True)
    for key, seconds in ranked:
        pct = (float(seconds) / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
        print(f"  {key:16s} {float(seconds):8.2f}s  {pct:6.2f}%")
    print()
    print("stage counts:")
    for key, value in stage_counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
