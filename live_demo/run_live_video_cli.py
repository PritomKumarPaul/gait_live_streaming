import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT = ROOT / "OpenGait"
DEMO_LIBS = OPENGAIT / "demo" / "libs"
LIVE_ROOT = ROOT / "live_demo"
OUTPUT_ROOT = LIVE_ROOT / "output"
DEFAULT_VIDEO = ROOT / "clean_demo_v2" / "probes" / "test1probe.mp4"

sys.path.insert(0, str(DEMO_LIBS))
sys.path.insert(0, str(OPENGAIT))


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    return ((37 * track_id) % 255, (137 * track_id) % 255, (211 * track_id) % 255)


def display_name_for_entry(entry_index: int) -> str:
    demo_names = {1: "pritom", 2: "coco"}
    return demo_names.get(entry_index, f"person{entry_index}")


def draw_panel(frame: np.ndarray, lines: List[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.82
    thickness = 2
    line_height = 34
    width = 560
    height = 24 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (18 + width, 18 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for idx, line in enumerate(lines):
        y = 52 + idx * line_height
        cv2.putText(frame, line, (34, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_label(frame: np.ndarray, x: int, y: int, text: str, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 12)
    cv2.rectangle(frame, (x, y0), (x + tw + 14, y0 + th + baseline + 12), color, -1)
    cv2.putText(frame, text, (x + 7, y0 + th + 3), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def make_browser_playable_video(input_video: Path, output_video: Path) -> Path:
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return input_video
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not output_video.exists() or output_video.stat().st_size == 0:
        print("[warn] ffmpeg conversion failed; keeping raw mp4")
        print(proc.stdout[-1000:])
        return input_video
    return output_video


def parse_args():
    parser = argparse.ArgumentParser(description="Debug live video tracking without Gradio.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Video to treat as a live stream.")
    parser.add_argument("--gpu-id", default=None, help="GPU id to expose, or 'cpu'. Set before tracker import.")
    parser.add_argument("--process-every-n", type=int, default=3, help="Process every N frames.")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="0 means full video.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output folder root.")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N processed frames.")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = ROOT / video_path
    video_path = video_path.resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root = output_root.resolve()

    if args.gpu_id:
        if args.gpu_id.lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.chdir(OPENGAIT)
    print(f"[live-cli] torch_cuda={torch.cuda.is_available()} visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
    if torch.cuda.is_available():
        print(f"[live-cli] cuda_device={torch.cuda.get_device_name(0)}", flush=True)

    print("[live-cli] importing tracker components", flush=True)
    from demo.libs.track import exp, model, track_cfgs  # noqa: E402
    from tracker.byte_tracker import BYTETracker  # noqa: E402
    from tracking_utils.predictor import Predictor  # noqa: E402
    from tracking_utils.timer import Timer  # noqa: E402

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    run_root = output_root / f"live_cli_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    raw_video = run_root / "live_tracking_raw.mp4"
    final_video = run_root / "live_tracking.mp4"
    event_log_path = run_root / "events_log.txt"

    device = torch.device("cuda" if track_cfgs["device"] == "gpu" and torch.cuda.is_available() else "cpu")
    predictor = Predictor(model, exp, None, None, device, device.type == "cuda")
    tracker = BYTETracker(frame_rate=30)
    timer = Timer()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames
    if args.max_seconds and args.max_seconds > 0:
        max_frames = min(max_frames or int(args.max_seconds * input_fps), int(args.max_seconds * input_fps))

    process_every_n = max(1, args.process_every_n)
    output_fps = max(1.0, input_fps / process_every_n)
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

    seen_tracks: Dict[int, int] = {}
    track_names: Dict[int, str] = {}
    event_lines: List[str] = []
    frame_id = 0
    processed = 0
    first_track_offset = None

    print(f"[live-cli] video={video_path}", flush=True)
    print(f"[live-cli] total_frames={total_frames} fps={input_fps:.2f} max_frames={max_frames} process_every_n={process_every_n}", flush=True)
    print(f"[live-cli] raw_output={raw_video}", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_id >= max_frames:
                break
            if frame_id % process_every_n != 0:
                frame_id += 1
                continue

            outputs, img_info = predictor.inference(frame, timer)
            active = set()
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info["height"], img_info["width"]], exp.test_size)
                for target in online_targets:
                    tlwh = target.tlwh
                    raw_track_id = int(target.track_id)
                    if first_track_offset is None:
                        first_track_offset = raw_track_id - 1
                    track_id = raw_track_id - first_track_offset
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] <= 10 or vertical:
                        continue
                    active.add(track_id)
                    if track_id not in seen_tracks:
                        seen_tracks[track_id] = len(seen_tracks) + 1
                        track_names[track_id] = display_name_for_entry(seen_tracks[track_id])
                        event_lines.append(
                            f"t={frame_id / input_fps:.2f}s frame={frame_id}: new entry "
                            f"{track_names[track_id]} (track {track_id:03d})"
                        )
                    x, y, w, h = tlwh
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    color = color_for_id(track_id)
                    name = track_names[track_id]
                    label = f"{name} | count {seen_tracks[track_id]} | track {track_id:03d}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                    draw_label(frame, x1, max(0, y1 - 4), label, color)
                timer.toc()

            seen_summary = ", ".join(track_names[tid] for tid in sorted(track_names)) or "none yet"
            draw_panel(
                frame,
                [
                    f"Total entries seen: {len(seen_tracks)}",
                    f"Active tracks now: {len(active)}",
                    f"Seen so far: {seen_summary}",
                ],
            )
            writer.write(frame)
            processed += 1
            if processed % max(1, args.log_every) == 0:
                print(f"[live-cli] processed={processed} frame={frame_id} active={len(active)} total_entries={len(seen_tracks)}", flush=True)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    playable = make_browser_playable_video(raw_video, final_video)
    summary_lines = [
        "Live tracking event log",
        f"video={video_path}",
        f"processed_frames={processed}",
        f"total_entries={len(seen_tracks)}",
        "",
        "Entries:",
    ]
    for track_id in sorted(seen_tracks):
        summary_lines.append(f"- {track_names[track_id]}: entry_count={seen_tracks[track_id]}, track_id={track_id:03d}")
    summary_lines.extend(["", "Timeline:"])
    summary_lines.extend(event_lines or ["No entries detected."])
    event_log_path.write_text("\n".join(summary_lines))
    print("[live-cli] finished", flush=True)
    print(f"[live-cli] processed_frames={processed}", flush=True)
    print(f"[live-cli] total_entries={len(seen_tracks)}", flush=True)
    print(f"[live-cli] output_video={playable}", flush=True)
    print(f"[live-cli] event_log={event_log_path}", flush=True)
    print(f"[live-cli] output_folder={run_root}", flush=True)


if __name__ == "__main__":
    main()
