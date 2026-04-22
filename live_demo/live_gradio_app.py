import shutil
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT = ROOT / "OpenGait"
DEMO_LIBS = OPENGAIT / "demo" / "libs"
LIVE_ROOT = ROOT / "live_demo"
OUTPUT_ROOT = LIVE_ROOT / "output"
DEFAULT_SAMPLE_VIDEO = ROOT / "clean_demo_v2" / "probes" / "test1probe.mp4"

GPU_ID = os.environ.get("LIVE_DEMO_GPU_ID")
if GPU_ID not in (None, "", "cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
elif GPU_ID in ("cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# The original demo code uses relative checkpoint paths such as ./demo/checkpoints.
sys.path.insert(0, str(DEMO_LIBS))
sys.path.insert(0, str(OPENGAIT))


def make_browser_playable_video(input_video: Path, output_video: Path) -> Path:
    """Convert OpenCV MP4 output to browser-friendly H.264 when ffmpeg is available."""
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
        return input_video
    return output_video


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


def resize_for_display(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    largest = max(width, height)
    if largest <= max_side:
        return frame
    scale = max_side / largest
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def import_tracker_components():
    """Import original All-in-One-Gait/ByteTrack components after cwd is set."""
    os.chdir(OPENGAIT)
    from demo.libs.track import exp, model, track_cfgs  # noqa: E402
    from tracker.byte_tracker import BYTETracker  # noqa: E402
    from tracking_utils.predictor import Predictor  # noqa: E402
    from tracking_utils.timer import Timer  # noqa: E402

    return exp, model, track_cfgs, BYTETracker, Predictor, Timer


def resolve_gradio_path(file_obj) -> Optional[Path]:
    """Handle Gradio filepath strings and FileData-like objects."""
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    for attr in ("path", "name"):
        value = getattr(file_obj, attr, None)
        if value:
            return Path(value)
    if isinstance(file_obj, dict):
        for key in ("path", "name"):
            value = file_obj.get(key)
            if value:
                return Path(value)
    return None


def live_track_video(
    input_mode: str,
    uploaded_file,
    process_every_n: int,
    max_seconds: float,
    display_max_side: int,
    preview_every_n: int,
) -> Iterable[Tuple[Optional[np.ndarray], str, Optional[str], Optional[str]]]:
    """Stream a video as if it were live and update tracking/count overlay."""
    if input_mode == "Default sample video":
        video_path = DEFAULT_SAMPLE_VIDEO
    elif input_mode == "Upload video":
        video_path = resolve_gradio_path(uploaded_file)
    else:
        video_path = None

    if video_path is None:
        if input_mode == "Upload video":
            raise gr.Error("Upload mode is selected. Please upload a video first, or choose Default sample video.")
        raise gr.Error("Please provide a video for the selected input mode.")
    if not video_path.exists():
        raise gr.Error(f"Video does not exist: {video_path}")

    process_every_n = max(1, int(process_every_n))
    preview_every_n = max(1, int(preview_every_n))
    display_max_side = max(240, int(display_max_side))

    run_root = OUTPUT_ROOT / f"live_video_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    raw_video = run_root / "live_tracking_raw.mp4"
    final_video = run_root / "live_tracking.mp4"
    event_log_path = run_root / "events_log.txt"

    device_note = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    yield None, (
        f"Loading ByteTrack detector/tracker for {video_path.name}. This can take a moment on the first run...\n"
        f"CUDA_VISIBLE_DEVICES={device_note}"
    ), None, None
    exp, det_model, track_cfgs, BYTETracker, Predictor, Timer = import_tracker_components()

    device = torch.device("cuda" if track_cfgs["device"] == "gpu" and torch.cuda.is_available() else "cpu")
    predictor = Predictor(det_model, exp, None, None, device, device.type == "cuda")
    tracker = BYTETracker(frame_rate=30)
    timer = Timer()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error(f"Could not open video: {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames
    if max_seconds and max_seconds > 0:
        max_frames = min(max_frames or int(max_seconds * input_fps), int(max_seconds * input_fps))

    output_fps = max(1.0, input_fps / process_every_n)
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

    seen_tracks: Dict[int, int] = {}
    track_names: Dict[int, str] = {}
    event_lines: List[str] = []
    active_tracks = set()
    frame_id = 0
    processed_frames = 0
    first_track_offset = None

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
            active_tracks.clear()
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

                    active_tracks.add(track_id)
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

            elapsed = frame_id / input_fps
            status = (
                f"Live simulation running\\n"
                f"Frame: {frame_id + 1}/{total_frames or '?'} | Time: {elapsed:.1f}s\\n"
                f"Processed FPS target: {output_fps:.2f}\\n"
                f"Active tracks now: {len(active_tracks)}\\n"
                f"Total entries seen: {len(seen_tracks)}\\n"
                f"Output folder: {run_root}"
            )
            seen_summary = ", ".join(track_names[tid] for tid in sorted(track_names)) or "none yet"
            draw_panel(
                frame,
                [
                    f"Total entries seen: {len(seen_tracks)}",
                    f"Active tracks now: {len(active_tracks)}",
                    f"Seen so far: {seen_summary}",
                ],
            )

            writer.write(frame)
            processed_frames += 1
            if processed_frames % preview_every_n == 0:
                display = resize_for_display(frame, display_max_side)
                yield cv2.cvtColor(display, cv2.COLOR_BGR2RGB), status, None, None
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    playable_video = make_browser_playable_video(raw_video, final_video)
    summary_lines = [
        "Live tracking event log",
        f"video={video_path}",
        f"processed_frames={processed_frames}",
        f"total_entries={len(seen_tracks)}",
        "",
        "Entries:",
    ]
    for track_id in sorted(seen_tracks):
        summary_lines.append(f"- {track_names[track_id]}: entry_count={seen_tracks[track_id]}, track_id={track_id:03d}")
    summary_lines.extend(["", "Timeline:"])
    summary_lines.extend(event_lines or ["No entries detected."])
    event_log_path.write_text("\n".join(summary_lines))
    final_status = (
        f"Finished live video simulation\\n"
        f"Processed frames: {processed_frames}\\n"
        f"Total entries seen: {len(seen_tracks)}\\n"
        f"Seen: {', '.join(track_names[tid] for tid in sorted(track_names)) or 'none'}\\n"
        f"Annotated video: {playable_video}\\n"
        f"Event log: {event_log_path}\\n"
        f"Output folder: {run_root}"
    )
    yield None, final_status, str(playable_video), str(event_log_path)


def build_demo() -> gr.Blocks:
    def update_input_visibility(mode: str):
        return (
            gr.update(visible=mode == "Upload video"),
            gr.update(visible=mode == "Default sample video"),
        )

    with gr.Blocks(title="Live Gait Tracking Prototype") as demo:
        gr.Markdown(
            """
            # Live Gait Tracking Prototype

            This is the stable live-simulation prototype. It runs the original ByteTrack detector/tracker
            frame-by-frame, updates progress while it runs, and returns an annotated MP4.

            Webcam streaming is intentionally not shown here yet because Gradio webcam input records a clip
            before sending it to the server. We will add a separate true webcam-streaming prototype later.
            """
        )
        with gr.Row():
            with gr.Column():
                input_mode = gr.Radio(
                    ["Upload video", "Default sample video"],
                    value="Upload video",
                    label="Input mode",
                )
                uploaded_file = gr.File(
                    label="Upload video file",
                    file_types=["video"],
                    visible=True,
                )
                default_video_note = gr.Markdown(
                    f"Default sample path: `{DEFAULT_SAMPLE_VIDEO}` "
                    f"({'found' if DEFAULT_SAMPLE_VIDEO.exists() else 'not found yet'})",
                    visible=False,
                )
                process_every_n = gr.Slider(1, 15, value=5, step=1, label="Process every N frames")
                preview_every_n = gr.Slider(
                    1,
                    30,
                    value=10,
                    step=1,
                    label="Update browser preview every N processed frames",
                )
                max_seconds = gr.Slider(0, 120, value=30, step=1, label="Max seconds to process, 0 = full video")
                display_max_side = gr.Slider(360, 1080, value=720, step=60, label="Live preview max side")
                run_button = gr.Button("Run Live Simulation", variant="primary")
            with gr.Column():
                live_frame = gr.Image(label="Live annotated preview", height=520)
                status = gr.Textbox(label="Status", lines=8)
                output_video = gr.Video(label="Final annotated video", interactive=False)
                event_log = gr.File(label="Event log")

        input_mode.change(
            fn=update_input_visibility,
            inputs=[input_mode],
            outputs=[uploaded_file, default_video_note],
        )

        run_button.click(
            fn=live_track_video,
            inputs=[input_mode, uploaded_file, process_every_n, max_seconds, display_max_side, preview_every_n],
            outputs=[live_frame, status, output_video, event_log],
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue(max_size=1).launch(server_name="0.0.0.0", server_port=7861, share=True)
