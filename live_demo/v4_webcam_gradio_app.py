import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr


ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = ROOT / "live_demo"
DEFAULT_GALLERY = LIVE_ROOT / "cache" / "pritom_coco_gallery.npz"

GPU_ID = os.environ.get("LIVE_DEMO_GPU_ID")
if GPU_ID not in (None, "", "cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
elif GPU_ID in ("cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(LIVE_ROOT))

from run_buffered_live_probe import run_buffered_live_probe  # noqa: E402


def gallery_for_model(model_key: str) -> Path:
    if model_key == "grew_gaitbase":
        return DEFAULT_GALLERY
    return LIVE_ROOT / "cache" / f"pritom_coco_gallery_{model_key}.npz"


def resolve_gradio_path(file_obj) -> Optional[Path]:
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


def resize_rgb_preview(frame, max_side: int):
    height, width = frame.shape[:2]
    largest = max(width, height)
    if largest > max_side:
        scale = max_side / largest
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def find_ffmpeg():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def compress_probe(input_video: Path, output_video: Path, max_side: int, fps: float = 30.0, crf: int = 28) -> Path:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return input_video
    output_video.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = (
        f"fps={fps},"
        f"scale='if(gt(iw,ih),min({max_side},iw),-2)':"
        f"'if(gt(ih,iw),min({max_side},ih),-2)'"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not output_video.exists() or output_video.stat().st_size == 0:
        return input_video
    return output_video


def inspect_video(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return "Could not inspect video."
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frames / fps if fps else 0.0
    return f"{width}x{height}, {fps:.2f} FPS, {duration:.1f}s, {path.stat().st_size / (1024 * 1024):.2f} MB"


def run_webcam_probe(
    webcam_video,
    model_key: str,
    max_seconds: float,
    compress_input: bool,
    compressed_max_side: int,
    process_every_n: int,
    assigned_process_every_n: int,
    identity_buffer_frames: int,
    preview_every_n: int,
    display_max_side: int,
):
    gallery_path = gallery_for_model(model_key)
    if not gallery_path.exists():
        raise gr.Error(
            f"Missing gallery cache for {model_key}: {gallery_path}\n\n"
            f"Build it first with:\n"
            f"CUDA_VISIBLE_DEVICES=<gpu> python live_demo/build_pritom_coco_gallery.py --model {model_key}"
        )

    input_path = resolve_gradio_path(webcam_video)
    if input_path is None or not input_path.exists():
        raise gr.Error("Please record or upload a webcam video first.")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    prepared_root = LIVE_ROOT / "output" / f"v4_webcam_input_{run_stamp}"
    prepared_root.mkdir(parents=True, exist_ok=True)
    probe_path = input_path
    if compress_input:
        probe_path = compress_probe(
            input_video=input_path,
            output_video=prepared_root / "webcam_probe_prepared.mp4",
            max_side=int(compressed_max_side),
            fps=30.0,
            crf=28,
        )

    logs = [
        "V4 webcam gait recognition started.",
        f"Raw webcam video: {input_path}",
        f"Raw info: {inspect_video(input_path)}",
        f"Processed probe: {probe_path}",
        f"Processed info: {inspect_video(probe_path)}",
        f"Gallery cache: {gallery_path}",
        f"Model: {model_key}",
        f"Max seconds: {max_seconds}",
        f"Compression: {compress_input}, max_side={compressed_max_side}",
        f"Stride: dense={process_every_n}, assigned={assigned_process_every_n}",
        f"Silhouette buffer frames: {identity_buffer_frames}",
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
    ]

    yield None, "\n".join(logs), "Starting delayed recognition...", None

    gen = run_buffered_live_probe(
        video_path=probe_path,
        gallery_path=gallery_path,
        model_key=model_key,
        process_every_n=int(process_every_n),
        assigned_process_every_n=int(assigned_process_every_n),
        detector_input_size=0,
        work_frame_max_side=0,
        min_detection_score=0,
        output_max_side=480,
        write_output_video=True,
        silhouette_every_n_processed=1,
        identity_buffer_frames=int(identity_buffer_frames),
        max_seconds=float(max_seconds),
        log_every=0,
        progress_callback=None,
        quiet=True,
    )

    last_preview = None
    processed = 0
    last_status = "Running..."
    try:
        while True:
            update = next(gen)
            processed = int(update["processed"])
            counts = update["counts"]
            realtime_factor = float(update.get("realtime_factor", 0.0))
            last_status = (
                f"Running delayed recognition\n"
                f"Processed frames: {processed}\n"
                f"Video time processed: {float(update.get('video_seconds', 0.0)):.2f}s\n"
                f"Wall time elapsed: {float(update.get('elapsed_seconds', 0.0)):.2f}s\n"
                f"Realtime factor: {realtime_factor:.2f}x\n"
                f"Counts: pritom={counts.get('pritom', 0)} | coco={counts.get('coco', 0)}\n"
                f"Run folder: {update['run_root']}"
            )
            if processed % max(1, int(preview_every_n)) == 0:
                last_preview = resize_rgb_preview(update["frame"], int(display_max_side))
                yield last_preview, "\n".join(logs), last_status, None
    except StopIteration as finished:
        result = finished.value or {}

    counts = result.get("counts", {})
    stage_times = result.get("stage_times", {})
    final_status = (
        f"Finished\n"
        f"Processed frames: {result.get('processed', processed)}\n"
        f"Video time processed: {float(result.get('video_seconds', 0.0)):.2f}s\n"
        f"Processing wall time: {float(result.get('elapsed_seconds', 0.0)):.2f}s\n"
        f"Total wall time incl. video conversion: {float(result.get('total_elapsed_seconds', 0.0)):.2f}s\n"
        f"Realtime factor: {float(result.get('realtime_factor', 0.0)):.2f}x\n"
        f"Stage seconds: detect={float(stage_times.get('detect', 0.0)):.1f}, "
        f"seg={float(stage_times.get('segmentation', 0.0)):.1f}, "
        f"recog={float(stage_times.get('recognition', 0.0)):.1f}, "
        f"draw/write={float(stage_times.get('draw_write', 0.0)):.1f}\n"
        f"Final counts: pritom={counts.get('pritom', 0)} | coco={counts.get('coco', 0)}\n"
        f"Output folder: {result.get('run_root')}\n"
        f"Annotated video: {result.get('output_video')}\n"
        f"Event log: {result.get('log_txt')}"
    )
    yield last_preview, "\n".join(logs), final_status, str(result.get("output_video")) if result.get("output_video") else None


def build_demo():
    with gr.Blocks(title="V4 Webcam Gait Recognition") as demo:
        gr.Markdown(
            """
            # V4 Webcam Gait Recognition

            Record from the browser webcam on the left, then run the same v3 delayed-recognition
            pipeline on the recorded webcam video. The right side shows delayed processing preview
            and the final annotated video.

            Note: this Gradio setup records a browser webcam video first, then processes it. For
            true frame-by-frame browser streaming we would need a WebRTC-style frontend.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                webcam_video = gr.Video(
                    sources=["webcam", "upload"],
                    label="Webcam input or uploaded webcam-style video",
                    include_audio=False,
                    max_length=300,
                    height=520,
                )
                run_button = gr.Button("Run V4 Webcam Recognition", variant="primary")
                model_key = gr.Dropdown(
                    ["grew_gaitbase", "grew_gaitgl", "current_gaitbase"],
                    value="grew_gaitbase",
                    label="Gait model",
                )
                max_seconds = gr.Slider(5, 300, value=300, step=5, label="Max webcam seconds to process")
                compress_input = gr.Checkbox(value=True, label="Prepare webcam video to fixed low resolution/FPS first")
                compressed_max_side = gr.Dropdown([480, 720, 1080], value=720, label="Prepared video max side")
                process_every_n = gr.Slider(1, 15, value=5, step=1, label="Manual frame stride")
                assigned_process_every_n = gr.Slider(1, 45, value=10, step=1, label="Assigned-track stride")
                identity_buffer_frames = gr.Slider(2, 64, value=5, step=1, label="Silhouette buffer frames")
                logs = gr.Textbox(label="Run setup log", lines=13)
            with gr.Column(scale=1):
                preview = gr.Image(label="Delayed recognition preview", height=520)
                status = gr.Textbox(label="Run status", lines=10)
                annotated_video = gr.Video(label="Final delayed annotated video", interactive=False)

        run_button.click(
            run_webcam_probe,
            inputs=[
                webcam_video,
                model_key,
                max_seconds,
                compress_input,
                compressed_max_side,
                process_every_n,
                assigned_process_every_n,
                identity_buffer_frames,
                gr.Number(value=5, visible=False),
                gr.Number(value=720, visible=False),
            ],
            outputs=[preview, logs, status, annotated_video],
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue(max_size=1).launch(server_name="0.0.0.0", server_port=7864, share=True)
