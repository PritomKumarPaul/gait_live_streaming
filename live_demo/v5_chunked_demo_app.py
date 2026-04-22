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
ORIGINAL_DEFAULT_PROBE = ROOT / "clean_demo_v2" / "probes" / "test1probe.mp4"
LOWRES_DEFAULT_PROBE = LIVE_ROOT / "prepared_inputs" / "test1probe_720p30.mp4"
DEFAULT_PROBE = LOWRES_DEFAULT_PROBE if LOWRES_DEFAULT_PROBE.exists() else ORIGINAL_DEFAULT_PROBE
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
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    return f"{width}x{height}, {fps:.2f} FPS, {duration:.1f}s, {size_mb:.2f} MB"


def choose_probe(mode: str, uploaded_probe, webcam_chunk):
    if mode == "Default probe video":
        return DEFAULT_PROBE, "default"
    if mode == "Upload probe video":
        return resolve_gradio_path(uploaded_probe), "upload"
    return resolve_gradio_path(webcam_chunk), "webcam_chunk"


def run_v5_demo(
    input_mode: str,
    uploaded_probe,
    webcam_chunk,
    model_key: str,
    compress_input: bool,
    compressed_max_side: int,
    max_seconds: float,
    process_every_n: int,
    assigned_process_every_n: int,
    work_frame_max_side: int,
    min_detection_score: float,
    output_max_side: int,
    write_output_video: bool,
    quiet_speed_mode: bool,
    target_processing_fps: float,
    silhouette_every_n_processed: int,
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

    probe_video, source_kind = choose_probe(input_mode, uploaded_probe, webcam_chunk)
    if probe_video is None:
        if input_mode == "Upload probe video":
            raise gr.Error("Please upload a probe video.")
        raise gr.Error("Please record a short webcam chunk first, then press Run V5 Demo.")
    if not probe_video.exists():
        raise gr.Error(f"Probe video does not exist: {probe_video}")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    prepared_root = LIVE_ROOT / "output" / f"v5_input_{run_stamp}"
    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared_probe = probe_video
    if compress_input:
        prepared_probe = compress_probe(
            input_video=probe_video,
            output_video=prepared_root / f"{source_kind}_prepared.mp4",
            max_side=int(compressed_max_side),
            fps=30.0,
            crf=28,
        )

    if target_processing_fps and target_processing_fps > 0:
        cap = cv2.VideoCapture(str(prepared_probe))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        process_every_n = max(1, int(round(input_fps / target_processing_fps)))
    else:
        process_every_n = max(1, int(process_every_n))

    logs = [
        "V5 chunked demo started.",
        f"Input mode: {input_mode}",
        f"Original probe: {probe_video}",
        f"Original probe info: {inspect_video(probe_video)}",
        f"Prepared probe: {prepared_probe}",
        f"Prepared probe info: {inspect_video(prepared_probe)}",
        f"Gallery cache: {gallery_path}",
        f"Model: {model_key}",
        f"Compression enabled: {compress_input}",
        f"Prepared max side: {compressed_max_side}",
        f"Process every N frames: {process_every_n}",
        f"Assigned-track stride: {assigned_process_every_n}",
        "Detector input size: original demo default",
        f"Working frame max side: {work_frame_max_side}",
        f"Minimum detection score: {min_detection_score}",
        f"Output max side: {output_max_side}",
        f"Write annotated video: {write_output_video}",
        f"Quiet mode: {quiet_speed_mode}",
        f"Target processing FPS: {target_processing_fps if target_processing_fps > 0 else 'manual stride'}",
        f"Segmentation every N processed frames: {silhouette_every_n_processed}",
        f"Identity buffer frames: {identity_buffer_frames}",
        f"Max seconds to process: {max_seconds if max_seconds > 0 else 'full video'}",
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
    ]

    yield None, "\n".join(logs), "Starting V5 delayed recognition...", None

    gen = run_buffered_live_probe(
        video_path=prepared_probe,
        gallery_path=gallery_path,
        model_key=model_key,
        process_every_n=process_every_n,
        assigned_process_every_n=assigned_process_every_n,
        detector_input_size=0,
        work_frame_max_side=work_frame_max_side,
        min_detection_score=min_detection_score,
        output_max_side=output_max_side,
        write_output_video=write_output_video,
        silhouette_every_n_processed=silhouette_every_n_processed,
        identity_buffer_frames=identity_buffer_frames,
        max_seconds=max_seconds,
        log_every=0 if quiet_speed_mode else 1,
        progress_callback=None if quiet_speed_mode else lambda message: logs.append(message),
        quiet=quiet_speed_mode,
    )

    last_preview = None
    processed = 0
    try:
        while True:
            update = next(gen)
            processed = int(update["processed"])
            counts = update["counts"]
            status = (
                f"Running V5 delayed recognition\n"
                f"Processed frames: {processed}\n"
                f"Frame id: {update['frame_id']}\n"
                f"Current stride: {update.get('current_stride')} "
                f"(dense={update.get('dense_stride')}, assigned={update.get('assigned_stride')})\n"
                f"Video time processed: {float(update.get('video_seconds', 0.0)):.2f}s\n"
                f"Wall time elapsed: {float(update.get('elapsed_seconds', 0.0)):.2f}s\n"
                f"Realtime factor: {float(update.get('realtime_factor', 0.0)):.2f}x\n"
                f"Active tracks: {update['active_count']}\n"
                f"Assigned tracks: {update['assigned_count']}\n"
                f"Counts: pritom={counts.get('pritom', 0)} | coco={counts.get('coco', 0)}\n"
                f"Run folder: {update['run_root']}"
            )
            if processed % max(1, int(preview_every_n)) == 0:
                last_preview = resize_rgb_preview(update["frame"], int(display_max_side))
            yield last_preview, "\n".join(logs[-25:]), status, None
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
        f"track={float(stage_times.get('track_update', 0.0)):.1f}, "
        f"seg={float(stage_times.get('segmentation', 0.0)):.1f}, "
        f"recog={float(stage_times.get('recognition', 0.0)):.1f}, "
        f"draw/write={float(stage_times.get('draw_write', 0.0)):.1f}, "
        f"ffmpeg={float(stage_times.get('ffmpeg_convert', 0.0)):.1f}\n"
        f"Final counts: pritom={counts.get('pritom', 0)} | coco={counts.get('coco', 0)}\n"
        f"Output folder: {result.get('run_root')}\n"
        f"Annotated video: {result.get('output_video') or 'not written'}\n"
        f"Event log: {result.get('log_txt')}"
    )
    output_video = result.get("output_video")
    yield last_preview, "\n".join(logs[-40:]), final_status, str(output_video) if output_video else None


def update_input_visibility(mode: str):
    return (
        gr.update(visible=mode == "Upload probe video"),
        gr.update(visible=mode == "Default probe video"),
        gr.update(visible=mode == "Webcam chunk mode"),
        gr.update(
            visible=mode == "Webcam chunk mode",
            value=(
                "Webcam chunk mode is the reliable Gradio fallback.\n\n"
                "1. Record a short webcam clip, ideally 5 to 10 seconds.\n"
                "2. Stop the recording.\n"
                "3. Press Run V5 Demo.\n\n"
                "This is chunk-based, not true live streaming, but it is much more stable on a remote server."
            ),
        ),
    )


def build_demo():
    with gr.Blocks(title="V5 Chunked Gait Demo") as demo:
        gr.Markdown(
            f"""
            # V5 Chunked Gait Demo

            This is the practical, reliable demo version.

            It supports:
            - uploaded probe video
            - default probe video
            - webcam chunk mode

            Webcam chunk mode is intentionally **record first, then process** because the earlier
            true-stream browser webcam path did not reliably deliver frames to the backend on this setup.

            Matching mode: **closed-set best cosine only**. No threshold, no margin.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_mode = gr.Radio(
                    ["Default probe video", "Upload probe video", "Webcam chunk mode"],
                    value="Default probe video",
                    label="Input mode",
                )
                uploaded_probe = gr.File(label="Upload probe video", file_types=["video"], visible=False)
                default_note = gr.Markdown(
                    f"Default probe: `{DEFAULT_PROBE}` ({'found' if DEFAULT_PROBE.exists() else 'not found'})",
                    visible=True,
                )
                webcam_chunk = gr.Video(
                    sources=["webcam", "upload"],
                    label="Record webcam chunk or upload a webcam-style clip",
                    include_audio=False,
                    max_length=30,
                    height=360,
                    visible=False,
                )
                webcam_help = gr.Markdown(visible=False)
                run_button = gr.Button("Run V5 Demo", variant="primary")
                model_key = gr.Dropdown(
                    ["grew_gaitbase", "grew_gaitgl", "current_gaitbase"],
                    value="grew_gaitbase",
                    label="Gait model",
                )
                compress_input = gr.Checkbox(value=True, label="Prepare input video to fixed low resolution/FPS first")
                compressed_max_side = gr.Dropdown([480, 720, 1080], value=720, label="Prepared input max side")
                process_every_n = gr.Slider(1, 15, value=5, step=1, label="Manual frame stride: process every N frames")
                assigned_process_every_n = gr.Slider(1, 45, value=10, step=1, label="Adaptive stride after visible tracks are assigned")
                work_frame_max_side = gr.Dropdown([360, 480, 640, 720, 1080, 0], value=0, label="Working frame max side, 0 keeps original aspect/size")
                min_detection_score = gr.Slider(0, 0.9, value=0, step=0.05, label="Minimum detection score before tracking")
                output_max_side = gr.Dropdown([360, 480, 640, 720, 1080, 0], value=480, label="Output video max side, 0 keeps original size")
                write_output_video = gr.Checkbox(value=True, label="Write final annotated MP4")
                quiet_speed_mode = gr.Checkbox(value=True, label="Quiet speed mode")
                target_processing_fps = gr.Slider(0, 15, value=0, step=1, label="Target processing FPS, 0 uses manual stride")
                silhouette_every_n_processed = gr.Slider(1, 10, value=1, step=1, label="Run segmentation every N processed frames per unassigned track")
                identity_buffer_frames = gr.Slider(2, 64, value=5, step=1, label="Silhouette buffer frames before identity assignment")
                max_seconds = gr.Slider(0, 180, value=30, step=1, label="Max seconds to process, 0 means full clip")
                preview_every_n = gr.Slider(1, 20, value=5, step=1, label="Update preview every N processed frames")
                display_max_side = gr.Slider(240, 1080, value=720, step=40, label="Preview max side")
                logs = gr.Textbox(label="Run log", lines=18)
            with gr.Column(scale=1):
                preview = gr.Image(label="Delayed recognition preview", height=420)
                status = gr.Textbox(label="Run status", lines=16)
                annotated_video = gr.Video(label="Final annotated video", interactive=False)

        input_mode.change(
            update_input_visibility,
            inputs=[input_mode],
            outputs=[uploaded_probe, default_note, webcam_chunk, webcam_help],
            show_progress="hidden",
        )
        run_button.click(
            run_v5_demo,
            inputs=[
                input_mode,
                uploaded_probe,
                webcam_chunk,
                model_key,
                compress_input,
                compressed_max_side,
                max_seconds,
                process_every_n,
                assigned_process_every_n,
                work_frame_max_side,
                min_detection_score,
                output_max_side,
                write_output_video,
                quiet_speed_mode,
                target_processing_fps,
                silhouette_every_n_processed,
                identity_buffer_frames,
                preview_every_n,
                display_max_side,
            ],
            outputs=[preview, logs, status, annotated_video],
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue(max_size=8).launch(server_name="0.0.0.0", server_port=7866, share=True)
