import os
import sys
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


def run_buffered_app(
    input_mode: str,
    uploaded_probe,
    model_key: str,
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
    max_seconds: float,
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

    if input_mode == "Default probe video":
        probe_video = DEFAULT_PROBE
    else:
        probe_video = resolve_gradio_path(uploaded_probe)
    if probe_video is None:
        raise gr.Error("Upload mode is selected. Please upload a probe video or choose Default probe video.")
    if not probe_video.exists():
        raise gr.Error(f"Probe video does not exist: {probe_video}")

    if target_processing_fps and target_processing_fps > 0:
        cap = cv2.VideoCapture(str(probe_video))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        process_every_n = max(1, int(round(input_fps / target_processing_fps)))
    else:
        process_every_n = max(1, int(process_every_n))

    preview_every_n = max(1, int(preview_every_n))
    display_max_side = max(240, int(display_max_side))
    logs = [
        "Buffered live gait recognition started.",
        f"Probe: {probe_video}",
        f"Gallery cache: {gallery_path}",
        f"Model: {model_key}",
        f"Process every N frames: {process_every_n}",
        f"Assigned-track process every N frames: {assigned_process_every_n}",
        "Detector input size: original demo default",
        f"Working frame max side: {work_frame_max_side}",
        f"Minimum detection score: {min_detection_score}",
        f"Output max side: {output_max_side}",
        f"Write output video: {write_output_video}",
        f"Quiet speed mode: {quiet_speed_mode}",
        f"Target processing FPS: {target_processing_fps if target_processing_fps > 0 else 'manual stride'}",
        f"Identity buffer frames: {identity_buffer_frames}",
        f"Max probe seconds: {max_seconds if max_seconds > 0 else 'full video'}",
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
    ]

    yield None, "\n".join(logs), "Starting...", None

    gen = run_buffered_live_probe(
        video_path=probe_video,
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

    last_status = "Running..."
    last_preview = None
    processed = 0
    try:
        while True:
            update = next(gen)
            processed = int(update["processed"])
            counts = update["counts"]
            video_seconds = float(update.get("video_seconds", 0.0))
            elapsed_seconds = float(update.get("elapsed_seconds", 0.0))
            realtime_factor = float(update.get("realtime_factor", 0.0))
            last_status = (
                f"Running\n"
                f"Processed frames: {processed}\n"
                f"Frame id: {update['frame_id']}\n"
                f"Current stride: {update.get('current_stride')} "
                f"(dense={update.get('dense_stride')}, assigned={update.get('assigned_stride')})\n"
                f"Video time processed: {video_seconds:.2f}s\n"
                f"Wall time elapsed: {elapsed_seconds:.2f}s\n"
                f"Realtime factor: {realtime_factor:.2f}x\n"
                f"Active tracks: {update['active_count']}\n"
                f"Assigned tracks: {update['assigned_count']}\n"
                f"Counts: pritom={counts.get('pritom', 0)} | coco={counts.get('coco', 0)}\n"
                f"Run folder: {update['run_root']}"
            )
            if processed % preview_every_n == 0:
                last_preview = resize_rgb_preview(update["frame"], display_max_side)
                yield last_preview, "\n".join(logs[-20:]), last_status, None
            elif not quiet_speed_mode:
                yield last_preview, "\n".join(logs[-20:]), last_status, None
    except StopIteration as finished:
        result = finished.value or {}

    final_counts = result.get("counts", {})
    final_video_seconds = float(result.get("video_seconds", 0.0))
    final_elapsed_seconds = float(result.get("elapsed_seconds", 0.0))
    final_total_elapsed_seconds = float(result.get("total_elapsed_seconds", final_elapsed_seconds))
    final_ffmpeg_elapsed_seconds = float(result.get("ffmpeg_elapsed_seconds", 0.0))
    final_realtime_factor = float(result.get("realtime_factor", 0.0))
    stage_times = result.get("stage_times", {})
    stage_summary = (
        "Stage seconds: "
        f"detect={float(stage_times.get('detect', 0.0)):.1f}, "
        f"track={float(stage_times.get('track_update', 0.0)):.1f}, "
        f"seg={float(stage_times.get('segmentation', 0.0)):.1f}, "
        f"recog={float(stage_times.get('recognition', 0.0)):.1f}, "
        f"draw/write={float(stage_times.get('draw_write', 0.0)):.1f}, "
        f"ffmpeg={final_ffmpeg_elapsed_seconds:.1f}"
    )
    final_status = (
        f"Finished\n"
        f"Processed frames: {result.get('processed', processed)}\n"
        f"Video time processed: {final_video_seconds:.2f}s\n"
        f"Processing wall time: {final_elapsed_seconds:.2f}s\n"
        f"Total wall time incl. video conversion: {final_total_elapsed_seconds:.2f}s\n"
        f"Realtime factor: {final_realtime_factor:.2f}x\n"
        f"{stage_summary}\n"
        f"Final counts: pritom={final_counts.get('pritom', 0)} | coco={final_counts.get('coco', 0)}\n"
        f"Output folder: {result.get('run_root')}\n"
        f"Annotated video: {result.get('output_video') or 'not written'}\n"
        f"Event log: {result.get('log_txt')}"
    )
    output_video = result.get("output_video")
    yield last_preview, "\n".join(logs[-30:]), final_status, str(output_video) if output_video else None


def build_demo():
    def update_input_visibility(mode: str):
        return (
            gr.update(visible=mode == "Upload probe video"),
            gr.update(visible=mode == "Default probe video"),
        )

    with gr.Blocks(title="Buffered Live Gait Recognition") as demo:
        gr.Markdown(
            f"""
            # Buffered Live Gait Recognition

            This prototype simulates live processing from a probe video:
            detection/tracking starts immediately, silhouettes are buffered per track,
            and identity is assigned once the buffer is full.

            Gallery cache is selected per model. GaitBase and GaitGL cannot share gallery embeddings.

            Matching mode: **closed-set best cosine only**. No threshold, no margin.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_mode = gr.Radio(
                    ["Upload probe video", "Default probe video"],
                    value="Default probe video",
                    label="Input mode",
                )
                uploaded_probe = gr.File(label="Upload probe video", file_types=["video"], visible=False)
                default_note = gr.Markdown(
                    f"Default probe: `{DEFAULT_PROBE}` ({'found' if DEFAULT_PROBE.exists() else 'not found'})\n\n"
                    f"Low-res preferred path: `{LOWRES_DEFAULT_PROBE}`\n\n"
                    f"Original fallback path: `{ORIGINAL_DEFAULT_PROBE}`",
                    visible=False,
                )
                run_button = gr.Button("Run Buffered Live Probe", variant="primary")
                model_key = gr.Dropdown(
                    ["grew_gaitbase", "grew_gaitgl", "current_gaitbase"],
                    value="grew_gaitbase",
                    label="Gait model",
                )
                process_every_n = gr.Slider(
                    1,
                    15,
                    value=5,
                    step=1,
                    label="Manual frame stride: process every N frames",
                )
                assigned_process_every_n = gr.Slider(
                    1,
                    45,
                    value=10,
                    step=1,
                    label="Adaptive stride after visible tracks are assigned",
                )
                work_frame_max_side = gr.Dropdown(
                    [360, 480, 640, 720, 1080, 0],
                    value=0,
                    label="Working frame max side, 0 keeps original aspect/size",
                )
                min_detection_score = gr.Slider(
                    0,
                    0.9,
                    value=0,
                    step=0.05,
                    label="Minimum detection score before tracking",
                )
                output_max_side = gr.Dropdown(
                    [360, 480, 640, 720, 1080, 0],
                    value=480,
                    label="Output video max side, 0 keeps original size",
                )
                write_output_video = gr.Checkbox(
                    value=True,
                    label="Write final annotated MP4, slower; off is better for realtime testing",
                )
                quiet_speed_mode = gr.Checkbox(
                    value=True,
                    label="Quiet speed mode, fewer terminal/UI log updates",
                )
                target_processing_fps = gr.Slider(
                    0,
                    15,
                    value=0,
                    step=1,
                    label="Target processing FPS, 0 uses manual stride",
                )
                silhouette_every_n_processed = gr.Slider(
                    1,
                    10,
                    value=1,
                    step=1,
                    label="Run segmentation every N processed frames per unassigned track",
                )
                identity_buffer_frames = gr.Slider(
                    2,
                    64,
                    value=5,
                    step=1,
                    label="Silhouette buffer frames before identity assignment",
                )
                max_seconds = gr.Slider(
                    0,
                    180,
                    value=30,
                    step=1,
                    label="Max probe seconds, 0 means full video",
                )
                preview_every_n = gr.Slider(
                    1,
                    50,
                    value=5,
                    step=1,
                    label="Update preview every N processed frames",
                )
                display_max_side = gr.Slider(
                    360,
                    1080,
                    value=720,
                    step=60,
                    label="Preview max side",
                )
                live_log = gr.Textbox(label="Live log", lines=16)

            with gr.Column(scale=1):
                preview = gr.Image(label="Buffered live preview", height=520)
                status = gr.Textbox(label="Run status", lines=8)
                annotated_video = gr.Video(label="Final annotated video", interactive=False)

        input_mode.change(update_input_visibility, inputs=[input_mode], outputs=[uploaded_probe, default_note])
        run_button.click(
            run_buffered_app,
            inputs=[
                input_mode,
                uploaded_probe,
                model_key,
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
                max_seconds,
                preview_every_n,
                display_max_side,
            ],
            outputs=[preview, live_log, status, annotated_video],
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue(max_size=1).launch(server_name="0.0.0.0", server_port=7863, share=True)
