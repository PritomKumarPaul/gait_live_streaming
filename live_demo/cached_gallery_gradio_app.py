import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr


ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = ROOT / "live_demo"
DEFAULT_PROBE = ROOT / "clean_demo_v2" / "probes" / "test1probe.mp4"
DEFAULT_GALLERY = LIVE_ROOT / "cache" / "pritom_coco_gallery.npz"
DEFAULT_GALLERY_META = LIVE_ROOT / "cache" / "pritom_coco_gallery.json"

GPU_ID = os.environ.get("LIVE_DEMO_GPU_ID")
if GPU_ID not in (None, "", "cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
elif GPU_ID in ("cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(LIVE_ROOT))

from run_cached_gallery_probe import run_cached_gallery_probe  # noqa: E402


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


def format_summary(payload, run_root: Path, annotated_video: Path) -> str:
    counts = payload["summary"]["counts"]
    lines = [
        "Cached Pritom/Coco Gallery Probe Result",
        "",
        f"Mode: {payload['mode']}",
        f"Model: {payload['model']}",
        f"Gallery: {payload['gallery_npz']}",
        f"Probe: {payload['probe_video']}",
        f"Run folder: {run_root}",
        f"Annotated video: {annotated_video}",
        "",
        "Counts",
        f"pritom: {counts.get('pritom', 0)}",
        f"coco: {counts.get('coco', 0)}",
        "",
        f"Processed entries: {payload['summary']['processed_entries']}",
        f"Skipped entries: {payload['summary']['skipped_entries']}",
        "",
        "Per-entry decisions",
    ]
    for entry in payload["entries"]:
        if entry["status"] != "ok":
            lines.append(f"{entry['entry_key']}: skipped {entry['status']}")
            continue
        lines.append(
            f"{entry['entry_key']}: {entry['assigned_identity']} "
            f"best={entry['best_score']:.4f} "
            f"second={entry['second_identity']}:{entry['second_score']:.4f}"
        )
    return "\n".join(lines)


def run_probe(input_mode: str, uploaded_file, model_key: str, min_frames: int):
    if not DEFAULT_GALLERY.exists() or not DEFAULT_GALLERY_META.exists():
        raise gr.Error(
            "Gallery cache is missing. Build it first on the server with:\n"
            "CUDA_VISIBLE_DEVICES=<gpu> python live_demo/build_pritom_coco_gallery.py"
        )

    if input_mode == "Default probe video":
        probe_video = DEFAULT_PROBE
    else:
        probe_video = resolve_gradio_path(uploaded_file)
    if probe_video is None:
        raise gr.Error("Please upload a probe video or choose Default probe video.")
    if not probe_video.exists():
        raise gr.Error(f"Probe video does not exist: {probe_video}")

    result = run_cached_gallery_probe(
        probe_video=probe_video,
        gallery_npz=DEFAULT_GALLERY,
        gallery_meta=DEFAULT_GALLERY_META,
        model_key=model_key,
        min_frames=int(min_frames),
        force_rebuild_gallery=False,
    )
    summary = format_summary(result["payload"], result["run_root"], result["annotated_video"])
    return summary, str(result["annotated_video"]), str(result["result_json"]), str(result["log_txt"])


def build_demo():
    def update_input_visibility(mode: str):
        return (
            gr.update(visible=mode == "Upload probe video"),
            gr.update(visible=mode == "Default probe video"),
        )

    with gr.Blocks(title="Cached Pritom/Coco Probe Recognition") as demo:
        gr.Markdown(
            f"""
            # Cached Pritom/Coco Probe Recognition

            This app uses the precomputed gallery cache:

            `{DEFAULT_GALLERY}`

            It does **closed-set best-cosine matching only**:
            every valid probe entry is assigned to either `pritom` or `coco`.
            There is **no threshold** and **no margin**.
            """
        )
        with gr.Row():
            with gr.Column():
                input_mode = gr.Radio(
                    ["Upload probe video", "Default probe video"],
                    value="Upload probe video",
                    label="Input mode",
                )
                uploaded_file = gr.File(label="Upload probe video", file_types=["video"], visible=True)
                default_note = gr.Markdown(
                    f"Default probe: `{DEFAULT_PROBE}` ({'found' if DEFAULT_PROBE.exists() else 'not found'})",
                    visible=False,
                )
                model_key = gr.Dropdown(
                    ["grew_gaitbase", "grew_gaitgl", "current_gaitbase"],
                    value="grew_gaitbase",
                    label="Model",
                )
                min_frames = gr.Slider(5, 80, value=20, step=1, label="Minimum silhouette frames per entry")
                run_button = gr.Button("Run Cached Gallery Probe", variant="primary")
            with gr.Column():
                summary = gr.Textbox(label="Summary", lines=18)
                annotated_video = gr.Video(label="Annotated output video", interactive=False)
                result_json = gr.File(label="Result JSON")
                log_txt = gr.File(label="Event log TXT")

        input_mode.change(update_input_visibility, inputs=[input_mode], outputs=[uploaded_file, default_note])
        run_button.click(
            run_probe,
            inputs=[input_mode, uploaded_file, model_key, min_frames],
            outputs=[summary, annotated_video, result_json, log_txt],
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue(max_size=1).launch(server_name="0.0.0.0", server_port=7862, share=True)
