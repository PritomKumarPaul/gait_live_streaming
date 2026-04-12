import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import gradio as gr
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = ROOT / "clean_demo_v2"
OPENGAIT_TOOLS = ROOT / "OpenGait" / "tools"
sys.path.insert(0, str(OPENGAIT_TOOLS))

from probe_only_entry_analysis import analyze_video, build_model, cosine_similarity, log  # noqa: E402


def copy_upload(src, dst: Path):
    src_path = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dst)
    return dst


def timestamped_run_root() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = CLEAN_ROOT / "output" / f"gradio_multi_gallery_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def load_tracking_file(path: Path) -> Dict[int, List[Dict]]:
    frame_tracks: Dict[int, List[Dict]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found: {path}")
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame_id = int(float(parts[0]))
        track_id = int(float(parts[1]))
        frame_tracks.setdefault(frame_id, []).append(
            {
                "track_id": f"{track_id:03d}",
                "bbox": [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])],
            }
        )
    return frame_tracks


def color_for_label(label: str):
    if label == "unknown":
        return (150, 150, 150)
    try:
        idx = int(label.replace("person", ""))
    except ValueError:
        idx = 1
    return ((37 * idx) % 255, (137 * idx) % 255, (211 * idx) % 255)


def draw_label(frame, x: int, y: int, text: str, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 8, y0 + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (x + 4, y0 + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def clip_video(input_path: Path, output_path: Path, max_seconds: float) -> Path:
    if max_seconds <= 0:
        return input_path

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open probe video for clipping: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    if written == 0:
        raise RuntimeError("Probe clipping produced zero frames.")
    return output_path


def render_annotations(
    video_path: Path,
    tracking_txt: Path,
    decisions: List[Dict],
    output_video: Path,
    title: str,
    max_side: int = 720,
    target_fps: float = 24.0,
):
    decision_by_track = {item["track_id"]: item for item in decisions}
    frame_tracks = load_tracking_file(tracking_txt)

    cap = cv2.VideoCapture(str(video_path))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(input_fps / target_fps)))
    fps = input_fps / stride
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    largest_side = max(width, height)
    if largest_side > max_side:
        scale = max_side / largest_side
        out_width = int(width * scale)
        out_height = int(height * scale)
    else:
        out_width = width
        out_height = height
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))

    frame_id = 0
    written_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % stride != 0:
            frame_id += 1
            continue
        cv2.putText(frame, title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        for item in frame_tracks.get(frame_id, []):
            decision = decision_by_track.get(item["track_id"])
            if decision is None:
                continue
            x, y, w, h = item["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            label = decision["assigned_identity"]
            color = color_for_label(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{label} count={decision['display_count']} score={decision['best_score']:.3f}"
            draw_label(frame, x1, max(0, y1 - 4), text, color)
        if (out_width, out_height) != (width, height):
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written_frames += 1
        frame_id += 1

    cap.release()
    writer.release()
    if written_frames == 0:
        raise RuntimeError("Annotated video rendering produced zero frames.")


def make_browser_playable_video(input_video: Path, output_video: Path) -> Path:
    """Convert OpenCV MP4 output to browser-friendly H.264 if ffmpeg is available."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return input_video

    output_video.parent.mkdir(parents=True, exist_ok=True)
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


def build_gallery(model, gallery_files: List[str], run_root: Path, min_frames: int, progress=None):
    gallery = []
    metadata = []
    total = max(len(gallery_files), 1)
    for idx, gallery_file in enumerate(gallery_files, start=1):
        label = f"person{idx}"
        if progress is not None:
            progress(
                0.10 + 0.45 * ((idx - 1) / total),
                desc=f"Building gallery embedding for {label}",
            )
        copied_path = copy_upload(
            gallery_file,
            run_root / "inputs" / "gallery" / f"{label}{Path(gallery_file).suffix or '.mp4'}",
        )
        log(f"[gradio] building gallery for {label}: {copied_path}")
        entries = analyze_video(
            model=model,
            video_path=copied_path,
            work_root=run_root / "work" / "gallery" / label,
            min_frames=min_frames,
        )
        ok_entries = [entry for entry in entries if entry["status"] == "ok"]
        if not ok_entries:
            raise gr.Error(f"No valid walking sequence found in gallery video for {label}.")
        embeddings = np.stack([entry["embedding"].astype(np.float32) for entry in ok_entries], axis=0)
        prototype = embeddings.mean(axis=0)
        gallery.append(
            {
                "label": label,
                "source_file": str(copied_path),
                "embedding": prototype,
                "valid_entries": len(ok_entries),
            }
        )
        metadata.append(
            {
                "label": label,
                "source_file": str(copied_path),
                "valid_entries": len(ok_entries),
                "entry_keys": [entry["entry_key"] for entry in ok_entries],
                "frame_counts": [entry["frame_count"] for entry in ok_entries],
            }
        )
        if progress is not None:
            progress(
                0.10 + 0.45 * (idx / total),
                desc=f"Finished gallery embedding for {label}",
            )
    return gallery, metadata


def score_probe_to_gallery(embedding: np.ndarray, gallery: List[Dict]) -> List[Dict]:
    scores = []
    for item in gallery:
        scores.append(
            {
                "label": item["label"],
                "score": cosine_similarity(embedding, item["embedding"]),
            }
        )
    return sorted(scores, key=lambda item: item["score"], reverse=True)


def assign_probe_entries(
    probe_entries: List[Dict],
    gallery: List[Dict],
    threshold: float,
) -> List[Dict]:
    decisions = []
    counts = Counter()
    closed_set = len(gallery) >= 2

    for entry in sorted(probe_entries, key=lambda item: item["track_id"]):
        if entry["status"] != "ok":
            decisions.append(
                {
                    "entry_key": entry["entry_key"],
                    "video": entry["video"],
                    "track_id": entry["track_id"],
                    "frame_count": entry["frame_count"],
                    "status": entry["status"],
                    "assigned_identity": "skipped",
                }
            )
            continue

        scores = score_probe_to_gallery(entry["embedding"], gallery)
        best = scores[0]
        second = scores[1] if len(scores) > 1 else {"label": "none", "score": -1.0}

        if closed_set:
            assigned = best["label"]
        else:
            assigned = best["label"] if best["score"] >= threshold else "unknown"

        counts[assigned] += 1
        decisions.append(
            {
                "entry_key": entry["entry_key"],
                "video": entry["video"],
                "track_id": entry["track_id"],
                "frame_count": entry["frame_count"],
                "status": "ok",
                "assigned_identity": assigned,
                "display_count": counts[assigned],
                "best_score": float(best["score"]),
                "best_identity": best["label"],
                "second_identity": second["label"],
                "second_score": float(second["score"]),
                "all_scores": {item["label"]: float(item["score"]) for item in scores},
            }
        )
    return decisions


def format_ui_summary(ui_summary: Dict) -> str:
    lines = []
    mode_label = (
        "Closed-set gallery matching"
        if ui_summary["mode"] == "closed_set_best_match"
        else "Single-gallery threshold fallback"
    )
    lines.append("Gait Recognition Result")
    lines.append("=" * 24)
    lines.append(f"Mode: {mode_label}")
    lines.append(f"Model: {ui_summary['model']}")
    lines.append("")
    lines.append("Gallery Mapping")
    for label, filename in ui_summary["gallery_mapping"].items():
        lines.append(f"- {label}: {filename}")
    lines.append("")
    lines.append("Entry Counts")
    for label, count in sorted(ui_summary["counts"].items()):
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.append("Processing Summary")
    lines.append(f"- Processed entries: {ui_summary['processed_entries']}")
    lines.append(f"- Skipped entries: {ui_summary['skipped_entries']}")
    lines.append("")
    lines.append("Output")
    lines.append(f"- Run folder: {ui_summary['run_root']}")
    return "\n".join(lines)


def run_multi_gallery(gallery_files, probe_video, model_name, threshold, min_frames, max_probe_seconds, progress=gr.Progress()):
    if not gallery_files:
        raise gr.Error("Please upload at least one gallery video.")
    if probe_video is None:
        raise gr.Error("Please upload one probe video.")

    run_root = timestamped_run_root()
    progress(0.02, desc="Loading selected gait model")
    model, profile = build_model(model_name)

    progress(0.08, desc="Preparing gallery videos")
    gallery, gallery_metadata = build_gallery(
        model=model,
        gallery_files=gallery_files,
        run_root=run_root,
        min_frames=min_frames,
        progress=progress,
    )

    progress(0.58, desc="Copying probe video")
    probe_path = copy_upload(
        probe_video,
        run_root / "inputs" / "probe" / f"probe{Path(probe_video).suffix or '.mp4'}",
    )
    if max_probe_seconds and max_probe_seconds > 0:
        progress(0.60, desc=f"Clipping probe to first {max_probe_seconds:.1f} seconds")
        probe_path = clip_video(
            probe_path,
            run_root / "inputs" / "probe" / f"probe_first_{int(max_probe_seconds)}s.mp4",
            float(max_probe_seconds),
        )

    log(f"[gradio] analyzing probe: {probe_path}")
    progress(0.62, desc="Tracking, segmenting, and extracting probe embeddings")
    probe_entries = analyze_video(
        model=model,
        video_path=probe_path,
        work_root=run_root / "work" / "probe",
        min_frames=min_frames,
    )
    progress(0.82, desc="Matching probe entries against gallery")
    decisions = assign_probe_entries(probe_entries, gallery, threshold=threshold)
    counts = Counter(
        decision["assigned_identity"]
        for decision in decisions
        if decision["status"] == "ok"
    )

    tracking_txt = run_root / "work" / "probe" / "tracking" / probe_path.stem / f"{probe_path.stem}.txt"
    annotated_video_raw = run_root / "annotated_probe_display_raw.mp4"
    annotated_video = run_root / "annotated_probe_display.mp4"
    mode = "closed_set_best_match" if len(gallery) >= 2 else "target_vs_unknown_threshold"
    title = "Gait recognition: closed set" if len(gallery) >= 2 else f"Gait recognition: threshold={threshold:.3f}"
    progress(0.90, desc="Rendering annotated output video")
    render_annotations(probe_path, tracking_txt, decisions, annotated_video_raw, title)
    progress(0.96, desc="Converting annotated video for browser playback")
    annotated_video = make_browser_playable_video(annotated_video_raw, annotated_video)

    result = {
        "config": {
            "mode": mode,
            "model": profile["display_name"],
            "model_key": model_name,
            "threshold_if_single_gallery": threshold,
            "min_frames": min_frames,
            "max_probe_seconds": max_probe_seconds,
            "run_root": str(run_root),
            "probe_video": str(probe_path),
        },
        "gallery": gallery_metadata,
        "summary": {
            "gallery_identity_count": len(gallery),
            "processed_entries": sum(1 for decision in decisions if decision["status"] == "ok"),
            "skipped_entries": sum(1 for decision in decisions if decision["status"] != "ok"),
            "counts": dict(counts),
        },
        "entries": decisions,
    }

    result_json = run_root / "result.json"
    summary_txt = run_root / "summary.txt"
    result_json.write_text(json.dumps(result, indent=2))
    summary_txt.write_text(
        "\n".join(
            [
                "Generic multi-gallery gait recognition summary",
                f"Mode: {mode}",
                f"Model: {profile['display_name']}",
                f"Counts: {dict(counts)}",
                f"Processed entries: {result['summary']['processed_entries']}",
                f"Skipped entries: {result['summary']['skipped_entries']}",
                f"Run folder: {run_root}",
                f"Annotated video: {annotated_video}",
            ]
        )
    )

    ui_summary = {
        "mode": mode,
        "model": model_name,
        "counts": dict(counts),
        "processed_entries": result["summary"]["processed_entries"],
        "skipped_entries": result["summary"]["skipped_entries"],
        "gallery_mapping": {
            item["label"]: Path(item["source_file"]).name
            for item in gallery_metadata
        },
        "run_root": str(run_root),
        "probe_duration_limit_seconds": max_probe_seconds,
    }
    progress(1.0, desc="Done")
    return format_ui_summary(ui_summary), str(annotated_video), str(result_json), str(summary_txt)


with gr.Blocks(title="OpenGait Gallery/Probe Recognition") as demo:
    gr.Markdown(
        """
        # OpenGait Gallery/Probe Recognition

        Upload **any number of gallery videos** and one probe video.

        The app automatically labels gallery videos by upload order:
        `person1`, `person2`, `person3`, ...

        - If you upload **2 or more gallery videos**, the app runs closed-set recognition: every probe entry is assigned to the most similar gallery person.
        - If you upload **1 gallery video**, the app falls back to target-vs-unknown thresholding using `person1` vs `unknown`.

        The output includes counts and an annotated overlay video.
        """
    )

    with gr.Row():
        model = gr.Dropdown(
            choices=["grew_gaitbase", "grew_gaitgl", "current_gaitbase"],
            value="grew_gaitbase",
            label="Model",
        )
        threshold = gr.Slider(
            0.90,
            0.995,
            value=0.97,
            step=0.001,
            label="Threshold, used only when one gallery video is uploaded",
        )
        min_frames = gr.Slider(
            5,
            80,
            value=20,
            step=1,
            label="Minimum valid silhouette frames per entry",
        )
        max_probe_seconds = gr.Slider(
            0,
            180,
            value=0,
            step=1,
            label="Max probe duration in seconds, 0 means full video",
        )

    gallery_files = gr.File(
        label="Gallery videos, upload one or more",
        file_count="multiple",
        file_types=["video"],
    )
    probe_file = gr.File(label="Probe video", file_types=["video"])
    run_button = gr.Button("Run Recognition", variant="primary")

    summary = gr.Textbox(label="Counts and run summary", lines=18)
    annotated = gr.Video(label="Annotated output video", format="mp4", height=420, width=520, interactive=False)
    result_file = gr.File(label="Download result JSON")
    summary_file = gr.File(label="Download summary TXT")

    run_button.click(
        run_multi_gallery,
        inputs=[gallery_files, probe_file, model, threshold, min_frames, max_probe_seconds],
        outputs=[summary, annotated, result_file, summary_file],
    )


if __name__ == "__main__":
    demo.queue(max_size=1).launch(share=True)
