import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Generator, Optional, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT = ROOT / "OpenGait"
DEMO_LIBS = OPENGAIT / "demo" / "libs"
PADDLE_LIBS = DEMO_LIBS / "paddle"
LIVE_ROOT = ROOT / "live_demo"
OUTPUT_ROOT = LIVE_ROOT / "output"
DEFAULT_PROBE = ROOT / "clean_demo_v2" / "probes" / "test1probe.mp4"
DEFAULT_GALLERY = LIVE_ROOT / "cache" / "pritom_coco_gallery.npz"


def resolve_path(path: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def color_for_label(label: str) -> Tuple[int, int, int]:
    return {
        "pritom": (0, 220, 0),
        "coco": (255, 140, 0),
        "collecting": (255, 255, 255),
    }.get(label, (255, 255, 255))


def draw_label(frame, x: int, y: int, text: str, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 12)
    cv2.rectangle(frame, (x, y0), (x + tw + 14, y0 + th + baseline + 12), color, -1)
    cv2.putText(frame, text, (x + 7, y0 + th + 3), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_panel(frame, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.82
    thickness = 2
    line_height = 34
    width = 720
    height = 24 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (18 + width, 18 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (34, 52 + idx * line_height), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


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
        print("[warn] ffmpeg conversion failed; keeping raw mp4", flush=True)
        return input_video
    return output_video


def save_track_silhouette(
    frame,
    tlwh,
    video_width: int,
    video_height: int,
    track_id: int,
    frame_id: int,
    predictor,
    seg_config: Path,
    sil_root: Path,
) -> Optional[Path]:
    frame_height, frame_width = frame.shape[:2]
    video_width = min(int(video_width), int(frame_width))
    video_height = min(int(video_height), int(frame_height))
    x, y, width, height = tlwh
    x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    x1_new = max(0, int(x1 - 0.1 * w))
    x2_new = min(video_width, int(x2 + 0.1 * w))
    y1_new = max(0, int(y1 - 0.1 * h))
    y2_new = min(video_height, int(y2 + 0.1 * h))
    crop = frame[y1_new:y2_new, x1_new:x2_new, :]
    if crop.size == 0:
        return None

    new_w = x2_new - x1_new
    new_h = y2_new - y1_new
    side = max(new_w, new_h)
    padded = np.full((side, side, 3), 255, dtype=np.uint8)
    x_pad = int((side - new_w) / 2)
    y_pad = int((side - new_h) / 2)
    padded[y_pad:y_pad + new_h, x_pad:x_pad + new_w, :] = crop
    resized = cv2.resize(padded, (192, 192))

    bg_img = 255 * np.ones(resized.shape)
    _, out_mask = predictor.run(resized, bg_img)
    out_mask = np.where(out_mask < 80, 0, 255).astype(np.uint8)

    track_dir = sil_root / "live_probe" / f"{track_id:03d}" / "undefined"
    track_dir.mkdir(parents=True, exist_ok=True)
    out_path = track_dir / f"{track_id:03d}-{frame_id:06d}.png"
    cv2.imwrite(str(out_path), out_mask)
    return out_path


def track_silhouettes_to_input(track_dir: Path):
    """Build one OpenGait input tuple from a single live track directory."""
    image_paths = sorted(track_dir.glob("*.png"))
    frames = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            frames.append(image.astype("uint8"))
    if not frames:
        return None
    sequence = np.asarray(frames)
    track_id = track_dir.parent.name
    view = track_dir.name
    return ([[sequence]], ["live_probe"], [track_id], [view], np.array([[len(frames)]]))


def filter_detections_by_score(outputs, min_score: float):
    if min_score <= 0 or outputs is None or outputs[0] is None:
        return outputs
    detections = outputs[0]
    if detections.shape[1] == 5:
        scores = detections[:, 4]
    else:
        scores = detections[:, 4] * detections[:, 5]
    keep = scores >= min_score
    outputs[0] = detections[keep]
    if outputs[0].shape[0] == 0:
        outputs[0] = None
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Buffered live-style cached-gallery gait probe.")
    parser.add_argument("--video", default=str(DEFAULT_PROBE))
    parser.add_argument("--gallery-npz", default=str(DEFAULT_GALLERY))
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--process-every-n", type=int, default=2)
    parser.add_argument("--assigned-process-every-n", type=int, default=6)
    parser.add_argument("--detector-input-size", type=int, default=0, help="0 uses the original demo detector size.")
    parser.add_argument("--work-frame-max-side", type=int, default=0)
    parser.add_argument("--min-detection-score", type=float, default=0.0)
    parser.add_argument("--output-max-side", type=int, default=720)
    parser.add_argument("--write-output-video", action="store_true")
    parser.add_argument("--silhouette-every-n-processed", type=int, default=1)
    parser.add_argument("--identity-buffer-frames", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-frame terminal/progress logging.")
    return parser.parse_args()


def run_buffered_live_probe(
    video_path: Path,
    gallery_path: Path = DEFAULT_GALLERY,
    model_key: str = "grew_gaitbase",
    process_every_n: int = 2,
    assigned_process_every_n: int = 6,
    detector_input_size: int = 0,
    work_frame_max_side: int = 0,
    min_detection_score: float = 0.0,
    output_max_side: int = 720,
    write_output_video: bool = True,
    silhouette_every_n_processed: int = 1,
    identity_buffer_frames: int = 8,
    max_seconds: float = 30.0,
    log_every: int = 5,
    progress_callback: Optional[Callable[[str], None]] = None,
    quiet: bool = False,
) -> Generator[Dict, None, Dict]:
    video_path = Path(video_path).resolve()
    gallery_path = Path(gallery_path).resolve()
    if not gallery_path.exists():
        raise FileNotFoundError(f"Missing gallery cache: {gallery_path}. Run live_demo/build_pritom_coco_gallery.py first.")

    sys.path.insert(0, str(DEMO_LIBS))
    sys.path.insert(0, str(PADDLE_LIBS))
    sys.path.insert(0, str(OPENGAIT))
    sys.path.insert(0, str(OPENGAIT / "tools"))
    os.chdir(OPENGAIT)

    import torch
    from demo.libs.track import exp, model as det_model, track_cfgs
    from infer import Predictor_opengait
    from probe_only_entry_analysis import build_model, cosine_similarity, extract_embedding
    from tracker.byte_tracker import BYTETracker
    from tracking_utils.predictor import Predictor
    from tracking_utils.timer import Timer

    def emit(message: str, important: bool = False) -> None:
        if not quiet or important:
            print(message, flush=True)
        if progress_callback is not None and (not quiet or important):
            progress_callback(message)

    emit(f"[buffered-live] cuda={torch.cuda.is_available()} visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", important=True)
    emit(f"[buffered-live] video={video_path}", important=True)
    emit(f"[buffered-live] gallery={gallery_path}", important=True)
    emit("[buffered-live] Loading cached gallery and gait model.", important=True)

    gallery_data = np.load(gallery_path)
    gallery_embeddings = gallery_data["embeddings"].astype(np.float32)
    gallery_labels = gallery_data["labels"].astype(str)
    metadata_path = gallery_path.with_suffix(".json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
        gallery_model_key = metadata.get("model_key")
        if gallery_model_key and gallery_model_key != model_key:
            raise RuntimeError(
                f"Gallery cache was built for {gallery_model_key}, but the selected model is {model_key}. "
                f"Build a matching cache with: python live_demo/build_pritom_coco_gallery.py --model {model_key}"
            )
    gait_model, _ = build_model(model_key)

    seg_config = OPENGAIT / "demo" / "checkpoints" / "seg_model" / "human_pp_humansegv2_mobile_192x192_inference_model_with_softmax" / "deploy.yaml"
    seg_predictor = Predictor_opengait(str(seg_config))

    run_root = OUTPUT_ROOT / f"buffered_live_{time.strftime('%Y%m%d_%H%M%S')}"
    sil_root = run_root / "silhouettes"
    raw_video = run_root / "buffered_live_raw.mp4"
    final_video = run_root / "buffered_live.mp4"
    log_txt = run_root / "events_log.txt"
    run_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if track_cfgs["device"] == "gpu" and torch.cuda.is_available() else "cpu")
    predictor = Predictor(det_model, exp, None, None, device, device.type == "cuda")
    if int(detector_input_size) > 0:
        detector_input_size = max(224, int(detector_input_size))
        predictor.test_size = (detector_input_size, detector_input_size)
    detector_test_size = predictor.test_size
    tracker = BYTETracker(frame_rate=30)
    timer = Timer()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    work_frame_max_side = max(0, int(work_frame_max_side))
    work_scale = 1.0
    if work_frame_max_side > 0:
        work_scale = min(1.0, work_frame_max_side / max(width, height))
    work_width = max(1, int(round(width * work_scale)))
    work_height = max(1, int(round(height * work_scale)))
    output_max_side = max(0, int(output_max_side))
    output_scale = 1.0
    if output_max_side > 0:
        output_scale = min(1.0, output_max_side / max(work_width, work_height))
    output_width = max(1, int(round(work_width * output_scale)))
    output_height = max(1, int(round(work_height * output_scale)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames
    if max_seconds and max_seconds > 0:
        max_frames = min(max_frames or int(max_seconds * input_fps), int(max_seconds * input_fps))
    process_every_n = max(1, int(process_every_n))
    assigned_process_every_n = max(process_every_n, int(assigned_process_every_n))
    min_detection_score = max(0.0, float(min_detection_score))
    silhouette_every_n_processed = max(1, int(silhouette_every_n_processed))
    identity_buffer_frames = max(1, int(identity_buffer_frames))
    output_fps = max(1.0, input_fps / process_every_n)
    writer = None
    if write_output_video:
        writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (output_width, output_height))
    run_settings = {
        "video_path": str(video_path),
        "gallery_path": str(gallery_path),
        "model_key": model_key,
        "input_fps": input_fps,
        "input_width": width,
        "input_height": height,
        "work_frame_max_side": work_frame_max_side,
        "work_width": work_width,
        "work_height": work_height,
        "work_scale": work_scale,
        "detector_input_size": detector_input_size,
        "detector_test_size": detector_test_size,
        "min_detection_score": min_detection_score,
        "output_width": output_width,
        "output_height": output_height,
        "output_max_side": output_max_side,
        "output_scale": output_scale,
        "write_output_video": write_output_video,
        "quiet": quiet,
        "input_total_frames": total_frames,
        "max_seconds": max_seconds,
        "max_frames": max_frames,
        "dense_process_every_n": process_every_n,
        "assigned_process_every_n": assigned_process_every_n,
        "silhouette_every_n_processed": silhouette_every_n_processed,
        "identity_buffer_frames": identity_buffer_frames,
        "output_fps": output_fps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }

    first_track_offset = None
    processed = 0
    frame_id = 0
    track_sil_counts: Dict[int, int] = {}
    track_labels: Dict[int, str] = {}
    track_scores: Dict[int, float] = {}
    event_lines = ["Run settings: " + json.dumps(run_settings, sort_keys=True)]
    start_time = time.perf_counter()
    stage_times = {
        "read_skip": 0.0,
        "detect": 0.0,
        "track_update": 0.0,
        "segmentation": 0.0,
        "recognition": 0.0,
        "draw_write": 0.0,
        "ffmpeg_convert": 0.0,
    }
    stage_counts = {
        "processed_frames": 0,
        "detection_calls": 0,
        "track_update_calls": 0,
        "segmentation_calls": 0,
        "recognition_calls": 0,
        "written_frames": 0,
    }
    all_visible_assigned = False

    emit("[buffered-live] Starting frame loop. Detection runs immediately; identity appears after buffer fills.", important=True)
    emit("Run settings: " + json.dumps(run_settings, sort_keys=True), important=False)
    while True:
        current_stride = assigned_process_every_n if all_visible_assigned else process_every_n
        read_start = time.perf_counter()
        ok = cap.grab()
        should_process = ok and frame_id % current_stride == 0
        if should_process:
            ok, frame = cap.retrieve()
        stage_times["read_skip"] += time.perf_counter() - read_start
        if not ok:
            break
        if max_frames and frame_id >= max_frames:
            break
        if not should_process:
            frame_id += 1
            continue
        if work_scale < 1.0:
            frame = cv2.resize(frame, (work_width, work_height), interpolation=cv2.INTER_AREA)

        detect_start = time.perf_counter()
        outputs, img_info = predictor.inference(frame, timer)
        outputs = filter_detections_by_score(outputs, min_detection_score)
        stage_times["detect"] += time.perf_counter() - detect_start
        stage_counts["detection_calls"] += 1
        active = set()
        unassigned_active = set()
        if outputs[0] is not None:
            track_start = time.perf_counter()
            online_targets = tracker.update(outputs[0], [img_info["height"], img_info["width"]], detector_test_size)
            stage_times["track_update"] += time.perf_counter() - track_start
            stage_counts["track_update_calls"] += 1
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

                if track_id not in track_labels and processed % silhouette_every_n_processed == 0:
                    seg_start = time.perf_counter()
                    saved = save_track_silhouette(
                        frame=frame,
                        tlwh=tlwh,
                        video_width=frame.shape[1],
                        video_height=frame.shape[0],
                        track_id=track_id,
                        frame_id=frame_id,
                        predictor=seg_predictor,
                        seg_config=seg_config,
                        sil_root=sil_root,
                    )
                    stage_times["segmentation"] += time.perf_counter() - seg_start
                    stage_counts["segmentation_calls"] += 1
                    if saved is not None:
                        track_sil_counts[track_id] = track_sil_counts.get(track_id, 0) + 1
                        if progress_callback and not quiet:
                            progress_callback(
                                f"frame={frame_id}: track {track_id:03d} buffered "
                                f"{track_sil_counts[track_id]}/{identity_buffer_frames} silhouettes"
                            )

                    if track_sil_counts.get(track_id, 0) >= identity_buffer_frames:
                        track_input = track_silhouettes_to_input(
                            sil_root / "live_probe" / f"{track_id:03d}" / "undefined"
                        )
                        if track_input is not None:
                            rec_start = time.perf_counter()
                            embedding = extract_embedding(gait_model, track_input)
                            if embedding.shape != gallery_embeddings[0].shape:
                                raise RuntimeError(
                                    f"Probe embedding shape {embedding.shape} does not match gallery shape "
                                    f"{gallery_embeddings[0].shape}. Build a gallery cache for {model_key}."
                                )
                            scores = [
                                (str(label), cosine_similarity(embedding, gallery_embedding))
                                for label, gallery_embedding in zip(gallery_labels, gallery_embeddings)
                            ]
                            stage_times["recognition"] += time.perf_counter() - rec_start
                            stage_counts["recognition_calls"] += 1
                            best_label, best_score = sorted(scores, key=lambda item: item[1], reverse=True)[0]
                            track_labels[track_id] = best_label
                            track_scores[track_id] = float(best_score)
                            score_text = ", ".join(f"{label}={score:.4f}" for label, score in scores)
                            event = (
                                f"t={frame_id / input_fps:.2f}s frame={frame_id}: "
                                f"track {track_id:03d} assigned {best_label} score={best_score:.4f} "
                                f"all_scores=[{score_text}]"
                            )
                            event_lines.append(event)
                            emit(f"[buffered-live] {event}", important=False)
                if track_id not in track_labels:
                    unassigned_active.add(track_id)

                x, y, w, h = tlwh
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                label = track_labels.get(track_id)
                color = color_for_label(label or "collecting")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                if label is not None:
                    draw_label(frame, x1, max(0, y1 - 4), label, color)
            timer.toc()

        previous_all_visible_assigned = all_visible_assigned
        all_visible_assigned = bool(active) and not unassigned_active
        if all_visible_assigned and not previous_all_visible_assigned and assigned_process_every_n > process_every_n:
            event = (
                f"t={frame_id / input_fps:.2f}s frame={frame_id}: all visible tracks assigned; "
                f"switching stride {process_every_n}->{assigned_process_every_n}"
            )
            event_lines.append(event)
            emit(f"[buffered-live] {event}", important=False)
        elif unassigned_active and previous_all_visible_assigned:
            event = (
                f"t={frame_id / input_fps:.2f}s frame={frame_id}: new/unassigned track visible; "
                f"switching stride {assigned_process_every_n}->{process_every_n}"
            )
            event_lines.append(event)
            emit(f"[buffered-live] {event}", important=False)

        draw_start = time.perf_counter()
        counts = {label: sum(1 for value in track_labels.values() if value == label) for label in ["pritom", "coco"]}
        elapsed_seconds = time.perf_counter() - start_time
        video_seconds = frame_id / input_fps if input_fps else 0.0
        realtime_factor = video_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
        draw_panel(
            frame,
            [
                f"Counts: pritom={counts['pritom']} | coco={counts['coco']}",
            ],
        )
        if writer is not None:
            frame_to_write = frame
            if output_scale < 1.0:
                frame_to_write = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            writer.write(frame_to_write)
            stage_counts["written_frames"] += 1
        stage_times["draw_write"] += time.perf_counter() - draw_start
        processed += 1
        stage_counts["processed_frames"] = processed
        status = (
            f"[buffered-live] processed={processed} frame={frame_id} active={len(active)} "
            f"assigned={len(track_labels)} buffers={track_sil_counts}"
        )
        if log_every > 0 and processed % max(1, log_every) == 0:
            emit(status, important=False)
        yield {
            "frame": frame.copy(),
            "status": status,
            "processed": processed,
            "frame_id": frame_id,
            "current_stride": current_stride,
            "dense_stride": process_every_n,
            "assigned_stride": assigned_process_every_n,
            "video_seconds": video_seconds,
            "elapsed_seconds": elapsed_seconds,
            "realtime_factor": realtime_factor,
            "active_count": len(active),
            "assigned_count": len(track_labels),
            "counts": counts,
            "stage_times": dict(stage_times),
            "stage_counts": dict(stage_counts),
            "run_root": run_root,
        }
        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
    processing_elapsed_seconds = time.perf_counter() - start_time
    video_seconds = min(frame_id, max_frames or frame_id) / input_fps if input_fps else 0.0
    realtime_factor = video_seconds / processing_elapsed_seconds if processing_elapsed_seconds > 0 else 0.0
    playable = None
    if write_output_video and raw_video.exists():
        convert_start = time.perf_counter()
        playable = make_browser_playable_video(raw_video, final_video)
        stage_times["ffmpeg_convert"] += time.perf_counter() - convert_start
    total_elapsed_seconds = time.perf_counter() - start_time
    log_txt.write_text("\n".join(event_lines or ["No identities assigned."]))
    final_counts = {label: sum(1 for value in track_labels.values() if value == label) for label in ["pritom", "coco"]}
    metrics = {
        "settings": run_settings,
        "processed_frames": processed,
        "video_seconds": video_seconds,
        "elapsed_seconds": processing_elapsed_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
        "ffmpeg_elapsed_seconds": stage_times["ffmpeg_convert"],
        "realtime_factor": realtime_factor,
        "dense_stride": process_every_n,
        "assigned_stride": assigned_process_every_n,
        "identity_buffer_frames": identity_buffer_frames,
        "write_output_video": write_output_video,
        "stage_times": stage_times,
        "stage_counts": stage_counts,
        "stage_time_percentages": {
            key: (value / total_elapsed_seconds * 100.0 if total_elapsed_seconds > 0 else 0.0)
            for key, value in stage_times.items()
        },
        "track_labels": {str(key): value for key, value in track_labels.items()},
        "track_scores": {str(key): value for key, value in track_scores.items()},
        "counts": final_counts,
    }
    metrics_json = run_root / "metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2))
    emit("[buffered-live] finished", important=True)
    emit(f"[buffered-live] output_video={playable}", important=True)
    emit(f"[buffered-live] log={log_txt}", important=True)
    emit(f"[buffered-live] output_folder={run_root}", important=True)
    return {
        "output_video": playable,
        "log_txt": log_txt,
        "metrics_json": metrics_json,
        "run_root": run_root,
        "processed": processed,
        "video_seconds": video_seconds,
        "elapsed_seconds": processing_elapsed_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
        "ffmpeg_elapsed_seconds": stage_times["ffmpeg_convert"],
        "realtime_factor": realtime_factor,
        "stage_times": stage_times,
        "stage_counts": stage_counts,
        "counts": final_counts,
        "track_labels": track_labels,
        "track_scores": track_scores,
    }


def main():
    args = parse_args()
    if args.gpu_id:
        if args.gpu_id.lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    video_path = resolve_path(args.video)
    gallery_path = resolve_path(args.gallery_npz)
    gen = run_buffered_live_probe(
        video_path=video_path,
        gallery_path=gallery_path,
        model_key=args.model,
        process_every_n=args.process_every_n,
        assigned_process_every_n=args.assigned_process_every_n,
        detector_input_size=args.detector_input_size,
        work_frame_max_side=args.work_frame_max_side,
        min_detection_score=args.min_detection_score,
        output_max_side=args.output_max_side,
        write_output_video=args.write_output_video,
        silhouette_every_n_processed=args.silhouette_every_n_processed,
        identity_buffer_frames=args.identity_buffer_frames,
        max_seconds=args.max_seconds,
        log_every=args.log_every,
        quiet=args.quiet,
    )
    try:
        while True:
            next(gen)
    except StopIteration:
        pass


if __name__ == "__main__":
    main()
