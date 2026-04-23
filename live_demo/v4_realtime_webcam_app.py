import json
import os
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import gradio as gr
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT = ROOT / "OpenGait"
DEMO_LIBS = OPENGAIT / "demo" / "libs"
PADDLE_LIBS = DEMO_LIBS / "paddle"
LIVE_ROOT = ROOT / "live_demo"
OUTPUT_ROOT = LIVE_ROOT / "output"
DEFAULT_GALLERY = LIVE_ROOT / "cache" / "pritom_coco_gallery.npz"

GPU_ID = os.environ.get("LIVE_DEMO_GPU_ID")
if GPU_ID not in (None, "", "cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
elif GPU_ID in ("cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(LIVE_ROOT))
from run_buffered_live_probe import (  # noqa: E402
    color_for_label,
    draw_label,
    draw_panel,
    filter_detections_by_score,
    save_track_silhouette,
    track_silhouettes_to_input,
)


class WebcamFrameBuffer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames = deque()
        self.running = False
        self.stop_requested = False
        self.started_at: Optional[float] = None
        self.next_index = 0
        self.dropped_frames = 0
        self.max_seconds = 300.0
        self.max_buffer_frames = 900
        self.preview_seen_count = 0
        self.last_preview_at: Optional[float] = None

    def start(self, max_seconds: float, max_buffer_frames: int) -> None:
        with self.lock:
            self.frames.clear()
            self.running = True
            self.stop_requested = False
            self.started_at = time.perf_counter()
            self.next_index = 0
            self.dropped_frames = 0
            self.max_seconds = max(1.0, float(max_seconds))
            self.max_buffer_frames = max(30, int(max_buffer_frames))

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self.stop_requested = True

    def append(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        now = time.perf_counter()
        with self.lock:
            self.preview_seen_count += 1
            self.last_preview_at = now
            if not self.running or self.started_at is None:
                return False, self.idle_status_locked()
            elapsed = now - self.started_at
            if elapsed >= self.max_seconds:
                self.running = False
                self.stop_requested = True
                return False, f"Reached max webcam duration ({self.max_seconds:.0f}s)."
            self.frames.append(
                {
                    "index": self.next_index,
                    "time": elapsed,
                    "frame": frame_bgr,
                }
            )
            self.next_index += 1
            while len(self.frames) > self.max_buffer_frames:
                self.frames.popleft()
                self.dropped_frames += 1
            return True, self.status_locked()

    def pop_oldest(self):
        with self.lock:
            if self.frames:
                return self.frames.popleft()
            return None

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "stop_requested": self.stop_requested,
                "buffered": len(self.frames),
                "captured": self.next_index,
                "dropped": self.dropped_frames,
                "elapsed": (time.perf_counter() - self.started_at) if self.started_at else 0.0,
                "oldest_time": self.frames[0]["time"] if self.frames else None,
                "newest_time": self.frames[-1]["time"] if self.frames else None,
                "preview_seen_count": self.preview_seen_count,
                "last_preview_age": (time.perf_counter() - self.last_preview_at) if self.last_preview_at else None,
            }

    def status_locked(self) -> str:
        elapsed = (time.perf_counter() - self.started_at) if self.started_at else 0.0
        span = 0.0
        if len(self.frames) >= 2:
            span = self.frames[-1]["time"] - self.frames[0]["time"]
        return (
            f"captured={self.next_index} buffered={len(self.frames)} "
            f"buffer_span={span:.2f}s dropped={self.dropped_frames} elapsed={elapsed:.1f}s"
        )

    def idle_status_locked(self) -> str:
        age = (time.perf_counter() - self.last_preview_at) if self.last_preview_at else None
        age_text = f"{age:.2f}s ago" if age is not None else "never"
        return (
            f"webcam_live_frames={self.preview_seen_count} "
            f"last_frame={age_text} "
            f"queue=idle press Start Live Recognition when ready"
        )


STREAM = WebcamFrameBuffer()
PIPELINE = None


def gallery_for_model(model_key: str) -> Path:
    if model_key == "grew_gaitbase":
        return DEFAULT_GALLERY
    return LIVE_ROOT / "cache" / f"pritom_coco_gallery_{model_key}.npz"


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    largest = max(width, height)
    if largest <= max_side:
        return frame
    scale = max_side / largest
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def capture_webcam_frame(frame_rgb, capture_max_side: int):
    if frame_rgb is None:
        return "Waiting for browser webcam frames.", None
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_bgr = resize_max_side(frame_bgr, int(capture_max_side))
    _, status = STREAM.append(frame_bgr)
    mirrored = cv2.cvtColor(resize_max_side(frame_bgr, 480), cv2.COLOR_BGR2RGB)
    return status, mirrored


def stop_live_stream():
    STREAM.stop()
    return "Stop requested. The processor will finish after the current frame/buffer drains."


class RealtimeGaitPipeline:
    def __init__(
        self,
        model_key: str,
        gallery_path: Path,
        detector_input_size: int = 0,
        min_detection_score: float = 0.0,
    ) -> None:
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

        self.torch = torch
        self.cosine_similarity = cosine_similarity
        self.extract_embedding = extract_embedding
        self.gallery_path = Path(gallery_path).resolve()
        if not self.gallery_path.exists():
            raise FileNotFoundError(f"Missing gallery cache: {self.gallery_path}")

        gallery_data = np.load(self.gallery_path)
        self.gallery_embeddings = gallery_data["embeddings"].astype(np.float32)
        self.gallery_labels = gallery_data["labels"].astype(str)
        metadata_path = self.gallery_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                metadata = {}
            gallery_model_key = metadata.get("model_key")
            if gallery_model_key and gallery_model_key != model_key:
                raise RuntimeError(
                    f"Gallery cache was built for {gallery_model_key}, but selected model is {model_key}."
                )
        self.gait_model, _ = build_model(model_key)

        seg_config = (
            OPENGAIT
            / "demo"
            / "checkpoints"
            / "seg_model"
            / "human_pp_humansegv2_mobile_192x192_inference_model_with_softmax"
            / "deploy.yaml"
        )
        self.seg_predictor = Predictor_opengait(str(seg_config))

        device = torch.device("cuda" if track_cfgs["device"] == "gpu" and torch.cuda.is_available() else "cpu")
        self.predictor = Predictor(det_model, exp, None, None, device, device.type == "cuda")
        if int(detector_input_size) > 0:
            detector_input_size = max(224, int(detector_input_size))
            self.predictor.test_size = (detector_input_size, detector_input_size)
        self.detector_test_size = self.predictor.test_size
        self.tracker = BYTETracker(frame_rate=30)
        self.timer = Timer()
        self.seg_config = seg_config
        self.min_detection_score = max(0.0, float(min_detection_score))

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        processed_index: int,
        run_root: Path,
        identity_buffer_frames: int,
        silhouette_every_n_processed: int,
        first_track_offset: Optional[int],
        track_sil_counts: Dict[int, int],
        track_labels: Dict[int, str],
        track_scores: Dict[int, float],
        event_lines,
        stage_times: Dict[str, float],
        stage_counts: Dict[str, int],
        track_seen_counts: Optional[Dict[int, int]] = None,
        segmentation_start_after_frames: int = 0,
    ):
        detect_start = time.perf_counter()
        outputs, img_info = self.predictor.inference(frame, self.timer)
        outputs = filter_detections_by_score(outputs, self.min_detection_score)
        stage_times["detect"] += time.perf_counter() - detect_start
        stage_counts["detection_calls"] += 1

        active = set()
        unassigned_active = set()
        if outputs[0] is not None:
            track_start = time.perf_counter()
            online_targets = self.tracker.update(outputs[0], [img_info["height"], img_info["width"]], self.detector_test_size)
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
                if track_seen_counts is not None:
                    track_seen_counts[track_id] = track_seen_counts.get(track_id, 0) + 1

                if (
                    track_id not in track_labels
                    and processed_index % max(1, silhouette_every_n_processed) == 0
                    and (track_seen_counts is None or track_seen_counts.get(track_id, 0) > max(0, int(segmentation_start_after_frames)))
                ):
                    seg_start = time.perf_counter()
                    saved = save_track_silhouette(
                        frame=frame,
                        tlwh=tlwh,
                        video_width=frame.shape[1],
                        video_height=frame.shape[0],
                        track_id=track_id,
                        frame_id=frame_index,
                        predictor=self.seg_predictor,
                        seg_config=self.seg_config,
                        sil_root=run_root / "silhouettes",
                    )
                    stage_times["segmentation"] += time.perf_counter() - seg_start
                    stage_counts["segmentation_calls"] += 1
                    if saved is not None:
                        track_sil_counts[track_id] = track_sil_counts.get(track_id, 0) + 1

                    if track_sil_counts.get(track_id, 0) >= identity_buffer_frames:
                        track_input = track_silhouettes_to_input(
                            run_root / "silhouettes" / "live_probe" / f"{track_id:03d}" / "undefined"
                        )
                        if track_input is not None:
                            rec_start = time.perf_counter()
                            embedding = self.extract_embedding(self.gait_model, track_input)
                            if embedding.shape != self.gallery_embeddings[0].shape:
                                raise RuntimeError(
                                    f"Probe embedding shape {embedding.shape} does not match gallery shape "
                                    f"{self.gallery_embeddings[0].shape}."
                                )
                            scores = [
                                (str(label), self.cosine_similarity(embedding, gallery_embedding))
                                for label, gallery_embedding in zip(self.gallery_labels, self.gallery_embeddings)
                            ]
                            stage_times["recognition"] += time.perf_counter() - rec_start
                            stage_counts["recognition_calls"] += 1
                            best_label, best_score = sorted(scores, key=lambda item: item[1], reverse=True)[0]
                            track_labels[track_id] = best_label
                            track_scores[track_id] = float(best_score)
                            score_text = ", ".join(f"{label}={score:.4f}" for label, score in scores)
                            event_lines.append(
                                f"frame={frame_index}: track {track_id:03d} assigned {best_label} "
                                f"score={best_score:.4f} all_scores=[{score_text}]"
                            )

                if track_id not in track_labels:
                    unassigned_active.add(track_id)

                x, y, w, h = tlwh
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                label = track_labels.get(track_id)
                color = color_for_label(label or "collecting")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                if label is not None:
                    draw_label(frame, x1, max(0, y1 - 4), label, color)
            self.timer.toc()

        return first_track_offset, active, unassigned_active


def load_pipeline(model_key: str, detector_input_size: int, min_detection_score: float):
    global PIPELINE
    gallery_path = gallery_for_model(model_key)
    cache_key = (model_key, str(gallery_path), int(detector_input_size), float(min_detection_score))
    if PIPELINE is None or getattr(PIPELINE, "cache_key", None) != cache_key:
        PIPELINE = RealtimeGaitPipeline(
            model_key=model_key,
            gallery_path=gallery_path,
            detector_input_size=int(detector_input_size),
            min_detection_score=float(min_detection_score),
        )
        PIPELINE.cache_key = cache_key
    return PIPELINE


def run_realtime_processor(
    model_key: str,
    max_seconds: float,
    headstart_seconds: float,
    capture_max_side: int,
    max_buffer_frames: int,
    process_every_n: int,
    assigned_process_every_n: int,
    identity_buffer_frames: int,
    silhouette_every_n_processed: int,
    detector_input_size: int,
    min_detection_score: float,
    preview_every_n: int,
    preview_max_side: int,
):
    gallery_path = gallery_for_model(model_key)
    if not gallery_path.exists():
        raise gr.Error(
            f"Missing gallery cache for {model_key}: {gallery_path}\n\n"
            f"Build it first with:\n"
            f"CUDA_VISIBLE_DEVICES=<gpu> python live_demo/build_pritom_coco_gallery.py --model {model_key}"
        )

    warmup_deadline = time.perf_counter() + 3.0
    initial_snap = STREAM.snapshot()
    while initial_snap["preview_seen_count"] <= 0 and time.perf_counter() < warmup_deadline:
        time.sleep(0.05)
        initial_snap = STREAM.snapshot()
    if initial_snap["preview_seen_count"] <= 0:
        raise gr.Error(
            "No webcam frames have reached the server yet. "
            "If the left webcam panel is visible but the server mirror stays blank, Gradio is not receiving webcam frames from the browser. "
            "Try refreshing the page, allowing webcam permission again, and waiting for the queue status to change before pressing Start."
        )

    STREAM.start(max_seconds=float(max_seconds), max_buffer_frames=int(max_buffer_frames))
    run_root = OUTPUT_ROOT / f"v4_realtime_webcam_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    event_lines = []
    settings = {
        "mode": "v4_realtime_webcam_queue",
        "model_key": model_key,
        "gallery_path": str(gallery_path),
        "max_seconds": float(max_seconds),
        "headstart_seconds": float(headstart_seconds),
        "capture_max_side": int(capture_max_side),
        "max_buffer_frames": int(max_buffer_frames),
        "process_every_n": int(process_every_n),
        "assigned_process_every_n": int(assigned_process_every_n),
        "identity_buffer_frames": int(identity_buffer_frames),
        "silhouette_every_n_processed": int(silhouette_every_n_processed),
        "detector_input_size": int(detector_input_size),
        "min_detection_score": float(min_detection_score),
        "preview_every_n": int(preview_every_n),
        "preview_max_side": int(preview_max_side),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    event_lines.append("Run settings: " + json.dumps(settings, sort_keys=True))

    yield None, (
        "Live webcam recognition started.\n"
        "Please allow the browser webcam. The processor is waiting for the head-start buffer.\n"
        f"Run folder: {run_root}\n"
        f"Settings: {json.dumps(settings, sort_keys=True)}"
    )

    pipeline = load_pipeline(model_key, detector_input_size, min_detection_score)
    process_every_n = max(1, int(process_every_n))
    assigned_process_every_n = max(process_every_n, int(assigned_process_every_n))
    identity_buffer_frames = max(1, int(identity_buffer_frames))
    preview_every_n = max(1, int(preview_every_n))
    headstart_seconds = max(0.0, float(headstart_seconds))

    start_time = time.perf_counter()
    processed = 0
    consumed = 0
    skipped = 0
    first_track_offset = None
    track_sil_counts: Dict[int, int] = {}
    track_labels: Dict[int, str] = {}
    track_scores: Dict[int, float] = {}
    stage_times = {
        "wait_for_buffer": 0.0,
        "detect": 0.0,
        "track_update": 0.0,
        "segmentation": 0.0,
        "recognition": 0.0,
        "draw": 0.0,
    }
    stage_counts = {
        "detection_calls": 0,
        "track_update_calls": 0,
        "segmentation_calls": 0,
        "recognition_calls": 0,
    }
    all_visible_assigned = False
    last_frame_rgb = None

    while True:
        snap = STREAM.snapshot()
        if snap["stop_requested"] and snap["buffered"] == 0:
            break
        if snap["elapsed"] >= float(max_seconds) and snap["buffered"] == 0:
            STREAM.stop()
            break
        if snap["newest_time"] is None:
            wait_start = time.perf_counter()
            time.sleep(0.05)
            stage_times["wait_for_buffer"] += time.perf_counter() - wait_start
            if time.perf_counter() - start_time > 5.0 and snap["preview_seen_count"] <= 0:
                STREAM.stop()
                raise gr.Error(
                    "Recognition started, but no live webcam frames arrived from the browser. "
                    "Please refresh the page, allow webcam access, and wait for the queue status to show live frames."
                )
            continue
        if snap["newest_time"] < headstart_seconds:
            wait_start = time.perf_counter()
            time.sleep(0.05)
            stage_times["wait_for_buffer"] += time.perf_counter() - wait_start
            if int((time.perf_counter() - start_time) * 2) % 2 == 0:
                yield last_frame_rgb, (
                    f"Buffering webcam input before delayed processing...\n"
                    f"Captured: {snap['captured']} frames\n"
                    f"Buffered: {snap['buffered']} frames\n"
                    f"Buffered time: {snap['newest_time']:.2f}s / {headstart_seconds:.2f}s\n"
                    f"Dropped: {snap['dropped']} frames"
                )
            continue

        item = STREAM.pop_oldest()
        if item is None:
            wait_start = time.perf_counter()
            time.sleep(0.03)
            stage_times["wait_for_buffer"] += time.perf_counter() - wait_start
            continue

        consumed += 1
        frame = item["frame"].copy()
        frame_index = int(item["index"])
        current_stride = assigned_process_every_n if all_visible_assigned else process_every_n
        if frame_index % current_stride != 0:
            skipped += 1
            continue

        first_track_offset, active, unassigned_active = pipeline.process_frame(
            frame=frame,
            frame_index=frame_index,
            processed_index=processed,
            run_root=run_root,
            identity_buffer_frames=identity_buffer_frames,
            silhouette_every_n_processed=silhouette_every_n_processed,
            first_track_offset=first_track_offset,
            track_sil_counts=track_sil_counts,
            track_labels=track_labels,
            track_scores=track_scores,
            event_lines=event_lines,
            stage_times=stage_times,
            stage_counts=stage_counts,
        )

        previous_all_visible_assigned = all_visible_assigned
        all_visible_assigned = bool(active) and not unassigned_active
        if all_visible_assigned and not previous_all_visible_assigned and assigned_process_every_n > process_every_n:
            event_lines.append(
                f"frame={frame_index}: all visible tracks assigned; "
                f"switching stride {process_every_n}->{assigned_process_every_n}"
            )
        elif unassigned_active and previous_all_visible_assigned:
            event_lines.append(
                f"frame={frame_index}: new/unassigned track visible; "
                f"switching stride {assigned_process_every_n}->{process_every_n}"
            )

        draw_start = time.perf_counter()
        counts = {label: sum(1 for value in track_labels.values() if value == label) for label in ["pritom", "coco"]}
        elapsed = time.perf_counter() - start_time
        input_time = item["time"]
        realtime_factor = input_time / elapsed if elapsed > 0 else 0.0
        draw_panel(frame, [f"Counts: pritom={counts['pritom']} | coco={counts['coco']}"])
        stage_times["draw"] += time.perf_counter() - draw_start

        processed += 1
        if processed % preview_every_n == 0:
            preview = resize_max_side(frame, int(preview_max_side))
            last_frame_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            latest = STREAM.snapshot()
            yield last_frame_rgb, (
                f"Running delayed webcam recognition\n"
                f"Captured frames: {latest['captured']}\n"
                f"Buffered frames waiting: {latest['buffered']}\n"
                f"Consumed frames: {consumed}\n"
                f"Processed frames: {processed}\n"
                f"Skipped frames: {skipped}\n"
                f"Dropped old buffered frames: {latest['dropped']}\n"
                f"Input time processed: {input_time:.2f}s\n"
                f"Wall time: {elapsed:.2f}s\n"
                f"Realtime factor so far: {realtime_factor:.2f}x\n"
                f"Active tracks: {len(active)}\n"
                f"Assigned tracks: {len(track_labels)}\n"
                f"Counts: pritom={counts['pritom']} | coco={counts['coco']}\n"
                f"Silhouette buffers: {track_sil_counts}\n"
                f"Run folder: {run_root}"
            )

    STREAM.stop()
    elapsed = time.perf_counter() - start_time
    final_counts = {label: sum(1 for value in track_labels.values() if value == label) for label in ["pritom", "coco"]}
    metrics = {
        "settings": settings,
        "processed_frames": processed,
        "consumed_frames": consumed,
        "skipped_frames": skipped,
        "elapsed_seconds": elapsed,
        "stage_times": stage_times,
        "stage_counts": stage_counts,
        "track_labels": {str(key): value for key, value in track_labels.items()},
        "track_scores": {str(key): value for key, value in track_scores.items()},
        "counts": final_counts,
        "stream_snapshot": STREAM.snapshot(),
    }
    (run_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_root / "events_log.txt").write_text("\n".join(event_lines or ["No identities assigned."]))
    yield last_frame_rgb, (
        f"Finished\n"
        f"Processed frames: {processed}\n"
        f"Consumed frames: {consumed}\n"
        f"Skipped frames: {skipped}\n"
        f"Wall time: {elapsed:.2f}s\n"
        f"Stage seconds: detect={stage_times['detect']:.1f}, "
        f"seg={stage_times['segmentation']:.1f}, recog={stage_times['recognition']:.1f}, "
        f"wait={stage_times['wait_for_buffer']:.1f}, draw={stage_times['draw']:.1f}\n"
        f"Final counts: pritom={final_counts['pritom']} | coco={final_counts['coco']}\n"
        f"Output folder: {run_root}\n"
        f"Metrics: {run_root / 'metrics.json'}\n"
        f"Event log: {run_root / 'events_log.txt'}"
    )


def build_demo():
    with gr.Blocks(title="V4 Realtime Webcam Gait Recognition") as demo:
        gr.Markdown(
            """
            # V4 Realtime Webcam Gait Recognition

            This version uses the browser webcam as a live frame source. Frames are added to a
            producer queue, the recognizer waits for a small head-start buffer, then processes from
            the queue without trying to read frames that have not arrived yet.

            Keep the webcam visible on the left, press **Start Live Recognition**, and watch the
            delayed recognition preview on the right.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Live browser webcam",
                    height=420,
                )
                with gr.Row():
                    start_button = gr.Button("Start Live Recognition", variant="primary")
                    stop_button = gr.Button("Stop", variant="stop")
                buffer_status = gr.Textbox(label="Input queue status", lines=4)
                server_mirror = gr.Image(label="Server-confirmed webcam mirror", height=280)
                gr.Markdown("### Run Controls")
                model_key = gr.Dropdown(["grew_gaitbase", "grew_gaitgl", "current_gaitbase"], value="grew_gaitbase", label="Gait model")
                max_seconds = gr.Slider(5, 300, value=300, step=5, label="Max webcam run seconds")
                headstart_seconds = gr.Slider(0, 10, value=2, step=0.5, label="Head-start buffer seconds")
                capture_max_side = gr.Slider(240, 1080, value=720, step=40, label="Captured webcam frame max side")
                max_buffer_frames = gr.Slider(60, 1800, value=900, step=30, label="Maximum queued frames before dropping oldest")
                process_every_n = gr.Slider(1, 12, value=5, step=1, label="Manual frame stride: process every N captured frames")
                assigned_process_every_n = gr.Slider(1, 24, value=10, step=1, label="Adaptive stride after visible tracks are assigned")
                identity_buffer_frames = gr.Slider(1, 12, value=5, step=1, label="Silhouette buffer frames before identity assignment")
                silhouette_every_n_processed = gr.Slider(1, 5, value=1, step=1, label="Run segmentation every N processed frames per unassigned track")
                detector_input_size = gr.Slider(0, 960, value=0, step=32, label="Detector input size, 0 uses original demo")
                min_detection_score = gr.Slider(0, 0.9, value=0, step=0.05, label="Minimum detection score before tracking")
                preview_every_n = gr.Slider(1, 20, value=3, step=1, label="Update delayed preview every N processed frames")
                preview_max_side = gr.Slider(240, 1080, value=720, step=40, label="Delayed preview max side")
            with gr.Column(scale=1):
                processed_preview = gr.Image(label="Delayed recognition preview", height=520)
                run_status = gr.Textbox(label="Recognition status and final summary", lines=22)

        webcam.stream(
            capture_webcam_frame,
            inputs=[webcam, capture_max_side],
            outputs=[buffer_status, server_mirror],
            show_progress="hidden",
            queue=False,
            trigger_mode="multiple",
            concurrency_limit=None,
        )
        webcam.change(
            capture_webcam_frame,
            inputs=[webcam, capture_max_side],
            outputs=[buffer_status, server_mirror],
            show_progress="hidden",
            queue=False,
        )
        webcam.input(
            capture_webcam_frame,
            inputs=[webcam, capture_max_side],
            outputs=[buffer_status, server_mirror],
            show_progress="hidden",
            queue=False,
        )
        start_button.click(
            run_realtime_processor,
            inputs=[
                model_key,
                max_seconds,
                headstart_seconds,
                capture_max_side,
                max_buffer_frames,
                process_every_n,
                assigned_process_every_n,
                identity_buffer_frames,
                silhouette_every_n_processed,
                detector_input_size,
                min_detection_score,
                preview_every_n,
                preview_max_side,
            ],
            outputs=[processed_preview, run_status],
            show_progress="minimal",
            concurrency_limit=1,
        )
        stop_button.click(stop_live_stream, outputs=[buffer_status], show_progress="hidden")
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue(max_size=64, default_concurrency_limit=4)
    requested_port = int(os.environ.get("LIVE_DEMO_PORT", "7865"))
    try:
        app.launch(server_name="0.0.0.0", server_port=requested_port, share=True)
    except OSError as exc:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            fallback_port = sock.getsockname()[1]
        print(
            f"[v4-webcam] port {requested_port} unavailable ({exc}); retrying on {fallback_port}.",
            flush=True,
        )
        app.launch(server_name="0.0.0.0", server_port=fallback_port, share=True)
