import json
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pyngrok import ngrok
import uvicorn


ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = ROOT / "live_demo_clean"
LEGACY_LIVE_ROOT = ROOT / "live_demo"

GPU_ID = os.environ.get("LIVE_DEMO_GPU_ID")
if GPU_ID not in (None, "", "cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
elif GPU_ID in ("cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(LEGACY_LIVE_ROOT))

import v4_realtime_webcam_app as legacy_v4  # noqa: E402
from v4_realtime_webcam_app import RealtimeGaitPipeline  # noqa: E402


def color_for_label(label: str):
    if label == "collecting":
        return (255, 255, 255)
    idx = 1
    if label.startswith("person"):
        try:
            idx = int(label.replace("person", ""))
        except ValueError:
            idx = 1
    return ((37 * idx) % 255, (137 * idx) % 255, (211 * idx) % 255)


legacy_v4.color_for_label = color_for_label


def draw_label(frame, x: int, y: int, text: str, color):
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
    width = 900
    height = 24 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (18 + width, 18 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (34, 52 + idx * line_height), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def gallery_for_model(model_key: str) -> Path:
    override = os.environ.get("LIVE_DEMO_CLEAN_GALLERY_NPZ")
    if override:
        return Path(override).expanduser().resolve()
    stem = "generic_gallery" if model_key == "grew_gaitbase" else f"generic_gallery_{model_key}"
    return (LIVE_ROOT / "cache" / f"{stem}.npz").resolve()


class LiveState:
    def __init__(self):
        self.lock = threading.RLock()
        self.queue = deque()
        self.running = False
        self.worker = None
        self.started_at = None
        self.frame_index = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.max_buffer_frames = 240
        self.headstart_seconds = 2.0
        self.capture_interval_ms = 50
        self.min_hold_frames = 10
        self.max_live_lag_frames = 30
        self.max_seconds = 300.0
        self.model_key = "grew_gaitbase"
        self.process_every_n = 1
        self.assigned_process_every_n = 10
        self.identity_buffer_frames = 15
        self.segmentation_start_after_frames = 0
        self.silhouette_every_n_processed = 1
        self.min_detection_score = 0.0
        self.preview_max_side = 720
        self.input_width = 0
        self.input_height = 0
        self.gallery_labels = []
        self.gallery_mapping = {}
        self.track_labels = {}
        self.counts = {}
        self.track_score_details = {}
        self.incomplete_tracks = {}
        self.mark_incomplete_tracks = True
        self.last_preview_jpg = None
        self.last_status = "Idle."
        self.preview_seen = 0
        self.last_preview_at = None
        self.logs = []
        self.run_root = None
        self.stop_requested = False
        self.last_error = None
        self.worker_phase = "idle"
        self.worker_phase_since = None
        self.last_processed_frame_index = None

    def reset_for_run(self, settings: dict, gallery_labels):
        self.queue.clear()
        self.running = True
        self.stop_requested = False
        self.started_at = time.perf_counter()
        self.frame_index = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.gallery_labels = list(gallery_labels)
        self.gallery_mapping = dict(settings.get("gallery_mapping", {}))
        self.track_labels = {}
        self.counts = {label: 0 for label in self.gallery_labels}
        self.track_score_details = {}
        self.incomplete_tracks = {}
        self.last_preview_jpg = None
        self.preview_seen = 0
        self.last_preview_at = None
        self.logs = ["Run settings: " + json.dumps(settings, sort_keys=True)]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_root = LIVE_ROOT / "output" / f"fastapi_live_{stamp}"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.last_status = "Starting live worker..."
        self.last_error = None
        self.worker_phase = "starting"
        self.worker_phase_since = time.perf_counter()
        self.last_processed_frame_index = None


STATE = LiveState()
PIPELINES = {}


def get_pipeline(model_key: str):
    gallery_path = gallery_for_model(model_key)
    cache_key = (model_key, str(gallery_path))
    if cache_key not in PIPELINES:
        PIPELINES[cache_key] = RealtimeGaitPipeline(
            model_key=model_key,
            gallery_path=gallery_path,
            detector_input_size=0,
            min_detection_score=0.0,
        )
    return PIPELINES[cache_key]


def load_gallery_mapping(model_key: str) -> Dict[str, str]:
    gallery_path = gallery_for_model(model_key)
    metadata_path = gallery_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return {}
    identities = metadata.get("identities", {})
    mapping = {}
    for label, payload in identities.items():
        video_name = payload.get("video_name")
        video_path = payload.get("video")
        mapping[str(label)] = video_name or (Path(video_path).name if video_path else "unknown")
    return mapping


def encode_jpg(frame_bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode preview JPEG.")
    return buf.tobytes()


def set_worker_phase(phase: str):
    if STATE.worker_phase != phase:
        STATE.worker_phase = phase
        STATE.worker_phase_since = time.perf_counter()


def counts_line(counts: Dict[str, int]) -> str:
    if not counts:
        return "Counts: none"
    ordered = [f"{label}={counts.get(label, 0)}" for label in sorted(counts)]
    return "Counts: " + " | ".join(ordered)


def incomplete_lines_from_payload(payload: Dict[str, Dict[str, int]]) -> str:
    if not payload:
        return "No unassigned tracks flagged."
    lines = []
    for track_id in sorted(payload, key=lambda value: int(value)):
        item = payload[track_id]
        lines.append(
            f"track {int(track_id):03d} -> unassigned | "
            f"tracked={item['seen_frames']} silhouettes={item['silhouette_frames']}/{item['required_silhouette_frames']}"
        )
    return "\n".join(lines)


ASSIGN_EVENT_RE = re.compile(
    r"track\s+(?P<track_id>\d+)\s+assigned\s+(?P<label>\S+)\s+score=(?P<best>[0-9.]+)\s+all_scores=\[(?P<scores>.*)\]"
)


def parse_assignment_event(line: str):
    match = ASSIGN_EVENT_RE.search(line)
    if not match:
        return None
    score_map = {}
    raw_scores = match.group("scores").strip()
    if raw_scores:
        for part in raw_scores.split(","):
            part = part.strip()
            if "=" not in part:
                continue
            label, value = part.split("=", 1)
            try:
                score_map[label.strip()] = float(value.strip())
            except ValueError:
                continue
    return {
        "track_id": int(match.group("track_id")),
        "label": match.group("label"),
        "best_score": float(match.group("best")),
        "scores": score_map,
    }


def worker_loop():
    try:
        with STATE.lock:
            model_key = STATE.model_key
            headstart_seconds = STATE.headstart_seconds
            max_seconds = STATE.max_seconds
            min_hold_frames = STATE.min_hold_frames
            max_live_lag_frames = STATE.max_live_lag_frames
            process_every_n = STATE.process_every_n
            assigned_process_every_n = STATE.assigned_process_every_n
            identity_buffer_frames = STATE.identity_buffer_frames
            segmentation_start_after_frames = STATE.segmentation_start_after_frames
            silhouette_every_n_processed = STATE.silhouette_every_n_processed
            preview_max_side = STATE.preview_max_side
            run_root = STATE.run_root
            mark_incomplete_tracks = STATE.mark_incomplete_tracks
        pipeline = get_pipeline(model_key)
        pipeline.min_detection_score = float(STATE.min_detection_score)
        gallery_labels = [str(label) for label in pipeline.gallery_labels]

        first_track_offset = None
        track_sil_counts: Dict[int, int] = {}
        track_labels: Dict[int, str] = {}
        track_scores: Dict[int, float] = {}
        track_score_details: Dict[int, Dict[str, float]] = {}
        track_seen_counts: Dict[int, int] = {}
        counted_track_ids = set()
        incomplete_tracks: Dict[int, Dict[str, int]] = {}
        cumulative_counts = {label: 0 for label in gallery_labels}
        all_visible_assigned = False
        previous_active_tracks = set()

        while True:
            set_worker_phase("loop_top")
            with STATE.lock:
                if not STATE.running:
                    break
                elapsed = time.perf_counter() - STATE.started_at if STATE.started_at else 0.0
                buffered = len(STATE.queue)
                newest_elapsed = STATE.queue[-1]["elapsed"] if STATE.queue else None
                if STATE.stop_requested and buffered == 0:
                    STATE.running = False
                    STATE.last_status = (
                        f"Stopped after draining queue\nProcessed frames: {STATE.processed_frames}\n"
                        f"{counts_line(STATE.counts)}\n"
                        f"Unassigned tracks:\n{incomplete_lines_from_payload(STATE.incomplete_tracks)}\n"
                        f"Output folder: {STATE.run_root}"
                    )
                    break
                if elapsed >= max_seconds and buffered == 0:
                    STATE.running = False
                    STATE.last_status = "Finished: queue drained after max duration."
                    break

            if newest_elapsed is None or newest_elapsed < headstart_seconds:
                set_worker_phase("waiting_for_headstart")
                time.sleep(0.03)
                with STATE.lock:
                    STATE.last_status = (
                        f"Buffering live webcam... server_frames={STATE.preview_seen} "
                        f"buffered={len(STATE.queue)} "
                        f"buffer_time={(newest_elapsed or 0.0):.2f}/{headstart_seconds:.2f}s"
                    )
                continue

            with STATE.lock:
                set_worker_phase("buffer_control")
                buffered_now = len(STATE.queue)
                if buffered_now > max_live_lag_frames:
                    drop_count = max(0, buffered_now - max_live_lag_frames)
                    for _ in range(drop_count):
                        if STATE.queue:
                            STATE.queue.popleft()
                            STATE.dropped_frames += 1
                    buffered_now = len(STATE.queue)
                    STATE.last_status = (
                        f"Dropping stale live frames to catch up... "
                        f"buffered={buffered_now} target_lag={max_live_lag_frames} "
                        f"dropped_total={STATE.dropped_frames}"
                    )

                if buffered_now <= min_hold_frames and not STATE.stop_requested:
                    STATE.last_status = (
                        f"Holding live buffer... buffered={buffered_now} min_hold={min_hold_frames} "
                        f"captured={STATE.preview_seen} processed={STATE.processed_frames}"
                    )
                    need_wait = True
                else:
                    need_wait = False
            if need_wait:
                set_worker_phase("holding_min_buffer")
                time.sleep(0.02)
                continue

            with STATE.lock:
                set_worker_phase("pop_queue")
                queued_before_pop = len(STATE.queue)
                item = STATE.queue.popleft() if STATE.queue else None
            if item is None:
                set_worker_phase("waiting_for_queue_item")
                time.sleep(0.02)
                continue

            frame = item["frame"].copy()
            frame_index = item["index"]
            current_stride = assigned_process_every_n if all_visible_assigned else process_every_n
            lag_after_pop = max(1, queued_before_pop - min_hold_frames)
            effective_stride = max(1, min(current_stride, lag_after_pop))
            if frame_index % effective_stride != 0:
                set_worker_phase("stride_skip")
                continue

            set_worker_phase("pipeline_process_frame")
            event_lines = []
            first_track_offset, active, unassigned_active = pipeline.process_frame(
                frame=frame,
                frame_index=frame_index,
                processed_index=STATE.processed_frames,
                run_root=run_root,
                identity_buffer_frames=identity_buffer_frames,
                silhouette_every_n_processed=silhouette_every_n_processed,
                first_track_offset=first_track_offset,
                track_sil_counts=track_sil_counts,
                track_labels=track_labels,
                track_scores=track_scores,
                event_lines=event_lines,
                stage_times={"detect": 0.0, "track_update": 0.0, "segmentation": 0.0, "recognition": 0.0},
                stage_counts={"detection_calls": 0, "track_update_calls": 0, "segmentation_calls": 0, "recognition_calls": 0},
                track_seen_counts=track_seen_counts,
                segmentation_start_after_frames=segmentation_start_after_frames,
            )

            for event in event_lines:
                parsed = parse_assignment_event(event)
                if not parsed:
                    continue
                track_id = parsed["track_id"]
                track_score_details[track_id] = parsed["scores"]
                track_scores[track_id] = parsed["best_score"]
                if track_id not in track_labels:
                    track_labels[track_id] = parsed["label"]

            for track_id in list(active):
                if track_id in track_labels and track_id not in counted_track_ids:
                    cumulative_counts[track_labels[track_id]] += 1
                    counted_track_ids.add(track_id)

            if mark_incomplete_tracks:
                disappeared_tracks = previous_active_tracks - set(active)
                for track_id in sorted(disappeared_tracks):
                    if track_id in track_labels or track_id in incomplete_tracks:
                        continue
                    seen_frames = int(track_seen_counts.get(track_id, 0))
                    sil_frames = int(track_sil_counts.get(track_id, 0))
                    if seen_frames > 0 and sil_frames < identity_buffer_frames:
                        incomplete_tracks[track_id] = {
                            "seen_frames": seen_frames,
                            "silhouette_frames": sil_frames,
                            "required_silhouette_frames": int(identity_buffer_frames),
                        }

            previous_active_tracks = set(active)

            previous_all_visible_assigned = all_visible_assigned
            all_visible_assigned = bool(active) and not unassigned_active
            if previous_all_visible_assigned != all_visible_assigned:
                set_worker_phase("stride_switch")

            display_frame = frame
            if preview_max_side > 0:
                h, w = frame.shape[:2]
                scale = min(1.0, preview_max_side / max(w, h))
                if scale < 1.0:
                    display_frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

            draw_panel(display_frame, [counts_line(cumulative_counts)])
            preview_jpg = encode_jpg(display_frame)

            with STATE.lock:
                STATE.processed_frames += 1
                STATE.last_processed_frame_index = frame_index
                STATE.last_preview_jpg = preview_jpg
                STATE.last_preview_at = time.perf_counter()
                STATE.counts = dict(cumulative_counts)
                STATE.track_labels = {str(key): value for key, value in track_labels.items()}
                STATE.track_score_details = {
                    str(track_id): {label: float(score) for label, score in score_map.items()}
                    for track_id, score_map in track_score_details.items()
                }
                STATE.incomplete_tracks = {
                    str(track_id): payload for track_id, payload in incomplete_tracks.items()
                }
                detail_lines = []
                for track_id in sorted(track_labels):
                    detail = STATE.track_score_details.get(str(track_id), {})
                    if detail:
                        ranked = sorted(detail.items(), key=lambda item: item[1], reverse=True)
                        top = ranked[:2]
                        if len(top) == 1:
                            summary = f"{top[0][0]}={top[0][1]:.4f}"
                        else:
                            summary = f"{top[0][0]} vs {top[1][0]} | {top[0][1]:.4f} vs {top[1][1]:.4f}"
                        detail_lines.append(f"track {track_id:03d} -> {track_labels[track_id]} | {summary}")
                    else:
                        detail_lines.append(f"track {track_id:03d} -> {track_labels[track_id]} | score details pending")
                detail_block = "\n".join(detail_lines) if detail_lines else "No assigned track score details yet."
                incomplete_lines = []
                for track_id in sorted(incomplete_tracks):
                    payload = incomplete_tracks[track_id]
                    incomplete_lines.append(
                        f"track {track_id:03d} -> unassigned | "
                        f"tracked={payload['seen_frames']} silhouettes={payload['silhouette_frames']}/{payload['required_silhouette_frames']}"
                    )
                incomplete_block = "\n".join(incomplete_lines) if incomplete_lines else "No unassigned tracks flagged."
                mapping_block = "\n".join(
                    f"{label} <- {video_name}" for label, video_name in sorted(STATE.gallery_mapping.items())
                ) or "No gallery mapping metadata found."
                input_seconds = 0.0
                if STATE.capture_interval_ms > 0:
                    input_seconds = frame_index * STATE.capture_interval_ms / 1000.0
                STATE.last_status = (
                    f"Running live webcam recognition\n"
                    f"Captured frames: {STATE.preview_seen}\n"
                    f"Buffered frames: {len(STATE.queue)} / {STATE.max_buffer_frames}\n"
                    f"Minimum hold buffer: {STATE.min_hold_frames}\n"
                    f"Maximum live lag: {STATE.max_live_lag_frames}\n"
                    f"Processed frames: {STATE.processed_frames}\n"
                    f"Last processed frame index: {STATE.last_processed_frame_index}\n"
                    f"Dropped frames: {STATE.dropped_frames}\n"
                    f"Worker phase: {STATE.worker_phase} ({worker_phase_age_seconds():.2f}s)\n"
                    f"Input time processed: {input_seconds:.2f}s\n"
                    f"Seg warmup tracked frames: {STATE.segmentation_start_after_frames}\n"
                    f"Gallery mapping:\n{mapping_block}\n"
                    f"Counts: {counts_line(cumulative_counts).replace('Counts: ', '')}\n"
                    f"Score details:\n{detail_block}\n"
                    f"Unassigned tracks:\n{incomplete_block}\n"
                    f"Run folder: {STATE.run_root}"
                )

        with STATE.lock:
            STATE.running = False
            if STATE.last_error is None and not STATE.last_status.startswith("Finished"):
                STATE.last_status = (
                    f"Finished\nProcessed frames: {STATE.processed_frames}\n"
                    f"{counts_line(STATE.counts)}\n"
                    f"Unassigned tracks:\n{incomplete_lines_from_payload(STATE.incomplete_tracks)}\n"
                    f"Output folder: {STATE.run_root}"
                )
    except Exception as exc:
        with STATE.lock:
            STATE.last_error = str(exc)
            STATE.running = False
            STATE.last_status = f"Worker error: {exc}"


def worker_phase_age_seconds() -> float:
    if STATE.worker_phase_since is None:
        return 0.0
    return time.perf_counter() - STATE.worker_phase_since


def app_html():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Generic Live Gait Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #101417; color: #f2f5f7; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
    .panel { background: #182026; border-radius: 14px; padding: 16px; }
    video, img { width: 100%; max-height: 520px; background: #000; border-radius: 10px; object-fit: contain; }
    button, select, input { padding: 10px 12px; border-radius: 10px; border: none; margin-right: 8px; margin-bottom: 8px; }
    button { background: #2a3943; color: #f2f5f7; cursor: pointer; transition: transform 0.08s ease, background 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease; }
    button:hover { background: #35505f; box-shadow: 0 8px 22px rgba(0,0,0,0.22); }
    button:active, button.clicked { transform: scale(0.96); background: #4b6c7f; }
    button.busy { opacity: 0.72; }
    .debug { white-space: pre-wrap; font-family: monospace; font-size: 13px; background: #0d1114; padding: 12px; border-radius: 10px; max-height: 420px; overflow: auto; }
    .toast { position: fixed; top: 20px; right: 20px; background: #2e7d32; color: white; padding: 12px 16px; border-radius: 10px; opacity: 0; transition: opacity 0.2s; }
    .toast.show { opacity: 1; }
    .controls { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px 18px; margin-bottom: 12px; }
    .field { display: flex; flex-direction: column; gap: 6px; }
    .field label { font-size: 13px; color: #d3dde4; }
    .buttons { margin-top: 8px; }
  </style>
</head>
<body>
  <div id="toast" class="toast"></div>
  <h1>Generic Live Gait Demo</h1>
  <p>Use any number of gallery identities. Build the gallery cache first, then run the webcam live demo.</p>

  <div class="panel">
    <div class="controls">
      <div class="field">
        <label for="model">Gait Model</label>
        <select id="model">
          <option value="grew_gaitbase">grew_gaitbase</option>
          <option value="current_gaitbase">current_gaitbase</option>
          <option value="grew_gaitgl">grew_gaitgl</option>
        </select>
      </div>
      <div class="field">
        <label for="headstart">Head-start buffer (s)</label>
        <input id="headstart" type="number" value="2" step="0.5" />
      </div>
      <div class="field">
        <label for="capture_ms">Capture interval (ms)</label>
        <input id="capture_ms" type="number" value="50" step="10" />
      </div>
      <div class="field">
        <label for="max_seconds">Max seconds</label>
        <input id="max_seconds" type="number" value="300" step="10" />
      </div>
      <div class="field">
        <label for="process_every_n">Dense stride</label>
        <input id="process_every_n" type="number" value="1" step="1" />
      </div>
      <div class="field">
        <label for="assigned_process_every_n">Assigned stride</label>
        <input id="assigned_process_every_n" type="number" value="10" step="1" />
      </div>
      <div class="field">
        <label for="identity_buffer_frames">Identity buffer frames</label>
        <input id="identity_buffer_frames" type="number" value="15" step="1" />
      </div>
      <div class="field">
        <label for="segmentation_start_after_frames">Segmentation starts after tracked frames</label>
        <input id="segmentation_start_after_frames" type="number" value="0" step="1" />
      </div>
      <div class="field">
        <label for="mark_incomplete_tracks">Mark incomplete tracks</label>
        <select id="mark_incomplete_tracks">
          <option value="1" selected>on</option>
          <option value="0">off</option>
        </select>
      </div>
    </div>
    <div class="buttons">
      <button id="cameraOnBtn" onclick="cameraOn(this)">Turn Camera On</button>
      <button id="cameraOffBtn" onclick="cameraOff(this)">Turn Camera Off</button>
      <button id="startBtn" onclick="startRun(this)">Start Live Demo</button>
      <button id="stopBtn" onclick="stopRun(this)">Stop Live Demo</button>
    </div>
  </div>

  <div class="grid">
    <div class="panel">
      <h3>Raw Webcam</h3>
      <video id="video" autoplay playsinline muted></video>
    </div>
    <div class="panel">
      <h3>Processed Preview</h3>
      <img id="preview" />
    </div>
  </div>

  <div class="panel">
    <h3>Detailed Debug</h3>
    <div id="debug" class="debug">Idle.</div>
  </div>

  <script>
    let mediaStream = null;
    let sendTimer = null;
    let pollTimer = null;
    let previewTimer = null;
    let lastFrameBlob = null;
    let clientFramesSent = 0;
    let clientSendFailures = 0;
    let lastSendAt = null;
    let lastPollAt = null;
    let lastPreviewAt = null;
    const video = document.getElementById("video");

    function toast(msg) {
      const el = document.getElementById("toast");
      el.textContent = msg;
      el.classList.add("show");
      setTimeout(() => el.classList.remove("show"), 1800);
    }

    function animateButton(btn) {
      if (!btn) return;
      btn.classList.add("clicked");
      setTimeout(() => btn.classList.remove("clicked"), 180);
    }

    async function cameraOn(btn) {
      animateButton(btn);
      if (mediaStream) return;
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = mediaStream;
      startSendingFrames();
      toast("Camera on");
    }

    function cameraOff(btn) {
      animateButton(btn);
      if (sendTimer) clearInterval(sendTimer);
      sendTimer = null;
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
      }
      mediaStream = null;
      video.srcObject = null;
      toast("Camera off");
    }

    function startSendingFrames() {
      if (sendTimer) clearInterval(sendTimer);
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      sendTimer = setInterval(async () => {
        if (!mediaStream || !video.videoWidth || !video.videoHeight) return;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.8));
        if (!blob) return;
        lastFrameBlob = blob;
        try {
          const res = await fetch("/frame", { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
          if (res.ok) {
            clientFramesSent += 1;
            lastSendAt = performance.now();
          } else {
            clientSendFailures += 1;
          }
        } catch (_) {
          clientSendFailures += 1;
        }
      }, parseInt(document.getElementById("capture_ms").value || "50", 10));
    }

    async function startRun(btn) {
      animateButton(btn);
      if (btn) btn.classList.add("busy");
      toast("Starting demo");
      const payload = {
        model_key: document.getElementById("model").value,
        headstart_seconds: parseFloat(document.getElementById("headstart").value || "2"),
        capture_interval_ms: parseInt(document.getElementById("capture_ms").value || "50", 10),
        max_seconds: parseFloat(document.getElementById("max_seconds").value || "300"),
        process_every_n: parseInt(document.getElementById("process_every_n").value || "1", 10),
        assigned_process_every_n: parseInt(document.getElementById("assigned_process_every_n").value || "10", 10),
        identity_buffer_frames: parseInt(document.getElementById("identity_buffer_frames").value || "15", 10),
        segmentation_start_after_frames: parseInt(document.getElementById("segmentation_start_after_frames").value || "0", 10),
        mark_incomplete_tracks: document.getElementById("mark_incomplete_tracks").value === "1"
      };
      const res = await fetch("/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        if (btn) btn.classList.remove("busy");
        toast(data.detail || "Start failed");
        return;
      }
      if (btn) btn.classList.remove("busy");
      toast("Live demo started");
      beginPolling();
    }

    async function stopRun(btn) {
      animateButton(btn);
      await fetch("/stop", { method: "POST" });
      toast("Stop requested");
    }

    function beginPolling() {
      if (pollTimer) clearInterval(pollTimer);
      if (previewTimer) clearInterval(previewTimer);
      pollTimer = setInterval(async () => {
        const res = await fetch("/status");
        const data = await res.json();
        lastPollAt = performance.now();
        const now = performance.now();
        const lastSendAge = lastSendAt ? ((now - lastSendAt) / 1000).toFixed(2) : "null";
        const lastPollAge = lastPollAt ? ((performance.now() - lastPollAt) / 1000).toFixed(2) : "null";
        const lastPreviewAge = lastPreviewAt ? ((now - lastPreviewAt) / 1000).toFixed(2) : "null";
        document.getElementById("debug").textContent =
          `client_frames_sent=${clientFramesSent}\n` +
          `client_send_failures=${clientSendFailures}\n` +
          `client_last_send_age_s=${lastSendAge}\n` +
          `client_last_poll_age_s=${lastPollAge}\n` +
          `client_last_preview_age_s=${lastPreviewAge}\n` +
          `server_preview_seen=${data.preview_seen}\n` +
          `queue=${data.queue_size}\n` +
          `processed=${data.processed_frames}\n` +
          `dropped=${data.dropped_frames}\n` +
          `worker_phase=${data.worker_phase}\n` +
          `worker_phase_age_s=${data.worker_phase_age_s}\n` +
          `last_processed_frame_index=${data.last_processed_frame_index}\n` +
          `last_error=${JSON.stringify(data.last_error)}\n` +
          `counts=${JSON.stringify(data.counts)}\n` +
          `track_score_details=${JSON.stringify(data.track_score_details)}\n` +
          `backend_settings=${JSON.stringify(data.settings)}\n\n` +
          `${data.status}`;
      }, 500);
      previewTimer = setInterval(() => {
        document.getElementById("preview").src = "/preview.jpg?ts=" + Date.now();
        lastPreviewAt = performance.now();
      }, 600);
    }
  </script>
</body>
</html>
"""


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    return app_html()


@app.post("/frame")
async def frame(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty frame payload.")
    jpg = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode JPEG frame.")
    with STATE.lock:
        STATE.preview_seen += 1
        STATE.last_preview_at = time.perf_counter()
        if STATE.running and not STATE.stop_requested:
            STATE.queue.append({"index": STATE.frame_index, "elapsed": time.perf_counter() - STATE.started_at, "frame": frame})
            STATE.frame_index += 1
            while len(STATE.queue) > STATE.max_buffer_frames:
                STATE.queue.popleft()
                STATE.dropped_frames += 1
    return JSONResponse({"ok": True})


@app.post("/start")
async def start(request: Request):
    payload = await request.json()
    model_key = payload.get("model_key", "grew_gaitbase")
    gallery_path = gallery_for_model(model_key)
    if not gallery_path.exists():
        raise HTTPException(status_code=400, detail=f"Missing gallery cache: {gallery_path}")
    pipeline = get_pipeline(model_key)
    settings = {
        "model_key": model_key,
        "gallery_path": str(gallery_path),
        "gallery_mapping": load_gallery_mapping(model_key),
        "headstart_seconds": float(payload.get("headstart_seconds", 2.0)),
        "capture_interval_ms": int(payload.get("capture_interval_ms", 50)),
        "max_seconds": float(payload.get("max_seconds", 300.0)),
        "process_every_n": int(payload.get("process_every_n", 1)),
        "assigned_process_every_n": int(payload.get("assigned_process_every_n", 10)),
        "identity_buffer_frames": int(payload.get("identity_buffer_frames", 15)),
        "segmentation_start_after_frames": int(payload.get("segmentation_start_after_frames", 0)),
        "mark_incomplete_tracks": bool(payload.get("mark_incomplete_tracks", True)),
    }
    with STATE.lock:
        if STATE.running:
            raise HTTPException(status_code=400, detail="A live run is already in progress.")
        STATE.model_key = settings["model_key"]
        STATE.headstart_seconds = settings["headstart_seconds"]
        STATE.capture_interval_ms = settings["capture_interval_ms"]
        STATE.max_seconds = settings["max_seconds"]
        STATE.process_every_n = settings["process_every_n"]
        STATE.assigned_process_every_n = settings["assigned_process_every_n"]
        STATE.identity_buffer_frames = settings["identity_buffer_frames"]
        STATE.segmentation_start_after_frames = settings["segmentation_start_after_frames"]
        STATE.mark_incomplete_tracks = settings["mark_incomplete_tracks"]
        STATE.min_hold_frames = max(10, int(round(STATE.headstart_seconds * 1000.0 / max(1, STATE.capture_interval_ms))))
        STATE.max_live_lag_frames = max(STATE.min_hold_frames + 10, STATE.min_hold_frames * 2)
        STATE.max_buffer_frames = max(240, STATE.max_live_lag_frames * 3)
        STATE.min_detection_score = 0.0
        STATE.preview_max_side = 720
        STATE.reset_for_run(settings, pipeline.gallery_labels)
        STATE.worker = threading.Thread(target=worker_loop, daemon=True)
        STATE.worker.start()
    return JSONResponse({"ok": True, "gallery_labels": [str(label) for label in pipeline.gallery_labels]})


@app.post("/stop")
def stop():
    with STATE.lock:
        STATE.stop_requested = True
        STATE.last_status = (
            f"Stop requested. Finishing buffered queue...\nProcessed frames: {STATE.processed_frames}\n"
            f"{counts_line(STATE.counts)}\n"
            f"Unassigned tracks:\n{incomplete_lines_from_payload(STATE.incomplete_tracks)}\n"
            f"Output folder: {STATE.run_root}"
        )
    return JSONResponse({"ok": True})


@app.get("/status")
def status():
    with STATE.lock:
        return JSONResponse(
            {
                "running": STATE.running,
                "preview_seen": STATE.preview_seen,
                "queue_size": len(STATE.queue),
                "processed_frames": STATE.processed_frames,
                "dropped_frames": STATE.dropped_frames,
                "worker_phase": STATE.worker_phase,
                "worker_phase_age_s": round(worker_phase_age_seconds(), 2),
                "last_processed_frame_index": STATE.last_processed_frame_index,
                "last_error": STATE.last_error,
                "counts": STATE.counts,
                "track_score_details": STATE.track_score_details,
                "incomplete_tracks": STATE.incomplete_tracks,
                "settings": {
                    "model_key": STATE.model_key,
                    "headstart_seconds": STATE.headstart_seconds,
                    "capture_interval_ms": STATE.capture_interval_ms,
                    "process_every_n": STATE.process_every_n,
                    "assigned_process_every_n": STATE.assigned_process_every_n,
                    "identity_buffer_frames": STATE.identity_buffer_frames,
                    "segmentation_start_after_frames": STATE.segmentation_start_after_frames,
                    "mark_incomplete_tracks": STATE.mark_incomplete_tracks,
                    "min_hold_frames": STATE.min_hold_frames,
                    "max_live_lag_frames": STATE.max_live_lag_frames,
                    "gallery_labels": STATE.gallery_labels,
                    "gallery_mapping": STATE.gallery_mapping,
                },
                "status": STATE.last_status,
            }
        )


@app.get("/preview.jpg")
def preview():
    with STATE.lock:
        if STATE.last_preview_jpg is None:
            raise HTTPException(status_code=404, detail="No preview yet.")
        return Response(content=STATE.last_preview_jpg, media_type="image/jpeg")


def main():
    port = int(os.environ.get("LIVE_DEMO_PORT", "8011"))
    public = os.environ.get("LIVE_DEMO_PUBLIC", "1") != "0"
    if public:
        authtoken = os.environ.get("NGROK_AUTHTOKEN")
        if authtoken:
            try:
                ngrok.set_auth_token(authtoken)
            except Exception as exc:
                print(f"[live-demo-clean] failed to set ngrok authtoken: {exc}", flush=True)
        try:
            public_url = ngrok.connect(port).public_url
            print(f"[live-demo-clean] public_url={public_url}", flush=True)
        except Exception as exc:
            print(
                "[live-demo-clean] public tunnel failed. "
                "Set NGROK_AUTHTOKEN or run with LIVE_DEMO_PUBLIC=0.\n"
                f"[live-demo-clean] tunnel_error={exc}",
                flush=True,
            )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
