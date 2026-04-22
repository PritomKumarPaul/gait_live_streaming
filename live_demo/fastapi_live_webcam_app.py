import json
import os
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
import uvicorn
from pyngrok import ngrok


ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = ROOT / "live_demo"

GPU_ID = os.environ.get("LIVE_DEMO_GPU_ID")
if GPU_ID not in (None, "", "cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
elif GPU_ID in ("cpu", "CPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(LIVE_ROOT))

from run_buffered_live_probe import color_for_label, draw_label, draw_panel, track_silhouettes_to_input  # noqa: E402
from v4_realtime_webcam_app import RealtimeGaitPipeline  # noqa: E402


def gallery_for_model(model_key: str) -> Path:
    default_gallery = LIVE_ROOT / "cache" / "pritom_coco_gallery.npz"
    if model_key == "grew_gaitbase":
        return default_gallery
    return LIVE_ROOT / "cache" / f"pritom_coco_gallery_{model_key}.npz"


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
        self.process_every_n = 5
        self.assigned_process_every_n = 10
        self.identity_buffer_frames = 5
        self.silhouette_every_n_processed = 1
        self.min_detection_score = 0.0
        self.preview_max_side = 720
        self.input_width = 0
        self.input_height = 0
        self.track_labels = {}
        self.counts = {"pritom": 0, "coco": 0}
        self.track_score_details = {}
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

    def reset_for_run(self, settings: dict):
        self.queue.clear()
        self.running = True
        self.stop_requested = False
        self.started_at = time.perf_counter()
        self.frame_index = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.track_labels = {}
        self.counts = {"pritom": 0, "coco": 0}
        self.track_score_details = {}
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
    cache_key = model_key
    if cache_key not in PIPELINES:
        PIPELINES[cache_key] = RealtimeGaitPipeline(
            model_key=model_key,
            gallery_path=gallery_for_model(model_key),
            detector_input_size=0,
            min_detection_score=0.0,
        )
    return PIPELINES[cache_key]


def encode_jpg(frame_bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode preview JPEG.")
    return buf.tobytes()


def set_worker_phase(phase: str):
    if STATE.worker_phase != phase:
        STATE.worker_phase = phase
        STATE.worker_phase_since = time.perf_counter()


def worker_loop():
    try:
        with STATE.lock:
            model_key = STATE.model_key
            headstart_seconds = STATE.headstart_seconds
            max_seconds = STATE.max_seconds
            max_buffer_frames = STATE.max_buffer_frames
            min_hold_frames = STATE.min_hold_frames
            max_live_lag_frames = STATE.max_live_lag_frames
            process_every_n = STATE.process_every_n
            assigned_process_every_n = STATE.assigned_process_every_n
            identity_buffer_frames = STATE.identity_buffer_frames
            silhouette_every_n_processed = STATE.silhouette_every_n_processed
            preview_max_side = STATE.preview_max_side
            run_root = STATE.run_root
        pipeline = get_pipeline(model_key)
        pipeline.min_detection_score = float(STATE.min_detection_score)

        first_track_offset = None
        track_sil_counts: Dict[int, int] = {}
        track_labels: Dict[int, str] = {}
        track_scores: Dict[int, float] = {}
        track_score_details: Dict[int, Dict[str, float]] = {}
        counted_track_ids = set()
        cumulative_counts = {"pritom": 0, "coco": 0}
        stage_times = {
            "wait": 0.0,
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

        while True:
            set_worker_phase("loop_top")
            with STATE.lock:
                if not STATE.running:
                    break
                elapsed = time.perf_counter() - STATE.started_at if STATE.started_at else 0.0
                buffered = len(STATE.queue)
                newest_elapsed = STATE.queue[-1]["elapsed"] if STATE.queue else None
                if elapsed >= max_seconds and buffered == 0:
                    STATE.running = False
                    STATE.last_status = "Finished: queue drained after max duration."
                    break

            if newest_elapsed is None or newest_elapsed < headstart_seconds:
                set_worker_phase("waiting_for_headstart")
                t0 = time.perf_counter()
                time.sleep(0.03)
                stage_times["wait"] += time.perf_counter() - t0
                with STATE.lock:
                    STATE.last_status = (
                        f"Buffering live webcam... "
                        f"server_frames={STATE.preview_seen} buffered={len(STATE.queue)} "
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
                        f"Holding live buffer... "
                        f"buffered={buffered_now} min_hold={min_hold_frames} "
                        f"captured={STATE.preview_seen} processed={STATE.processed_frames}"
                    )
                    need_wait = True
                else:
                    need_wait = False
            if need_wait:
                set_worker_phase("holding_min_buffer")
                t0 = time.perf_counter()
                time.sleep(0.02)
                stage_times["wait"] += time.perf_counter() - t0
                continue

            with STATE.lock:
                set_worker_phase("pop_queue")
                queued_before_pop = len(STATE.queue)
                if not STATE.queue:
                    item = None
                else:
                    item = STATE.queue.popleft()
            if item is None:
                set_worker_phase("waiting_for_queue_item")
                t0 = time.perf_counter()
                time.sleep(0.02)
                stage_times["wait"] += time.perf_counter() - t0
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
                event_lines=STATE.logs,
                stage_times=stage_times,
                stage_counts=stage_counts,
            )
            for track_id in track_labels.keys():
                if track_id in track_score_details:
                    continue
                track_dir = run_root / "silhouettes" / "live_probe" / f"{track_id:03d}" / "undefined"
                if not track_dir.exists():
                    continue
                track_input = track_silhouettes_to_input(track_dir)
                if track_input is None:
                    continue
                embedding = pipeline.extract_embedding(pipeline.gait_model, track_input)
                score_map = {
                    str(label): float(pipeline.cosine_similarity(embedding, gallery_embedding))
                    for label, gallery_embedding in zip(pipeline.gallery_labels, pipeline.gallery_embeddings)
                }
                track_score_details[track_id] = score_map
                with STATE.lock:
                    STATE.logs.append(
                        f"track {track_id:03d} score_details="
                        + ", ".join(f"{k}={v:.4f}" for k, v in score_map.items())
                    )
            for track_id, label in track_labels.items():
                if track_id not in counted_track_ids:
                    counted_track_ids.add(track_id)
                    if label in cumulative_counts:
                        cumulative_counts[label] += 1
            all_visible_assigned = bool(active) and not unassigned_active
            set_worker_phase("draw_preview")
            draw_t0 = time.perf_counter()
            counts = dict(cumulative_counts)
            draw_panel(frame, [f"Counts: pritom={counts['pritom']} | coco={counts['coco']}"])
            stage_times["draw"] += time.perf_counter() - draw_t0
            if preview_max_side > 0:
                h, w = frame.shape[:2]
                largest = max(h, w)
                if largest > preview_max_side:
                    scale = preview_max_side / largest
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            with STATE.lock:
                set_worker_phase("encode_and_publish")
                STATE.processed_frames += 1
                STATE.last_processed_frame_index = frame_index
                STATE.track_labels = dict(track_labels)
                STATE.counts = counts
                STATE.track_score_details = {
                    str(track_id): {label: float(score) for label, score in score_map.items()}
                    for track_id, score_map in track_score_details.items()
                }
                STATE.last_preview_jpg = encode_jpg(frame, quality=80)
                wall = time.perf_counter() - STATE.started_at if STATE.started_at else 0.0
                realtime = item["elapsed"] / wall if wall > 0 else 0.0
                phase_age = (
                    time.perf_counter() - STATE.worker_phase_since
                    if STATE.worker_phase_since is not None
                    else 0.0
                )
                score_lines = []
                for track_id, label in sorted(STATE.track_labels.items()):
                    detail = STATE.track_score_details.get(str(track_id), {})
                    if detail:
                        detail_text = ", ".join(f"{k}={v:.4f}" for k, v in detail.items())
                        score_lines.append(f"track {track_id:03d} -> {label} | {detail_text}")
                score_block = (("Score details:\n" + "\n".join(score_lines) + "\n") if score_lines else "")
                STATE.last_status = (
                    f"Running live webcam recognition\n"
                    f"Captured frames: {STATE.preview_seen}\n"
                    f"Buffered frames: {len(STATE.queue)} / {max_buffer_frames}\n"
                    f"Minimum hold buffer: {min_hold_frames}\n"
                    f"Maximum live lag: {max_live_lag_frames}\n"
                    f"Process stride: dense={process_every_n}, assigned={assigned_process_every_n}\n"
                    f"Effective stride this frame: {effective_stride}\n"
                    f"Identity buffer frames: {identity_buffer_frames}\n"
                    f"Segmentation every N processed frames: {silhouette_every_n_processed}\n"
                    f"Min detection score: {STATE.min_detection_score:.2f}\n"
                    f"Processed frames: {STATE.processed_frames}\n"
                    f"Last processed frame index: {STATE.last_processed_frame_index}\n"
                    f"Dropped frames: {STATE.dropped_frames}\n"
                    f"Worker phase: {STATE.worker_phase} ({phase_age:.2f}s)\n"
                    f"Input time processed: {item['elapsed']:.2f}s\n"
                    f"Wall time: {wall:.2f}s\n"
                    f"Realtime factor: {realtime:.2f}x\n"
                    f"Counts: pritom={counts['pritom']} | coco={counts['coco']}\n"
                    f"{score_block}"
                    f"Run folder: {STATE.run_root}"
                )
            set_worker_phase("published")
    except Exception as exc:
        with STATE.lock:
            STATE.running = False
            STATE.last_error = repr(exc)
            STATE.last_status = f"Worker error: {type(exc).__name__}: {exc}"
            if STATE.run_root:
                try:
                    (STATE.run_root / "error_log.txt").write_text(STATE.last_status)
                except Exception:
                    pass


app = FastAPI(title="Live Webcam Gait Demo")


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Live Webcam Gait Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #111; color: #eee; }
    .row { display: flex; gap: 20px; align-items: flex-start; }
    .col { flex: 1; min-width: 320px; }
    video, img { width: 100%; max-width: 640px; border: 1px solid #444; background: #000; }
    button, select, input { margin: 6px 6px 6px 0; padding: 8px 12px; }
    pre { white-space: pre-wrap; background: #1a1a1a; padding: 12px; border: 1px solid #333; min-height: 220px; }
    #toast {
      position: fixed;
      top: 18px;
      right: 18px;
      background: rgba(30, 30, 30, 0.95);
      color: #fff;
      border: 1px solid #555;
      padding: 12px 16px;
      border-radius: 10px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
      z-index: 9999;
      max-width: 360px;
    }
    #toast.show { opacity: 1; }
  </style>
</head>
<body>
  <h1>Live Webcam Gait Demo</h1>
  <p>This version uses browser webcam capture plus backend queue processing. Left is raw webcam, right is delayed processed preview.</p>
  <div id="toast"></div>
  <div>
    <button id="camera-on">Turn Camera On</button>
    <button id="camera-off">Turn Camera Off</button>
    <label>Model</label>
    <select id="model">
      <option value="grew_gaitbase">grew_gaitbase</option>
      <option value="grew_gaitgl">grew_gaitgl</option>
      <option value="current_gaitbase">current_gaitbase</option>
    </select>
    <label>Head-start buffer (s)</label>
    <input id="headstart" type="number" value="2" min="0" max="10" step="0.5"/>
    <label>Capture interval (ms)</label>
    <input id="interval" type="number" value="50" min="50" max="1000" step="10"/>
    <label>Max seconds</label>
    <input id="maxseconds" type="number" value="300" min="5" max="300" step="5"/>
    <label>Dense stride</label>
    <input id="dense_stride" type="number" value="5" min="1" max="20" step="1"/>
    <label>Assigned stride</label>
    <input id="assigned_stride" type="number" value="10" min="1" max="40" step="1"/>
    <label>Identity buffer</label>
    <input id="identity_buffer" type="number" value="5" min="1" max="20" step="1"/>
    <label>Seg every N</label>
    <input id="seg_every" type="number" value="1" min="1" max="10" step="1"/>
    <label>Min det score</label>
    <input id="min_det" type="number" value="0" min="0" max="0.95" step="0.05"/>
    <label>Preview max side</label>
    <input id="preview_max_side" type="number" value="720" min="240" max="1080" step="40"/>
    <button id="start">Start Live Demo</button>
    <button id="stop">Stop</button>
  </div>
  <div class="row">
    <div class="col">
      <h3>Raw Webcam</h3>
      <video id="video" autoplay muted playsinline></video>
    </div>
    <div class="col">
      <h3>Delayed Processed Preview</h3>
      <img id="preview" />
    </div>
  </div>
  <div class="row">
    <div class="col">
      <h3>Status</h3>
      <pre id="status">Idle.</pre>
    </div>
    <div class="col">
      <h3>Detailed Debug</h3>
      <pre id="client">Not started.</pre>
    </div>
  </div>
  <script>
    const video = document.getElementById('video');
    const statusEl = document.getElementById('status');
    const clientEl = document.getElementById('client');
    const preview = document.getElementById('preview');
    const toastEl = document.getElementById('toast');
    const canvas = document.createElement('canvas');
    let stream = null;
    let sendTimer = null;
    let pollTimer = null;
    let started = false;
    let clientFramesSent = 0;
    let clientSendFailures = 0;
    let clientLastSendAt = null;
    let clientLastPollAt = null;
    let clientLastPreviewAt = null;
    let clientStartedAt = null;

    function showToast(message, ms = 1800) {
      toastEl.textContent = message;
      toastEl.classList.add('show');
      setTimeout(() => toastEl.classList.remove('show'), ms);
    }

    async function ensureCamera() {
      if (stream) return;
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      await video.play();
      clientEl.textContent = 'Camera is on.';
      showToast('Camera turned on');
    }

    function stopCamera() {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      video.srcObject = null;
      clientEl.textContent = 'Camera is off.';
      showToast('Camera turned off');
    }

    async function startDemo() {
      if (!stream) {
        statusEl.textContent = 'Turn the camera on first.';
        return;
      }
      const payload = {
        model_key: document.getElementById('model').value,
        headstart_seconds: parseFloat(document.getElementById('headstart').value),
        capture_interval_ms: parseInt(document.getElementById('interval').value),
        max_seconds: parseFloat(document.getElementById('maxseconds').value),
        max_buffer_frames: 240,
        process_every_n: parseInt(document.getElementById('dense_stride').value),
        assigned_process_every_n: parseInt(document.getElementById('assigned_stride').value),
        identity_buffer_frames: parseInt(document.getElementById('identity_buffer').value),
        silhouette_every_n_processed: parseInt(document.getElementById('seg_every').value),
        min_detection_score: parseFloat(document.getElementById('min_det').value),
        preview_max_side: parseInt(document.getElementById('preview_max_side').value)
      };
      const res = await fetch('/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        statusEl.textContent = data.detail || 'Failed to start.';
        showToast('Live demo failed to start');
        return;
      }
      started = true;
      clientFramesSent = 0;
      clientSendFailures = 0;
      clientStartedAt = performance.now();
      clientEl.textContent = 'Live capture active.';
      showToast('Live demo started');
      beginSending(payload.capture_interval_ms);
      beginPolling();
    }

    function beginSending(intervalMs) {
      if (sendTimer) clearInterval(sendTimer);
      sendTimer = setInterval(async () => {
        if (!started || !video.videoWidth || !video.videoHeight) return;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.75));
        if (!blob) return;
        const form = new FormData();
        form.append('frame', blob, 'frame.jpg');
        fetch('/push_frame', { method: 'POST', body: form })
          .then(() => {
            clientFramesSent += 1;
            clientLastSendAt = performance.now();
          })
          .catch(() => {
            clientSendFailures += 1;
          });
      }, intervalMs);
    }

    function beginPolling() {
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(async () => {
        const res = await fetch('/status');
        const data = await res.json();
        clientLastPollAt = performance.now();
        statusEl.textContent = data.status;
        const now = performance.now();
        const runtimeSec = clientStartedAt ? (now - clientStartedAt) / 1000.0 : 0.0;
        const clientSendRate = runtimeSec > 0 ? (clientFramesSent / runtimeSec) : 0.0;
        const lastSendAge = clientLastSendAt ? ((now - clientLastSendAt) / 1000.0).toFixed(2) : 'n/a';
        const lastPollAge = clientLastPollAt ? ((now - clientLastPollAt) / 1000.0).toFixed(2) : 'n/a';
        const lastPreviewAge = clientLastPreviewAt ? ((now - clientLastPreviewAt) / 1000.0).toFixed(2) : 'n/a';
        clientEl.textContent =
          `client_frames_sent=${clientFramesSent}\\n` +
          `client_send_failures=${clientSendFailures}\\n` +
          `client_send_rate_fps=${clientSendRate.toFixed(2)}\\n` +
          `client_last_send_age_s=${lastSendAge}\\n` +
          `client_last_poll_age_s=${lastPollAge}\\n` +
          `client_last_preview_age_s=${lastPreviewAge}\\n` +
          `server_preview_seen=${data.preview_seen}\\n` +
          `queue=${data.buffered}\\n` +
          `processed=${data.processed_frames}\\n` +
          `dropped=${data.dropped_frames}\\n` +
          `worker_phase=${data.worker_phase}\\n` +
          `worker_phase_age_s=${(data.worker_phase_age ?? 0).toFixed(2)}\\n` +
          `last_processed_frame_index=${data.last_processed_frame_index}\\n` +
          `last_error=${data.last_error}\\n` +
          `counts=${JSON.stringify(data.counts)}\\n` +
          `track_score_details=${JSON.stringify(data.track_score_details)}\\n` +
          `backend_settings=${JSON.stringify(data.settings)}`;
        if (data.has_preview) {
          preview.src = '/preview.jpg?t=' + Date.now();
          clientLastPreviewAt = performance.now();
        }
        if (!data.running && started) {
          started = false;
          if (sendTimer) clearInterval(sendTimer);
          showToast('Live demo stopped');
        }
      }, 400);
    }

    async function stopDemo() {
      started = false;
      if (sendTimer) clearInterval(sendTimer);
      const res = await fetch('/stop', { method: 'POST' });
      const data = await res.json();
      statusEl.textContent = data.status;
      clientEl.textContent = 'Stopped.';
      showToast('Stop requested');
    }

    document.getElementById('start').onclick = startDemo;
    document.getElementById('stop').onclick = stopDemo;
    document.getElementById('camera-on').onclick = () => ensureCamera().catch(err => { statusEl.textContent = 'Camera error: ' + err; });
    document.getElementById('camera-off').onclick = stopCamera;
  </script>
</body>
</html>
"""


@app.post("/start")
async def start_live(request: Request):
    data = await request.json()
    model_key = data.get("model_key", "grew_gaitbase")
    gallery_path = gallery_for_model(model_key)
    if not gallery_path.exists():
        raise HTTPException(status_code=400, detail=f"Missing gallery cache: {gallery_path}")
    settings = {
        "model_key": model_key,
        "headstart_seconds": float(data.get("headstart_seconds", 2.0)),
        "capture_interval_ms": int(data.get("capture_interval_ms", 50)),
        "max_seconds": float(data.get("max_seconds", 300.0)),
        "max_buffer_frames": int(data.get("max_buffer_frames", 240)),
        "process_every_n": int(data.get("process_every_n", 5)),
        "assigned_process_every_n": int(data.get("assigned_process_every_n", 10)),
        "identity_buffer_frames": int(data.get("identity_buffer_frames", 5)),
        "silhouette_every_n_processed": int(data.get("silhouette_every_n_processed", 1)),
        "min_detection_score": float(data.get("min_detection_score", 0.0)),
        "preview_max_side": int(data.get("preview_max_side", 720)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    computed_hold_frames = max(
        3,
        int(round(settings["headstart_seconds"] * 1000.0 / max(1, settings["capture_interval_ms"]))),
    )
    computed_live_lag_frames = max(computed_hold_frames + 6, computed_hold_frames * 2)
    with STATE.lock:
        STATE.model_key = settings["model_key"]
        STATE.headstart_seconds = settings["headstart_seconds"]
        STATE.capture_interval_ms = settings["capture_interval_ms"]
        STATE.max_seconds = settings["max_seconds"]
        STATE.process_every_n = max(1, settings["process_every_n"])
        STATE.assigned_process_every_n = max(STATE.process_every_n, settings["assigned_process_every_n"])
        STATE.identity_buffer_frames = max(1, settings["identity_buffer_frames"])
        STATE.silhouette_every_n_processed = max(1, settings["silhouette_every_n_processed"])
        STATE.min_detection_score = max(0.0, settings["min_detection_score"])
        STATE.preview_max_side = max(0, settings["preview_max_side"])
        STATE.min_hold_frames = computed_hold_frames
        STATE.max_live_lag_frames = computed_live_lag_frames
        STATE.max_buffer_frames = max(int(settings["max_buffer_frames"]), computed_hold_frames + 10)
        settings["min_hold_frames"] = STATE.min_hold_frames
        settings["max_live_lag_frames"] = STATE.max_live_lag_frames
        settings["effective_max_buffer_frames"] = STATE.max_buffer_frames
        STATE.reset_for_run(settings)
        STATE.worker = threading.Thread(target=worker_loop, daemon=True)
        STATE.worker.start()
    return JSONResponse({"ok": True, "status": "Live demo started."})


@app.post("/push_frame")
async def push_frame(request: Request):
    form = await request.form()
    upload = form.get("frame")
    if upload is None:
        raise HTTPException(status_code=400, detail="Missing frame upload.")
    body = await upload.read()
    arr = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid JPEG frame.")
    with STATE.lock:
        STATE.preview_seen += 1
        STATE.last_preview_at = time.perf_counter()
        if not STATE.running or STATE.started_at is None:
            return JSONResponse({"ok": True, "queued": False})
        elapsed = time.perf_counter() - STATE.started_at
        item = {
            "index": STATE.frame_index,
            "elapsed": elapsed,
            "frame": frame,
        }
        STATE.frame_index += 1
        STATE.queue.append(item)
        while len(STATE.queue) > STATE.max_buffer_frames:
            STATE.queue.popleft()
            STATE.dropped_frames += 1
    return JSONResponse({"ok": True, "queued": True})


@app.get("/status")
async def status():
    with STATE.lock:
        phase_age = (
            time.perf_counter() - STATE.worker_phase_since
            if STATE.worker_phase_since is not None
            else None
        )
        return JSONResponse(
            {
                "running": STATE.running,
                "buffered": len(STATE.queue),
                "processed_frames": STATE.processed_frames,
                "preview_seen": STATE.preview_seen,
                "dropped_frames": STATE.dropped_frames,
                "counts": STATE.counts,
                "track_score_details": STATE.track_score_details,
                "status": STATE.last_status,
                "has_preview": STATE.last_preview_jpg is not None,
                "run_root": str(STATE.run_root) if STATE.run_root else None,
                "worker_phase": STATE.worker_phase,
                "worker_phase_age": phase_age,
                "last_processed_frame_index": STATE.last_processed_frame_index,
                "last_error": STATE.last_error,
                "settings": {
                    "model_key": STATE.model_key,
                    "headstart_seconds": STATE.headstart_seconds,
                    "capture_interval_ms": STATE.capture_interval_ms,
                    "max_seconds": STATE.max_seconds,
                    "process_every_n": STATE.process_every_n,
                    "assigned_process_every_n": STATE.assigned_process_every_n,
                    "identity_buffer_frames": STATE.identity_buffer_frames,
                    "silhouette_every_n_processed": STATE.silhouette_every_n_processed,
                    "min_detection_score": STATE.min_detection_score,
                    "preview_max_side": STATE.preview_max_side,
                    "min_hold_frames": STATE.min_hold_frames,
                    "max_live_lag_frames": STATE.max_live_lag_frames,
                },
            }
        )


@app.get("/preview.jpg")
async def preview_jpg():
    with STATE.lock:
        if STATE.last_preview_jpg is None:
            raise HTTPException(status_code=404, detail="No preview yet.")
        return Response(content=STATE.last_preview_jpg, media_type="image/jpeg")


@app.post("/stop")
async def stop():
    with STATE.lock:
        STATE.running = False
        STATE.stop_requested = True
        STATE.last_status = "Stop requested."
    return JSONResponse({"ok": True, "status": "Stop requested."})


if __name__ == "__main__":
    port = int(os.environ.get("LIVE_DEMO_PORT", "8010"))
    public_url = None
    use_public_tunnel = os.environ.get("LIVE_DEMO_PUBLIC", "1") != "0"
    if use_public_tunnel:
        try:
            authtoken = os.environ.get("NGROK_AUTHTOKEN")
            if authtoken:
                ngrok.set_auth_token(authtoken)
            public_url = ngrok.connect(addr=port, proto="http").public_url
            print(f"[live-demo] public_url={public_url}", flush=True)
        except Exception as exc:
            print(
                "[live-demo] public tunnel failed. "
                "If you want a public URL, set NGROK_AUTHTOKEN first. "
                f"Details: {exc}",
                flush=True,
            )
    uvicorn.run(app, host="0.0.0.0", port=port)
