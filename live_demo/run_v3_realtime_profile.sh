#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ppaul11/All-in-One-Gait"
cd "$ROOT"

GPU_ID="${GPU_ID:-1}"
MODEL="${MODEL:-grew_gaitbase}"
PROBE="${PROBE:-live_demo/prepared_inputs/test1probe_720p30.mp4}"

echo "[v3-realtime-profile] Realtime presentation profile."
echo "[v3-realtime-profile] Correct run reference: buffered_live_20260420_160447"
echo "[v3-realtime-profile] Expected: pritom=3 coco=3, about 1.0x realtime on this server."
echo "[v3-realtime-profile] GPU_ID=$GPU_ID MODEL=$MODEL PROBE=$PROBE"

CUDA_VISIBLE_DEVICES="$GPU_ID" python live_demo/run_buffered_live_probe.py \
  --video "$PROBE" \
  --model "$MODEL" \
  --process-every-n 5 \
  --assigned-process-every-n 10 \
  --detector-input-size 0 \
  --work-frame-max-side 0 \
  --min-detection-score 0 \
  --output-max-side 480 \
  --silhouette-every-n-processed 1 \
  --identity-buffer-frames 5 \
  --max-seconds 30 \
  --log-every 0 \
  --quiet \
  --write-output-video

echo "[v3-realtime-profile] latest metrics:"
python live_demo/summarize_live_metrics.py
