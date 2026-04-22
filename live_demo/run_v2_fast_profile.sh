#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ppaul11/All-in-One-Gait"
cd "$ROOT"

GPU_ID="${GPU_ID:-1}"
MODEL="${MODEL:-grew_gaitbase}"
PROBE="${PROBE:-live_demo/prepared_inputs/test1probe_1080p30.mp4}"

echo "[v2-fast-profile] Combined speed profile with restored assignment logic."
echo "[v2-fast-profile] Uses reduced-resolution probe, original detector size, quiet logs,"
echo "[v2-fast-profile] minimal overlay, and no final MP4 writing."
echo "[v2-fast-profile] GPU_ID=$GPU_ID MODEL=$MODEL PROBE=$PROBE"

CUDA_VISIBLE_DEVICES="$GPU_ID" python live_demo/run_buffered_live_probe.py \
  --video "$PROBE" \
  --model "$MODEL" \
  --process-every-n 2 \
  --assigned-process-every-n 6 \
  --detector-input-size 0 \
  --work-frame-max-side 0 \
  --min-detection-score 0 \
  --output-max-side 480 \
  --silhouette-every-n-processed 1 \
  --identity-buffer-frames 8 \
  --max-seconds 30 \
  --log-every 0 \
  --quiet

echo "[v2-fast-profile] latest metrics:"
python live_demo/summarize_live_metrics.py
