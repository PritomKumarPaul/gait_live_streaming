#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ppaul11/All-in-One-Gait"
cd "$ROOT"

GPU_ID="${GPU_ID:-1}"
MODEL="${MODEL:-grew_gaitbase}"

echo "[v2-ablation-baseline] This is the correctness-first baseline."
echo "[v2-ablation-baseline] It uses the original probe, original detector size, no frame resize,"
echo "[v2-ablation-baseline] frequent detection, logs enabled, and annotated MP4 writing enabled."
echo "[v2-ablation-baseline] GPU_ID=$GPU_ID MODEL=$MODEL"

CUDA_VISIBLE_DEVICES="$GPU_ID" python live_demo/run_buffered_live_probe.py \
  --video clean_demo_v2/probes/test1probe.mp4 \
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
  --log-every 5 \
  --write-output-video

echo "[v2-ablation-baseline] latest metrics:"
python live_demo/summarize_live_metrics.py
