#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
CLEAN_ROOT="$ROOT/clean_demo_v2"
OUT_ROOT="$CLEAN_ROOT/output/pritom_coco_test1_closed_$TS"
WORK_ROOT="$OUT_ROOT/work"

MODEL="${MODEL:-grew_gaitbase}"

mkdir -p "$OUT_ROOT"

echo "[pritom-coco-closed] timestamp=$TS"
echo "[pritom-coco-closed] model=$MODEL"
echo "[pritom-coco-closed] gallery pritom: $CLEAN_ROOT/gallery/pritomgallery.mp4"
echo "[pritom-coco-closed] gallery coco:   $CLEAN_ROOT/gallery/cocogallery.mp4"
echo "[pritom-coco-closed] probe:          $CLEAN_ROOT/probes/test1probe.mp4"

python clean_demo_v2/tools/build_two_identity_gallery.py \
  --pritom-video "$CLEAN_ROOT/gallery/pritomgallery.mp4" \
  --coco-video "$CLEAN_ROOT/gallery/cocogallery.mp4" \
  --model "$MODEL" \
  --work-root "$WORK_ROOT/gallery_build" \
  --gallery-out "$OUT_ROOT/pritom_coco_gallery.npz" \
  --metadata-out "$OUT_ROOT/pritom_coco_gallery.json"

python clean_demo_v2/tools/identify_probe_with_gallery.py \
  --gallery-npz "$OUT_ROOT/pritom_coco_gallery.npz" \
  --video-path "$CLEAN_ROOT/probes/test1probe.mp4" \
  --model "$MODEL" \
  --threshold -1 \
  --margin 0 \
  --match-mode max \
  --work-root "$WORK_ROOT/probe_test1" \
  --output-json "$OUT_ROOT/test1_pritom_coco_closed_result.json"

python - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
result_path = out_root / "test1_pritom_coco_closed_result.json"
payload = json.loads(result_path.read_text())
summary_path = out_root / "test1_pritom_coco_closed_summary.txt"
lines = [
    "Pritom/Coco closed-set Test1 summary",
    f"Counts: {payload['summary']['counts']}",
    f"Processed entries: {payload['summary']['processed_entries']}",
    f"Skipped entries: {payload['summary']['skipped_entries']}",
    "",
    "Per-entry assignments:",
]
for entry in payload["entries"]:
    if entry["status"] != "ok":
        lines.append(f"{entry['entry_key']}: skipped {entry['status']}")
        continue
    lines.append(
        f"{entry['entry_key']}: {entry['assigned_identity']} "
        f"best={entry['best_score']:.4f} second={entry['second_identity']}:{entry['second_score']:.4f}"
    )
summary_path.write_text("\n".join(lines))
print(json.dumps(payload["summary"], indent=2))
print(f"[pritom-coco-closed] saved summary: {summary_path}")
PY

python clean_demo_v2/tools/render_closed_set_annotations.py \
  --video-path "$CLEAN_ROOT/probes/test1probe.mp4" \
  --tracking-txt "$WORK_ROOT/probe_test1/tracking/test1probe/test1probe.txt" \
  --result-json "$OUT_ROOT/test1_pritom_coco_closed_result.json" \
  --output-video "$OUT_ROOT/test1_pritom_coco_closed_annotated.mp4" \
  --title "Pritom vs Coco closed-set"

echo "[pritom-coco-closed] finished"
echo "[pritom-coco-closed] output folder: $OUT_ROOT"
echo "[pritom-coco-closed] result JSON: $OUT_ROOT/test1_pritom_coco_closed_result.json"
echo "[pritom-coco-closed] annotated video: $OUT_ROOT/test1_pritom_coco_closed_annotated.mp4"
