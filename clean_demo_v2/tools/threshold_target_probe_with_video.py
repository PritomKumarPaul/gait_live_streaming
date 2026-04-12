import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "OpenGait" / "tools"))

from probe_only_entry_analysis import analyze_video, build_model, cosine_similarity, log  # noqa: E402


def load_gallery(path: Path):
    data = np.load(path)
    return data["embeddings"].astype(np.float32), data["labels"].astype(str)


def score_to_gallery(embedding: np.ndarray, gallery_embeddings: np.ndarray) -> float:
    return float(max(cosine_similarity(embedding, gallery_embedding) for gallery_embedding in gallery_embeddings))


def load_tracking_file(path: Path) -> Dict[int, List[Dict]]:
    frame_tracks: Dict[int, List[Dict]] = {}
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


def draw_label(frame, x: int, y: int, text: str, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 8, y0 + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (x + 4, y0 + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def render(video_path: Path, tracking_txt: Path, decisions: List[Dict], output_video: Path, threshold: float):
    decision_by_track = {item["track_id"]: item for item in decisions}
    frame_tracks = load_tracking_file(tracking_txt)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(
            frame,
            f"Pritom vs Unknown | threshold={threshold}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        for item in frame_tracks.get(frame_id, []):
            decision = decision_by_track.get(item["track_id"])
            if decision is None:
                continue
            x, y, w, h = item["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            label = decision["assigned_label"]
            color = (0, 220, 0) if label == "pritom" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{label} count={decision['display_count']} score={decision['score_to_target']:.3f}"
            draw_label(frame, x1, max(0, y1 - 4), text, color)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="Threshold one target gallery against one probe and render readable labels."
    )
    parser.add_argument("--gallery-npz", required=True)
    parser.add_argument("--probe-video", required=True)
    parser.add_argument("--target-label", default="pritom")
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-txt", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    args = parser.parse_args()

    gallery_npz = Path(args.gallery_npz).resolve()
    probe_video = Path(args.probe_video).resolve()
    output_json = Path(args.output_json).resolve()
    summary_txt = Path(args.summary_txt).resolve()
    output_video = Path(args.output_video).resolve()
    work_root = Path(args.work_root).resolve()

    gallery_embeddings, labels = load_gallery(gallery_npz)
    target_embeddings = gallery_embeddings[labels == args.target_label]
    if target_embeddings.shape[0] == 0:
        raise ValueError(f"No embeddings found for target label {args.target_label}")

    model, profile = build_model(args.model)
    entries = analyze_video(model=model, video_path=probe_video, work_root=work_root, min_frames=args.min_frames)

    counts = Counter()
    decisions = []
    skipped = []
    for entry in sorted(entries, key=lambda item: item["track_id"]):
        if entry["status"] != "ok":
            skipped.append(
                {
                    "entry_key": entry["entry_key"],
                    "track_id": entry["track_id"],
                    "frame_count": entry["frame_count"],
                    "status": entry["status"],
                }
            )
            continue
        score = score_to_gallery(entry["embedding"], target_embeddings)
        assigned = args.target_label if score >= args.threshold else "unknown"
        counts[assigned] += 1
        decisions.append(
            {
                "entry_key": entry["entry_key"],
                "video": entry["video"],
                "track_id": entry["track_id"],
                "frame_count": entry["frame_count"],
                "score_to_target": score,
                "assigned_label": assigned,
                "display_count": counts[assigned],
            }
        )

    tracking_txt = work_root / "tracking" / probe_video.stem / f"{probe_video.stem}.txt"
    render(probe_video, tracking_txt, decisions, output_video, args.threshold)

    payload = {
        "config": {
            "gallery_npz": str(gallery_npz),
            "probe_video": str(probe_video),
            "target_label": args.target_label,
            "threshold": args.threshold,
            "model": profile["display_name"],
            "model_key": args.model,
            "output_video": str(output_video),
        },
        "summary": {
            "processed_entries": len(decisions),
            "skipped_entries": len(skipped),
            "counts": dict(counts),
        },
        "entries": decisions,
        "skipped_entries": skipped,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text(
        "\n".join(
            [
                "Clean Demo V2: Pritom vs Unknown",
                f"Probe: {probe_video}",
                f"Threshold: {args.threshold}",
                f"Counts: {dict(counts)}",
                f"Processed entries: {len(decisions)}",
                f"Skipped entries: {len(skipped)}",
                f"Output video: {output_video}",
            ]
        )
    )
    log(json.dumps(payload["summary"], indent=2))
    log(f"[v2] saved JSON to {output_json}")
    log(f"[v2] saved summary to {summary_txt}")
    log(f"[v2] saved annotated video to {output_video}")


if __name__ == "__main__":
    main()
