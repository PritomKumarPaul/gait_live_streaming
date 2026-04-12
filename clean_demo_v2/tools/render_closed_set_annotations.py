import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2


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


def color_for_label(label: str):
    colors = {
        "pritom": (0, 220, 0),
        "coco": (255, 140, 0),
        "jeevith": (0, 165, 255),
        "unknown": (180, 180, 180),
    }
    return colors.get(label, (255, 255, 255))


def draw_label(frame, x: int, y: int, text: str, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 8, y0 + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (x + 4, y0 + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def build_decisions(result_json: Path) -> Dict[str, Dict]:
    payload = json.loads(result_json.read_text())
    counts = Counter()
    decisions = {}
    for entry in sorted(payload["entries"], key=lambda item: item["track_id"]):
        if entry["status"] != "ok":
            continue
        label = entry["assigned_identity"]
        counts[label] += 1
        decisions[entry["track_id"]] = {
            "label": label,
            "count": counts[label],
            "score": entry["best_score"],
            "second_identity": entry["second_identity"],
            "second_score": entry["second_score"],
        }
    return decisions


def main():
    parser = argparse.ArgumentParser(
        description="Render a closed-set annotated video from an existing result JSON and tracking TXT."
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--tracking-txt", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--title", default="Closed-set gait recognition")
    args = parser.parse_args()

    video_path = Path(args.video_path).resolve()
    tracking_txt = Path(args.tracking_txt).resolve()
    result_json = Path(args.result_json).resolve()
    output_video = Path(args.output_video).resolve()

    frame_tracks = load_tracking_file(tracking_txt)
    decisions = build_decisions(result_json)

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
        cv2.putText(frame, args.title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        for item in frame_tracks.get(frame_id, []):
            decision = decisions.get(item["track_id"])
            if decision is None:
                continue
            x, y, w, h = item["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            color = color_for_label(decision["label"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{decision['label']} count={decision['count']} score={decision['score']:.3f}"
            draw_label(frame, x1, max(0, y1 - 4), text, color)
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"[render] saved annotated video to {output_video}", flush=True)


if __name__ == "__main__":
    main()
