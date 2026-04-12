import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from probe_only_entry_analysis import (
    PROJECT_ROOT,
    REPO_ROOT,
    analyze_video,
    build_model,
    cosine_similarity,
    log,
)


def load_gallery(gallery_npz: Path, target_label: str) -> Dict:
    data = np.load(gallery_npz)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(str)
    target_mask = labels == target_label
    if not np.any(target_mask):
        raise ValueError(f"Target label '{target_label}' was not found in {gallery_npz}")
    impostor_mask = labels != target_label
    return {
        "target_embeddings": embeddings[target_mask],
        "impostor_embeddings": embeddings[impostor_mask],
        "impostor_labels": labels[impostor_mask],
    }


def score_against_gallery(embedding: np.ndarray, gallery_embeddings: np.ndarray) -> float:
    scores = [cosine_similarity(embedding, gallery_embedding) for gallery_embedding in gallery_embeddings]
    return float(max(scores)) if scores else -1.0


def safe_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 10**9


def build_decisions(
    entries: List[Dict],
    target_gallery: np.ndarray,
    impostor_gallery: np.ndarray,
    target_label: str,
    threshold: float,
    margin: float,
) -> Dict:
    ok_entries = [entry for entry in entries if entry["status"] == "ok"]
    ok_entries = sorted(ok_entries, key=lambda entry: safe_int(entry["track_id"]))

    modes = {
        "no_threshold": [],
        "with_threshold": [],
    }

    counts = {
        "no_threshold": Counter(),
        "with_threshold": Counter(),
    }

    for entry in ok_entries:
        target_score = score_against_gallery(entry["embedding"], target_gallery)
        impostor_score = score_against_gallery(entry["embedding"], impostor_gallery)
        score_margin = target_score - impostor_score

        counts["no_threshold"][target_label] += 1
        modes["no_threshold"].append(
            {
                "entry_key": entry["entry_key"],
                "video": entry["video"],
                "track_id": entry["track_id"],
                "frame_count": entry["frame_count"],
                "score_to_target": target_score,
                "best_impostor_score": impostor_score,
                "score_margin": score_margin,
                "assigned_label": target_label,
                "display_count": counts["no_threshold"][target_label],
                "accepted_by_threshold": True,
                "accepted_by_margin": True,
            }
        )

        accepted_by_threshold = target_score >= threshold
        accepted_by_margin = score_margin >= margin
        assigned_label = target_label if accepted_by_threshold and accepted_by_margin else "unknown"
        counts["with_threshold"][assigned_label] += 1
        modes["with_threshold"].append(
            {
                "entry_key": entry["entry_key"],
                "video": entry["video"],
                "track_id": entry["track_id"],
                "frame_count": entry["frame_count"],
                "score_to_target": target_score,
                "best_impostor_score": impostor_score,
                "score_margin": score_margin,
                "assigned_label": assigned_label,
                "display_count": counts["with_threshold"][assigned_label],
                "accepted_by_threshold": accepted_by_threshold,
                "accepted_by_margin": accepted_by_margin,
            }
        )

    skipped = [
        {
            "entry_key": entry["entry_key"],
            "video": entry["video"],
            "track_id": entry["track_id"],
            "frame_count": entry["frame_count"],
            "status": entry["status"],
        }
        for entry in entries
        if entry["status"] != "ok"
    ]

    return {
        "summary": {
            mode: {
                "processed_entries": len(modes[mode]),
                "skipped_entries": len(skipped),
                "counts": dict(counts[mode]),
            }
            for mode in modes
        },
        "entries": modes,
        "skipped_entries": skipped,
    }


def load_tracking_file(path: Path) -> Dict[int, List[Dict]]:
    frame_tracks: Dict[int, List[Dict]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found: {path}")
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame_id = int(float(parts[0]))
        track_id = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        frame_tracks.setdefault(frame_id, []).append(
            {
                "track_id": f"{track_id:03d}",
                "bbox": [x, y, w, h],
            }
        )
    return frame_tracks


def color_for_label(label: str):
    if label == "unknown":
        return (0, 165, 255)
    return (0, 220, 0)


def draw_label(frame, x: int, y: int, text: str, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 8, y0 + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (x + 4, y0 + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def render_video(
    video_path: Path,
    tracking_txt: Path,
    decisions: List[Dict],
    output_path: Path,
    title: str,
):
    decision_by_track = {item["track_id"]: item for item in decisions}
    frame_tracks = load_tracking_file(tracking_txt)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        for item in frame_tracks.get(frame_id, []):
            decision = decision_by_track.get(item["track_id"])
            if decision is None:
                continue
            x, y, w, h = item["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            label = decision["assigned_label"]
            color = color_for_label(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{label} count={decision['display_count']} score={decision['score_to_target']:.3f}"
            draw_label(frame, x1, max(0, y1 - 4), text, color)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()


def write_text_summary(path: Path, payload: Dict):
    lines = []
    lines.append("Target vs Unknown Ablation Summary")
    lines.append(f"Target label: {payload['config']['target_label']}")
    lines.append(f"Threshold: {payload['config']['threshold']}")
    lines.append(f"Margin: {payload['config']['margin']}")
    lines.append("")
    for video_name, video_result in payload["videos"].items():
        lines.append(f"Video: {video_name}")
        for mode in ("no_threshold", "with_threshold"):
            summary = video_result["summary"][mode]
            lines.append(f"  Mode: {mode}")
            lines.append(f"    processed_entries: {summary['processed_entries']}")
            lines.append(f"    skipped_entries: {summary['skipped_entries']}")
            lines.append(f"    counts: {summary['counts']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Run Pritom-vs-unknown ablation with and without thresholding, including annotated videos."
    )
    parser.add_argument("--gallery-npz", default=str(REPO_ROOT / "output" / "multi_identity_gallery.npz"))
    parser.add_argument("--target-label", default="pritom")
    parser.add_argument("--video-dir", default=str(PROJECT_ROOT / "newvideos"))
    parser.add_argument("--video-path", default=None, help="Optional single test video. Defaults to Test1/Test2.")
    parser.add_argument("--video-paths", default=None, help="Optional comma-separated test video paths. Overrides --video-path and defaults.")
    parser.add_argument("--output-json", default=str(REPO_ROOT / "output" / "target_vs_unknown_ablation.json"))
    parser.add_argument("--summary-txt", default=str(REPO_ROOT / "output" / "target_vs_unknown_ablation.txt"))
    parser.add_argument("--output-video-dir", default=str(REPO_ROOT / "demo" / "output" / "target_vs_unknown_videos"))
    parser.add_argument("--work-root", default=str(REPO_ROOT / "demo" / "output" / "target_vs_unknown_work"))
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--min-frames", type=int, default=20)
    args = parser.parse_args()

    gallery_path = Path(args.gallery_npz).resolve()
    video_dir = Path(args.video_dir).resolve()
    output_json = Path(args.output_json).resolve()
    summary_txt = Path(args.summary_txt).resolve()
    output_video_dir = Path(args.output_video_dir).resolve()
    work_root = Path(args.work_root).resolve()

    if args.video_paths:
        video_paths = [Path(item.strip()).resolve() for item in args.video_paths.split(",") if item.strip()]
    elif args.video_path:
        video_paths = [Path(args.video_path).resolve()]
    else:
        video_paths = [video_dir / "Test1.mp4", video_dir / "Test2.mp4"]
    missing = [path for path in video_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing test videos: " + ", ".join(str(path) for path in missing))

    gallery = load_gallery(gallery_path, args.target_label)
    model, profile = build_model(args.model)

    payload = {
        "config": {
            "gallery_npz": str(gallery_path),
            "target_label": args.target_label,
            "threshold": args.threshold,
            "margin": args.margin,
            "model": profile["display_name"],
            "model_key": args.model,
            "work_root": str(work_root),
            "output_video_dir": str(output_video_dir),
            "target_gallery_size": int(gallery["target_embeddings"].shape[0]),
            "impostor_gallery_size": int(gallery["impostor_embeddings"].shape[0]),
        },
        "videos": {},
    }

    total_counts = {
        "no_threshold": Counter(),
        "with_threshold": Counter(),
    }

    for video_path in video_paths:
        video_name = video_path.stem
        log(f"[ablation] analyzing {video_path.name}")
        entries = analyze_video(
            model=model,
            video_path=video_path,
            work_root=work_root,
            min_frames=args.min_frames,
        )
        result = build_decisions(
            entries=entries,
            target_gallery=gallery["target_embeddings"],
            impostor_gallery=gallery["impostor_embeddings"],
            target_label=args.target_label,
            threshold=args.threshold,
            margin=args.margin,
        )
        payload["videos"][video_name] = result

        for mode in ("no_threshold", "with_threshold"):
            total_counts[mode].update(result["summary"][mode]["counts"])

        tracking_txt = work_root / "tracking" / video_name / f"{video_name}.txt"
        render_video(
            video_path=video_path,
            tracking_txt=tracking_txt,
            decisions=result["entries"]["no_threshold"],
            output_path=output_video_dir / f"{video_name}_no_threshold.mp4",
            title=f"{video_name}: no threshold",
        )
        render_video(
            video_path=video_path,
            tracking_txt=tracking_txt,
            decisions=result["entries"]["with_threshold"],
            output_path=output_video_dir / f"{video_name}_with_threshold.mp4",
            title=f"{video_name}: threshold={args.threshold}",
        )

    payload["summary"] = {
        mode: dict(counter)
        for mode, counter in total_counts.items()
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    write_text_summary(summary_txt, payload)

    log(json.dumps(payload["summary"], indent=2))
    log(f"[ablation] saved JSON to {output_json}")
    log(f"[ablation] saved text summary to {summary_txt}")
    log(f"[ablation] saved annotated videos to {output_video_dir}")


if __name__ == "__main__":
    main()
