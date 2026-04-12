import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from probe_only_entry_analysis import (
    PROJECT_ROOT,
    REPO_ROOT,
    analyze_video,
    build_model,
    cosine_similarity,
    log,
)


def load_prototype(path: Path) -> Dict:
    payload = json.loads(path.read_text())
    payload["prototype_embedding"] = np.asarray(payload["prototype_embedding"], dtype=np.float32)
    return payload


def score_entries_against_target(entries: List[Dict], prototype: np.ndarray, threshold: float) -> Dict:
    resolved_entries = []
    matched_entries = []
    for entry in entries:
        base = {
            "entry_key": entry["entry_key"],
            "video": entry["video"],
            "track_id": entry["track_id"],
            "sequence_name": entry["sequence_name"],
            "frame_count": entry["frame_count"],
            "status": entry["status"],
        }
        if entry["status"] != "ok":
            resolved_entries.append(base)
            continue

        score = float(cosine_similarity(entry["embedding"], prototype))
        is_target = score >= threshold
        base.update(
            {
                "cosine_similarity_to_target": score,
                "is_target_person": is_target,
            }
        )
        resolved_entries.append(base)
        if is_target:
            matched_entries.append(base)

    return {
        "summary": {
            "processed_entries": sum(1 for entry in resolved_entries if entry["status"] == "ok"),
            "skipped_entries": sum(1 for entry in resolved_entries if entry["status"] != "ok"),
            "target_entries_estimated": len(matched_entries),
        },
        "matched_entries": matched_entries,
        "entries": resolved_entries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Count entries of an enrolled target person from a saved prototype embedding."
    )
    parser.add_argument(
        "--video-dir",
        default=str(PROJECT_ROOT / "newvideos"),
        help="Folder containing Test1/Test2 videos.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional path to a single test video. Overrides the default test video set.",
    )
    parser.add_argument(
        "--prototype-json",
        default=str(REPO_ROOT / "output" / "pritom_prototype.json"),
        help="Saved prototype JSON produced by build_target_gallery.py",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "output" / "target_entry_count.json"),
        help="Where to save the target counting JSON.",
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "demo" / "output" / "target_count"),
        help="Working folder for tracking videos and silhouettes.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for cosine threshold. Defaults to the saved suggested threshold.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum number of silhouette frames required to keep an entry.",
    )
    parser.add_argument(
        "--model",
        choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"],
        default=None,
        help="Optional override. Defaults to the model stored in the prototype JSON.",
    )
    args = parser.parse_args()

    prototype_path = Path(args.prototype_json).resolve()
    output_json = Path(args.output_json).resolve()
    work_root = Path(args.work_root).resolve()
    video_dir = Path(args.video_dir).resolve()

    if not prototype_path.exists():
        raise FileNotFoundError(f"Prototype file does not exist: {prototype_path}")

    prototype_payload = load_prototype(prototype_path)
    model_name = args.model or prototype_payload["model_key"]
    threshold = args.threshold if args.threshold is not None else float(prototype_payload["suggested_threshold"])

    if args.video_path is not None:
        video_paths = [Path(args.video_path).resolve()]
    else:
        if not video_dir.exists():
            raise FileNotFoundError(f"Video folder does not exist: {video_dir}")
        video_paths = [video_dir / "Test1.mp4", video_dir / "Test2.mp4"]
        missing = [path for path in video_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing expected test videos: " + ", ".join(str(path) for path in missing)
            )

    model, profile = build_model(model_name)

    all_entries: List[Dict] = []
    for video_path in video_paths:
        log(f"[target] analyzing test video {video_path.name}")
        all_entries.extend(
            analyze_video(
                model=model,
                video_path=video_path,
                work_root=work_root,
                min_frames=args.min_frames,
            )
        )

    results = score_entries_against_target(
        entries=all_entries,
        prototype=prototype_payload["prototype_embedding"],
        threshold=threshold,
    )
    results["config"] = {
        "video_dir": str(video_dir),
        "video_paths": [str(path) for path in video_paths],
        "work_root": str(work_root),
        "output_json": str(output_json),
        "prototype_json": str(prototype_path),
        "target_person": prototype_payload["person_name"],
        "threshold": threshold,
        "model": profile["display_name"],
        "model_key": model_name,
        "checkpoint": str(profile["checkpoint"]),
        "config_path": str(profile["config"]),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2))
    log(json.dumps(results["summary"], indent=2))
    log(f"[target] saved target counting JSON to {output_json}")


if __name__ == "__main__":
    main()
