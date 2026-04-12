import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "OpenGait" / "tools"))

from probe_only_entry_analysis import (
    PROJECT_ROOT,
    REPO_ROOT,
    analyze_video,
    build_model,
    cosine_similarity,
    log,
)


def load_gallery(path: Path) -> Dict:
    data = np.load(path)
    return {
        "embeddings": data["embeddings"].astype(np.float32),
        "labels": data["labels"].astype(str),
        "entry_keys": data["entry_keys"].astype(str),
        "video_names": data["video_names"].astype(str),
    }


def score_identity(probe_embedding: np.ndarray, gallery_embeddings: np.ndarray, mode: str, top_k: int) -> float:
    scores = np.array([cosine_similarity(probe_embedding, emb) for emb in gallery_embeddings], dtype=np.float32)
    if scores.size == 0:
        return -1.0
    if mode == "max":
        return float(np.max(scores))
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "topk":
        k = min(top_k, scores.size)
        return float(np.mean(np.sort(scores)[-k:]))
    raise ValueError(f"Unsupported match mode: {mode}")


def identify_entry(
    embedding: np.ndarray,
    gallery: Dict,
    match_mode: str,
    top_k: int,
    threshold: float,
    margin: float,
) -> Dict:
    identity_scores = {}
    for identity in sorted(set(gallery["labels"])):
        mask = gallery["labels"] == identity
        identity_scores[identity] = score_identity(
            embedding,
            gallery["embeddings"][mask],
            mode=match_mode,
            top_k=top_k,
        )

    ranked = sorted(identity_scores.items(), key=lambda item: item[1], reverse=True)
    best_identity, best_score = ranked[0]
    second_identity, second_score = ranked[1] if len(ranked) > 1 else ("none", -1.0)
    score_margin = best_score - second_score
    accepted = best_score >= threshold and score_margin >= margin

    return {
        "assigned_identity": best_identity if accepted else "unknown",
        "accepted": accepted,
        "best_identity": best_identity,
        "best_score": float(best_score),
        "second_identity": second_identity,
        "second_score": float(second_score),
        "score_margin": float(score_margin),
        "identity_scores": {key: float(value) for key, value in identity_scores.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Identify entries in probe videos using a multi-identity gait gallery."
    )
    parser.add_argument("--gallery-npz", default=str(REPO_ROOT / "output" / "multi_identity_gallery.npz"))
    parser.add_argument("--video-dir", default=str(PROJECT_ROOT / "newvideos"))
    parser.add_argument("--video-path", default=None, help="Optional single probe video. Defaults to Test1/Test2.")
    parser.add_argument("--output-json", default=str(REPO_ROOT / "output" / "multi_gallery_probe_results.json"))
    parser.add_argument("--work-root", default=str(REPO_ROOT / "demo" / "output" / "multi_gallery_probe"))
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--match-mode", choices=["max", "mean", "topk"], default="max")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    gallery_path = Path(args.gallery_npz).resolve()
    video_dir = Path(args.video_dir).resolve()
    output_json = Path(args.output_json).resolve()
    work_root = Path(args.work_root).resolve()

    if not gallery_path.exists():
        raise FileNotFoundError(f"Gallery file does not exist: {gallery_path}")

    if args.video_path:
        video_paths = [Path(args.video_path).resolve()]
    else:
        video_paths = [video_dir / "Test1.mp4", video_dir / "Test2.mp4"]
        missing = [path for path in video_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing probe videos: " + ", ".join(str(path) for path in missing))

    gallery = load_gallery(gallery_path)
    model, profile = build_model(args.model)

    all_entries: List[Dict] = []
    for video_path in video_paths:
        log(f"[probe] analyzing {video_path.name}")
        all_entries.extend(
            analyze_video(
                model=model,
                video_path=video_path,
                work_root=work_root,
                min_frames=args.min_frames,
            )
        )

    resolved_entries = []
    for entry in all_entries:
        item = {
            "entry_key": entry["entry_key"],
            "video": entry["video"],
            "track_id": entry["track_id"],
            "sequence_name": entry["sequence_name"],
            "frame_count": entry["frame_count"],
            "status": entry["status"],
        }
        if entry["status"] == "ok":
            item.update(
                identify_entry(
                    entry["embedding"],
                    gallery=gallery,
                    match_mode=args.match_mode,
                    top_k=args.top_k,
                    threshold=args.threshold,
                    margin=args.margin,
                )
            )
        resolved_entries.append(item)

    counts = Counter(
        entry["assigned_identity"]
        for entry in resolved_entries
        if entry["status"] == "ok"
    )
    per_video_counts = {}
    for entry in resolved_entries:
        if entry["status"] != "ok":
            continue
        per_video_counts.setdefault(entry["video"], Counter())
        per_video_counts[entry["video"]][entry["assigned_identity"]] += 1

    results = {
        "config": {
            "gallery_npz": str(gallery_path),
            "video_paths": [str(path) for path in video_paths],
            "work_root": str(work_root),
            "output_json": str(output_json),
            "model": profile["display_name"],
            "model_key": args.model,
            "threshold": args.threshold,
            "margin": args.margin,
            "match_mode": args.match_mode,
            "top_k": args.top_k,
            "gallery_identities": sorted(set(gallery["labels"])),
        },
        "summary": {
            "processed_entries": sum(1 for entry in resolved_entries if entry["status"] == "ok"),
            "skipped_entries": sum(1 for entry in resolved_entries if entry["status"] != "ok"),
            "counts": dict(counts),
            "per_video_counts": {video: dict(counter) for video, counter in per_video_counts.items()},
        },
        "entries": resolved_entries,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2))
    log(json.dumps(results["summary"], indent=2))
    log(f"[probe] saved multi-gallery probe results to {output_json}")


if __name__ == "__main__":
    main()
