import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from probe_only_entry_analysis import REPO_ROOT, cosine_similarity, log


def load_single_embedding(path: Path) -> np.ndarray:
    embeddings = np.load(path)["embeddings"].astype(np.float32)
    if embeddings.shape[0] == 0:
        raise ValueError(f"No embeddings in {path}")
    if embeddings.shape[0] > 1:
        return embeddings.mean(axis=0)
    return embeddings[0]


def load_embeddings(per_video_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    identities = {}
    for identity_dir in sorted(path for path in per_video_root.iterdir() if path.is_dir()):
        identity = identity_dir.name
        videos = {}
        for npz_path in sorted(identity_dir.glob("*.npz")):
            videos[npz_path.stem] = load_single_embedding(npz_path)
        if videos:
            identities[identity] = videos
    return identities


def evaluate_candidate(identity: str, video_name: str, embedding: np.ndarray, identities: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    positives = [
        cosine_similarity(embedding, other_embedding)
        for other_video, other_embedding in identities[identity].items()
        if other_video != video_name
    ]
    negatives = [
        cosine_similarity(embedding, other_embedding)
        for other_identity, videos in identities.items()
        if other_identity != identity
        for other_embedding in videos.values()
    ]

    positive_min = float(min(positives)) if positives else None
    positive_mean = float(np.mean(positives)) if positives else None
    negative_max = float(max(negatives)) if negatives else None
    negative_mean = float(np.mean(negatives)) if negatives else None
    separation = positive_min - negative_max if positive_min is not None and negative_max is not None else None

    return {
        "identity": identity,
        "gallery_video": video_name,
        "positive_min": positive_min,
        "positive_mean": positive_mean,
        "negative_max": negative_max,
        "negative_mean": negative_mean,
        "separation": separation,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rank one-video gallery choices using saved per-video enrollment embeddings."
    )
    parser.add_argument(
        "--per-video-root",
        default=str(REPO_ROOT / "demo" / "output" / "multi_identity_gallery" / "per_video"),
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "output" / "single_gallery_choice_analysis.json"),
    )
    args = parser.parse_args()

    per_video_root = Path(args.per_video_root).resolve()
    output_json = Path(args.output_json).resolve()
    identities = load_embeddings(per_video_root)

    rankings = {}
    recommended = {}
    for identity, videos in identities.items():
        rows = [
            evaluate_candidate(identity, video_name, embedding, identities)
            for video_name, embedding in videos.items()
        ]
        rows.sort(
            key=lambda row: (
                row["separation"] if row["separation"] is not None else -999,
                row["positive_mean"] if row["positive_mean"] is not None else -999,
            ),
            reverse=True,
        )
        rankings[identity] = rows
        recommended[identity] = rows[0]["gallery_video"]

    payload = {
        "config": {
            "per_video_root": str(per_video_root),
            "output_json": str(output_json),
        },
        "recommended_single_gallery_videos": recommended,
        "rankings": rankings,
        "notes": [
            "Separation = minimum same-identity similarity minus maximum different-identity similarity.",
            "If separation is negative, a single gallery video cannot cleanly separate that identity from all negatives.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    log(json.dumps(payload["recommended_single_gallery_videos"], indent=2))
    log(f"[single-gallery] saved analysis to {output_json}")


if __name__ == "__main__":
    main()
