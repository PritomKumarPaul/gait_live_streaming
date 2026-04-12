import argparse
import json
from pathlib import Path

import numpy as np

from probe_only_entry_analysis import REPO_ROOT, log


def load_embedding(per_video_root: Path, identity: str, video_name: str) -> np.ndarray:
    path = per_video_root / identity / f"{video_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected gallery embedding: {path}")
    embeddings = np.load(path)["embeddings"].astype(np.float32)
    if embeddings.shape[0] == 0:
        raise ValueError(f"No embeddings found in {path}")
    return embeddings.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Build a compact one-video-per-person gallery from saved per-video embeddings."
    )
    parser.add_argument(
        "--per-video-root",
        default=str(REPO_ROOT / "demo" / "output" / "multi_identity_gallery" / "per_video"),
    )
    parser.add_argument("--pritom-video", default="Pritomtrain1")
    parser.add_argument("--coco-video", default="Coco1")
    parser.add_argument("--jeevith-video", default="Jeevith1")
    parser.add_argument("--gallery-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    args = parser.parse_args()

    per_video_root = Path(args.per_video_root).resolve()
    gallery_out = Path(args.gallery_out).resolve()
    metadata_out = Path(args.metadata_out).resolve()

    selected = {
        "pritom": args.pritom_video,
        "coco": args.coco_video,
        "jeevith": args.jeevith_video,
    }
    labels = []
    entry_keys = []
    video_names = []
    embeddings = []

    for identity, video_name in selected.items():
        embeddings.append(load_embedding(per_video_root, identity, video_name))
        labels.append(identity)
        entry_keys.append(f"{video_name}:selected")
        video_names.append(video_name)

    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gallery_out,
        embeddings=np.stack(embeddings, axis=0).astype(np.float32),
        labels=np.array(labels),
        entry_keys=np.array(entry_keys),
        video_names=np.array(video_names),
    )

    pairwise = {}
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if j <= i:
                continue
            pairwise[f"{left}_vs_{right}"] = float(np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))

    metadata = {
        "selected_gallery_videos": selected,
        "gallery_out": str(gallery_out),
        "metadata_out": str(metadata_out),
        "pairwise_cosine_similarity": pairwise,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2))
    log(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
