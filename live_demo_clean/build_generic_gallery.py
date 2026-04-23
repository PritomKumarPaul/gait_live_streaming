import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_TOOLS = ROOT / "OpenGait" / "tools"
LIVE_ROOT = ROOT / "live_demo_clean"
CACHE_ROOT = LIVE_ROOT / "cache"
OUTPUT_ROOT = LIVE_ROOT / "output"

sys.path.insert(0, str(OPENGAIT_TOOLS))

from probe_only_entry_analysis import analyze_video, build_model, cosine_similarity, log  # noqa: E402


def resolve_path(path: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def default_cache_paths(model_key: str):
    stem = "generic_gallery" if model_key == "grew_gaitbase" else f"generic_gallery_{model_key}"
    return CACHE_ROOT / f"{stem}.npz", CACHE_ROOT / f"{stem}.json"


def list_gallery_videos(gallery_dir: Path):
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(
        [path for path in gallery_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed],
        key=lambda path: path.name.lower(),
    )


def average_gallery_embedding(model, video_path: Path, label: str, work_root: Path, min_frames: int):
    log(f"[gallery-build] {label}: extracting tracking/silhouettes/embedding from {video_path}")
    entries = analyze_video(model=model, video_path=video_path, work_root=work_root / label, min_frames=min_frames)
    ok_entries = [entry for entry in entries if entry["status"] == "ok"]
    if not ok_entries:
        raise RuntimeError(f"No valid gallery entry found for {label}: {video_path}")
    embeddings = np.stack([entry["embedding"].astype(np.float32) for entry in ok_entries], axis=0)
    return embeddings.mean(axis=0), {
        "label": label,
        "video": str(video_path),
        "video_name": video_path.name,
        "valid_entries": len(ok_entries),
        "entry_keys": [entry["entry_key"] for entry in ok_entries],
        "frame_counts": [entry["frame_count"] for entry in ok_entries],
    }


def main():
    parser = argparse.ArgumentParser(description="Build a generic multi-person gallery cache for live demos.")
    parser.add_argument("--gallery-dir", default=str(LIVE_ROOT / "gallery"))
    parser.add_argument("--gallery-out", default=None)
    parser.add_argument("--metadata-out", default=None)
    parser.add_argument("--work-root", default=str(OUTPUT_ROOT / "gallery_cache_build"))
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache already exists.")
    args = parser.parse_args()

    gallery_dir = resolve_path(args.gallery_dir)
    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")
    gallery_videos = list_gallery_videos(gallery_dir)
    if len(gallery_videos) < 2:
        raise RuntimeError(
            f"Expected at least 2 gallery videos in {gallery_dir}, found {len(gallery_videos)}. "
            "Add videos such as personA.mp4, personB.mp4, etc."
        )

    default_gallery_out, default_metadata_out = default_cache_paths(args.model)
    gallery_out = resolve_path(args.gallery_out) if args.gallery_out else default_gallery_out.resolve()
    metadata_out = resolve_path(args.metadata_out) if args.metadata_out else default_metadata_out.resolve()
    work_root = resolve_path(args.work_root) / time.strftime("%Y%m%d_%H%M%S")

    if gallery_out.exists() and metadata_out.exists() and not args.force:
        print("[gallery-build] Gallery cache already exists. Nothing to rebuild.", flush=True)
        print(f"[gallery-build] gallery={gallery_out}", flush=True)
        print(f"[gallery-build] metadata={metadata_out}", flush=True)
        print("[gallery-build] Use --force to rebuild.", flush=True)
        return

    print("[gallery-build] Building generic gallery cache. Please wait; this can take a while.", flush=True)
    print(f"[gallery-build] gallery_dir={gallery_dir}", flush=True)
    for idx, video_path in enumerate(gallery_videos, start=1):
        print(f"[gallery-build] person{idx} <- {video_path}", flush=True)
    print(f"[gallery-build] output={gallery_out}", flush=True)
    print(f"[gallery-build] work_root={work_root}", flush=True)

    model, profile = build_model(args.model)
    embeddings = []
    labels = []
    entry_keys = []
    video_names = []
    identities = {}
    pairwise = {}

    for idx, video_path in enumerate(gallery_videos, start=1):
        label = f"person{idx}"
        embedding, meta = average_gallery_embedding(model, video_path, label, work_root, args.min_frames)
        embeddings.append(embedding.astype(np.float32))
        labels.append(label)
        entry_keys.append(f"{label}:cached_gallery")
        video_names.append(video_path.stem)
        identities[label] = meta

    gallery_embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    gallery_labels = np.array(labels)
    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gallery_out,
        embeddings=gallery_embeddings,
        labels=gallery_labels,
        entry_keys=np.array(entry_keys),
        video_names=np.array(video_names),
    )

    for left_idx, left_label in enumerate(labels):
        for right_idx in range(left_idx + 1, len(labels)):
            right_label = labels[right_idx]
            key = f"{left_label}_vs_{right_label}"
            pairwise[key] = float(cosine_similarity(gallery_embeddings[left_idx], gallery_embeddings[right_idx]))

    metadata = {
        "model": profile["display_name"],
        "model_key": args.model,
        "gallery_path": str(gallery_out),
        "gallery_dir": str(gallery_dir),
        "labels": labels,
        "identity_count": len(labels),
        "identities": identities,
        "pairwise_cosine": pairwise,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2))

    print("[gallery-build] Done.", flush=True)
    print(f"[gallery-build] gallery={gallery_out}", flush=True)
    print(f"[gallery-build] metadata={metadata_out}", flush=True)


if __name__ == "__main__":
    main()
