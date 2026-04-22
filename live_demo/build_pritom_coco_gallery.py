import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_TOOLS = ROOT / "OpenGait" / "tools"
LIVE_ROOT = ROOT / "live_demo"
CACHE_ROOT = LIVE_ROOT / "cache"
OUTPUT_ROOT = LIVE_ROOT / "output"
DEFAULT_GALLERY = CACHE_ROOT / "pritom_coco_gallery.npz"
DEFAULT_GALLERY_META = CACHE_ROOT / "pritom_coco_gallery.json"

sys.path.insert(0, str(OPENGAIT_TOOLS))

from probe_only_entry_analysis import analyze_video, build_model, cosine_similarity, log  # noqa: E402


def resolve_path(path: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def default_cache_paths(model_key: str):
    if model_key == "grew_gaitbase":
        return DEFAULT_GALLERY, DEFAULT_GALLERY_META
    return (
        CACHE_ROOT / f"pritom_coco_gallery_{model_key}.npz",
        CACHE_ROOT / f"pritom_coco_gallery_{model_key}.json",
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
        "valid_entries": len(ok_entries),
        "entry_keys": [entry["entry_key"] for entry in ok_entries],
        "frame_counts": [entry["frame_count"] for entry in ok_entries],
    }


def main():
    parser = argparse.ArgumentParser(description="Build the fixed Pritom/Coco gallery cache for live demos.")
    parser.add_argument("--gallery-out", default=None)
    parser.add_argument("--metadata-out", default=None)
    parser.add_argument("--pritom-gallery-video", default=str(ROOT / "clean_demo_v2" / "gallery" / "pritomgallery.mp4"))
    parser.add_argument("--coco-gallery-video", default=str(ROOT / "clean_demo_v2" / "gallery" / "cocogallery.mp4"))
    parser.add_argument("--work-root", default=str(OUTPUT_ROOT / "gallery_cache_build"))
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache already exists.")
    args = parser.parse_args()

    default_gallery_out, default_metadata_out = default_cache_paths(args.model)
    gallery_out = resolve_path(args.gallery_out) if args.gallery_out else default_gallery_out.resolve()
    metadata_out = resolve_path(args.metadata_out) if args.metadata_out else default_metadata_out.resolve()
    pritom_video = resolve_path(args.pritom_gallery_video)
    coco_video = resolve_path(args.coco_gallery_video)
    work_root = resolve_path(args.work_root) / time.strftime("%Y%m%d_%H%M%S")

    if gallery_out.exists() and metadata_out.exists() and not args.force:
        print("[gallery-build] Gallery cache already exists. Nothing to rebuild.", flush=True)
        print(f"[gallery-build] gallery={gallery_out}", flush=True)
        print(f"[gallery-build] metadata={metadata_out}", flush=True)
        print("[gallery-build] Use --force to rebuild.", flush=True)
        return

    print("[gallery-build] Building Pritom/Coco gallery cache. Please wait; this can take a while.", flush=True)
    print(f"[gallery-build] pritom_gallery={pritom_video}", flush=True)
    print(f"[gallery-build] coco_gallery={coco_video}", flush=True)
    print(f"[gallery-build] output={gallery_out}", flush=True)
    print(f"[gallery-build] work_root={work_root}", flush=True)

    model, profile = build_model(args.model)
    pritom_embedding, pritom_meta = average_gallery_embedding(model, pritom_video, "pritom", work_root, args.min_frames)
    coco_embedding, coco_meta = average_gallery_embedding(model, coco_video, "coco", work_root, args.min_frames)

    embeddings = np.stack([pritom_embedding, coco_embedding], axis=0).astype(np.float32)
    labels = np.array(["pritom", "coco"])
    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gallery_out,
        embeddings=embeddings,
        labels=labels,
        entry_keys=np.array(["pritom:cached_gallery", "coco:cached_gallery"]),
        video_names=np.array([pritom_video.stem, coco_video.stem]),
    )

    pritom_vs_coco = cosine_similarity(pritom_embedding, coco_embedding)
    metadata = {
        "model": profile["display_name"],
        "model_key": args.model,
        "gallery_path": str(gallery_out),
        "labels": labels.tolist(),
        "pritom_vs_coco_cosine": float(pritom_vs_coco),
        "identities": {
            "pritom": pritom_meta,
            "coco": coco_meta,
        },
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2))

    print("[gallery-build] Done.", flush=True)
    print(f"[gallery-build] gallery={gallery_out}", flush=True)
    print(f"[gallery-build] metadata={metadata_out}", flush=True)
    print(f"[gallery-build] pritom_vs_coco_cosine={pritom_vs_coco:.4f}", flush=True)


if __name__ == "__main__":
    main()
