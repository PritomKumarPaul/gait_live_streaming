import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "OpenGait" / "tools"))

from probe_only_entry_analysis import analyze_video, build_model, log  # noqa: E402


def process_gallery_video(model, video_path: Path, label: str, work_root: Path, min_frames: int):
    entries = analyze_video(
        model=model,
        video_path=video_path,
        work_root=work_root / label,
        min_frames=min_frames,
    )
    ok_entries = [entry for entry in entries if entry["status"] == "ok"]
    if not ok_entries:
        raise ValueError(f"No valid gallery entries were extracted for {label}: {video_path}")
    embeddings = np.stack([entry["embedding"].astype(np.float32) for entry in ok_entries], axis=0)
    prototype = embeddings.mean(axis=0)
    metadata = {
        "label": label,
        "video": str(video_path),
        "valid_entries": len(ok_entries),
        "entries": [
            {
                "entry_key": entry["entry_key"],
                "track_id": entry["track_id"],
                "frame_count": entry["frame_count"],
                "embedding_norm": float(np.linalg.norm(entry["embedding"])),
            }
            for entry in ok_entries
        ],
    }
    return prototype, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build a two-person Pritom/Coco gallery from clean_demo_v2 gallery videos."
    )
    parser.add_argument("--pritom-video", required=True)
    parser.add_argument("--coco-video", required=True)
    parser.add_argument("--gallery-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    args = parser.parse_args()

    pritom_video = Path(args.pritom_video).resolve()
    coco_video = Path(args.coco_video).resolve()
    gallery_out = Path(args.gallery_out).resolve()
    metadata_out = Path(args.metadata_out).resolve()
    work_root = Path(args.work_root).resolve()

    model, profile = build_model(args.model)
    pritom_embedding, pritom_meta = process_gallery_video(model, pritom_video, "pritom", work_root, args.min_frames)
    coco_embedding, coco_meta = process_gallery_video(model, coco_video, "coco", work_root, args.min_frames)

    embeddings = np.stack([pritom_embedding, coco_embedding], axis=0).astype(np.float32)
    labels = np.array(["pritom", "coco"])
    entry_keys = np.array(["pritomgallery:gallery", "cocogallery:gallery"])
    video_names = np.array([pritom_video.stem, coco_video.stem])

    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gallery_out,
        embeddings=embeddings,
        labels=labels,
        entry_keys=entry_keys,
        video_names=video_names,
    )

    pritom_flat = pritom_embedding.reshape(-1)
    coco_flat = coco_embedding.reshape(-1)
    cosine = float(np.dot(pritom_flat, coco_flat) / (np.linalg.norm(pritom_flat) * np.linalg.norm(coco_flat)))
    metadata = {
        "model": profile["display_name"],
        "model_key": args.model,
        "gallery_out": str(gallery_out),
        "identities": {
            "pritom": pritom_meta,
            "coco": coco_meta,
        },
        "pritom_vs_coco_cosine": cosine,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2))
    log(json.dumps({"gallery_out": str(gallery_out), "pritom_vs_coco_cosine": cosine}, indent=2))


if __name__ == "__main__":
    main()
