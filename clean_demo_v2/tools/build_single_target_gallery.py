import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "OpenGait" / "tools"))

from probe_only_entry_analysis import analyze_video, build_model, log  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Build a single-person target gallery from one gallery video."
    )
    parser.add_argument("--gallery-video", required=True)
    parser.add_argument("--target-label", default="pritom")
    parser.add_argument("--gallery-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    args = parser.parse_args()

    gallery_video = Path(args.gallery_video).resolve()
    gallery_out = Path(args.gallery_out).resolve()
    metadata_out = Path(args.metadata_out).resolve()
    work_root = Path(args.work_root).resolve()

    if not gallery_video.exists():
        raise FileNotFoundError(f"Gallery video does not exist: {gallery_video}")

    model, profile = build_model(args.model)
    entries = analyze_video(
        model=model,
        video_path=gallery_video,
        work_root=work_root,
        min_frames=args.min_frames,
    )
    ok_entries = [entry for entry in entries if entry["status"] == "ok"]
    if not ok_entries:
        raise ValueError("No valid gallery entries were extracted.")

    embeddings = np.stack([entry["embedding"].astype(np.float32) for entry in ok_entries], axis=0)
    labels = np.array([args.target_label] * embeddings.shape[0])
    entry_keys = np.array([entry["entry_key"] for entry in ok_entries])
    video_names = np.array([entry["video"] for entry in ok_entries])

    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gallery_out,
        embeddings=embeddings,
        labels=labels,
        entry_keys=entry_keys,
        video_names=video_names,
    )

    metadata = {
        "target_label": args.target_label,
        "gallery_video": str(gallery_video),
        "gallery_out": str(gallery_out),
        "model": profile["display_name"],
        "model_key": args.model,
        "valid_gallery_entries": len(ok_entries),
        "entries": [
            {
                "entry_key": entry["entry_key"],
                "video": entry["video"],
                "track_id": entry["track_id"],
                "frame_count": entry["frame_count"],
                "embedding_norm": float(np.linalg.norm(entry["embedding"])),
            }
            for entry in ok_entries
        ],
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2))
    log(json.dumps({"valid_gallery_entries": len(ok_entries), "gallery_out": str(gallery_out)}, indent=2))


if __name__ == "__main__":
    main()
