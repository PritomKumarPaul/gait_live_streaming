import argparse
import json
from pathlib import Path

import numpy as np

from probe_only_entry_analysis import (
    REPO_ROOT,
    analyze_video,
    build_model,
    log,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract valid gait entry embeddings for one video and save metadata plus an NPZ embedding file."
    )
    parser.add_argument("--video-path", required=True, help="Input video path.")
    parser.add_argument("--metadata-json", required=True, help="Output metadata JSON path.")
    parser.add_argument("--embeddings-npz", required=True, help="Output embeddings NPZ path.")
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "demo" / "output" / "extract_video_entries"),
        help="Working folder for tracking videos and silhouettes.",
    )
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument(
        "--model",
        choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"],
        default="grew_gaitbase",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path).resolve()
    metadata_json = Path(args.metadata_json).resolve()
    embeddings_npz = Path(args.embeddings_npz).resolve()
    work_root = Path(args.work_root).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    model, profile = build_model(args.model)
    entries = analyze_video(
        model=model,
        video_path=video_path,
        work_root=work_root,
        min_frames=args.min_frames,
    )

    metadata = []
    embeddings = []
    for entry in entries:
        item = {
            "entry_key": entry["entry_key"],
            "video": entry["video"],
            "track_id": entry["track_id"],
            "sequence_name": entry["sequence_name"],
            "frame_count": entry["frame_count"],
            "status": entry["status"],
        }
        if entry["status"] == "ok":
            item["embedding_index"] = len(embeddings)
            item["embedding_norm"] = float(np.linalg.norm(entry["embedding"]))
            embeddings.append(entry["embedding"].astype(np.float32))
        metadata.append(item)

    if embeddings:
        embedding_array = np.stack(embeddings, axis=0)
    else:
        embedding_array = np.empty((0,), dtype=np.float32)

    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "video_name": video_path.stem,
                "model": profile["display_name"],
                "model_key": args.model,
                "checkpoint": str(profile["checkpoint"]),
                "entries": metadata,
            },
            indent=2,
        )
    )
    np.savez_compressed(embeddings_npz, embeddings=embedding_array)

    log(
        json.dumps(
            {
                "video": video_path.name,
                "valid_embeddings": int(embedding_array.shape[0]),
                "metadata_json": str(metadata_json),
                "embeddings_npz": str(embeddings_npz),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
