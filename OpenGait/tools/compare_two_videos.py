import argparse
import json
from pathlib import Path

from probe_only_entry_analysis import (
    PROJECT_ROOT,
    REPO_ROOT,
    analyze_video,
    assign_identities,
    build_model,
    log,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run probe-only gait entry analysis jointly across exactly two videos."
    )
    parser.add_argument(
        "--video-a",
        required=True,
        help="Absolute path to the first video.",
    )
    parser.add_argument(
        "--video-b",
        required=True,
        help="Absolute path to the second video.",
    )
    parser.add_argument(
        "--model",
        choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"],
        default="grew_gaitbase",
        help="Which gait checkpoint/profile to use.",
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "demo" / "output" / "compare_two_videos"),
        help="Working folder for tracking and silhouettes.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "output" / "compare_two_videos.json"),
        help="Where to save the combined JSON result.",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.75,
        help="Cosine threshold for matching entries across the two videos.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum silhouette frames required to keep an entry.",
    )
    args = parser.parse_args()

    video_a = Path(args.video_a).resolve()
    video_b = Path(args.video_b).resolve()
    work_root = Path(args.work_root).resolve()
    output_json = Path(args.output_json).resolve()

    if not video_a.exists():
        raise FileNotFoundError(f"Video A does not exist: {video_a}")
    if not video_b.exists():
        raise FileNotFoundError(f"Video B does not exist: {video_b}")

    model, profile = build_model(args.model)
    log(f"[compare] video_a={video_a}")
    log(f"[compare] video_b={video_b}")

    all_entries = []
    all_entries.extend(
        analyze_video(
            model=model,
            video_path=video_a,
            work_root=work_root / video_a.stem,
            min_frames=args.min_frames,
        )
    )
    all_entries.extend(
        analyze_video(
            model=model,
            video_path=video_b,
            work_root=work_root / video_b.stem,
            min_frames=args.min_frames,
        )
    )

    log("[compare] assigning identities jointly across both videos")
    results = assign_identities(all_entries, args.cosine_threshold)
    results["config"] = {
        "project_root": str(PROJECT_ROOT),
        "video_a": str(video_a),
        "video_b": str(video_b),
        "work_root": str(work_root),
        "output_json": str(output_json),
        "model": profile["display_name"],
        "model_key": args.model,
        "checkpoint": str(profile["checkpoint"]),
        "cosine_threshold": args.cosine_threshold,
        "min_frames": args.min_frames,
        "notes": [
            "Entries are extracted independently per video and then matched jointly.",
            "This is useful for checking whether the same person is recognized across videos.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2))
    log(json.dumps(results["summary"], indent=2))
    log(f"[compare] saved combined JSON to {output_json}")


if __name__ == "__main__":
    main()
