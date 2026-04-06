import argparse
import json
from pathlib import Path

from compare_two_videos import REPO_ROOT
from probe_only_entry_analysis import analyze_video, assign_identities, build_model, log


def parse_thresholds(raw: str):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("No thresholds were provided.")
    return values


def count_people_for_video(entries, video_name: str):
    return len({entry["assigned_person_id"] for entry in entries if entry["video"] == video_name and entry["status"] == "ok"})


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple models and cosine thresholds for two-video identity separation."
    )
    parser.add_argument("--video-a", required=True, help="Absolute path to the first video.")
    parser.add_argument("--video-b", required=True, help="Absolute path to the second video.")
    parser.add_argument(
        "--models",
        default="grew_gaitbase,grew_gaitgl",
        help="Comma-separated model keys. Example: grew_gaitbase,grew_gaitgl",
    )
    parser.add_argument(
        "--thresholds",
        default="0.75,0.9,0.95,0.97,0.975,0.98",
        help="Comma-separated cosine thresholds to test.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum silhouette frames required to keep an entry.",
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "demo" / "output" / "threshold_eval"),
        help="Root folder for intermediate outputs.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "output" / "threshold_eval.json"),
        help="Where to save the summary JSON.",
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

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    thresholds = parse_thresholds(args.thresholds)

    summary_rows = []
    detailed_results = []

    for model_name in model_names:
        log(f"[eval] loading model={model_name}")
        model, profile = build_model(model_name)

        log(f"[eval] extracting entries for video_a={video_a.name}")
        entries_a = analyze_video(
            model=model,
            video_path=video_a,
            work_root=work_root / model_name / video_a.stem,
            min_frames=args.min_frames,
        )

        log(f"[eval] extracting entries for video_b={video_b.name}")
        entries_b = analyze_video(
            model=model,
            video_path=video_b,
            work_root=work_root / model_name / video_b.stem,
            min_frames=args.min_frames,
        )

        combined_entries = entries_a + entries_b
        processed_entries = sum(1 for entry in combined_entries if entry["status"] == "ok")
        log(f"[eval] model={model_name} extracted processed_entries={processed_entries}")

        for threshold in thresholds:
            log(f"[eval] model={model_name} threshold={threshold:.4f}")
            result = assign_identities(combined_entries, threshold)
            entries = result["entries"]
            joint_people = result["summary"]["unique_people_estimated"]
            video_a_people = count_people_for_video(entries, video_a.stem)
            video_b_people = count_people_for_video(entries, video_b.stem)
            meets_goal = (video_a_people == 1 and video_b_people == 1 and joint_people == 2)

            row = {
                "model_key": model_name,
                "model": profile["display_name"],
                "threshold": threshold,
                "video_a": video_a.name,
                "video_b": video_b.name,
                "video_a_people": video_a_people,
                "video_b_people": video_b_people,
                "joint_people": joint_people,
                "processed_entries": result["summary"]["processed_entries"],
                "meets_goal": meets_goal,
            }
            summary_rows.append(row)
            detailed_results.append(
                {
                    "summary": row,
                    "people": result["people"],
                    "entries": entries,
                }
            )

    payload = {
        "config": {
            "video_a": str(video_a),
            "video_b": str(video_b),
            "models": model_names,
            "thresholds": thresholds,
            "min_frames": args.min_frames,
            "work_root": str(work_root),
            "output_json": str(output_json),
            "goal": "one person within each video, but two different people jointly across the two videos",
        },
        "summary_rows": summary_rows,
        "detailed_results": detailed_results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    log(json.dumps(summary_rows, indent=2))
    log(f"[eval] saved threshold evaluation to {output_json}")


if __name__ == "__main__":
    main()
