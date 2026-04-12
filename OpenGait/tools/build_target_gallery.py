import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from probe_only_entry_analysis import (
    PROJECT_ROOT,
    REPO_ROOT,
    analyze_video,
    build_model,
    cosine_similarity,
    log,
)


def average_embeddings(entries: List[Dict]) -> np.ndarray:
    ok_entries = [entry for entry in entries if entry["status"] == "ok"]
    if not ok_entries:
        raise ValueError("No valid entries were found to average.")
    stacked = np.stack([entry["embedding"] for entry in ok_entries], axis=0)
    return stacked.mean(axis=0)


def build_video_prototypes(entries: List[Dict]) -> Dict[str, Dict]:
    by_video: Dict[str, List[Dict]] = {}
    for entry in entries:
        if entry["status"] != "ok":
            continue
        by_video.setdefault(entry["video"], []).append(entry)

    prototypes = {}
    for video_name, video_entries in sorted(by_video.items()):
        prototype = average_embeddings(video_entries)
        similarities = [
            cosine_similarity(entry["embedding"], prototype)
            for entry in video_entries
        ]
        prototypes[video_name] = {
            "prototype": prototype,
            "entry_count": len(video_entries),
            "entry_keys": [entry["entry_key"] for entry in video_entries],
            "similarity_to_video_prototype": {
                "min": float(min(similarities)),
                "max": float(max(similarities)),
                "mean": float(np.mean(similarities)),
            },
        }
    return prototypes


def average_video_prototypes(video_prototypes: Dict[str, Dict]) -> np.ndarray:
    if not video_prototypes:
        raise ValueError("No video prototypes were found to average.")
    stacked = np.stack(
        [item["prototype"] for item in video_prototypes.values()],
        axis=0,
    )
    return stacked.mean(axis=0)


def serialize_entries(entries: List[Dict]) -> List[Dict]:
    payload = []
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
            item["embedding_norm"] = float(np.linalg.norm(entry["embedding"]))
        payload.append(item)
    return payload


def collect_group_entries(
    model,
    video_paths: List[Path],
    work_root: Path,
    min_frames: int,
) -> List[Dict]:
    entries: List[Dict] = []
    for video_path in video_paths:
        log(f"[gallery] analyzing enrollment video {video_path.name}")
        entries.extend(
            analyze_video(
                model=model,
                video_path=video_path,
                work_root=work_root,
                min_frames=min_frames,
            )
        )
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Build target-person gallery prototypes from labeled training videos."
    )
    parser.add_argument(
        "--video-dir",
        default=str(PROJECT_ROOT / "newvideos"),
        help="Folder containing Pritom/Coco/Jeevith training videos.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "output" / "target_gallery_analysis.json"),
        help="Where to save the gallery analysis JSON.",
    )
    parser.add_argument(
        "--pritom-prototype-out",
        default=str(REPO_ROOT / "output" / "pritom_prototype.json"),
        help="Where to save the averaged Pritom prototype for later matching.",
    )
    parser.add_argument(
        "--work-root",
        default=str(REPO_ROOT / "demo" / "output" / "target_gallery"),
        help="Working folder for tracking videos and silhouettes.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum number of silhouette frames required to keep an entry.",
    )
    parser.add_argument(
        "--model",
        choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"],
        default="grew_gaitbase",
        help="Which gait checkpoint/profile to use.",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir).resolve()
    output_json = Path(args.output_json).resolve()
    pritom_prototype_out = Path(args.pritom_prototype_out).resolve()
    work_root = Path(args.work_root).resolve()

    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder does not exist: {video_dir}")

    groups = {
        "pritom": sorted(video_dir.glob("Pritomtrain*.mp4")),
        "coco": sorted(video_dir.glob("Coco*.mp4")),
        "jeevith": sorted(video_dir.glob("Jeevith*.mp4")),
    }

    missing_groups = [name for name, paths in groups.items() if not paths]
    if missing_groups:
        raise FileNotFoundError(f"Missing expected training videos for: {', '.join(missing_groups)}")

    model, profile = build_model(args.model)

    group_results = {}
    prototypes = {}

    for group_name, video_paths in groups.items():
        log(f"[gallery] building prototype for {group_name} from {len(video_paths)} videos")
        entries = collect_group_entries(
            model=model,
            video_paths=video_paths,
            work_root=work_root / group_name,
            min_frames=args.min_frames,
        )
        ok_entries = [entry for entry in entries if entry["status"] == "ok"]
        if not ok_entries:
            raise ValueError(f"No usable entries were found for group '{group_name}'.")

        video_prototypes = build_video_prototypes(entries)
        prototype = average_video_prototypes(video_prototypes)
        prototypes[group_name] = prototype
        within_scores = [
            cosine_similarity(entry["embedding"], prototype)
            for entry in ok_entries
        ]
        video_prototype_scores = {
            video_name: float(cosine_similarity(item["prototype"], prototype))
            for video_name, item in video_prototypes.items()
        }
        group_results[group_name] = {
            "video_count": len(video_paths),
            "video_names": [path.name for path in video_paths],
            "processed_entries": len(ok_entries),
            "skipped_entries": sum(1 for entry in entries if entry["status"] != "ok"),
            "entries": serialize_entries(entries),
            "video_prototypes": {
                video_name: {
                    "entry_count": item["entry_count"],
                    "entry_keys": item["entry_keys"],
                    "similarity_to_video_prototype": item["similarity_to_video_prototype"],
                    "similarity_to_group_prototype": video_prototype_scores[video_name],
                }
                for video_name, item in video_prototypes.items()
            },
            "prototype_norm": float(np.linalg.norm(prototype)),
            "similarity_to_group_prototype": {
                "min": float(min(within_scores)),
                "max": float(max(within_scores)),
                "mean": float(np.mean(within_scores)),
            },
            "video_prototype_similarity_to_group_prototype": {
                "min": float(min(video_prototype_scores.values())),
                "max": float(max(video_prototype_scores.values())),
                "mean": float(np.mean(list(video_prototype_scores.values()))),
            },
        }

    pairwise = {
        "pritom_vs_coco": float(cosine_similarity(prototypes["pritom"], prototypes["coco"])),
        "pritom_vs_jeevith": float(cosine_similarity(prototypes["pritom"], prototypes["jeevith"])),
        "coco_vs_jeevith": float(cosine_similarity(prototypes["coco"], prototypes["jeevith"])),
    }

    pritom_self_min = group_results["pritom"]["similarity_to_group_prototype"]["min"]
    impostor_max = max(pairwise["pritom_vs_coco"], pairwise["pritom_vs_jeevith"])
    suggested_threshold = float((pritom_self_min + impostor_max) / 2.0)

    payload = {
        "config": {
            "video_dir": str(video_dir),
            "work_root": str(work_root),
            "output_json": str(output_json),
            "pritom_prototype_out": str(pritom_prototype_out),
            "model": profile["display_name"],
            "model_key": args.model,
            "checkpoint": str(profile["checkpoint"]),
            "config_path": str(profile["config"]),
            "min_frames": args.min_frames,
        },
        "groups": group_results,
        "pairwise_prototype_cosine_similarity": pairwise,
        "threshold_analysis": {
            "pritom_self_min_similarity": float(pritom_self_min),
            "pritom_impostor_max_similarity": float(impostor_max),
            "suggested_midpoint_threshold": suggested_threshold,
            "notes": [
                "Each person's prototype is built by averaging per-video prototypes, so each enrollment video gets equal weight.",
                "A good threshold should usually stay below the minimum same-person similarity and above the maximum impostor similarity.",
                "If these overlap, you will need more enrollment data or a stricter evaluation setup.",
            ],
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    pritom_payload = {
        "person_name": "pritom",
        "model": profile["display_name"],
        "model_key": args.model,
        "checkpoint": str(profile["checkpoint"]),
        "config_path": str(profile["config"]),
        "prototype_embedding": prototypes["pritom"].astype(np.float32).tolist(),
        "training_videos": groups["pritom"] and [path.name for path in groups["pritom"]],
        "suggested_threshold": suggested_threshold,
        "pritom_self_min_similarity": float(pritom_self_min),
        "pritom_vs_coco": pairwise["pritom_vs_coco"],
        "pritom_vs_jeevith": pairwise["pritom_vs_jeevith"],
    }
    pritom_prototype_out.parent.mkdir(parents=True, exist_ok=True)
    pritom_prototype_out.write_text(json.dumps(pritom_payload, indent=2))

    log(json.dumps(payload["threshold_analysis"], indent=2))
    log(f"[gallery] saved gallery analysis to {output_json}")
    log(f"[gallery] saved Pritom prototype to {pritom_prototype_out}")


if __name__ == "__main__":
    main()
