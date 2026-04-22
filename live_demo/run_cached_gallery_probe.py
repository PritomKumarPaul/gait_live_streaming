import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_TOOLS = ROOT / "OpenGait" / "tools"
LIVE_ROOT = ROOT / "live_demo"
CACHE_ROOT = LIVE_ROOT / "cache"
OUTPUT_ROOT = LIVE_ROOT / "output"
DEFAULT_PROBE = ROOT / "clean_demo_v2" / "probes" / "test1probe.mp4"
DEFAULT_GALLERY = CACHE_ROOT / "pritom_coco_gallery.npz"
DEFAULT_GALLERY_META = CACHE_ROOT / "pritom_coco_gallery.json"

sys.path.insert(0, str(OPENGAIT_TOOLS))

from probe_only_entry_analysis import analyze_video, build_model, cosine_similarity, log  # noqa: E402


def resolve_path(path: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def make_browser_playable_video(input_video: Path, output_video: Path) -> Path:
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return input_video
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not output_video.exists() or output_video.stat().st_size == 0:
        return input_video
    return output_video


def average_gallery_embedding(model, video_path: Path, label: str, work_root: Path, min_frames: int):
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


def ensure_gallery_cache(
    model,
    gallery_path: Path,
    metadata_path: Path,
    work_root: Path,
    pritom_video: Path,
    coco_video: Path,
    min_frames: int,
    force_rebuild: bool,
) -> Dict:
    if gallery_path.exists() and metadata_path.exists() and not force_rebuild:
        log(f"[gallery-cache] using cached gallery: {gallery_path}")
        return json.loads(metadata_path.read_text())

    if not force_rebuild:
        raise FileNotFoundError(
            "Gallery cache is missing. Build it first with:\n"
            "  python live_demo/build_pritom_coco_gallery.py\n"
            f"Expected gallery: {gallery_path}\n"
            f"Expected metadata: {metadata_path}"
        )

    log("[gallery-cache] cache not found or rebuild requested; computing gallery embeddings")
    pritom_embedding, pritom_meta = average_gallery_embedding(model, pritom_video, "pritom", work_root, min_frames)
    coco_embedding, coco_meta = average_gallery_embedding(model, coco_video, "coco", work_root, min_frames)
    embeddings = np.stack([pritom_embedding, coco_embedding], axis=0).astype(np.float32)
    labels = np.array(["pritom", "coco"])
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gallery_path,
        embeddings=embeddings,
        labels=labels,
        entry_keys=np.array(["pritom:cached_gallery", "coco:cached_gallery"]),
        video_names=np.array([pritom_video.stem, coco_video.stem]),
    )
    meta = {
        "gallery_path": str(gallery_path),
        "labels": labels.tolist(),
        "identities": {
            "pritom": pritom_meta,
            "coco": coco_meta,
        },
    }
    metadata_path.write_text(json.dumps(meta, indent=2))
    log(f"[gallery-cache] saved gallery: {gallery_path}")
    return meta


def load_gallery(gallery_path: Path):
    data = np.load(gallery_path)
    return {
        "embeddings": data["embeddings"].astype(np.float32),
        "labels": data["labels"].astype(str),
    }


def best_match(embedding: np.ndarray, gallery: Dict) -> Dict:
    scores = []
    for label, gallery_embedding in zip(gallery["labels"], gallery["embeddings"]):
        scores.append((str(label), cosine_similarity(embedding, gallery_embedding)))
    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_label, second_score = ranked[1] if len(ranked) > 1 else ("none", -1.0)
    return {
        "assigned_identity": best_label,
        "best_identity": best_label,
        "best_score": float(best_score),
        "second_identity": second_label,
        "second_score": float(second_score),
        "identity_scores": {label: float(score) for label, score in scores},
    }


def load_tracking_file(path: Path) -> Dict[int, List[Dict]]:
    frame_tracks: Dict[int, List[Dict]] = {}
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame_id = int(float(parts[0]))
        track_id = f"{int(float(parts[1])):03d}"
        frame_tracks.setdefault(frame_id, []).append(
            {
                "track_id": track_id,
                "bbox": [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])],
            }
        )
    return frame_tracks


def color_for_label(label: str):
    return {
        "pritom": (0, 220, 0),
        "coco": (255, 140, 0),
    }.get(label, (255, 255, 255))


def draw_label(frame, x: int, y: int, text: str, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.95
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 12)
    cv2.rectangle(frame, (x, y0), (x + tw + 14, y0 + th + baseline + 12), color, -1)
    cv2.putText(frame, text, (x + 7, y0 + th + 3), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_panel(frame, lines: List[str]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.86
    thickness = 2
    line_height = 36
    width = 620
    height = 24 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (18 + width, 18 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (34, 54 + idx * line_height), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_video(video_path: Path, tracking_txt: Path, entries: List[Dict], output_video: Path):
    frame_tracks = load_tracking_file(tracking_txt)
    counts = Counter()
    decisions = {}
    for entry in sorted(entries, key=lambda item: item["track_id"]):
        if entry["status"] != "ok":
            continue
        label = entry["assigned_identity"]
        counts[label] += 1
        decisions[entry["track_id"]] = {
            "label": label,
            "count": counts[label],
            "best_score": entry["best_score"],
            "second_identity": entry["second_identity"],
            "second_score": entry["second_score"],
        }

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        draw_panel(
            frame,
            [
                "Closed-set gait recognition",
                f"Counts: pritom={counts.get('pritom', 0)} | coco={counts.get('coco', 0)}",
                "Gallery: cached Pritom/Coco | Decision: best cosine only",
            ],
        )
        for item in frame_tracks.get(frame_id, []):
            decision = decisions.get(item["track_id"])
            if decision is None:
                continue
            x, y, w, h = item["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            color = color_for_label(decision["label"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            text = f"{decision['label']} count={decision['count']} score={decision['best_score']:.3f}"
            draw_label(frame, x1, max(0, y1 - 4), text, color)
        writer.write(frame)
        frame_id += 1
    cap.release()
    writer.release()


def run_cached_gallery_probe(
    probe_video: Path,
    gallery_npz: Path = DEFAULT_GALLERY,
    gallery_meta: Path = DEFAULT_GALLERY_META,
    model_key: str = "grew_gaitbase",
    min_frames: int = 20,
    force_rebuild_gallery: bool = False,
) -> Dict:
    probe_video = Path(probe_video).resolve()
    gallery_npz = Path(gallery_npz).resolve()
    gallery_meta = Path(gallery_meta).resolve()
    run_root = OUTPUT_ROOT / f"cached_gallery_probe_{time.strftime('%Y%m%d_%H%M%S')}"
    work_root = run_root / "work"
    result_json = run_root / "result.json"
    raw_video = run_root / "annotated_raw.mp4"
    final_video = run_root / "annotated.mp4"
    log_txt = run_root / "events_log.txt"

    model, profile = build_model(model_key)
    ensure_gallery_cache(
        model=model,
        gallery_path=gallery_npz,
        metadata_path=gallery_meta,
        work_root=work_root / "gallery_build",
        pritom_video=ROOT / "clean_demo_v2" / "gallery" / "pritomgallery.mp4",
        coco_video=ROOT / "clean_demo_v2" / "gallery" / "cocogallery.mp4",
        min_frames=min_frames,
        force_rebuild=force_rebuild_gallery,
    )
    gallery = load_gallery(gallery_npz)

    entries = analyze_video(model=model, video_path=probe_video, work_root=work_root / "probe", min_frames=min_frames)
    resolved = []
    for entry in entries:
        item = {
            "entry_key": entry["entry_key"],
            "video": entry["video"],
            "track_id": entry["track_id"],
            "frame_count": entry["frame_count"],
            "status": entry["status"],
        }
        if entry["status"] == "ok":
            item.update(best_match(entry["embedding"], gallery))
        resolved.append(item)

    counts = Counter(entry["assigned_identity"] for entry in resolved if entry["status"] == "ok")
    payload = {
        "mode": "closed_set_cached_gallery_best_cosine",
        "model": profile["display_name"],
        "gallery_npz": str(gallery_npz),
        "probe_video": str(probe_video),
        "summary": {
            "counts": dict(counts),
            "processed_entries": sum(1 for entry in resolved if entry["status"] == "ok"),
            "skipped_entries": sum(1 for entry in resolved if entry["status"] != "ok"),
        },
        "entries": resolved,
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2))

    tracking_txt = work_root / "probe" / "tracking" / probe_video.stem / f"{probe_video.stem}.txt"
    render_video(probe_video, tracking_txt, resolved, raw_video)
    playable = make_browser_playable_video(raw_video, final_video)

    lines = [
        "Cached gallery closed-set probe result",
        f"mode={payload['mode']}",
        f"gallery={gallery_npz}",
        f"probe={probe_video}",
        f"counts={dict(counts)}",
        "",
        "Per-entry decisions:",
    ]
    for entry in resolved:
        if entry["status"] != "ok":
            lines.append(f"- {entry['entry_key']}: skipped {entry['status']}")
            continue
        lines.append(
            f"- {entry['entry_key']}: {entry['assigned_identity']} "
            f"best={entry['best_score']:.4f} second={entry['second_identity']}:{entry['second_score']:.4f}"
        )
    log_txt.write_text("\n".join(lines))

    return {
        "payload": payload,
        "run_root": run_root,
        "result_json": result_json,
        "annotated_video": playable,
        "log_txt": log_txt,
    }


def main():
    parser = argparse.ArgumentParser(description="Cached Pritom/Coco gallery closed-set probe matching.")
    parser.add_argument("--probe-video", default=str(DEFAULT_PROBE))
    parser.add_argument("--gallery-npz", default=str(DEFAULT_GALLERY))
    parser.add_argument("--gallery-meta", default=str(DEFAULT_GALLERY_META))
    parser.add_argument("--pritom-gallery-video", default=str(ROOT / "clean_demo_v2" / "gallery" / "pritomgallery.mp4"))
    parser.add_argument("--coco-gallery-video", default=str(ROOT / "clean_demo_v2" / "gallery" / "cocogallery.mp4"))
    parser.add_argument("--model", choices=["grew_gaitbase", "current_gaitbase", "grew_gaitgl"], default="grew_gaitbase")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--force-rebuild-gallery", action="store_true")
    args = parser.parse_args()

    probe_video = resolve_path(args.probe_video)
    gallery_npz = resolve_path(args.gallery_npz)
    gallery_meta = resolve_path(args.gallery_meta)
    pritom_video = resolve_path(args.pritom_gallery_video)
    coco_video = resolve_path(args.coco_gallery_video)
    # Preserve CLI overrides for gallery source videos only when force-rebuilding.
    if args.force_rebuild_gallery:
        model, _ = build_model(args.model)
        ensure_gallery_cache(
            model=model,
            gallery_path=gallery_npz,
            metadata_path=gallery_meta,
            work_root=OUTPUT_ROOT / f"gallery_rebuild_{time.strftime('%Y%m%d_%H%M%S')}",
            pritom_video=pritom_video,
            coco_video=coco_video,
            min_frames=args.min_frames,
            force_rebuild=True,
        )

    result = run_cached_gallery_probe(
        probe_video=probe_video,
        gallery_npz=gallery_npz,
        gallery_meta=gallery_meta,
        model_key=args.model,
        min_frames=args.min_frames,
        force_rebuild_gallery=False,
    )
    payload = result["payload"]

    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"[cached-gallery] gallery={gallery_npz}", flush=True)
    print(f"[cached-gallery] result_json={result['result_json']}", flush=True)
    print(f"[cached-gallery] annotated_video={result['annotated_video']}", flush=True)
    print(f"[cached-gallery] log={result['log_txt']}", flush=True)


if __name__ == "__main__":
    main()
