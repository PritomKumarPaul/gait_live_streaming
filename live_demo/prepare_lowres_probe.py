import argparse
import shutil
import subprocess
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "clean_demo_v2" / "probes" / "test1probe.mp4"
DEFAULT_OUTPUT = ROOT / "live_demo" / "prepared_inputs" / "test1probe_720p30.mp4"


def resolve_path(path: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or imageio-ffmpeg first.")
        return ffmpeg


def inspect_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frames / fps if fps else 0.0
    return {"fps": fps, "width": width, "height": height, "frames": frames, "duration": duration}


def main():
    parser = argparse.ArgumentParser(description="Create a lower-resolution/lower-FPS probe for live demo speed tests.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=28, help="H.264 quality. Higher is smaller/lower quality.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.force:
        print(f"[prepare-lowres] already exists: {output_path}")
        print("[prepare-lowres] use --force to rebuild")
        print(f"[prepare-lowres] output_info={inspect_video(output_path)}")
        return

    input_info = inspect_video(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    scale_filter = (
        f"fps={args.fps},"
        f"scale='if(gt(iw,ih),min({args.max_side},iw),-2)':"
        f"'if(gt(ih,iw),min({args.max_side},ih),-2)'"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    print(f"[prepare-lowres] input={input_path}")
    print(f"[prepare-lowres] input_info={input_info}")
    print(f"[prepare-lowres] output={output_path}")
    print(f"[prepare-lowres] max_side={args.max_side} fps={args.fps} crf={args.crf}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError("ffmpeg compression failed")
    output_info = inspect_video(output_path)
    print(f"[prepare-lowres] output_info={output_info}")
    print(f"[prepare-lowres] size_bytes={output_path.stat().st_size}")


if __name__ == "__main__":
    main()
