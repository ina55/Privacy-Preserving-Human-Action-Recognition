from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2


DEFAULT_SRC_ROOT = Path("/path/to/project/datasets/hmdb51/hmdb51_1")
DEFAULT_OUT_ROOT = Path("/path/to/project/datasets/hmdb51/hmdb51_low")


@dataclass(frozen=True)
class ConversionResult:
    rel_path: str
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a low-resolution HMDB51 variant by downscaling videos."
    )
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N videos. Useful for quick validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate outputs even if they already exist.",
    )
    return parser.parse_args()


def iter_video_paths(src_root: Path):
    for path in sorted(src_root.rglob("*.avi")):
        if path.is_file():
            yield path


def open_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    codecs = ("XVID", "MJPG")
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open a VideoWriter for {output_path}")


def convert_video(
    src_path: Path,
    src_root: Path,
    out_root: Path,
    width: int,
    height: int,
    overwrite: bool,
) -> ConversionResult:
    rel_path = src_path.relative_to(src_root)
    out_path = out_root / rel_path

    if out_path.exists() and not overwrite:
        return ConversionResult(str(rel_path), "skipped")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(src_path))
    if not capture.isOpened():
        return ConversionResult(str(rel_path), "error", "could not open input video")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    try:
        writer = open_writer(out_path, fps, width, height)
    except Exception as exc:
        capture.release()
        return ConversionResult(str(rel_path), "error", str(exc))

    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            lowres = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(lowres)
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        out_path.unlink(missing_ok=True)
        return ConversionResult(str(rel_path), "error", "video had zero readable frames")

    return ConversionResult(str(rel_path), "converted", f"{frame_count} frames")


def main() -> int:
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Width and height must be positive integers.")

    if not args.src_root.exists():
        raise SystemExit(f"Source root does not exist: {args.src_root}")

    videos = list(iter_video_paths(args.src_root))
    if args.limit is not None:
        videos = videos[: args.limit]

    total = len(videos)
    if total == 0:
        print("No .avi videos found.")
        return 0

    print(f"Source root: {args.src_root}")
    print(f"Output root: {args.out_root}")
    print(f"Target size: {args.width}x{args.height}")
    print(f"Jobs: {args.jobs}")
    print(f"Videos queued: {total}")

    converted = 0
    skipped = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                convert_video,
                video,
                args.src_root,
                args.out_root,
                args.width,
                args.height,
                args.overwrite,
            )
            for video in videos
        ]

        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result.status == "converted":
                converted += 1
            elif result.status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"ERROR {result.rel_path}: {result.message}")

            if index % 50 == 0 or index == total:
                print(
                    f"Progress: {index}/{total} | converted={converted} skipped={skipped} errors={errors}"
                )

    print("Finished.")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())