import os
import re
import shutil
import subprocess
from pathlib import Path

import cv2


SRC_DIR = Path("/path/to/project/datasets/ucf101/ucf101_1")
OUT_DIR = Path("/path/to/project/datasets/ucf101/ucf101_cartoon")
CARTOON_ROOT = Path("/tmp/VideoToCartoon")
VIDEO_INPUT_DIR = CARTOON_ROOT / "video_input"
VIDEO_OUTPUT_DIR = CARTOON_ROOT / "video_output"
FAILED_LOG = OUT_DIR / "cartoonize_failures.txt"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Batch cartoonize UCF101 videos")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N videos")
    return parser.parse_args()


def clean_temp_dirs():
    VIDEO_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (VIDEO_INPUT_DIR, VIDEO_OUTPUT_DIR):
        for item in folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()


def can_open_video(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    cap.release()
    return ok


def run_cartoonize(cartoon_main: Path, safe_name: str, use_cpu: bool) -> bool:
    env = os.environ.copy()
    if use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        print(f"Retrying on CPU for {safe_name}")
    result = subprocess.run(
        ["python3", str(cartoon_main), "--input", safe_name],
        cwd=CARTOON_ROOT,
        env=env,
        check=False,
    )
    return result.returncode == 0


def append_failure(rel_path: str, reason: str):
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{rel_path}\t{reason}\n")


def main():
    args = parse_args()
    cartoon_main = CARTOON_ROOT / "main.py"
    if not cartoon_main.exists():
        raise SystemExit(f"Cartoon entrypoint missing: {cartoon_main}")

    processed = 0
    for src_path in sorted(SRC_DIR.rglob("*.avi")):
        if not src_path.is_file():
            continue

        if args.limit > 0 and processed >= args.limit:
            break

        rel_path = src_path.relative_to(SRC_DIR)
        out_path = OUT_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"Skipping existing output: {out_path}")
            continue

        print(f"Cartoonizing {rel_path} -> {out_path}")
        clean_temp_dirs()

        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", src_path.name)
        temp_input = VIDEO_INPUT_DIR / safe_name
        temp_output = VIDEO_OUTPUT_DIR / safe_name

        shutil.copy2(src_path, temp_input)
        if not temp_input.is_file():
            print(f"ERROR: File not found after copy: {temp_input}")
            append_failure(str(rel_path), "copy_failed")
            continue

        if not can_open_video(temp_input):
            print(f"ERROR: Cannot open video file: {temp_input}")
            append_failure(str(rel_path), "input_not_readable")
            continue

        success = run_cartoonize(cartoon_main, safe_name, use_cpu=False)
        if not success:
            success = run_cartoonize(cartoon_main, safe_name, use_cpu=True)

        if not success:
            print(f"ERROR: Cartoonization failed for {rel_path}")
            append_failure(str(rel_path), "cartoonize_failed_gpu_and_cpu")
            continue

        if not temp_output.exists():
            print(f"ERROR: Output missing for {rel_path}: {temp_output}")
            append_failure(str(rel_path), "output_missing")
            continue

        shutil.move(str(temp_output), str(out_path))
        processed += 1

    print("UCF101 cartoonization batch complete.")


if __name__ == "__main__":
    main()
