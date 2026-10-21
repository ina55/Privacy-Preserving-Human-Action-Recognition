import os
import re
import shutil
import subprocess
from pathlib import Path

import cv2


CARTOON_ROOT = Path("/tmp/VideoToCartoon")
VIDEO_INPUT_DIR = CARTOON_ROOT / "video_input"
VIDEO_OUTPUT_DIR = CARTOON_ROOT / "video_output"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Batch cartoonize UPB videos")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("/path/to/project/datasets/upb"),
        help="Source UPB root directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/path/to/project/datasets/upb/upb_cartoon"),
        help="Output cartoonized UPB directory",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N videos")
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "DatasetHAR", "HAR_ActiuniNoi"],
        help="Process only one UPB subset",
    )
    parser.add_argument(
        "--action-code",
        type=str,
        default="",
        help="Optional UPB action code filter (e.g., A42)",
    )
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


def append_failure(failed_log: Path, rel_path: str, reason: str):
    failed_log.parent.mkdir(parents=True, exist_ok=True)
    with failed_log.open("a", encoding="utf-8") as handle:
        handle.write(f"{rel_path}\t{reason}\n")


def iter_videos(src_dir: Path, subset: str):
    if subset == "all":
        roots = [src_dir / "DatasetHAR", src_dir / "HAR_ActiuniNoi"]
    else:
        roots = [src_dir / subset]

    for root in roots:
        if not root.exists():
            continue
        for src_path in sorted(root.rglob("*")):
            if src_path.is_file() and src_path.suffix.lower() in VIDEO_EXTENSIONS:
                yield src_path


def match_action_code(path: Path, action_code: str) -> bool:
    if not action_code:
        return True
    stem_parts = path.stem.split("_")
    if len(stem_parts) < 3:
        return False
    return stem_parts[-1] == action_code


def main():
    args = parse_args()

    src_dir: Path = args.src_dir
    out_dir: Path = args.out_dir
    failed_log = out_dir / "cartoonize_failures.txt"

    cartoon_main = CARTOON_ROOT / "main.py"
    if not cartoon_main.exists():
        raise SystemExit(f"Cartoon entrypoint missing: {cartoon_main}")

    videos = [
        path
        for path in iter_videos(src_dir, args.subset)
        if match_action_code(path, args.action_code)
    ]
    print(f"Found {len(videos)} videos in source: {src_dir}")

    processed = 0
    skipped_existing = 0
    failed = 0

    for src_path in videos:
        if args.limit > 0 and processed >= args.limit:
            break

        rel_path = src_path.relative_to(src_dir)
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            skipped_existing += 1
            continue

        print(f"Cartoonizing {rel_path} -> {out_path}")
        clean_temp_dirs()

        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", src_path.name)
        temp_input = VIDEO_INPUT_DIR / safe_name
        temp_output = VIDEO_OUTPUT_DIR / safe_name

        shutil.copy2(src_path, temp_input)
        if not temp_input.is_file():
            failed += 1
            append_failure(failed_log, str(rel_path), "copy_failed")
            continue

        if not can_open_video(temp_input):
            failed += 1
            append_failure(failed_log, str(rel_path), "input_not_readable")
            continue

        success = run_cartoonize(cartoon_main, safe_name, use_cpu=False)
        if not success:
            success = run_cartoonize(cartoon_main, safe_name, use_cpu=True)

        if not success:
            failed += 1
            append_failure(failed_log, str(rel_path), "cartoonize_failed_gpu_and_cpu")
            continue

        if not temp_output.exists():
            failed += 1
            append_failure(failed_log, str(rel_path), "output_missing")
            continue

        shutil.move(str(temp_output), str(out_path))
        processed += 1

    print("UPB cartoonization batch complete.")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
