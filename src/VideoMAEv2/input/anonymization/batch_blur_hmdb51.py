from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SRC_ROOT = Path("/path/to/project/datasets/hmdb51/hmdb51_1")
DEFAULT_OUT_ROOT = Path("/path/to/project/datasets/hmdb51/hmdb51_faceblurred")
DEFAULT_HELPER = Path("/path/to/repo/src/VideoMAEv2/input/anonymization/blur_faces.py")
DEFAULT_PYTHON = Path("/usr/bin/python3.9")


@dataclass(frozen=True)
class BlurJobResult:
    rel_path: str
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch blur all HMDB51 videos and replace symlinks with real files."
    )
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--helper-script", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_videos(src_root: Path):
    for path in sorted(src_root.rglob("*.avi")):
        if path.is_file() and not path.name.startswith("._"):
            yield path


def blur_one(
    src_path: Path,
    src_root: Path,
    out_root: Path,
    helper_script: Path,
    python_path: Path,
    overwrite: bool,
) -> BlurJobResult:
    rel_path = src_path.relative_to(src_root)
    out_path = out_root / rel_path.with_suffix(".mp4")

    if out_path.exists() and not out_path.is_symlink() and not overwrite:
        return BlurJobResult(str(rel_path), "skipped")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()

    cmd = [str(python_path), str(helper_script), str(src_path), str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        out_path.unlink(missing_ok=True)
        return BlurJobResult(str(rel_path), "error", message)

    if not out_path.exists():
        return BlurJobResult(str(rel_path), "error", "output file missing after helper completed")

    return BlurJobResult(str(rel_path), "blurred")


def main() -> int:
    args = parse_args()

    if not args.src_root.exists():
        raise SystemExit(f"Source root does not exist: {args.src_root}")
    if not args.helper_script.exists():
        raise SystemExit(f"Helper script does not exist: {args.helper_script}")
    if not args.python.exists():
        raise SystemExit(f"Python runtime does not exist: {args.python}")

    videos = list(iter_videos(args.src_root))
    if args.limit is not None:
        videos = videos[: args.limit]

    total = len(videos)
    if total == 0:
        print("No source videos found.")
        return 0

    print(f"Source root: {args.src_root}")
    print(f"Output root: {args.out_root}")
    print(f"Helper script: {args.helper_script}")
    print(f"Python runtime: {args.python}")
    print(f"Jobs: {args.jobs}")
    print(f"Videos queued: {total}")

    blurred = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(
                blur_one,
                src_path,
                args.src_root,
                args.out_root,
                args.helper_script,
                args.python,
                args.overwrite,
            )
            for src_path in videos
        ]

        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result.status == "blurred":
                blurred += 1
            elif result.status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"ERROR {result.rel_path}: {result.message}")

            if index % 10 == 0 or index == total:
                print(
                    f"Progress: {index}/{total} | blurred={blurred} skipped={skipped} errors={errors}"
                )

    print("Finished.")
    print(f"Blurred: {blurred}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())