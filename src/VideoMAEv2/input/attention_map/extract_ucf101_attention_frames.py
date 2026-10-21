import argparse
import json
from pathlib import Path

from attention_map import load_model, visualize_video_attention


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "ucf101"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "ucf101_attention_frames"
DEFAULT_CKPT = PROJECT_ROOT / "data" / "VideoMAEv2_data" / "vit_b_k710_dl_from_giant.pth"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-frame attention overlays for selected UCF101 classes across one or more dataset variants."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing UCF101 variants such as ucf101_1, ucf101_low56, and ucf101_low28.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["ucf101_1", "ucf101_low56", "ucf101_low28"],
        help="Variant directories to process under dataset_root.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["LongJump", "Haircut"],
        help="Class folder names to process inside each variant.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where attention overlay frames will be written.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help="Checkpoint path forwarded to the attention-map loader.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample from each video.")
    parser.add_argument(
        "--frames_to_visualize",
        type=int,
        default=10,
        help="Number of sampled frames to export per video.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Spatial size used before computing attention maps.",
    )
    parser.add_argument(
        "--max_videos_per_class",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--video_contains",
        default=None,
        help="Optional substring filter applied to video stems.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess videos even when the target directory already contains jpg outputs.",
    )
    parser.add_argument(
        "--annotate_labels",
        action="store_true",
        help="Annotate overlays when a face falls inside a high-attention region.",
    )
    return parser.parse_args()


def collect_videos(class_dir, stem_filter=None):
    videos = [
        path for path in sorted(class_dir.iterdir()) if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if stem_filter:
        videos = [path for path in videos if stem_filter.lower() in path.stem.lower()]
    return videos


def should_skip(output_dir, expected_frames, overwrite):
    if overwrite or not output_dir.exists():
        return False
    existing = sorted(output_dir.glob("*.jpg"))
    return len(existing) >= expected_frames


def main():
    args = parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device=args.device)
    manifest = []
    stats = {"processed": 0, "skipped": 0, "missing_class_dirs": 0, "missing_videos": 0}

    for variant in args.variants:
        variant_dir = args.dataset_root / variant
        if not variant_dir.exists():
            print(f"Missing variant directory: {variant_dir}")
            continue

        for class_name in args.classes:
            class_dir = variant_dir / class_name
            if not class_dir.exists():
                print(f"Missing class directory: {class_dir}")
                stats["missing_class_dirs"] += 1
                continue

            videos = collect_videos(class_dir, stem_filter=args.video_contains)
            if args.max_videos_per_class is not None:
                videos = videos[: args.max_videos_per_class]

            if not videos:
                print(f"No videos found in {class_dir}")
                stats["missing_videos"] += 1
                continue

            for video_path in videos:
                target_dir = args.output_root / variant / class_name / video_path.stem
                if should_skip(target_dir, args.frames_to_visualize, args.overwrite):
                    print(f"Skipping existing output: {target_dir}")
                    stats["skipped"] += 1
                    manifest.append(
                        {
                            "variant": variant,
                            "class": class_name,
                            "video": str(video_path),
                            "output_dir": str(target_dir),
                            "status": "skipped",
                        }
                    )
                    continue

                print(f"Processing {video_path}")
                target_dir.mkdir(parents=True, exist_ok=True)
                visualize_video_attention(
                    video_path=video_path,
                    model=model,
                    device=args.device,
                    num_frames=args.num_frames,
                    input_size=args.input_size,
                    frames_to_visualize=args.frames_to_visualize,
                    output_dir=target_dir,
                    det_model=None,
                    annotate_labels=args.annotate_labels,
                )
                generated = sorted(target_dir.glob("*.jpg"))
                manifest.append(
                    {
                        "variant": variant,
                        "class": class_name,
                        "video": str(video_path),
                        "output_dir": str(target_dir),
                        "frames_written": len(generated),
                        "status": "processed",
                    }
                )
                stats["processed"] += 1

    manifest_path = args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"stats": stats, "runs": manifest}, handle, indent=2)

    print("Finished attention extraction")
    print(json.dumps({"stats": stats, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()