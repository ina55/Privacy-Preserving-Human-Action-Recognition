import argparse
import json
import os
from pathlib import Path

from evaluate_hmdb51_local import (
    anonymize_video,
    build_label_map_from_csv,
    compute_metrics,
    load_label_map,
    load_model,
    run_inference_on_video,
)


def build_video_index(dataset_dir):
    valid_exts = {'.avi', '.mp4', '.mov', '.mkv', '.webm', '.m4v'}
    index = {}

    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        class_index = {}
        for file_path in class_dir.iterdir():
            if not file_path.is_file() or file_path.name.startswith('._'):
                continue
            if file_path.suffix.lower() not in valid_exts:
                continue

            stem = file_path.stem
            class_index.setdefault(stem, file_path)
            if stem.startswith('anonymized_'):
                class_index.setdefault(stem[len('anonymized_'):], file_path)

        index[class_dir.name] = class_index

    return index


def resolve_ucf101_video_path(dataset_dir, rel_path, video_index=None):
    rel_path_obj = Path(rel_path)
    direct_path = dataset_dir / rel_path_obj
    if direct_path.exists() and direct_path.is_file() and not direct_path.name.startswith('._'):
        return direct_path

    class_name = rel_path_obj.parent.name
    stem = rel_path_obj.stem

    if video_index is not None:
        return video_index.get(class_name, {}).get(stem)

    return None


def write_report(original_metrics, anonymized_metrics, label_names, output_dir, report_name, include_anonymized=True):
    report_path = output_dir / report_name

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        if include_anonymized:
            f.write("UCF101 EVALUATION REPORT (ORIGINAL VS ANONYMIZED)\n")
        else:
            f.write("UCF101 EVALUATION REPORT (ORIGINAL ONLY)\n")
        f.write("=" * 80 + "\n\n")

        f.write("--- OVERALL SUMMARY ---\n")

        ob_total = original_metrics['overall']['total']
        ob_top1 = original_metrics['overall'].get('top1_acc', 0)
        ob_top5 = original_metrics['overall'].get('top5_acc', 0)

        f.write(f"Total Videos Evaluated: {ob_total}\n\n")
        f.write("Original Videos:\n")
        f.write(
            f"  Top-1 Accuracy: {ob_top1:.2f}% ({original_metrics['overall']['top1_correct']}/{ob_total})\n"
        )
        f.write(
            f"  Top-5 Accuracy: {ob_top5:.2f}% ({original_metrics['overall']['top5_correct']}/{ob_total})\n\n"
        )

        if include_anonymized:
            an_total = anonymized_metrics['overall']['total']
            an_top1 = anonymized_metrics['overall'].get('top1_acc', 0)
            an_top5 = anonymized_metrics['overall'].get('top5_acc', 0)

            f.write("Anonymized (Blurred) Videos:\n")
            f.write(
                f"  Top-1 Accuracy: {an_top1:.2f}% ({anonymized_metrics['overall']['top1_correct']}/{an_total})\n"
            )
            f.write(
                f"  Top-5 Accuracy: {an_top5:.2f}% ({anonymized_metrics['overall']['top5_correct']}/{an_total})\n\n"
            )

            f.write("Difference (Original - Anonymized):\n")
            f.write(f"  Top-1 Drop: {ob_top1 - an_top1:.2f}%\n")
            f.write(f"  Top-5 Drop: {ob_top5 - an_top5:.2f}%\n\n")

        f.write("--- PER-CLASS BREAKDOWN ---\n")
        if include_anonymized:
            f.write(
                f"{'Class Name':<20} | {'Orig Top-1':<10} | {'Anon Top-1':<10} | {'Diff':<8} || {'Orig Top-5':<10} | {'Anon Top-5':<10} | {'Total':<5}\n"
            )
            f.write("-" * 95 + "\n")
        else:
            f.write(f"{'Class Name':<20} | {'Top-1':<10} | {'Top-5':<10} | {'Total':<5}\n")
            f.write("-" * 60 + "\n")

        for class_name in label_names:
            orig_data = original_metrics['per_class'].get(
                class_name,
                {'total': 0, 'top1_acc': 0, 'top5_acc': 0},
            )
            if orig_data['total'] == 0:
                continue

            orig_t1 = orig_data.get('top1_acc', 0)
            orig_t5 = orig_data.get('top5_acc', 0)

            if include_anonymized:
                anon_data = anonymized_metrics['per_class'].get(
                    class_name,
                    {'total': 0, 'top1_acc': 0, 'top5_acc': 0},
                )
                anon_t1 = anon_data.get('top1_acc', 0)
                anon_t5 = anon_data.get('top5_acc', 0)
                diff_t1 = orig_t1 - anon_t1
                f.write(
                    f"{class_name:<20} | {orig_t1:>9.2f}% | {anon_t1:>9.2f}% | {diff_t1:>7.2f}% || {orig_t5:>9.2f}% | {anon_t5:>9.2f}% | {orig_data['total']:>5}\n"
                )
            else:
                f.write(
                    f"{class_name:<20} | {orig_t1:>9.2f}% | {orig_t5:>9.2f}% | {orig_data['total']:>5}\n"
                )

    print(f"\nReport successfully saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate UCF101 local dataset")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--model', type=str, default='vit_base', choices=['vit_base', 'vit_giant'], help="Model architecture")
    parser.add_argument('--limit', type=int, default=0, help="Limit number of videos to evaluate (for debugging)")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for inference")
    parser.add_argument('--ucf101-dir', type=str, default='', help="Path to the UCF101 dataset directory containing train.csv/val.csv")
    parser.add_argument('--eval-split', type=str, default='val.csv', help="Split file to evaluate inside the dataset directory")
    parser.add_argument('--report-name', type=str, default='eval_report_ucf101.txt', help="Report filename written inside the results directory")
    parser.add_argument('--results-name', type=str, default='eval_results_raw_ucf101.json', help="Raw JSON results filename written inside the results directory")
    parser.add_argument('--anonymize-video', dest='anonymize_video', action='store_true', help="Run anonymization and compare original vs anonymized")
    parser.add_argument('--no-anonymize-video', dest='anonymize_video', action='store_false', help="Skip anonymization and evaluate only original videos")
    parser.set_defaults(anonymize_video=True)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    workspace_root = base_dir.parents[1]

    ucf101_dir = Path(args.ucf101_dir) if args.ucf101_dir else (workspace_root / "datasets/ucf101/ucf101_1")
    anonymization_script = base_dir / "input/anonymization/anonymization.py"
    anonymized_output_env = os.getenv("ANON_OUTPUT_DIR", "").strip()
    anonymized_output_dir = Path(anonymized_output_env) if anonymized_output_env else (workspace_root / "datasets/ucf101/ucf101_anonimized")
    results_dir = workspace_root / "datasets/ucf101/ucf101_results"

    if args.anonymize_video:
        anonymized_output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset dir: {ucf101_dir}")
    print(f"Anonymization enabled: {args.anonymize_video}")

    label_map_file = ucf101_dir / "label_map.txt"
    train_csv = ucf101_dir / "train.csv"
    eval_csv = ucf101_dir / args.eval_split

    if label_map_file.exists():
        label_names = load_label_map(label_map_file)
    elif train_csv.exists():
        label_names = build_label_map_from_csv(train_csv)
    elif eval_csv.exists():
        label_names = build_label_map_from_csv(eval_csv)
    else:
        raise FileNotFoundError(
            f"Could not find label_map.txt, train.csv, or {args.eval_split} in {ucf101_dir}"
        )
    print(f"Loaded {len(label_names)} classes.")

    video_index = build_video_index(ucf101_dir)

    videos_to_eval = []

    with open(eval_csv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            rel_path = parts[0]
            if rel_path.startswith("ucf101_1/"):
                rel_path = rel_path[len("ucf101_1/"):]

            full_path = resolve_ucf101_video_path(ucf101_dir, rel_path, video_index)
            label = int(parts[1])
            class_name = label_names[label]

            if full_path is not None:
                videos_to_eval.append({
                    'path': full_path,
                    'label': label,
                    'class_name': class_name,
                })

                if args.limit > 0 and len(videos_to_eval) >= args.limit:
                    break

    print(f"Found {len(videos_to_eval)} videos to evaluate.")

    model = load_model(
        args.checkpoint,
        model_type=args.model,
        num_classes=len(label_names),
        device=args.device,
    )

    original_results = []
    anonymized_results = []

    try:
        for idx, video_info in enumerate(videos_to_eval, 1):
            video_path = video_info['path']
            label = video_info['label']
            class_name = video_info['class_name']

            print(f"\n[{idx}/{len(videos_to_eval)}] Processing: {video_path.name} (Class: {class_name})")

            res_orig = run_inference_on_video(video_path, model, args.device)
            if res_orig:
                res_orig['label'] = label
                res_orig['class_name'] = class_name
                original_results.append(res_orig)
                is_correct = res_orig['top1_pred'] == label
                print(f"  ✓ Original Top-1: {res_orig['top1_pred']} (Correct: {is_correct})")
            else:
                print("  ✗ Original inference failed")
                continue

            if not args.anonymize_video:
                continue

            anon_path = anonymize_video(video_path, anonymization_script, anonymized_output_dir)
            if anon_path:
                res_anon = run_inference_on_video(anon_path, model, args.device)
                if res_anon:
                    res_anon['label'] = label
                    res_anon['class_name'] = class_name
                    anonymized_results.append(res_anon)
                    is_correct = res_anon['top1_pred'] == label
                    print(f"  ✓ Anonymized Top-1: {res_anon['top1_pred']} (Correct: {is_correct})")
                else:
                    print("  ✗ Anonymized inference failed")
            else:
                print("  ✗ Anonymization failed")

    except KeyboardInterrupt:
        print("\n\n[!] Evaluation interrupted by user! Saving partial results...")

    finally:
        orig_metrics = compute_metrics(original_results)
        anon_metrics = compute_metrics(anonymized_results)

        with open(results_dir / args.results_name, 'w') as f:
            json.dump(
                {
                    'original': original_results,
                    'anonymized': anonymized_results,
                    'metrics_original': orig_metrics,
                    'metrics_anonymized': anon_metrics,
                },
                f,
                indent=2,
            )

        write_report(
            orig_metrics,
            anon_metrics,
            label_names,
            results_dir,
            args.report_name,
            include_anonymized=args.anonymize_video,
        )


if __name__ == "__main__":
    main()
