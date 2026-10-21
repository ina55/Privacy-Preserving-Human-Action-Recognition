import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_hmdb51_local import build_label_map_from_csv, load_model, run_inference_on_video


UPB_ACTIONS = {
    "DatasetHAR": {
        "A41": "sneeze/cough",
        "A42": "staggering",
        "A43": "falling down",
        "A44": "headache",
        "A45": "chest pain",
        "A46": "back pain",
        "A47": "neck pain",
        "A48": "nausea/vomiting",
        "A49": "fan self",
        "A103": "yawn",
        "A104": "stretch oneself",
        "A105": "blow nose",
    },
    "HAR_ActiuniNoi": {
        "A1-v1": "drink water from bottle",
        "A1-v2": "drink water from cup",
        "A2": "eat meal",
        "A28": "phone call",
        "A8": "sit down",
        "A9": "stand up",
        "A80": "squat down",
        "A97": "arm circles",
        "A98": "arm swings",
        "A101": "cross toe touch",
        "A33": "check time (from watch)",
        "A31": "point to something",
        "C1": "sit down, stand up, phone call",
        "C2": "stand up, drink water from cup, sit down",
    },
}


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def load_mapping(mapping_path):
    with open(mapping_path, "r") as handle:
        return json.load(handle)


def validate_mapping(mapping, hmdb51_labels):
    hmdb51_set = set(hmdb51_labels)
    bad_entries = []
    for subset_name, subset_mapping in mapping.items():
        if subset_name in {"mapping_version", "notes"}:
            continue
        for action_code, spec in subset_mapping.items():
            for hmdb51_class in spec.get("hmdb51", []):
                if hmdb51_class not in hmdb51_set:
                    bad_entries.append((subset_name, action_code, hmdb51_class))
    if bad_entries:
        details = ", ".join(
            f"{subset}/{code}->{cls}" for subset, code, cls in bad_entries
        )
        raise ValueError(f"Unknown HMDB51 classes in mapping: {details}")


def enumerate_upb_videos(upb_root, subset):
    subset_names = [subset] if subset != "all" else ["DatasetHAR", "HAR_ActiuniNoi"]
    samples = []

    for subset_name in subset_names:
        subset_dir = upb_root / subset_name
        action_lookup = UPB_ACTIONS[subset_name]
        for path in sorted(subset_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            stem_parts = path.stem.split("_")
            if len(stem_parts) < 3:
                continue
            action_code = stem_parts[-1]
            samples.append(
                {
                    "subset": subset_name,
                    "path": path,
                    "action_code": action_code,
                    "upb_class_name": action_lookup.get(action_code, action_code),
                    "subject_id": stem_parts[-2],
                }
            )
    return samples


def filter_mapped_samples(samples, mapping):
    filtered = []
    for sample in samples:
        subset_mapping = mapping.get(sample["subset"], {})
        mapping_spec = subset_mapping.get(sample["action_code"], {})
        if mapping_spec.get("hmdb51", []):
            filtered.append(sample)
    return filtered


def score_prediction(top1_hmdb51, top5_hmdb51, mapped_hmdb51):
    mapped_set = set(mapped_hmdb51)
    return top1_hmdb51 in mapped_set, any(pred in mapped_set for pred in top5_hmdb51)


def compute_metrics(results):
    metrics = {
        "overall": {
            "total": 0,
            "mapped_total": 0,
            "unmapped_total": 0,
            "top1_correct": 0,
            "top5_correct": 0,
        },
        "per_class": {},
    }

    grouped_predictions = defaultdict(Counter)

    for res in results:
        class_name = res["upb_class_name"]
        class_stats = metrics["per_class"].setdefault(
            class_name,
            {
                "subset": res["subset"],
                "action_code": res["action_code"],
                "mapped_hmdb51": res["mapped_hmdb51"],
                "total": 0,
                "mapped_total": 0,
                "unmapped_total": 0,
                "top1_correct": 0,
                "top5_correct": 0,
            },
        )

        metrics["overall"]["total"] += 1
        class_stats["total"] += 1
        grouped_predictions[class_name][res["top1_hmdb51"]] += 1

        if not res["mapped_hmdb51"]:
            metrics["overall"]["unmapped_total"] += 1
            class_stats["unmapped_total"] += 1
            continue

        metrics["overall"]["mapped_total"] += 1
        class_stats["mapped_total"] += 1

        if res["top1_correct"]:
            metrics["overall"]["top1_correct"] += 1
            class_stats["top1_correct"] += 1
        if res["top5_correct"]:
            metrics["overall"]["top5_correct"] += 1
            class_stats["top5_correct"] += 1

    mapped_total = metrics["overall"]["mapped_total"]
    if mapped_total > 0:
        metrics["overall"]["top1_acc"] = 100.0 * metrics["overall"]["top1_correct"] / mapped_total
        metrics["overall"]["top5_acc"] = 100.0 * metrics["overall"]["top5_correct"] / mapped_total
    else:
        metrics["overall"]["top1_acc"] = 0.0
        metrics["overall"]["top5_acc"] = 0.0

    for class_name, class_stats in metrics["per_class"].items():
        mapped_class_total = class_stats["mapped_total"]
        if mapped_class_total > 0:
            class_stats["top1_acc"] = 100.0 * class_stats["top1_correct"] / mapped_class_total
            class_stats["top5_acc"] = 100.0 * class_stats["top5_correct"] / mapped_class_total
        else:
            class_stats["top1_acc"] = 0.0
            class_stats["top5_acc"] = 0.0
        class_stats["top1_prediction_histogram"] = grouped_predictions[class_name].most_common(5)

    return metrics


def write_report(report_path, checkpoint_path, mapping_path, hmdb51_train_csv, metrics):
    overall = metrics["overall"]
    with open(report_path, "w") as handle:
        handle.write("=" * 90 + "\n")
        handle.write("UPB EVALUATION WITH HMDB51-FINETUNED VIDEOMAE V2\n")
        handle.write("=" * 90 + "\n\n")
        handle.write(f"Checkpoint: {checkpoint_path}\n")
        handle.write(f"HMDB51 train.csv: {hmdb51_train_csv}\n")
        handle.write(f"Mapping file: {mapping_path}\n\n")

        handle.write("--- OVERALL ---\n")
        handle.write(f"Total UPB videos: {overall['total']}\n")
        handle.write(f"Mapped videos: {overall['mapped_total']}\n")
        handle.write(f"Unmapped videos: {overall['unmapped_total']}\n")
        handle.write(
            f"Mapped Top-1 Accuracy: {overall['top1_acc']:.2f}% ({overall['top1_correct']}/{overall['mapped_total']})\n"
        )
        handle.write(
            f"Mapped Top-5 Accuracy: {overall['top5_acc']:.2f}% ({overall['top5_correct']}/{overall['mapped_total']})\n\n"
        )

        handle.write("--- PER-CLASS ---\n")
        handle.write(
            f"{'Subset':<16} | {'Action':<8} | {'UPB Class':<34} | {'Mapped HMDB51':<24} | {'Top-1':<8} | {'Top-5':<8} | {'Mapped':<6} | {'Total':<5}\n"
        )
        handle.write("-" * 140 + "\n")

        for class_name, class_stats in sorted(metrics["per_class"].items()):
            mapped_str = ", ".join(class_stats["mapped_hmdb51"]) if class_stats["mapped_hmdb51"] else "UNMAPPED"
            handle.write(
                f"{class_stats['subset']:<16} | {class_stats['action_code']:<8} | {class_name:<34} | {mapped_str:<24} | "
                f"{class_stats['top1_acc']:>6.2f}% | {class_stats['top5_acc']:>6.2f}% | {class_stats['mapped_total']:>6} | {class_stats['total']:>5}\n"
            )

            if class_stats["top1_prediction_histogram"]:
                preds = ", ".join(
                    f"{name}:{count}" for name, count in class_stats["top1_prediction_histogram"]
                )
                handle.write(f"  top1 predictions: {preds}\n")
        handle.write("\n")


def main():
    workspace_root = SCRIPT_DIR.parents[2]
    default_checkpoint = workspace_root / "work_dirs/hmdb51_vit_b_finetune/checkpoint-best.pth"
    default_upb_dir = workspace_root / "datasets/upb"
    default_hmdb_train_csv = workspace_root / "datasets/hmdb51/hmdb51_1/train.csv"
    default_mapping = SCRIPT_DIR / "upb_hmdb51_mapping.json"
    default_output_dir = default_upb_dir / "results_videomaev2_hmdb51_transfer"

    parser = argparse.ArgumentParser(description="Evaluate UPB videos with an HMDB51-finetuned VideoMAE V2 checkpoint")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint), help="Path to the HMDB51-finetuned checkpoint")
    parser.add_argument("--model", type=str, default="vit_base", choices=["vit_base", "vit_giant"], help="VideoMAE V2 architecture")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Inference device")
    parser.add_argument("--upb-dir", type=str, default=str(default_upb_dir), help="Path to the UPB dataset root")
    parser.add_argument("--subset", type=str, default="all", choices=["all", "DatasetHAR", "HAR_ActiuniNoi"], help="Which UPB subset to evaluate")
    parser.add_argument("--hmdb51-train-csv", type=str, default=str(default_hmdb_train_csv), help="HMDB51 train.csv used to recover the checkpoint label order")
    parser.add_argument("--mapping", type=str, default=str(default_mapping), help="JSON file describing the HMDB51-to-UPB class mapping")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir), help="Directory for reports and raw results")
    parser.add_argument("--report-name", type=str, default="upb_eval_report.txt", help="Text report filename")
    parser.add_argument("--results-name", type=str, default="upb_eval_results.json", help="Raw JSON results filename")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of evaluated videos for debugging")
    parser.add_argument("--mapped-only", dest="mapped_only", action="store_true", help="Evaluate only UPB classes that have at least one HMDB51 mapping")
    parser.add_argument("--include-unmapped", dest="mapped_only", action="store_false", help="Also run inference on UPB classes that have no HMDB51 mapping")
    parser.set_defaults(mapped_only=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    upb_dir = Path(args.upb_dir)
    hmdb51_train_csv = Path(args.hmdb51_train_csv)
    mapping_path = Path(args.mapping)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hmdb51_labels = build_label_map_from_csv(hmdb51_train_csv)
    mapping = load_mapping(mapping_path)
    validate_mapping(mapping, hmdb51_labels)

    samples = enumerate_upb_videos(upb_dir, args.subset)
    all_sample_count = len(samples)
    if args.mapped_only:
        samples = filter_mapped_samples(samples, mapping)
    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"Loaded {len(hmdb51_labels)} HMDB51 labels from {hmdb51_train_csv}")
    if args.mapped_only:
        print(f"Found {all_sample_count} UPB videos, evaluating {len(samples)} mapped videos.")
    else:
        print(f"Found {len(samples)} UPB videos to evaluate.")
    print(f"Using device: {args.device}")

    model = load_model(
        checkpoint_path,
        model_type=args.model,
        num_classes=len(hmdb51_labels),
        device=args.device,
    )

    results = []
    for index, sample in enumerate(samples, start=1):
        subset_mapping = mapping.get(sample["subset"], {})
        mapping_spec = subset_mapping.get(sample["action_code"], {})
        mapped_hmdb51 = mapping_spec.get("hmdb51", [])

        print(
            f"[{index}/{len(samples)}] {sample['subset']} | {sample['path'].name} | "
            f"UPB={sample['upb_class_name']} | HMDB51 targets={mapped_hmdb51 or 'UNMAPPED'}"
        )

        inference = run_inference_on_video(sample["path"], model, args.device)
        if inference is None:
            continue

        top1_hmdb51 = hmdb51_labels[inference["top1_pred"]]
        top5_hmdb51 = [hmdb51_labels[idx] for idx in inference["top5_preds"]]
        top1_correct, top5_correct = score_prediction(top1_hmdb51, top5_hmdb51, mapped_hmdb51)

        results.append(
            {
                "subset": sample["subset"],
                "video_path": str(sample["path"]),
                "action_code": sample["action_code"],
                "upb_class_name": sample["upb_class_name"],
                "mapped_hmdb51": mapped_hmdb51,
                "top1_hmdb51": top1_hmdb51,
                "top5_hmdb51": top5_hmdb51,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "probabilities": inference["probs"],
            }
        )

    metrics = compute_metrics(results)

    results_payload = {
        "checkpoint": str(checkpoint_path),
        "hmdb51_train_csv": str(hmdb51_train_csv),
        "mapping": str(mapping_path),
        "results": results,
        "metrics": metrics,
    }

    results_path = output_dir / args.results_name
    report_path = output_dir / args.report_name
    with open(results_path, "w") as handle:
        json.dump(results_payload, handle, indent=2)
    write_report(report_path, checkpoint_path, mapping_path, hmdb51_train_csv, metrics)

    print(f"Saved raw results to: {results_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()