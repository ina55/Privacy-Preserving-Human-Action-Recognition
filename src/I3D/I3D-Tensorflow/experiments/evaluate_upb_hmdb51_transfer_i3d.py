import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


SCRIPT_DIR = Path(__file__).resolve().parent
I3D_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[3]

if str(I3D_ROOT) not in sys.path:
    sys.path.insert(0, str(I3D_ROOT))

import input_data
from i3d import InceptionI3d


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


def parse_args():
    default_checkpoint_dir = WORKSPACE_ROOT / "work_dirs/i3d_hmdb51_rgb_video_paper/checkpoints"
    default_upb_dir = WORKSPACE_ROOT / "datasets/upb"
    default_hmdb_train_csv = WORKSPACE_ROOT / "datasets/hmdb51/hmdb51_1/train.csv"
    default_mapping = WORKSPACE_ROOT / "project/VideoMAEv2/scripts/upb_hmdb51_mapping.json"
    default_output_dir = default_upb_dir / "results_i3d_hmdb51_transfer"

    parser = argparse.ArgumentParser(
        description="Evaluate UPB videos with an HMDB51-finetuned I3D RGB checkpoint"
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(default_checkpoint_dir), help="Directory containing TensorFlow checkpoint files")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit TensorFlow checkpoint prefix")
    parser.add_argument("--upb-dir", type=str, default=str(default_upb_dir), help="Path to the UPB dataset root")
    parser.add_argument("--subset", type=str, default="all", choices=["all", "DatasetHAR", "HAR_ActiuniNoi"], help="Which UPB subset to evaluate")
    parser.add_argument("--hmdb51-train-csv", type=str, default=str(default_hmdb_train_csv), help="HMDB51 train.csv used to recover label order")
    parser.add_argument("--mapping", type=str, default=str(default_mapping), help="JSON file describing the HMDB51-to-UPB class mapping")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir), help="Directory for reports and raw results")
    parser.add_argument("--report-name", type=str, default="upb_eval_report.txt", help="Text report filename")
    parser.add_argument("--results-name", type=str, default="upb_eval_results.json", help="Raw JSON results filename")
    parser.add_argument("--num-frames", type=int, default=32, help="Frames per clip")
    parser.add_argument("--sample-rate", type=int, default=2, help="Temporal sampling rate")
    parser.add_argument("--crop-size", type=int, default=224, help="Center crop size")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value; set to empty string to force CPU")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of evaluated videos for debugging")
    parser.add_argument("--log-every", type=int, default=25, help="Progress log frequency in videos")
    parser.add_argument("--mapped-only", dest="mapped_only", action="store_true", help="Evaluate only UPB classes that have at least one HMDB51 mapping")
    parser.add_argument("--include-unmapped", dest="mapped_only", action="store_false", help="Also run inference on UPB classes that have no HMDB51 mapping")
    parser.set_defaults(mapped_only=True)
    return parser.parse_args()


def build_label_map_from_csv(csv_path):
    label_to_class = {}
    with open(csv_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rel_path, label = line.split()[:2]
            label = int(label)
            class_name = rel_path.split("/")[0]
            label_to_class.setdefault(label, class_name)
    return [label_to_class[index] for index in range(max(label_to_class) + 1)]


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
        details = ", ".join(f"{subset}/{code}->{label}" for subset, code, label in bad_entries)
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
        handle.write("UPB EVALUATION WITH HMDB51-FINETUNED I3D RGB\n")
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


def resolve_checkpoint_path(checkpoint_dir, checkpoint_path):
    if checkpoint_path:
        return checkpoint_path
    ckpt = tf.train.get_checkpoint_state(str(checkpoint_dir))
    if ckpt and ckpt.model_checkpoint_path:
        return ckpt.model_checkpoint_path
    raise ValueError(f"No checkpoint found in {checkpoint_dir}")


def preprocess_video(video_path, num_frames, sample_rate, crop_size):
    rgb_frames, _, _ = input_data.get_frames_data(
        str(video_path),
        num_frames,
        sample_rate,
        add_flow=False,
    )
    if not rgb_frames:
        return None
    return np.array(input_data.data_process(rgb_frames, crop_size), dtype=np.float32)


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, args.checkpoint_path)
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
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    with tf.Graph().as_default():
        rgb_images_placeholder = tf.placeholder(
            tf.float32,
            shape=(1, args.num_frames, args.crop_size, args.crop_size, 3),
            name="rgb_images",
        )
        is_training = tf.placeholder(tf.bool, name="is_training")

        with tf.variable_scope("RGB"):
            rgb_logits, _ = InceptionI3d(
                num_classes=len(hmdb51_labels),
                spatial_squeeze=True,
                final_endpoint="Logits",
            )(rgb_images_placeholder, is_training)

        probabilities = tf.nn.softmax(rgb_logits)
        saver = tf.train.Saver()
        session_config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            print(f"Loading evaluation checkpoint {checkpoint_path}")
            saver.restore(sess, checkpoint_path)

            results = []
            for index, sample in enumerate(samples, start=1):
                subset_mapping = mapping.get(sample["subset"], {})
                mapping_spec = subset_mapping.get(sample["action_code"], {})
                mapped_hmdb51 = mapping_spec.get("hmdb51", [])

                if index % args.log_every == 0 or index == 1 or index == len(samples):
                    print(
                        f"[{index}/{len(samples)}] {sample['subset']} | {sample['path'].name} | "
                        f"UPB={sample['upb_class_name']} | HMDB51 targets={mapped_hmdb51 or 'UNMAPPED'}"
                    )

                clip = preprocess_video(sample["path"], args.num_frames, args.sample_rate, args.crop_size)
                if clip is None:
                    print(f"Skipping unreadable video: {sample['path']}")
                    continue

                probs = sess.run(
                    probabilities,
                    feed_dict={
                        rgb_images_placeholder: clip[np.newaxis, ...],
                        is_training: False,
                    },
                )[0]
                top5_indices = np.argsort(probs)[::-1][:5]
                top1_index = int(top5_indices[0])
                top1_hmdb51 = hmdb51_labels[top1_index]
                top5_hmdb51 = [hmdb51_labels[int(idx)] for idx in top5_indices]
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
                        "probabilities": [float(probs[int(idx)]) for idx in top5_indices],
                    }
                )

    metrics = compute_metrics(results)
    results_payload = {
        "checkpoint": checkpoint_path,
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