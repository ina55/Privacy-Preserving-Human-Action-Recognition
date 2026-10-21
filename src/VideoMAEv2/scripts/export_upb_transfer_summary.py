import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a compact CSV and Markdown summary from UPB transfer evaluation JSON results"
    )
    parser.add_argument("results_json", type=str, help="Path to upb_eval_results.json")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional output CSV path (default: alongside input JSON as upb_eval_summary.csv)",
    )
    parser.add_argument(
        "--md-path",
        type=str,
        default=None,
        help="Optional output Markdown path (default: alongside input JSON as upb_eval_summary.md)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="UPB transfer evaluation summary",
        help="Markdown title",
    )
    return parser.parse_args()


def build_rows(metrics):
    rows = []
    for class_name, class_stats in sorted(
        metrics["per_class"].items(),
        key=lambda item: (item[1]["subset"], item[1]["action_code"], item[0]),
    ):
        mapped = class_stats.get("mapped_hmdb51", [])
        histogram = class_stats.get("top1_prediction_histogram", [])
        rows.append(
            {
                "subset": class_stats["subset"],
                "action_code": class_stats["action_code"],
                "upb_class_name": class_name,
                "mapped_hmdb51": ", ".join(mapped) if mapped else "UNMAPPED",
                "mapped_total": class_stats["mapped_total"],
                "total": class_stats["total"],
                "top1_correct": class_stats["top1_correct"],
                "top5_correct": class_stats["top5_correct"],
                "top1_acc": round(class_stats["top1_acc"], 2),
                "top5_acc": round(class_stats["top5_acc"], 2),
                "top1_predictions": ", ".join(f"{name}:{count}" for name, count in histogram),
            }
        )
    return rows


def write_csv(csv_path, rows):
    fieldnames = [
        "subset",
        "action_code",
        "upb_class_name",
        "mapped_hmdb51",
        "mapped_total",
        "total",
        "top1_correct",
        "top5_correct",
        "top1_acc",
        "top5_acc",
        "top1_predictions",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(md_path, title, payload, rows):
    overall = payload["metrics"]["overall"]
    lines = [
        f"# {title}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total UPB videos | {overall['total']} |",
        f"| Mapped videos | {overall['mapped_total']} |",
        f"| Unmapped videos | {overall['unmapped_total']} |",
        f"| Mapped Top-1 Accuracy | {overall['top1_acc']:.2f}% ({overall['top1_correct']}/{overall['mapped_total']}) |",
        f"| Mapped Top-5 Accuracy | {overall['top5_acc']:.2f}% ({overall['top5_correct']}/{overall['mapped_total']}) |",
        "",
        "## Per-class",
        "",
        "| Subset | Action | UPB class | Mapped HMDB51 | Top-1 | Top-5 | Mapped | Top-1 predictions |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {subset} | {action_code} | {upb_class_name} | {mapped_hmdb51} | {top1_acc:.2f}% | {top5_acc:.2f}% | {mapped_total} | {top1_predictions} |".format(
                **row
            )
        )
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    results_json = Path(args.results_json)
    output_dir = results_json.parent
    csv_path = Path(args.csv_path) if args.csv_path else output_dir / "upb_eval_summary.csv"
    md_path = Path(args.md_path) if args.md_path else output_dir / "upb_eval_summary.md"

    with open(results_json, "r") as handle:
        payload = json.load(handle)

    rows = build_rows(payload["metrics"])
    write_csv(csv_path, rows)
    write_markdown(md_path, args.title, payload, rows)

    print(f"saved_csv {csv_path}")
    print(f"saved_markdown {md_path}")


if __name__ == "__main__":
    main()