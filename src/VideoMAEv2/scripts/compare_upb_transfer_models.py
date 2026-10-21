import argparse
import csv
from pathlib import Path


KEY_FIELDS = ("subset", "action_code", "upb_class_name")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two UPB transfer summary CSV files and export a single report table"
    )
    parser.add_argument("--model-a-name", required=True, help="Label for the first model, e.g. VideoMAE V2")
    parser.add_argument("--model-a-csv", required=True, help="Path to the first model summary CSV")
    parser.add_argument("--model-b-name", required=True, help="Label for the second model, e.g. I3D")
    parser.add_argument("--model-b-csv", required=True, help="Path to the second model summary CSV")
    parser.add_argument("--output-csv", required=True, help="Output path for the comparison CSV")
    parser.add_argument("--output-md", required=True, help="Output path for the comparison Markdown table")
    parser.add_argument(
        "--title",
        default="UPB transfer comparison",
        help="Markdown document title",
    )
    return parser.parse_args()


def read_summary(path):
    rows = {}
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = tuple(row[field] for field in KEY_FIELDS)
            rows[key] = row
    return rows


def parse_float(row, field):
    return float(row[field]) if row and row.get(field) not in (None, "") else 0.0


def compare_rows(rows_a, rows_b, model_a_name, model_b_name):
    keys = sorted(set(rows_a) | set(rows_b))
    compared = []
    for key in keys:
        row_a = rows_a.get(key)
        row_b = rows_b.get(key)
        subset, action_code, upb_class_name = key

        mapped_hmdb51 = ""
        mapped_total = ""
        if row_a:
            mapped_hmdb51 = row_a["mapped_hmdb51"]
            mapped_total = row_a["mapped_total"]
        elif row_b:
            mapped_hmdb51 = row_b["mapped_hmdb51"]
            mapped_total = row_b["mapped_total"]

        top1_a = parse_float(row_a, "top1_acc")
        top5_a = parse_float(row_a, "top5_acc")
        top1_b = parse_float(row_b, "top1_acc")
        top5_b = parse_float(row_b, "top5_acc")

        compared.append(
            {
                "subset": subset,
                "action_code": action_code,
                "upb_class_name": upb_class_name,
                "mapped_hmdb51": mapped_hmdb51,
                "mapped_total": mapped_total,
                f"{model_a_name}_top1": round(top1_a, 2),
                f"{model_a_name}_top5": round(top5_a, 2),
                f"{model_b_name}_top1": round(top1_b, 2),
                f"{model_b_name}_top5": round(top5_b, 2),
                "delta_top1": round(top1_b - top1_a, 2),
                "delta_top5": round(top5_b - top5_a, 2),
            }
        )

    compared.sort(key=lambda row: (row["subset"], row["action_code"], row["upb_class_name"]))
    return compared


def write_csv(path, rows, model_a_name, model_b_name):
    fieldnames = [
        "subset",
        "action_code",
        "upb_class_name",
        "mapped_hmdb51",
        "mapped_total",
        f"{model_a_name}_top1",
        f"{model_a_name}_top5",
        f"{model_b_name}_top1",
        f"{model_b_name}_top5",
        "delta_top1",
        "delta_top5",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, title, model_a_name, model_b_name):
    lines = [
        f"# {title}",
        "",
        "| Subset | Action | UPB class | Mapped HMDB51 | {} Top-1 | {} Top-5 | {} Top-1 | {} Top-5 | Delta Top-1 | Delta Top-5 |".format(
            model_a_name,
            model_a_name,
            model_b_name,
            model_b_name,
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {subset} | {action_code} | {upb_class_name} | {mapped_hmdb51} | {a1:.2f}% | {a5:.2f}% | {b1:.2f}% | {b5:.2f}% | {d1:+.2f} pp | {d5:+.2f} pp |".format(
                subset=row["subset"],
                action_code=row["action_code"],
                upb_class_name=row["upb_class_name"],
                mapped_hmdb51=row["mapped_hmdb51"],
                a1=row[f"{model_a_name}_top1"],
                a5=row[f"{model_a_name}_top5"],
                b1=row[f"{model_b_name}_top1"],
                b5=row[f"{model_b_name}_top5"],
                d1=row["delta_top1"],
                d5=row["delta_top5"],
            )
        )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows_a = read_summary(args.model_a_csv)
    rows_b = read_summary(args.model_b_csv)
    compared = compare_rows(rows_a, rows_b, args.model_a_name, args.model_b_name)

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    write_csv(output_csv, compared, args.model_a_name, args.model_b_name)
    write_markdown(output_md, compared, args.title, args.model_a_name, args.model_b_name)

    print(f"saved_csv {output_csv}")
    print(f"saved_markdown {output_md}")


if __name__ == "__main__":
    main()