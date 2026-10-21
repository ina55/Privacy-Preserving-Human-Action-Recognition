#!/usr/bin/env python3

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build strict paper-grade UCF101 splits from train.csv and val.csv.'
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('/path/to/project/datasets/ucf101/ucf101_1'),
        help='UCF101 dataset directory containing train.csv and val.csv.',
    )
    parser.add_argument(
        '--train-csv',
        default='train.csv',
        help='Training CSV inside the dataset root.',
    )
    parser.add_argument(
        '--official-test-csv',
        default='val.csv',
        help='Held-out final test CSV inside the dataset root.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory where train.csv, val.csv, test.csv and metadata.json will be written.',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Per-class ratio taken from train.csv to form the train-only validation split.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for deterministic per-class validation sampling.',
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(' ', 1)
            rows.append((rel_path, int(label)))
    return rows


def write_rows(rows, csv_path: Path):
    with csv_path.open('w', encoding='utf-8') as handle:
        for rel_path, label in rows:
            handle.write(f'{rel_path} {label}\n')


def main():
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    train_csv = (dataset_root / args.train_csv).resolve()
    official_test_csv = (dataset_root / args.official_test_csv).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(train_csv)
    official_test_rows = load_rows(official_test_csv)

    grouped = defaultdict(list)
    for rel_path, label in train_rows:
        grouped[label].append((rel_path, label))

    rng = random.Random(args.seed)
    paper_train_rows = []
    paper_val_rows = []
    per_class_val_counts = {}

    for label in sorted(grouped):
        label_rows = sorted(grouped[label])
        if len(label_rows) <= 1:
            raise ValueError(f'Class {label} has too few samples to split safely.')
        rng.shuffle(label_rows)
        val_count = max(1, int(round(len(label_rows) * args.val_ratio)))
        val_count = min(val_count, len(label_rows) - 1)
        per_class_val_counts[label] = val_count
        paper_val_rows.extend(sorted(label_rows[:val_count]))
        paper_train_rows.extend(sorted(label_rows[val_count:]))

    paper_train_rows.sort()
    paper_val_rows.sort()
    official_test_rows = sorted(official_test_rows)

    train_paths = {rel for rel, _ in paper_train_rows}
    val_paths = {rel for rel, _ in paper_val_rows}
    test_paths = {rel for rel, _ in official_test_rows}
    overlaps = {
        'train_val': len(train_paths & val_paths),
        'train_test': len(train_paths & test_paths),
        'val_test': len(val_paths & test_paths),
    }
    if any(overlaps.values()):
        raise ValueError(f'Generated splits overlap: {overlaps}')

    train_counts = Counter(label for _, label in paper_train_rows)
    val_counts = Counter(label for _, label in paper_val_rows)
    test_counts = Counter(label for _, label in official_test_rows)

    write_rows(paper_train_rows, output_dir / 'train.csv')
    write_rows(paper_val_rows, output_dir / 'val.csv')
    write_rows(official_test_rows, output_dir / 'test.csv')

    metadata = {
        'dataset_root': str(dataset_root),
        'train_csv': str(train_csv),
        'official_test_csv': str(official_test_csv),
        'output_dir': str(output_dir),
        'seed': args.seed,
        'val_ratio': args.val_ratio,
        'counts': {
            'paper_train': len(paper_train_rows),
            'paper_val': len(paper_val_rows),
            'official_test': len(official_test_rows),
        },
        'per_class_counts': {
            'paper_train': {str(label): train_counts[label] for label in sorted(train_counts)},
            'paper_val': {str(label): val_counts[label] for label in sorted(val_counts)},
            'official_test': {str(label): test_counts[label] for label in sorted(test_counts)},
        },
        'per_class_val_counts': {str(label): per_class_val_counts[label] for label in sorted(per_class_val_counts)},
        'overlaps': overlaps,
        'notes': [
            'Validation split is sampled only from train.csv.',
            'The original val.csv is reserved as the final untouched test split.',
        ],
    }
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f'Wrote {len(paper_train_rows)} paper-train rows to {output_dir / "train.csv"}')
    print(f'Wrote {len(paper_val_rows)} paper-val rows to {output_dir / "val.csv"}')
    print(f'Wrote {len(official_test_rows)} official-test rows to {output_dir / "test.csv"}')


if __name__ == '__main__':
    main()