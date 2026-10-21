#!/usr/bin/env python3

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build paper-grade HMDB51 split files from the recovered split-1 CSVs.'
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('/path/to/project/datasets/hmdb51/hmdb51_1'),
        help='HMDB51 dataset directory containing the existing train.csv and test.csv files.',
    )
    parser.add_argument(
        '--combined-train-csv',
        default='train.csv',
        help='CSV that currently contains the union of official train and test samples.',
    )
    parser.add_argument(
        '--official-test-csv',
        default='test.csv',
        help='CSV that contains the official HMDB51 split-1 test samples.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory where train.csv, val.csv, test.csv, and metadata.json will be written.',
    )
    parser.add_argument(
        '--val-per-class',
        type=int,
        default=10,
        help='How many samples per class to reserve from the official training pool for validation.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed used for deterministic per-class validation sampling.',
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
    combined_train_csv = (dataset_root / args.combined_train_csv).resolve()
    official_test_csv = (dataset_root / args.official_test_csv).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = load_rows(combined_train_csv)
    official_test_rows = load_rows(official_test_csv)

    official_test_paths = {rel_path for rel_path, _ in official_test_rows}

    official_train_rows = [
        (rel_path, label)
        for rel_path, label in combined_rows
        if rel_path not in official_test_paths
    ]

    missing_official_test_in_combined = sorted(
        rel_path for rel_path, _ in official_test_rows if rel_path not in {path for path, _ in combined_rows}
    )

    if len(official_train_rows) != len({rel_path for rel_path, _ in official_train_rows}):
        raise ValueError('Duplicate entries detected in derived official training rows.')

    grouped_train_rows = defaultdict(list)
    for rel_path, label in official_train_rows:
        grouped_train_rows[label].append((rel_path, label))

    rng = random.Random(args.seed)
    paper_train_rows = []
    paper_val_rows = []

    for label in sorted(grouped_train_rows):
        label_rows = sorted(grouped_train_rows[label])
        if len(label_rows) <= args.val_per_class:
            raise ValueError(
                f'Class {label} has only {len(label_rows)} samples; cannot reserve '
                f'{args.val_per_class} validation samples.'
            )
        rng.shuffle(label_rows)
        val_rows = sorted(label_rows[:args.val_per_class])
        train_rows = sorted(label_rows[args.val_per_class:])
        paper_val_rows.extend(val_rows)
        paper_train_rows.extend(train_rows)

    paper_train_rows.sort()
    paper_val_rows.sort()
    official_test_rows = sorted(official_test_rows)

    train_paths = {rel_path for rel_path, _ in paper_train_rows}
    val_paths = {rel_path for rel_path, _ in paper_val_rows}
    test_paths = {rel_path for rel_path, _ in official_test_rows}

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
        'combined_train_csv': str(combined_train_csv),
        'official_test_csv': str(official_test_csv),
        'output_dir': str(output_dir),
        'seed': args.seed,
        'val_per_class': args.val_per_class,
        'counts': {
            'combined_train': len(combined_rows),
            'paper_train': len(paper_train_rows),
            'paper_val': len(paper_val_rows),
            'official_test': len(official_test_rows),
            'missing_official_test_in_combined': len(missing_official_test_in_combined),
        },
        'per_class_counts': {
            'paper_train': {str(label): train_counts[label] for label in sorted(train_counts)},
            'paper_val': {str(label): val_counts[label] for label in sorted(val_counts)},
            'official_test': {str(label): test_counts[label] for label in sorted(test_counts)},
        },
        'overlaps': overlaps,
        'notes': [
            'Recovered official HMDB51 split-1 training set as combined train.csv minus official test.csv.',
            'Validation split is sampled only from the recovered training pool.',
        ],
    }
    if missing_official_test_in_combined:
        metadata['missing_official_test_in_combined'] = missing_official_test_in_combined
        metadata['notes'].append(
            'Some official test entries were not present in combined train.csv and were left only in the held-out test split.'
        )
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f'Wrote {len(paper_train_rows)} paper-train rows to {output_dir / "train.csv"}')
    print(f'Wrote {len(paper_val_rows)} paper-val rows to {output_dir / "val.csv"}')
    print(f'Wrote {len(official_test_rows)} official-test rows to {output_dir / "test.csv"}')


if __name__ == '__main__':
    main()