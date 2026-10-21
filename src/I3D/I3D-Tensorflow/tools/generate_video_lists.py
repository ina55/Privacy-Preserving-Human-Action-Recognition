import argparse
import json
import os
import random


VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mov', '.mkv', '.webm')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate I3D video lists from dataset CSV splits.')
    parser.add_argument('--dataset-root', required=True, help='Dataset variant root, e.g. /path/to/project/datasets/hmdb51/hmdb51_1')
    parser.add_argument('--train-csv', default='train.csv', help='Training split CSV path relative to dataset root or absolute path')
    parser.add_argument('--test-csv', default=None, help='Test split CSV path relative to dataset root or absolute path')
    parser.add_argument('--output-dir', required=True, help='Directory where train_rgb.list/test_rgb.list will be written')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio used only when no test CSV is available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for fallback train/val split')
    parser.add_argument('--skip-missing', action='store_true', help='Skip CSV entries whose video files cannot be resolved')
    return parser.parse_args()


def resolve_csv_path(dataset_root, csv_path):
    if csv_path is None:
        return None
    if os.path.isabs(csv_path):
        return csv_path
    return os.path.join(dataset_root, csv_path)


def resolve_video_path(dataset_root, relative_path):
    candidate = os.path.join(dataset_root, relative_path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    class_dir = os.path.dirname(candidate)
    basename = os.path.basename(relative_path)
    stem, _ = os.path.splitext(basename)
    candidates = [stem]
    if stem.startswith('v_'):
        candidates.append('anonymized_' + stem)

    if os.path.isdir(class_dir):
        for entry in sorted(os.listdir(class_dir)):
            entry_stem, entry_ext = os.path.splitext(entry)
            if entry_ext.lower() not in VIDEO_EXTENSIONS:
                continue
            if entry_stem in candidates:
                return os.path.abspath(os.path.join(class_dir, entry))

    for ext in VIDEO_EXTENSIONS:
        swapped = os.path.join(class_dir, stem + ext)
        if os.path.exists(swapped):
            return os.path.abspath(swapped)

    raise FileNotFoundError('Could not resolve video path for {}'.format(relative_path))


def load_entries(csv_path, dataset_root, skip_missing=False):
    entries = []
    missing_entries = []
    with open(csv_path, 'r') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            relative_path, label = line.rsplit(' ', 1)
            try:
                entries.append((resolve_video_path(dataset_root, relative_path), int(label)))
            except FileNotFoundError:
                if not skip_missing:
                    raise
                missing_entries.append(relative_path)
    return entries, missing_entries


def stratified_split(entries, val_ratio, seed):
    grouped = {}
    for path, label in entries:
        grouped.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    train_entries = []
    val_entries = []
    for label, label_entries in grouped.items():
        rng.shuffle(label_entries)
        if len(label_entries) <= 1:
            train_entries.extend(label_entries)
            continue
        val_count = max(1, int(round(len(label_entries) * val_ratio)))
        val_count = min(val_count, len(label_entries) - 1)
        val_entries.extend(label_entries[:val_count])
        train_entries.extend(label_entries[val_count:])
    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries


def write_list(entries, output_path):
    with open(output_path, 'w') as handle:
        for path, label in entries:
            handle.write('{} {}\n'.format(path, label))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_csv_path = resolve_csv_path(args.dataset_root, args.train_csv)
    test_csv_path = resolve_csv_path(args.dataset_root, args.test_csv)

    train_entries, missing_train_entries = load_entries(train_csv_path, args.dataset_root, skip_missing=args.skip_missing)
    if test_csv_path and os.path.exists(test_csv_path):
        test_entries, missing_test_entries = load_entries(test_csv_path, args.dataset_root, skip_missing=args.skip_missing)
    else:
        train_entries, test_entries = stratified_split(train_entries, args.val_ratio, args.seed)
        missing_test_entries = []
    if not (test_csv_path and os.path.exists(test_csv_path)):
        missing_train_entries = []

    train_output = os.path.join(args.output_dir, 'train_rgb.list')
    test_output = os.path.join(args.output_dir, 'test_rgb.list')
    write_list(train_entries, train_output)
    write_list(test_entries, test_output)

    metadata = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'train_csv': os.path.abspath(train_csv_path),
        'test_csv': os.path.abspath(test_csv_path) if test_csv_path and os.path.exists(test_csv_path) else None,
        'train_samples': len(train_entries),
        'test_samples': len(test_entries),
        'num_classes': len({label for _, label in train_entries + test_entries}),
        'used_generated_validation_split': not (test_csv_path and os.path.exists(test_csv_path)),
        'seed': args.seed,
        'val_ratio': args.val_ratio,
        'skipped_missing_train_entries': missing_train_entries,
        'skipped_missing_test_entries': missing_test_entries,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2)

    print('Wrote {} training entries to {}'.format(len(train_entries), train_output))
    print('Wrote {} validation entries to {}'.format(len(test_entries), test_output))
    if missing_train_entries:
        print('Skipped {} missing training entries'.format(len(missing_train_entries)))
    if missing_test_entries:
        print('Skipped {} missing test entries'.format(len(missing_test_entries)))


if __name__ == '__main__':
    main()