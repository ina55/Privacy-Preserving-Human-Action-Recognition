#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DATASET_ROOT="${1:-/path/to/project/datasets/hmdb51/hmdb51_1}"
OUTPUT_DIR="${2:-/path/to/project/work_dirs/i3d_hmdb51_rgb_video_paper}"
GPU_ID="${3:-0}"
VAL_PER_CLASS="${4:-10}"
SEED="${5:-42}"

ANNOTATION_DIR="${DATASET_ROOT}/paper_split_seed${SEED}_val${VAL_PER_CLASS}"
LIST_DIR="$OUTPUT_DIR/lists"

mkdir -p "$OUTPUT_DIR"

python3 "$PROJECT_ROOT/processing/generate_hmdb51_paper_splits.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$ANNOTATION_DIR" \
  --val-per-class "$VAL_PER_CLASS" \
  --seed "$SEED"

python -u "$SCRIPT_DIR/tools/generate_video_lists.py" \
  --dataset-root "$DATASET_ROOT" \
  --train-csv "$ANNOTATION_DIR/train.csv" \
  --test-csv "$ANNOTATION_DIR/val.csv" \
  --output-dir "$LIST_DIR"

python -u "$SCRIPT_DIR/experiments/train_video_rgb.py" \
  --train-list "$LIST_DIR/train_rgb.list" \
  --test-list "$LIST_DIR/test_rgb.list" \
  --num-classes 51 \
  --output-dir "$OUTPUT_DIR" \
  --pretrained-dir "$SCRIPT_DIR/checkpoints/rgb_imagenet" \
  --gpu "$GPU_ID"