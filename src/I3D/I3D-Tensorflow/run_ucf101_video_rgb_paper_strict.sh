#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DATASET_ROOT="${1:-/path/to/project/datasets/ucf101/ucf101_1}"
OUTPUT_DIR="${2:-/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict}"
GPU_ID="${3:-0}"
VAL_RATIO="${4:-0.1}"
SEED="${5:-42}"

ANNOTATION_DIR="${DATASET_ROOT}/paper_split_seed${SEED}_valratio${VAL_RATIO}"
LIST_DIR="$OUTPUT_DIR/lists"

mkdir -p "$OUTPUT_DIR"

python3 "$PROJECT_ROOT/processing/generate_ucf101_paper_splits.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$ANNOTATION_DIR" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED"

python -u "$SCRIPT_DIR/tools/generate_video_lists.py" \
  --dataset-root "$DATASET_ROOT" \
  --train-csv "$ANNOTATION_DIR/train.csv" \
  --test-csv "$ANNOTATION_DIR/val.csv" \
  --output-dir "$LIST_DIR"

python -u "$SCRIPT_DIR/experiments/train_video_rgb.py" \
  --train-list "$LIST_DIR/train_rgb.list" \
  --test-list "$LIST_DIR/test_rgb.list" \
  --num-classes 101 \
  --output-dir "$OUTPUT_DIR" \
  --pretrained-dir "$SCRIPT_DIR/checkpoints/rgb_imagenet" \
  --gpu "$GPU_ID"