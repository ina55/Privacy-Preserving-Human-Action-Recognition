#!/usr/bin/env bash

set -euo pipefail

RESULTS="/path/to/project/datasets/ucf101/ucf101_results"
LIST_DIR="$RESULTS/i3d_variant_lists/low56"
MASTER_LOG="$RESULTS/final_strict_i3d_eval_runner.log"
LOG_PATH="$RESULTS/final_strict_i3d_eval_low56.log"
REPORT_PATH="$RESULTS/final_strict_i3d_eval_report_low56.txt"
RAW_JSON_PATH="$RESULTS/final_strict_i3d_eval_results_low56.json"
PYTHON_BIN="python"
LIST_SCRIPT="/path/to/repo/src/I3D/I3D-Tensorflow/tools/generate_video_lists.py"
EVAL_SCRIPT="/path/to/repo/src/I3D/I3D-Tensorflow/experiments/evaluate_video_rgb.py"
DATASET_ROOT="/path/to/project/datasets/ucf101/ucf101_low56"
CHECKPOINT_DIR="/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict/checkpoints"
CHECKPOINT_PATH="/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict/checkpoints/i3d_rgb-9999"
GPU_ID="${1:-2}"
START_POS="${2:-0}"
SUFFIX=""

if [ "$START_POS" != "0" ]; then
  SUFFIX="_resume_from_${START_POS}"
fi

LOG_PATH="$RESULTS/final_strict_i3d_eval_low56${SUFFIX}.log"
REPORT_PATH="$RESULTS/final_strict_i3d_eval_report_low56${SUFFIX}.txt"
RAW_JSON_PATH="$RESULTS/final_strict_i3d_eval_results_low56${SUFFIX}.json"
PARTIAL_JSON_PATH="$RESULTS/final_strict_i3d_eval_results_low56${SUFFIX}.partial.json"

mkdir -p "$RESULTS" "$LIST_DIR"
touch "$MASTER_LOG"

if [ ! -s "$LIST_DIR/test_rgb.list" ]; then
  "$PYTHON_BIN" -u "$LIST_SCRIPT" \
    --dataset-root "$DATASET_ROOT" \
    --train-csv val.csv \
    --test-csv val.csv \
    --output-dir "$LIST_DIR" \
    > "$LIST_DIR/generate_lists.log" 2>&1
fi

echo "[$(date '+%F %T')] START low56 dataset=$DATASET_ROOT checkpoint=$CHECKPOINT_PATH start_pos=$START_POS" | tee -a "$MASTER_LOG"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$EVAL_SCRIPT" \
  --test-list "$LIST_DIR/test_rgb.list" \
  --num-classes 101 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --gpu "$GPU_ID" \
  --start-pos "$START_POS" \
  --report-path "$REPORT_PATH" \
  --raw-json-path "$RAW_JSON_PATH" \
  --partial-json-path "$PARTIAL_JSON_PATH" \
  --save-partial-every 25 \
  > "$LOG_PATH" 2>&1

"$PYTHON_BIN" -c "import json; data=json.load(open('$RAW_JSON_PATH')); s=data['summary']; print(f'[low56] top1={s[\"top1_acc\"]*100:.2f}% ({s[\"top1_correct\"]}/{s[\"samples_evaluated\"]}), loss={s[\"avg_loss\"]:.5f}')" | tee -a "$MASTER_LOG"

echo "[$(date '+%F %T')] DONE low56" | tee -a "$MASTER_LOG"