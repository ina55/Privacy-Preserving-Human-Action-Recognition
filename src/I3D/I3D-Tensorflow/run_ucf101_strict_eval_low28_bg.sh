#!/usr/bin/env bash

set -euo pipefail

RESULTS="/path/to/project/datasets/ucf101/ucf101_results"
MASTER_LOG="$RESULTS/final_strict_i3d_eval_runner.log"
LOG_PATH="$RESULTS/final_strict_i3d_eval_low28.log"
REPORT_PATH="$RESULTS/final_strict_i3d_eval_report_low28.txt"
RAW_JSON_PATH="$RESULTS/final_strict_i3d_eval_results_low28.json"
PYTHON_BIN="python"
EVAL_SCRIPT="/path/to/repo/src/I3D/I3D-Tensorflow/experiments/evaluate_video_rgb.py"
TEST_LIST="/path/to/project/datasets/ucf101/ucf101_results/i3d_variant_lists/low28/test_rgb.list"
CHECKPOINT_DIR="/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict/checkpoints"
CHECKPOINT_PATH="/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict/checkpoints/i3d_rgb-9999"
GPU_ID="${1:-2}"

mkdir -p "$RESULTS"
touch "$MASTER_LOG"

echo "[$(date '+%F %T')] START low28 dataset=/path/to/project/datasets/ucf101/ucf101_low28 checkpoint=$CHECKPOINT_PATH" | tee -a "$MASTER_LOG"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$EVAL_SCRIPT" \
  --test-list "$TEST_LIST" \
  --num-classes 101 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --gpu "$GPU_ID" \
  --report-path "$REPORT_PATH" \
  --raw-json-path "$RAW_JSON_PATH" \
  > "$LOG_PATH" 2>&1

"$PYTHON_BIN" -c "import json; data=json.load(open('$RAW_JSON_PATH')); s=data['summary']; print(f'[low28] top1={s[\"top1_acc\"]*100:.2f}% ({s[\"top1_correct\"]}/{s[\"samples_evaluated\"]}), loss={s[\"avg_loss\"]:.5f}')" | tee -a "$MASTER_LOG"

echo "[$(date '+%F %T')] DONE low28" | tee -a "$MASTER_LOG"