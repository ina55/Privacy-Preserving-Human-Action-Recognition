#!/usr/bin/env bash

set -euo pipefail

RESULTS="/path/to/project/datasets/ucf101/ucf101_results"
MASTER_LOG="$RESULTS/final_strict_i3d_eval_runner.log"
LOG_PATH="$RESULTS/final_strict_i3d_eval_faceblurred.log"
REPORT_PATH="$RESULTS/final_strict_i3d_eval_report_faceblurred.txt"
RAW_JSON_PATH="$RESULTS/final_strict_i3d_eval_results_faceblurred.json"
PYTHON_BIN="python"
EVAL_SCRIPT="/path/to/repo/src/I3D/I3D-Tensorflow/experiments/evaluate_video_rgb.py"
TEST_LIST="/path/to/project/datasets/ucf101/ucf101_results/i3d_variant_lists/faceblurred/test_rgb.list"
CHECKPOINT_DIR="/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict/checkpoints"
CHECKPOINT_PATH="/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict/checkpoints/i3d_rgb-9999"
GPU_ID="${1:-0}"
START_POS="${2:-0}"
SUFFIX=""

if [ "$START_POS" != "0" ]; then
  SUFFIX="_resume_from_${START_POS}"
fi

LOG_PATH="$RESULTS/final_strict_i3d_eval_faceblurred${SUFFIX}.log"
REPORT_PATH="$RESULTS/final_strict_i3d_eval_report_faceblurred${SUFFIX}.txt"
RAW_JSON_PATH="$RESULTS/final_strict_i3d_eval_results_faceblurred${SUFFIX}.json"
PARTIAL_JSON_PATH="$RESULTS/final_strict_i3d_eval_results_faceblurred${SUFFIX}.partial.json"

mkdir -p "$RESULTS"
touch "$MASTER_LOG"

echo "[$(date '+%F %T')] START faceblurred dataset=/path/to/project/datasets/ucf101/ucf101_faceblurred checkpoint=$CHECKPOINT_PATH start_pos=$START_POS" | tee -a "$MASTER_LOG"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$EVAL_SCRIPT" \
  --test-list "$TEST_LIST" \
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

"$PYTHON_BIN" -c "import json; data=json.load(open('$RAW_JSON_PATH')); s=data['summary']; print(f'[faceblurred] top1={s[\"top1_acc\"]*100:.2f}% ({s[\"top1_correct\"]}/{s[\"samples_evaluated\"]}), loss={s[\"avg_loss\"]:.5f}')" | tee -a "$MASTER_LOG"

echo "[$(date '+%F %T')] DONE faceblurred" | tee -a "$MASTER_LOG"