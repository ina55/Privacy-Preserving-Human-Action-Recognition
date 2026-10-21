#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python"
RESULTS_DIR="/path/to/project/datasets/ucf101/ucf101_results"
VARIANT_LIST_ROOT="$RESULTS_DIR/i3d_variant_lists_strict"
CHECKPOINT_DIR="/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict/checkpoints"
CHECKPOINT_PATH="/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper_strict/checkpoints/i3d_rgb-9999"
MASTER_LOG="$RESULTS_DIR/final_strict_i3d_eval_runner.log"
GPU_ID="${1:-2}"
shift || true
VARIANTS=("$@")

if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(raw faceblurred low56 low28)
fi

mkdir -p "$RESULTS_DIR" "$VARIANT_LIST_ROOT"
touch "$MASTER_LOG"
echo "[$(date '+%F %T')] RUN variants=${VARIANTS[*]} gpu=$GPU_ID" >> "$MASTER_LOG"

generate_lists() {
  local dataset_root="$1"
  local output_dir="$2"

  mkdir -p "$output_dir"
  "$PYTHON_BIN" -u "$SCRIPT_DIR/tools/generate_video_lists.py" \
    --dataset-root "$dataset_root" \
    --train-csv train.csv \
    --test-csv val.csv \
    --output-dir "$output_dir" \
    > "$output_dir/generate_lists.log" 2>&1
}

run_eval() {
  local name="$1"
  local dataset_root="$2"
  local list_dir="$VARIANT_LIST_ROOT/$name"
  local report_path="$RESULTS_DIR/final_strict_i3d_eval_report_${name}.txt"
  local raw_json_path="$RESULTS_DIR/final_strict_i3d_eval_results_${name}.json"
  local log_path="$RESULTS_DIR/final_strict_i3d_eval_${name}.log"

  echo "[$(date '+%F %T')] START $name dataset=$dataset_root checkpoint=$CHECKPOINT_PATH" | tee -a "$MASTER_LOG"
  generate_lists "$dataset_root" "$list_dir"

  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$SCRIPT_DIR/experiments/evaluate_video_rgb.py" \
    --test-list "$list_dir/test_rgb.list" \
    --num-classes 101 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --gpu "$GPU_ID" \
    --report-path "$report_path" \
    --raw-json-path "$raw_json_path" \
    > "$log_path" 2>&1

  "$PYTHON_BIN" -c "import json,sys; data=json.load(open(sys.argv[1])); s=data['summary']; print(f'[{sys.argv[2]}] top1={s[\"top1_acc\"]*100:.2f}% ({s[\"top1_correct\"]}/{s[\"samples_evaluated\"]}), loss={s[\"avg_loss\"]:.5f}')" \
    "$raw_json_path" "$name" | tee -a "$MASTER_LOG"
  echo "[$(date '+%F %T')] DONE $name" | tee -a "$MASTER_LOG"
}

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    raw)
      run_eval raw /path/to/project/datasets/ucf101/ucf101_1
      ;;
    faceblurred)
      run_eval faceblurred /path/to/project/datasets/ucf101/ucf101_faceblurred
      ;;
    low56)
      run_eval low56 /path/to/project/datasets/ucf101/ucf101_low56
      ;;
    low28)
      run_eval low28 /path/to/project/datasets/ucf101/ucf101_low28
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac
done

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$MASTER_LOG"