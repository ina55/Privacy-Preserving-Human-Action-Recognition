#!/bin/bash
# Fine-tune VideoMAE V2 (vit_base_patch16_224) on UCF101 with the configuration reported in GPECOM 2026 / SYNASC 2026.

# Activate venv
source /path/to/repo/src/VideoMAEv2/.venv/bin/activate || echo "Warning: Could not activate .venv"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set paths
PROJ_ROOT="/path/to/repo/src/VideoMAEv2"
DATA_ROOT="/path/to/project/datasets/ucf101/videos"
OUTPUT_DIR="/path/to/project/work_dirs/ucf101_vit_b_finetune"
MODEL_PATH="${PROJ_ROOT}/data/VideoMAEv2_data/vit_b_k710_dl_from_giant.pth"

cd ${PROJ_ROOT}

mkdir -p ${OUTPUT_DIR}

# Determine number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found ${NUM_GPUS} GPUs"

if [ "$NUM_GPUS" -gt "1" ]; then
  DIST_CMD="python3 -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=23456"
else
    DIST_CMD="python3"
fi

nohup ${DIST_CMD} run_class_finetuning.py \
  --model vit_base_patch16_224 \
  --data_set UCF101 \
  --nb_classes 101 \
  --data_path ${DATA_ROOT} \
  --data_root ${DATA_ROOT} \
  --finetune ${MODEL_PATH} \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --with_checkpoint \
  --batch_size 1 \
  --num_sample 1 \
  --input_size 224 \
  --short_side_size 224 \
  --save_ckpt_freq 5 \
  --num_frames 16 \
  --sampling_rate 2 \
  --opt adamw \
  --lr 5e-4 \
  --layer_decay 0.90 \
  --num_workers 2 \
  --opt_betas 0.9 0.999 \
  --weight_decay 0.05 \
  --epochs 30 \
  --drop_path 0.35 \
  --head_drop_rate 0.5 \
  --mixup 0.0 \
  --cutmix 0.0 \
  --dist_eval \
  > ${OUTPUT_DIR}/training_ucf101.log 2>&1 &

PID=$!
echo $PID > ${OUTPUT_DIR}/train.pid
echo "Training started with PID: $PID"
echo "Logs: ${OUTPUT_DIR}/training_ucf101.log"
