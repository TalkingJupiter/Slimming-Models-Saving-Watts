#!/usr/bin/env bash
#SBATCH --job-name=base_sft_model
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --array=0-4
#SBATCH --output=Base/logs/base/%x_%j.out
#SBATCH --error=Base/logs/base/%x_%j.err

# ---------------------------------------------
# 0. Environment / NCCL / Conda
# ---------------------------------------------
set -euo pipefail

source scripts/_env_single_node.sh

module load cuda
export CUDA_HOME=$CUDA_HOME 
echo "CUDA_HOME = $CUDA_HOME"


# ---------------------------------------------
# 1. Training config
# ---------------------------------------------
MODEL=${MODEL:-}
DATA_PATH="data/shards.jsonl"
MODEL_NAME=${MODEL_NAME:-}
OUT_DIR=${OUT_DIR:-}
TELEM_DIR="$MODEL_NAME/$SLURM_ARRAY_TASK_ID/telemetry"

PER_DEVICE_BS=2
GRAD_ACCUM=16
EPOCHS=1
MAX_LEN=2048
LR=1e-4

# Telemetry sampling interval (sec)
TELEM_INTERVAL=1.0

mkdir -p logs/base
mkdir -p "$TELEM_DIR"

# ---------------------------------------------
# 2. Launch training with Accelerate
# ---------------------------------------------
accelerate launch \
  --num_processes=1 \
  Base/train_base_from_shards.py \
  --model $MODEL \
  --data_path $DATA_PATH \
  --output_dir $OUT_DIR \
  --batch_size $PER_DEVICE_BS \
  --num_train_epochs $EPOCHS \
  --max_length $MAX_LEN \
  --learning_rate $LR \
  --telemetry \
  --telemetry_output $TELEM_DIR/train.jsonl \
  --telemetry_interval $TELEM_INTERVAL