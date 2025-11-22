#!/usr/bin/env bash
#SBATCH --job-name=base_sft_8B
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
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
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH="data/shards.jsonl"
OUT_DIR="Base/llama3.1-8B-sft_from_shards"
TELEM_DIR="Base/telemetry/base_8B_sft"

PER_DEVICE_BS=1
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
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --data_path data/shards.jsonl \
  --output_dir Base/base_ckpt \
  --batch_size 1 \
  --num_train_epochs 1 \
  --max_length 2048 \
  --learning_rate 1e-4 \
  --telemetry \
  --telemetry_output telemetry/base_8B_sft/train.jsonl \
  --telemetry_interval 5