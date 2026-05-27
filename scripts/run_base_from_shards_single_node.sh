#!/usr/bin/env bash
# ---------------------------------------------
# 0. Environment / NCCL / Conda
# ---------------------------------------------
set -euo pipefail

source scripts/_env_single_node.sh

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

module load cuda
export CUDA_HOME=$CUDA_HOME 
echo "CUDA_HOME = $CUDA_HOME"
# ---------------------------------------------
# 1. Training config
# ---------------------------------------------
MODEL=${MODEL:-${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}}
MODEL_SOURCE=$(resolve_hf_model "$MODEL")
echo "[INFO] Student model: $MODEL"
echo "[INFO] Student model source: $MODEL_SOURCE"
DATA_PATH="${DATA_PATH:-data/shards.jsonl}"
MODEL_NAME=${MODEL_NAME:-${SAFE_STUDENT_NAME:-base_compat}}
OUT_DIR=${OUT_DIR:-traditional-model/checkpoints/$MODEL_NAME/$SLURM_ARRAY_TASK_ID}
TELEM_DIR="${TELEM_DIR:-traditional-model/telemetry/${MODEL_NAME:-base_compat}/$SLURM_ARRAY_TASK_ID}"

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
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  traditional-model/train_sft.py \
  --model "$MODEL_SOURCE" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT_DIR" \
  --batch_size $PER_DEVICE_BS \
  --grad_accum $GRAD_ACCUM \
  --num_train_epochs $EPOCHS \
  --max_length $MAX_LEN \
  --learning_rate $LR \
  --optimizer adamw_8bit \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --telemetry \
  --telemetry_output $TELEM_DIR/train.jsonl \
  --telemetry_interval $TELEM_INTERVAL
