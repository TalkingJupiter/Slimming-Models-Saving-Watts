#!/usr/bin/env bash
set -euo pipefail

trap 'rc=$?; echo "[ERROR] traditional SFT failed with exit code $rc at ${BASH_SOURCE[0]}:${LINENO}" >&2' ERR

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source ~/.bashrc

export CUDA_HOME=/opt/apps/nfs/spack-1.0.1/opt/spack/linux-sapphirerapids/cuda-12.9.0-u2ppthmoi4r6ddyxdqesrj5oqja3byph
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "CUDA_HOME in job: $CUDA_HOME"
which nvcc || echo "nvcc not found in PATH"

conda activate kd
source scripts/_env_single_node.sh

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_NAME="${MODEL_NAME:-${SAFE_STUDENT_NAME:-traditional_student}}"
MODEL_ID="${MODEL:-${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}}"
MODEL_SOURCE=$(resolve_hf_model "$MODEL_ID")
SHARDS_FILE="${SHARDS_FILE:-data/shards.jsonl}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STUDENT_RUN_NAME="${SAFE_STUDENT_NAME:-$MODEL_NAME}"
OUTPUT_DIR="${OUT_DIR:-traditional_student/$STUDENT_RUN_NAME/$ARRAY_TASK_ID}"
# TELEMETRY_OUTPUT="${TELEMETRY_OUTPUT:-results/traditional_student/$STUDENT_RUN_NAME/$ARRAY_TASK_ID/telemetry.json}"
TELEMETRY_OUTPUT="${TELEMETRY_OUTPUT:-results/$STUDENT_RUN_NAME/traditional/$ARRAY_TASK_ID/telemetry.json}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-50}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

echo "[INFO] Student model: $MODEL_ID"
echo "[INFO] Student model source: $MODEL_SOURCE"
echo "[INFO] Shards file: $SHARDS_FILE"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] Telemetry output: $TELEMETRY_OUTPUT"
echo "[INFO] num_epochs=$NUM_EPOCHS save_every=$SAVE_EVERY log_every=$LOG_EVERY"

if [[ ! -s "$SHARDS_FILE" ]]; then
  echo "[ERROR] Shards file is missing or empty: $SHARDS_FILE" >&2
  exit 1
fi

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  traditional-model/train_sft.py \
  --model_name "$MODEL_SOURCE" \
  --shards_file "$SHARDS_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-5 \
  --optimizer adamw_8bit \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --num_epochs "$NUM_EPOCHS" \
  --max_length 2048 \
  --log_every "$LOG_EVERY" \
  --save_every "$SAVE_EVERY" \
  --telemetry \
  --telemetry_output "$TELEMETRY_OUTPUT" \
  --telemetry_interval 1
