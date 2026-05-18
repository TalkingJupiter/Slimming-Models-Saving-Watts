#!/usr/bin/env bash


set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

source ~/.bashrc

export CUDA_HOME=/opt/apps/nfs/spack-1.0.1/opt/spack/linux-sapphirerapids/cuda-12.9.0-u2ppthmoi4r6ddyxdqesrj5oqja3byph
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "CUDA_HOME in job: $CUDA_HOME"
which nvcc || echo "nvcc not found in PATH"

conda activate kd

MODEL_NAME="${MODEL_NAME:-${SAFE_STUDENT_NAME:-traditional_student}}"
MODEL_ID="${MODEL:-${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}}"
SHARDS_FILE="${SHARDS_FILE:-data/shards.jsonl}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STUDENT_RUN_NAME="${SAFE_STUDENT_NAME:-$MODEL_NAME}"
OUTPUT_DIR="${OUT_DIR:-traditional_student/$STUDENT_RUN_NAME/$ARRAY_TASK_ID}"
TELEMETRY_OUTPUT="${TELEMETRY_OUTPUT:-results/traditional_student/$STUDENT_RUN_NAME/$ARRAY_TASK_ID/telemetry.json}"

accelerate launch traditional-model/train_sft.py \
  --model_name "$MODEL_ID" \
  --shards_file "$SHARDS_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-5 \
  --num_epochs 1 \
  --max_length 2048 \
  --save_every 1000 \
  --telemetry \
  --telemetry_output "$TELEMETRY_OUTPUT" \
  --telemetry_interval 1
