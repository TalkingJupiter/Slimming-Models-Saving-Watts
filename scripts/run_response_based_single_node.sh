#!/usr/bin/env bash


set -euo pipefail
source scripts/_env_single_node.sh

STUDENT_MODEL=${STUDENT_MODEL:-}
TEACHER_DATA=${TEACHER_DATA:-}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-}

echo "[INFO] $STUDENT_MODEL Response-Based KD | node=1 | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES"

RUN_DIR="serialization_dir/$SAFE_STUDENT_NAME/feature/$SLURM_ARRAY_TASK_ID"
TELEMETRY_OUT="logs/telemetry/$SAFE_STUDENT_NAME/response/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/telemetry.jsonl"
mkdir -p "$RUN_DIR"

accelerate launch \
  --num_machines 1 \
  --num_processes ${NUM_PROCESSES} \
  --deepspeed_config_file configs/ds_zero3.json \
  --module kd.train \
    --kd.mode rb \
    --student $STUDENT_MODEL \
    --data "data/$TEACHER_DATA/topk_k16/*.parquet" \
    --rb.topk 16 \
    --rb.temperature 2.0 \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr 1e-4 \
    --bash_size 2 \
    --save-dir "$RUN_DIR" \
    --save_every 200 \
    --max_steps 5000 \
    --telemetry \
    --telemetry_output "$TELEMETRY_OUT" \
    --telemetry_interval 1.0 \
    --resume auto 
    

echo "[INFO] $STUDENT_MODEL RB KD complete"
