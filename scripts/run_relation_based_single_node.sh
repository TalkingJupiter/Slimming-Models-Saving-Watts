#!/usr/bin/env bash
#SBATCH --job-name=kd_relation_based_single_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/2nd/relation/%x_%j.out
#SBATCH --error=logs/2nd/relation/%x_%j.err
#SBATCH --nodelist=rpg-93-5
#SBATCH --array=0-4

set -euo pipefail
source scripts/_env_single_node.sh

STUDENT_MODEL=${STUDENT_MODEL:-}
TEACHER_DATA=${TEACHER_DATA:-}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-}

echo "[INFO] $STUDENT_MODEL Relation-Based KD | node=1 | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES"

RUN_DIR="serialization_dir/$SAFE_STUDENT_NAME/relation/$SLURM_ARRAY_TASK_ID"
TELEMETRY_OUT="logs/telemetry/$SAFE_STUDENT_NAME/relation/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/telemetry.jsonl"
mkdir -p "$RUN_DIR"

accelerate launch \
  --num_machines 1 \
  --num_processes ${NUM_PROCESSES} \
  --deepspeed_config_file configs/ds_zero3.json \
  --module kd.train \
    --kd.mode relb \
    --student $STUDENT_MODEL \
    --data "data/$TEACHER_DATA/relb_embeds/*.parquet" \
    --relb.lambda_dist 1.0 \
    --relb.lambda_angle 0.5 \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr 1e-4 \
    --bash_size 4 \
    --save-dir "$RUN_DIR" \
    --save_every 200 \
    --max_steps 5000 \
    --telemetry \
    --telemetry_output "$TELEMETRY_OUT" \
    --telemetry_interval 1.0 \
    --resume auto 

echo "[INFO] $STUDENT_MODEL RelB KD complete"
