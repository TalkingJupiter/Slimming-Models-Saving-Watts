#!/usr/bin/env bash
#SBATCH --job-name=kd_feature_based_single_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/feature/%x_%j.out
#SBATCH --error=logs/feature/%x_%j.err
#SBATCH --array=0-4

set -euo pipefail
source scripts/_env_single_node.sh

STUDENT_MODEL=${STUDENT_MODEL:-}
TEACHER_DATA=${TEACHER_DATA:-}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-}

echo "[INFO] $STUDENT_MODEL Feature-Based KD | node=1 | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES"

mkdir -p logs/telemetry/"$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID"
python monitor.py --output logs/telemetry/$SAFE_STUDENT_NAME/feature/"$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID"/telemetry.jsonl --interval 1.0 &
MON_PID=$!

RUN_DIR="serialization_dir/$SAFE_STUDENT_NAME/feature/$SLURM_ARRAY_TASK_ID"
mkdir -p "$RUN_DIR"


accelerate launch \
  --num_machines 1 \
  --num_processes ${NUM_PROCESSES} \
  --deepspeed_config_file configs/ds_zero3.json \
  --module kd.train \
    --kd.mode fb \
    --student $STUDENT_MODEL \
    --data "data/$TEACHER_DATA/*.parquet" \
    --fb.teacher_layer 22 \
    --fb.student_layer 12 \
    --fb.token_subset_ratio 0.25 \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr 1e-4 \
    --bash_size 2 \
    --save-dir "$RUN_DIR" \
    --save_every 200 \
    --max_steps 2000 \
    --resume auto 

kill $MON_PID || true
echo "[INFO] $STUDENT_MODEL FB KD complete"

    # --max_steps 2000 \