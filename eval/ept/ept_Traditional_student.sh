#!/usr/bin/env bash
#SBATCH --job-name=ept_traditional_student
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=96:00:00
#SBATCH --output=eval/ept/benchmark/logs/%x_%j.out
#SBATCH --error=eval/ept/benchmark/logs/%x_%j.err

set -euo pipefail

# -----------------------------------------------------
# Manual Configuration (EDIT THESE DIRECTLY)
# -----------------------------------------------------

MODEL="traditional-model/checkpoints/epoch_0"   # Model to benchmark
NUM_PROMPTS=100                             # Number of Dolly prompts
BATCH_SIZE=4                                # Batch size for generation
GPU_INDEX=0                                 # GPU index to monitor

# Output location
OUTFILE="eval/ept/benchmark/results/ept_traditional_student_${SLURM_JOB_ID}.json"

# -----------------------------------------------------
# Initialization
# -----------------------------------------------------
mkdir -p eval/ept/benchmark logs results || true

echo "===================================================="
echo "           EPT-Bench: Energy-Per-Token"
echo "===================================================="
echo "[EPT] Model          : $MODEL"
echo "[EPT] Prompts        : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] GPU index      : $GPU_INDEX"
echo "[EPT] Output file    : $OUTFILE"
echo "[EPT] SLURM Job ID   : $SLURM_JOB_ID"
echo "----------------------------------------------------"
