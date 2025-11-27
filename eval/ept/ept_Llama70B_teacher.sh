#!/usr/bin/env bash
#SBATCH --job-name=ept_70B_teacher
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

MODEL="meta-llama/Llama-3.1-70B-Instruct"   # Model to benchmark
NUM_PROMPTS=100                             # Number of Dolly prompts
BATCH_SIZE=2                                # Batch size for generation
GPU_INDEX=0                                 # GPU index to monitor

# Output location
OUTFILE="eval/ept/benchmark/results/ept_teacher_${SLURM_JOB_ID}.json"

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

# -----------------------------------------------------
# Activate Environment
# -----------------------------------------------------
source ~/.bashrc || true
conda activate kd || true

# -----------------------------------------------------
# Run Benchmark
# -----------------------------------------------------
python eval/ept/benchmark/run_ept_benchmark.py \
  --model "$MODEL" \
  --use-dolly \
  --num-prompts "$NUM_PROMPTS" \
  --batch-size "$BATCH_SIZE" \
  --gpu-indices "$GPU_INDEX" \
  --out "$OUTFILE"

# -----------------------------------------------------
# Completion Message
# -----------------------------------------------------
echo "[EPT] Benchmark completed successfully."
echo "[EPT] Results saved to: $OUTFILE"
echo "===================================================="
