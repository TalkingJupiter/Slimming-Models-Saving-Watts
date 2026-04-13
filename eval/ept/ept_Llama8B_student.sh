#!/usr/bin/env bash
#SBATCH --job-name=ept_8B_base_student
#SBATCH --partition=toreador
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=eval/ept/benchmark/logs/%x_%j.out
#SBATCH --error=eval/ept/benchmark/logs/%x_%j.err

set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_PROMPTS=100
BATCH_SIZE=2
GPU_INDICES="0"

OUTFILE="eval/ept/benchmark/results/ept_base_student_${SLURM_JOB_ID}.json"

mkdir -p eval/ept/benchmark/logs eval/ept/benchmark/results

echo "===================================================="
echo "           EPT-Bench: Energy-Per-Token"
echo "===================================================="
echo "[EPT] Model          : $MODEL"
echo "[EPT] Prompts        : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] GPU indices    : $GPU_INDICES"
echo "[EPT] Output file    : $OUTFILE"
echo "[EPT] SLURM Job ID   : $SLURM_JOB_ID"
echo "----------------------------------------------------"

source ~/.bashrc || true
conda activate kd || true

python eval/ept/benchmark/run_ept_benchmark.py \
  --model "$MODEL" \
  --use-dolly \
  --num-prompts "$NUM_PROMPTS" \
  --batch-size "$BATCH_SIZE" \
  --gpu-indices "$GPU_INDICES" \
  --out "$OUTFILE"

echo "[EPT] Benchmark completed successfully."
echo "[EPT] Results saved to: $OUTFILE"
echo "===================================================="