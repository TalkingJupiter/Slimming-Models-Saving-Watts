#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------
# Manual Configuration (EDIT THESE DIRECTLY)
# -----------------------------------------------------

MODEL=${STUDENT:-meta-llama/Meta-Llama-3.1-8B}   # Model to benchmark
NUM_PROMPTS=100                             # Number of Dolly prompts
BATCH_SIZE=2                                # Batch size for generation
GPU_INDEX=0                                 # GPU index to monitor

# Output location
SAFE_MODEL=${MODEL//\//}
OUTFILE="results/${SAFE_MODEL}/BASE/EPT/ept_base${SLURM_ARRAY_TASK_ID}.json"

# -----------------------------------------------------
# Initialization
# -----------------------------------------------------
mkdir -p logs results/${SAFE_MODEL}/BASE/EPT || true

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
source scripts/_env_single_node.sh
MODEL_SOURCE=$(resolve_hf_model "$MODEL")
echo "[EPT] Model source   : $MODEL_SOURCE"

# -----------------------------------------------------
# Run Benchmark
# -----------------------------------------------------
python eval/ept/benchmark/run_ept_benchmark.py \
  --model "$MODEL_SOURCE" \
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
