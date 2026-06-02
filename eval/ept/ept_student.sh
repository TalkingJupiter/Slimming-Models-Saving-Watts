#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------
# Manual Configuration (EDIT THESE DIRECTLY)
# -----------------------------------------------------

MODEL=${STUDENT:-meta-llama/Meta-Llama-3.1-8B}   # Model to benchmark
NUM_PROMPTS=${EPT_NUM_PROMPTS:-100}              # Number of prompts
BATCH_SIZE=${EPT_BATCH_SIZE:-2}                 # Batch size for generation
GPU_INDEX=${EPT_GPU_INDICES:-0}                 # GPU index or comma-separated indices to monitor
MAX_NEW_TOKENS_LIST=${EPT_MAX_NEW_TOKENS_LIST:-32,64,128,256,512,1024}
WARMUP_BATCHES=${EPT_WARMUP_BATCHES:-2}
SAMPLE_INTERVAL=${EPT_SAMPLE_INTERVAL:-1.0}
EPT_PROMPTS=${EPT_PROMPTS:-}

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
echo "[EPT] GPU indices    : $GPU_INDEX"
echo "[EPT] Token sweep    : $MAX_NEW_TOKENS_LIST"
echo "[EPT] Warmup batches : $WARMUP_BATCHES"
echo "[EPT] Sample interval: $SAMPLE_INTERVAL"
if [[ -n "$EPT_PROMPTS" ]]; then
    echo "[EPT] Prompt file    : $EPT_PROMPTS"
else
    echo "[EPT] Prompt source  : Dolly seed 42"
fi
echo "[EPT] Output file    : $OUTFILE"
echo "[EPT] SLURM Job ID   : $SLURM_JOB_ID"
echo "----------------------------------------------------"

# -----------------------------------------------------
# Activate Environment
# -----------------------------------------------------
source ~/.bashrc || true
conda activate kd || true
source scripts/_env_single_node.sh
PROMPT_ARGS=()
if [[ -n "$EPT_PROMPTS" ]]; then
    PROMPT_ARGS+=(--prompts "$EPT_PROMPTS")
else
    PROMPT_ARGS+=(--use-dolly)
fi
MODEL_SOURCE=$(resolve_hf_model "$MODEL")
echo "[EPT] Model source   : $MODEL_SOURCE"

# -----------------------------------------------------
# Run Benchmark
# -----------------------------------------------------
python eval/ept/benchmark/run_ept_benchmark.py \
  --model "$MODEL_SOURCE" \
  --method "base_student" \
  --checkpoint "$MODEL_SOURCE" \
  "${PROMPT_ARGS[@]}" \
  --num-prompts "$NUM_PROMPTS" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens-list "$MAX_NEW_TOKENS_LIST" \
  --warmup-batches "$WARMUP_BATCHES" \
  --sample-interval "$SAMPLE_INTERVAL" \
  --gpu-indices "$GPU_INDEX" \
  --out "$OUTFILE"

# -----------------------------------------------------
# Completion Message
# -----------------------------------------------------
echo "[EPT] Benchmark completed successfully."
echo "[EPT] Results saved to: $OUTFILE"
echo "===================================================="
