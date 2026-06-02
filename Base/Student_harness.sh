#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0. Hard-coded HuggingFace pretrained model
###############################################################################

BASE=${STUDENT:-meta-llama/Meta-Llama-3.1-8B}
ADAPTER=""   # leave empty unless you want to plug a LoRA directory
TASKS="mmlu,hellaswag,bbh,arc_challenge"

echo "[INFO] Using HF model: $BASE"
echo "[INFO] Adapter: ${ADAPTER:-<none>}"
echo "[INFO] Tasks: $TASKS"

###############################################################################
# 1. Setup
###############################################################################
source ~/.bashrc 2>/dev/null || true
conda activate kd 2>/dev/null || true
[[ -f scripts/_env_single_node.sh ]] && source scripts/_env_single_node.sh

JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export HF_DATASETS_CACHE="$HF_HOME/datasets_eval/harness/${JOB_GROUP_ID}_${TASK_ID}"
mkdir -p "$HF_DATASETS_CACHE"
echo "[INFO] HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
BASE_SOURCE=$(resolve_hf_model "$BASE")
echo "[INFO] Model source: $BASE_SOURCE"

###############################################################################
# 2. Create run name
###############################################################################
SAFE_BASE="${BASE//\//_}"
RUN_NAME="$SAFE_BASE"

if [[ -n "$ADAPTER" ]]; then
    SAFE_ADAPTER="$(basename "$ADAPTER")"
    RUN_NAME="${RUN_NAME}__${SAFE_ADAPTER}"
fi


# OUTFILE="results/harness/base/harness_${RUN_NAME}_${SLURM_JOB_ID}.json"
OUTFILE="results/${RUN_NAME}/BASE/harness/eval_${RUN_NAME}.json"

mkdir -p logs results/${RUN_NAME}/BASE/harness

echo "[INFO] Output -> $OUTFILE"

###############################################################################
# 3. Build model args
###############################################################################

MODEL_ARGS="pretrained=${BASE_SOURCE},trust_remote_code=True,dtype=bfloat16"

if [[ -n "$ADAPTER" ]]; then
    ls -l "$ADAPTER" || { echo "[ERROR] Invalid adapter: $ADAPTER"; exit 1; }
    MODEL_ARGS="${MODEL_ARGS},peft=${ADAPTER}"
fi

###############################################################################
# 4. HF token
###############################################################################

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    echo "[INFO] Using HF_TOKEN"
else
    echo "[WARN] No HF_TOKEN set — private models may fail."
fi

###############################################################################
# 5. Run evaluation
###############################################################################

echo "[INFO] Running BASE CASE lm_eval..."
lm_eval \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size auto \
  --output_path "$OUTFILE"

echo "[INFO] Student harness bench completed. File saved to $OUTFILE"
