#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0. Hard-coded HuggingFace pretrained model
###############################################################################

BASE=${TEACHER:-meta-llama/Meta-Llama-3.1-70B}
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

# Use all GPUs requested by SLURM when CUDA_VISIBLE_DEVICES is not set.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_COUNT="${SLURM_GPUS_ON_NODE:-4}"
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
    export CUDA_VISIBLE_DEVICES
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

###############################################################################
# 2. Create run name
###############################################################################

SAFE_BASE="${BASE//\//_}"
RUN_NAME="$SAFE_BASE"

if [[ -n "$ADAPTER" ]]; then
    SAFE_ADAPTER="$(basename "$ADAPTER")"
    RUN_NAME="${RUN_NAME}__${SAFE_ADAPTER}"
fi

OUTFILE="results/${RUN_NAME}/BASE/harness/eval_${RUN_NAME}.json"

mkdir -p logs results/${RUN_NAME}/BASE/harness

echo "[INFO] Output -> $OUTFILE"

###############################################################################
# 3. Build model args (multi-GPU model parallel)
###############################################################################

# Key change: parallelize=True so 70B spans the GPUs allocated by SLURM
MODEL_ARGS="pretrained=${BASE_SOURCE},trust_remote_code=True,dtype=bfloat16,parallelize=True"

if [[ -n "$ADAPTER" ]]; then
    ls -l "$ADAPTER" || { echo "[ERROR] Invalid adapter: $ADAPTER"; exit 1; }
    MODEL_ARGS="${MODEL_ARGS},peft=${ADAPTER}"
fi

echo "[INFO] MODEL_ARGS: $MODEL_ARGS"

###############################################################################
# 4. HF token (env or cached)
###############################################################################

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    echo "[INFO] Using HF_TOKEN from environment."
elif [[ -f "$HOME/.huggingface/token" ]]; then
    echo "[INFO] Using cached HF token from ~/.huggingface/token"
else
    echo "[ERROR] No HuggingFace token found. Run: hf auth login"
    exit 1
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
  --apply_chat_template \
  --output_path "$OUTFILE"

echo "[INFO] Teacher harness bench completed. File saved to $OUTFILE"
