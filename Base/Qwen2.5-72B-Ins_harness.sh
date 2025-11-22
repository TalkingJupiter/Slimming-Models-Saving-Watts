#!/usr/bin/env bash
#SBATCH --job-name=base_Qwen/Qwen2.5-72B-Instruct_harness
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=eval/logs/%x_%j.out
#SBATCH --error=eval/logs/%x_%j.err

set -euo pipefail

###############################################################################
# 0. Hard-coded HuggingFace pretrained model
###############################################################################

BASE="Qwen/Qwen2.5-72B-Instruct"
ADAPTER=""   # leave empty unless you want to plug a LoRA directory
TASKS="mmlu,hellaswag,bbh,arc_challenge"

echo "[INFO] Using HF model: $BASE"
echo "[INFO] Adapter: ${ADAPTER:-<none>}"
echo "[INFO] Tasks: $TASKS"

###############################################################################
# 1. Setup
###############################################################################

mkdir -p eval/logs eval/results

source ~/.bashrc 2>/dev/null || true
conda activate kd 2>/dev/null || true
[[ -f scripts/_env_single_node.sh ]] && source scripts/_env_single_node.sh

# Use all 4 GPUs requested by SLURM (if not already set)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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

TS=$(date +%Y%m%d_%H%M%S)
OUTFILE="eval/results/harness_${RUN_NAME}_${TS}.json"

echo "[INFO] Output -> $OUTFILE"

###############################################################################
# 3. Build model args (multi-GPU model parallel)
###############################################################################

# Key change: parallelize=True and device_map_option=auto so 70B spans 4 GPUs
MODEL_ARGS="pretrained=${BASE},trust_remote_code=True,dtype=bfloat16,parallelize=True"

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

echo "[INFO] Completed. File saved to $OUTFILE"
