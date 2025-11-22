#!/usr/bin/env bash
#SBATCH --job-name=base_Qwen/Qwen2.5-7B-Instruct_harness
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=eval/logs/%x_%j.out
#SBATCH --error=eval/logs/%x_%j.err

#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0. Hard-coded HuggingFace pretrained model
###############################################################################

BASE="Qwen/Qwen2.5-7B-Instruct"
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
# 3. Build model args
###############################################################################

MODEL_ARGS="pretrained=${BASE},trust_remote_code=True,dtype=bfloat16"

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

echo "[INFO] Completed. File saved to $OUTFILE"
