#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_harness
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/eval/harness/adapters/%x_%j.out
#SBATCH --error=logs/eval/harness/adapters/%x_%j.err



set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BASE=${1:?Usage: sbatch kd_eval_harness.slurm <base_model_id> <adapter_dir> [extra flags]}
ADAPTER=${2:?Usage: sbatch kd_eval_harness.slurm <base_model_id> <adapter_dir> [extra flags]}
EXTRA_FLAGS=( "${@:3}" )  # e.g., --apply_chat_template --fewshot_as_multiturn

mkdir -p logs results
source ~/.bashrc || true
conda activate kd || true
[[ -f scripts/_env_single_node.sh ]] && source scripts/_env_single_node.sh

JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export HF_DATASETS_CACHE="$HF_HOME/datasets_eval/harness/${JOB_GROUP_ID}_${TASK_ID}"
mkdir -p "$HF_DATASETS_CACHE"
echo "[INFO] HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

SAFE_BASE="${BASE//\//_}"
ADAPTER_NAME="$(basename "$ADAPTER")"
HARNESS_METHOD="$(basename "$(dirname "$ADAPTER")")"
ADAPTER_PARENT="$(basename "$(dirname "$(dirname "$ADAPTER")")")"
SAFE_STUDENT_NAME="${SAFE_STUDENT_NAME:-$ADAPTER_PARENT}"
HARNESS_REPEAT="${HARNESS_REPEAT:-1}"

MODEL_NUMBER="$ADAPTER_NAME"

OUTDIR="results/${SAFE_STUDENT_NAME}/${HARNESS_METHOD}/${MODEL_NUMBER}/harness"
OUTFILE="${OUTDIR}/eval_repeat${HARNESS_REPEAT}.json"
RUN_NAME="${SAFE_BASE}__${HARNESS_METHOD}_${MODEL_NUMBER}_repeat${HARNESS_REPEAT}"
mkdir -p "$OUTDIR"

echo "[INFO] CWD: $(pwd)"
echo "[INFO] Base: $BASE"
echo "[INFO] Adapter: $ADAPTER"
echo "[INFO] Harness method: $HARNESS_METHOD"
echo "[INFO] Model number: $MODEL_NUMBER"
echo "[INFO] Repeat: $HARNESS_REPEAT"
echo "[INFO] Output: $OUTFILE"
echo "[INFO] Adapter listing:"
ls -l "$ADAPTER" || true

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE},peft=${ADAPTER},trust_remote_code=True,dtype=bfloat16" \
  "${EXTRA_FLAGS[@]}" \
  --tasks mmlu,hellaswag,bbh,arc_challenge \
  --batch_size auto \
  --output_path "$OUTFILE"

echo "[INFO] Done -> $OUTFILE"
