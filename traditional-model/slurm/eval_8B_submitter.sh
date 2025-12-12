#!/usr/bin/env bash
#SBATCH --job-name=8B_trad_eval
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=traditional-model/logs/eval_trad_%j.out
#SBATCH --error=traditional-model/logs/eval_trad_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BASE=${1:-./traditional-model/checkpoints/epoch_0/}
EXTRA_FLAGS=( "${@:2}" )

mkdir -p logs results
source ~/.bashrc || true
conda activate kd || true
[[ -f scripts/_env_single_node.sh ]] && source scripts/_env_single_node.sh

SAFE_BASE="${BASE//\//_}"
RUN_NAME="${SAFE_BASE}__traditional"
TS=$(date +%Y%m%d_%H%M%S)

echo "[INFO] CWD: $(pwd)"
echo "[INFO] Base (pretrained): $BASE"

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE},trust_remote_code=True,dtype=bfloat16" \
  "${EXTRA_FLAGS[@]}" \
  --tasks mmlu,hellaswag,bbh,arc_challenge \
  --batch_size auto \
  --output_path "results/${SLURM_JOB_ID}_harness_${RUN_NAME}_${TS}.json"

echo "[INFO] Done -> results/${SLURM_JOB_ID}_harness_${RUN_NAME}_${TS}.json"
