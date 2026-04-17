#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_harness
#SBATCH --reservation=cpufreq
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --nodelist=rpg-93-[3-4]
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=60:00:00
#SBATCH --output=eval/student_runner/logs/%x_%j.out
#SBATCH --error=eval/student_runner/logs/%x_%j.err



set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BASE=${1:?Usage: sbatch kd_eval_harness.slurm <base_model_id> <adapter_dir> [extra flags]}
ADAPTER=${2:?Usage: sbatch kd_eval_harness.slurm <base_model_id> <adapter_dir> [extra flags]}
EXTRA_FLAGS=( "${@:3}" )  # e.g., --apply_chat_template --fewshot_as_multiturn

mkdir -p logs results/eval/${BASE}
source ~/.bashrc || true

export CUDA_HOME=/opt/apps/nfs/spack-1.1.0/opt/spack/linux-sapphirerapids/cuda-12.9.1-tio2hjc6xnw4bpsn37tz5fmwkz4dabp7
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

conda activate kd || true
[[ -f scripts/_env_single_node.sh ]] && source scripts/_env_single_node.sh

SAFE_BASE="${BASE//\//_}"
RUN_NAME="${SAFE_BASE}__$(basename "$ADAPTER")"
TS=$(date +%Y%m%d_%H%M%S)

echo "[INFO] CWD: $(pwd)"
echo "[INFO] Base: $BASE"
echo "[INFO] Adapter: $ADAPTER"
echo "[INFO] Adapter listing:"
ls -l "$ADAPTER" || true

accelerate launch --multi_gpu \
  -m lm_eval \
    --model hf \
    --model_args "pretrained=${BASE},peft=${ADAPTER},trust_remote_code=True,dtype=bfloat16" \
    "${EXTRA_FLAGS[@]}" \
    --tasks mmlu,hellaswag,bbh,arc_challenge \
    --batch_size auto \
    --output_path "results/eval/${BASE}/${SLURM_JOB_ID}_harness_${RUN_NAME}_${TS}.json"

echo "[INFO] Done -> results/eval/${SLURM_JOB_ID}_harness_${RUN_NAME}_${TS}.json"
