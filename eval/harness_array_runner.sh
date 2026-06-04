#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_harness_array
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BASE=${1:?Usage: harness_array_runner.sh <base_model_id> [extra lm_eval flags...]}
EXTRA_FLAGS=( "${@:2}" )

source ~/.bashrc || true
conda activate kd || true
source scripts/_env_single_node.sh

STUDENT_MODEL=${STUDENT:-"meta-llama/Llama-3.1-8B"}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-${STUDENT_MODEL//\//_}}
HARNESS_METHOD=${HARNESS_METHOD:?Set HARNESS_METHOD to feature, relation, or response}
HARNESS_REPEATS=${HARNESS_REPEATS:-${EPT_REPEATS:-5}}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [[ "$HARNESS_REPEATS" -lt 1 ]]; then
  echo "[ERROR] HARNESS_REPEATS must be >= 1, got: $HARNESS_REPEATS" >&2
  exit 2
fi

ADAPTER_ROOT="serialization_dir/${SAFE_STUDENT_NAME}/${HARNESS_METHOD}"
ADAPTERS=()
if [[ -d "$ADAPTER_ROOT" ]]; then
  mapfile -t ADAPTERS < <(find "$ADAPTER_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#ADAPTERS[@]} -eq 0 ]]; then
  echo "[WARN] No adapters found under $ADAPTER_ROOT"
  exit 0
fi

MODEL_INDEX=$((TASK_ID / HARNESS_REPEATS))
HARNESS_REPEAT=$((TASK_ID % HARNESS_REPEATS + 1))

if [[ "$MODEL_INDEX" -ge "${#ADAPTERS[@]}" ]]; then
  echo "[WARN] No adapter for SLURM_ARRAY_TASK_ID=$TASK_ID MODEL_INDEX=$MODEL_INDEX under $ADAPTER_ROOT"
  exit 0
fi

ADAPTER="${ADAPTERS[$MODEL_INDEX]}"
ADAPTER_NAME="$(basename "$ADAPTER")"
if [[ "$ADAPTER_NAME" != "$MODEL_INDEX" ]]; then
  echo "[INFO] Adapter dir name is '$ADAPTER_NAME'; preserving it for output paths. Computed index=$MODEL_INDEX"
fi

export HARNESS_REPEAT

echo "[INFO] Harness method: $HARNESS_METHOD"
echo "[INFO] Adapter root: $ADAPTER_ROOT"
echo "[INFO] Model index: $MODEL_INDEX"
echo "[INFO] Repeat: $HARNESS_REPEAT/$HARNESS_REPEATS"
echo "[INFO] Adapter: $ADAPTER"

bash eval/harness_runner.sh "$BASE" "$ADAPTER" "${EXTRA_FLAGS[@]}"
