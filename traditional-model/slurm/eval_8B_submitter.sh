#!/usr/bin/env bash

set -euo pipefail
source scripts/_env_single_node.sh

abspath() { python - "$1" << 'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

STUDENT_MODEL=${STUDENT:-"meta-llama/Llama-3.1-8B"}
SAFE_STUDENT_NAME=${STUDENT_MODEL//\//_}
STUDENT_MODEL_SOURCE=$(resolve_hf_model "$STUDENT_MODEL")
echo "[INFO] Student model source: $STUDENT_MODEL_SOURCE"

CHAT_FLAGS=( --apply_chat_template --fewshot_as_multiturn)

# Collect traditional student final checkpoints:
# traditional_student/<safe_student_name>/<run_id>/final
TRAD_ROOT="$(abspath traditional_student/${SAFE_STUDENT_NAME})"

CHECKPOINTS_TRAD=()

mapfile -t CHECKPOINTS_TRAD < <(
  find "$TRAD_ROOT" -mindepth 2 -maxdepth 2 -type d -name final | sort
)

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
REPEATS="${HARNESS_REPEATS:-${EPT_REPEATS:-5}}"

if [[ "$REPEATS" -lt 1 ]]; then
  echo "[ERROR] HARNESS_REPEATS must be >= 1, got: $REPEATS" >&2
  exit 2
fi

MODEL_INDEX=$((ARRAY_TASK_ID / REPEATS))
REPEAT_INDEX=$((ARRAY_TASK_ID % REPEATS + 1))

if [[ "$MODEL_INDEX" -ge "${#CHECKPOINTS_TRAD[@]}" ]]; then
  echo "[WARN] No checkpoint for SLURM_ARRAY_TASK_ID=$ARRAY_TASK_ID MODEL_INDEX=$MODEL_INDEX under $TRAD_ROOT"
  exit 0
fi

# Avoid HF Datasets first-writer races when many eval array tasks build the
# same processed task cache at once.
JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
export HF_DATASETS_CACHE="$HF_HOME/datasets_eval/traditional/${JOB_GROUP_ID}_${ARRAY_TASK_ID}"
mkdir -p "$HF_DATASETS_CACHE"
echo "[INFO] HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
STRUCTURED_LOG_DIR="logs/eval/harness/${SAFE_STUDENT_NAME}/traditional/$MODEL_INDEX"
STRUCTURED_LOG_BASE="${STRUCTURED_LOG_DIR}/repeat${REPEAT_INDEX}_${JOB_GROUP_ID}_${ARRAY_TASK_ID}"
mkdir -p "$STRUCTURED_LOG_DIR"
exec > >(tee -a "${STRUCTURED_LOG_BASE}.out") 2> >(tee -a "${STRUCTURED_LOG_BASE}.err" >&2)
echo "[INFO] Structured log: ${STRUCTURED_LOG_BASE}.{out,err}"
echo "[INFO] Harness repeats/model: $REPEATS"
echo "[INFO] Model index: $MODEL_INDEX/${#CHECKPOINTS_TRAD[@]}"
echo "[INFO] Repeat: $REPEAT_INDEX/$REPEATS"

submit_group() {
  local base="$1"; shift
  local -a checkpoints=( "$@" )

  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "[WARN] No checkpoints found under traditional_student/${SAFE_STUDENT_NAME}"
    return
  fi

  mkdir -p "results/${SAFE_STUDENT_NAME}/traditional/$MODEL_INDEX/harness"

  for ckpt in "${checkpoints[@]}"; do
    run_id="$(basename "$(dirname "$ckpt")")"
    out_file="results/${SAFE_STUDENT_NAME}/traditional/$MODEL_INDEX/harness/eval_repeat${REPEAT_INDEX}.json"

    lm_eval \
      --model hf \
      --model_args "pretrained=${ckpt},trust_remote_code=True,dtype=bfloat16" \
      "${CHAT_FLAGS[@]}" \
      --tasks mmlu,hellaswag,bbh,arc_challenge \
      --batch_size auto \
      --output_path "$out_file"
    echo "[INFO] Done: BASE=$base CHECKPOINT=$ckpt REPEAT=$REPEAT_INDEX/$REPEATS OUT=$out_file"
  done
}

submit_group "$STUDENT_MODEL_SOURCE" "${CHECKPOINTS_TRAD[$MODEL_INDEX]}"
