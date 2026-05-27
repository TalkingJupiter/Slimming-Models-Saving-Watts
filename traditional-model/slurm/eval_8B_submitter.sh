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

if [[ "$ARRAY_TASK_ID" -ge "${#CHECKPOINTS_TRAD[@]}" ]]; then
  echo "[WARN] No checkpoint for SLURM_ARRAY_TASK_ID=$ARRAY_TASK_ID under $TRAD_ROOT"
  exit 0
fi

submit_group() {
  local base="$1"; shift
  local -a checkpoints=( "$@" )

  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "[WARN] No checkpoints found under traditional_student/${SAFE_STUDENT_NAME}"
    return
  fi

  mkdir -p "results/traditional_student/${SAFE_STUDENT_NAME}/harness"

  for ckpt in "${checkpoints[@]}"; do
    run_id="$(basename "$(dirname "$ckpt")")"
    out_file="results/traditional_student/${SAFE_STUDENT_NAME}/harness/run_${run_id}.json"

    lm_eval \
      --model hf \
      --model_args "pretrained=${ckpt},trust_remote_code=True,dtype=bfloat16" \
      "${CHAT_FLAGS[@]}" \
      --tasks mmlu,hellaswag,bbh,arc_challenge \
      --batch_size auto \
      --output_path "$out_file"
    echo "[INFO] Done: BASE=$base CHECKPOINT=$ckpt OUT=$out_file"
  done
}

submit_group "$STUDENT_MODEL_SOURCE" "${CHECKPOINTS_TRAD[$ARRAY_TASK_ID]}"
