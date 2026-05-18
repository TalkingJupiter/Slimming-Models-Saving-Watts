#!/usr/bin/env bash

# set -euo pipefail
# cd "${SLURM_SUBMIT_DIR:-$PWD}"

# BASE=${1:-./traditional_student/$SAFE_STUDENT_NAME/*/final/}
# EXTRA_FLAGS=( "${@:2}" )

# mkdir -p logs results
# source ~/.bashrc || true
# conda activate kd || true
# [[ -f scripts/_env_single_node.sh ]] && source scripts/_env_single_node.sh

# SAFE_BASE="${BASE//\//_}"
# RUN_NAME="${SAFE_BASE}__traditional"


# echo "[INFO] CWD: $(pwd)"
# echo "[INFO] Base (pretrained): $BASE"

# lm_eval \
#   --model hf \
#   --model_args "pretrained=${BASE},trust_remote_code=True,dtype=bfloat16" \
#   "${EXTRA_FLAGS[@]}" \
#   --tasks mmlu,hellaswag,bbh,arc_challenge \
#   --batch_size auto \
#   --output_path "results/traditional_student/${SAFE_STUDENT_NAME}_harness_${RUN_NAME}_${TS}.json"

# echo "[INFO] Done -> results/${SLURM_JOB_ID}_harness_${RUN_NAME}_${TS}.json"


set -euo pipefail
source scripts/_env_single_node.sh

abspath() { python - "$1" << 'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

STUDENT_MODEL=${STUDENT:-"meta-llama/Llama-3.1-8B"}
SAFE_STUDENT_NAME=${STUDENT_MODEL//\//_}

CHAT_FLAGS=( --apply_chat_template --fewshot_as_multiturn)

# Collect traditional student final checkpoints:
# traditional_student/<safe_student_name>/<run_id>/final
TRAD_ROOT="$(abspath traditional_student/${SAFE_STUDENT_NAME})"

CHECKPOINTS_TRAD=()

mapfile -t CHECKPOINTS_TRAD < <(
  find "$TRAD_ROOT" -mindepth 2 -maxdepth 2 -type d -name final | sort
)

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

submit_group "$STUDENT_MODEL" "${CHECKPOINTS_TRAD[@]}"
