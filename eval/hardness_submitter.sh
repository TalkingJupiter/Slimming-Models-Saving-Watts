#!/usr/bin/env bash

set -euo pipefail
source scripts/_env_single_node.sh

# -------------------------------------------------------------------
# Helper to get absolute paths (portable even if 'readlink -f' is missing)
# -------------------------------------------------------------------
abspath() { python - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

# -------------------------------------------------------------------
# Bases (Instruct variants; will use chat template flags)
# -------------------------------------------------------------------
STUDENT_MODEL=${STUDENT:-"meta-llama/Llama-3.1-8B"}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-${STUDENT_MODEL//\//_}}
STUDENT_MODEL_SOURCE=$(resolve_hf_model "$STUDENT_MODEL")
echo "[INFO] Student model source: $STUDENT_MODEL_SOURCE"

CHAT_FLAGS=( --apply_chat_template --fewshot_as_multiturn )
HARNESS_REPEATS="${HARNESS_REPEATS:-${EPT_REPEATS:-5}}"


# -------------------------------------------------------------------
# Collect adapters: all subdirs under serialization_dir/<student>/{feature,...}
# -------------------------------------------------------------------
SER_ROOT="$(abspath serialization_dir/${SAFE_STUDENT_NAME})"

ADAPTERS_FEATURE=()
ADAPTERS_RELATION=()
ADAPTERS_RESPONSE=()

if [[ -d "$SER_ROOT/feature" ]]; then
  # all immediate subdirs, sorted
  mapfile -t ADAPTERS_FEATURE < <(
    find "$SER_ROOT/feature" -mindepth 1 -maxdepth 1 -type d | sort
  )
fi

if [[ -d "$SER_ROOT/relation" ]]; then
  mapfile -t ADAPTERS_RELATION < <(
    find "$SER_ROOT/relation" -mindepth 1 -maxdepth 1 -type d | sort
  )
fi

if [[ -d "$SER_ROOT/response" ]]; then
  mapfile -t ADAPTERS_RESPONSE < <(
    find "$SER_ROOT/response" -mindepth 1 -maxdepth 1 -type d | sort
  )
fi

# -------------------------------------------------------------------
# Submission helper
# -------------------------------------------------------------------
submit_group () {
  local method="$1"
  local base="$2"
  shift 2
  local -a adapters=( "$@" )

  if [[ ${#adapters[@]} -eq 0 ]]; then
    echo "[WARN] No adapters for method '$method' and base '$base' - skipping group"
    return
  fi

  local array_max=$(( ${#adapters[@]} * HARNESS_REPEATS - 1 ))
  local log_dir="logs/eval/harness/${SAFE_STUDENT_NAME}/${method}/raw"
  mkdir -p "$log_dir"

  echo "[INFO] Submitting harness array: METHOD=$method BASE=$base ADAPTERS=${#adapters[@]} REPEATS=$HARNESS_REPEATS ARRAY=0-$array_max"
  sbatch \
    --array="0-${array_max}" \
    --output="${log_dir}/%x_%A_%a.out" \
    --error="${log_dir}/%x_%A_%a.err" \
    --export=ALL,HARNESS_METHOD="$method",HARNESS_REPEATS="$HARNESS_REPEATS" \
    eval/harness_array_runner.sh "$base" "${CHAT_FLAGS[@]}"
}

# -------------------------------------------------------------------
# Submit all groups: one Slurm array per distillation method
# -------------------------------------------------------------------
submit_group "feature" "$STUDENT_MODEL_SOURCE" "${ADAPTERS_FEATURE[@]}"
submit_group "response" "$STUDENT_MODEL_SOURCE" "${ADAPTERS_RESPONSE[@]}"
submit_group "relation" "$STUDENT_MODEL_SOURCE" "${ADAPTERS_RELATION[@]}"
