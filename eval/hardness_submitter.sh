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
  local base="$1"; shift
  local -a adapters=( "$@" )

  if [[ ${#adapters[@]} -eq 0 ]]; then
    echo "[WARN] No adapters for base '$base' - skipping group"
    return
  fi

  for ad in "${adapters[@]}"; do
    # validate adapter dir contains adapter_config.json
    if [[ ! -f "$ad/adapter_config.json" ]]; then
      echo "[WARN] Skipping '$ad' (missing adapter_config.json)"
      continue
    fi
    method="$(basename "$(dirname "$ad")")"
    adapter_name="$(basename "$ad")"
    if [[ "$adapter_name" =~ ^[0-9]+$ ]]; then
      model_number="$adapter_name"
    else
      model_number="$adapter_name"
    fi

    for repeat in $(seq 1 "$HARNESS_REPEATS"); do
      log_dir="logs/eval/harness/${SAFE_STUDENT_NAME}/${method}/${model_number}"
      mkdir -p "$log_dir"
      echo "[INFO] Submitting: BASE=$base  ADAPTER=$ad  REPEAT=$repeat/$HARNESS_REPEATS"
      sbatch \
        --output="${log_dir}/repeat${repeat}_%j.out" \
        --error="${log_dir}/repeat${repeat}_%j.err" \
        --export=ALL,HARNESS_REPEAT="$repeat" \
        eval/harness_runner.sh "$base" "$ad" "${CHAT_FLAGS[@]}"
    done
  done
}

# -------------------------------------------------------------------
# Submit all groups
# -------------------------------------------------------------------
submit_group "$STUDENT_MODEL_SOURCE"  "${ADAPTERS_FEATURE[@]}"
submit_group "$STUDENT_MODEL_SOURCE" "${ADAPTERS_RESPONSE[@]}"
submit_group "$STUDENT_MODEL_SOURCE" "${ADAPTERS_RELATION[@]}"
