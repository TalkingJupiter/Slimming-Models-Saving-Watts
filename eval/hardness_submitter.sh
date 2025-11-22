#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_submitter_response
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=eval/logs/student/%x_%j.out
#SBATCH --error=eval/logs/student/%x_%j.err

set -euo pipefail

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
# BASE_FEATURE="meta-llama/Llama-3.1-8B-Instruct"
# BASE_RESPONSE="meta-llama/Llama-3.1-8B-Instruct"
BASE_RELATION="meta-llama/Llama-3.1-8B-Instruct"

CHAT_FLAGS=( --apply_chat_template --fewshot_as_multiturn )

# -------------------------------------------------------------------
# Collect adapters: all subdirs under serialization_dir/{feature,...}
# -------------------------------------------------------------------
SER_ROOT="$(abspath serialization_dir)"

# ADAPTERS_FEATURE=()
ADAPTERS_RELATION=()
# ADAPTERS_RESPONSE=()

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
    echo "[INFO] Submitting: BASE=$base  ADAPTER=$ad"
    sbatch eval/harness_runner.sh "$base" "$ad" "${CHAT_FLAGS[@]}"
  done
}

# -------------------------------------------------------------------
# Submit all groups
# -------------------------------------------------------------------
# submit_group "$BASE_FEATURE"  "${ADAPTERS_FEATURE[@]}"
# submit_group "$BASE_RESPONSE" "${ADAPTERS_RESPONSE[@]}"
submit_group "$BASE_RELATION" "${ADAPTERS_RELATION[@]}"
