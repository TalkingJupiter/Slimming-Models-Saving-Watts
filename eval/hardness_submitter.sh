#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_submitter
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=eval/logs/student-eval/%x_%j.out
#SBATCH --error=eval/logs/student-eval/%x_%j.err

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
BASE_FEATURE="meta-llama/Llama-3.1-8B-Instruct"
BASE_RESPONSE="meta-llama/Llama-3.1-8B-Instruct"
BASE_RELATION="meta-llama/Llama-3.1-8B-Instruct"

CHAT_FLAGS=( --apply_chat_template )

# -------------------------------------------------------------------
# Collect adapters: all subdirs under serialization_dir/{feature,...}
# -------------------------------------------------------------------
SER_ROOT="$(abspath serialization_dir)"

ADAPTERS_FEATURE=()
ADAPTERS_RELATION=()
ADAPTERS_RESPONSE=()

if [[ -d "$SER_ROOT/feature" ]]; then
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
# Submit one KD group
# - All jobs in this group can run together
# - Entire group can optionally depend on previous group's jobs
# - Prints submitted job IDs (space-separated) to stdout
# -------------------------------------------------------------------
submit_group () {
  local base="$1"
  local dependency_string="$2"
  shift 2
  local -a adapters=( "$@" )

  local -a submitted_ids=()

  if [[ ${#adapters[@]} -eq 0 ]]; then
    echo "[WARN] No adapters for base '$base' - skipping group" >&2
    echo ""
    return
  fi

  for ad in "${adapters[@]}"; do
    if [[ ! -f "$ad/adapter_config.json" ]]; then
      echo "[WARN] Skipping '$ad' (missing adapter_config.json)" >&2
      continue
    fi

    echo "[INFO] Submitting: BASE=$base  ADAPTER=$ad" >&2

    local job_id
    if [[ -n "$dependency_string" ]]; then
      job_id=$(
        sbatch --parsable --dependency="$dependency_string" \
          eval/harness_runner.sh "$base" "$ad" "${CHAT_FLAGS[@]}"
      )
    else
      job_id=$(
        sbatch --parsable \
          eval/harness_runner.sh "$base" "$ad" "${CHAT_FLAGS[@]}"
      )
    fi

    echo "[INFO] Submitted job ID: $job_id" >&2
    submitted_ids+=( "$job_id" )
  done

  echo "${submitted_ids[*]}"
}

# -------------------------------------------------------------------
# Helper: build dependency string from a list of job IDs
# Example: afterok:12345:12346:12347
# -------------------------------------------------------------------
build_afterok_dependency () {
  local ids_str="$1"

  if [[ -z "$ids_str" ]]; then
    echo ""
    return
  fi

  read -r -a ids <<< "$ids_str"
  local dep="afterok:${ids[0]}"

  for ((i=1; i<${#ids[@]}; i++)); do
    dep="${dep}:${ids[i]}"
  done

  echo "$dep"
}

# -------------------------------------------------------------------
# Submit all groups in order:
# 1. feature
# 2. response   (after all feature jobs finish successfully)
# 3. relation   (after all response jobs finish successfully)
# -------------------------------------------------------------------
#FEATURE_JOB_IDS=$(submit_group "$BASE_FEATURE" "" "${ADAPTERS_FEATURE[@]}")
#FEATURE_DEP=$(build_afterok_dependency "$FEATURE_JOB_IDS")

#RESPONSE_JOB_IDS=$(submit_group "$BASE_RESPONSE" "" "${ADAPTERS_RESPONSE[@]}")
#RESPONSE_DEP=$(build_afterok_dependency "$RESPONSE_JOB_IDS")

RELATION_JOB_IDS=$(submit_group "$BASE_RELATION" "" "${ADAPTERS_RELATION[@]}")

#echo "[INFO] Feature job IDs : ${FEATURE_JOB_IDS:-<none>}"
#echo "[INFO] Response job IDs: ${RESPONSE_JOB_IDS:-<none>}"
echo "[INFO] Relation job IDs: ${RELATION_JOB_IDS:-<none>}"
echo "[INFO] Submission chain complete."
