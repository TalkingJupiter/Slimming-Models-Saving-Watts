#!/usr/bin/env bash
#SBATCH --job-name=pipeline_launcher
#SBATCH --partition=zen4              # CPU partition to run the launcher itself
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/h100/%x_%j.out
#SBATCH --error=logs/h100/%x_%j.err

# This job only SUBMITS other jobs using sbatch (very light). Those jobs do the real work.

set -euo pipefail

TARGET="${TARGET:-repacss}"
TEACHER="${TEACHER:-meta-llama/Meta-Llama-3.1-70B}"
STUDENT="${STUDENT:-meta-llama/Meta-Llama-3.1-8B}"
TEACHER_DATA="${TEACHER_DATA:-$TEACHER}"

LAST_JOB_ID=""

extract_job_id() {
  awk '
    /^Submitted batch job / { id=$4 }
    /^[0-9]+/ { id=$1 }
    END {
      if (id != "") {
        sub(/;.*/, "", id)
        print id
      }
    }
  '
}

submit_job() {
  local job_kind="$1"
  local dependency="${2:-}"
  local output=""

  echo "[SUBMIT] kind=$job_kind target=$TARGET dependency=${dependency:-none}"

  if [[ -n "$dependency" ]]; then
    output=$(
      SBATCH_DEPENDENCY="$dependency" \
      TEACHER="$TEACHER" \
      STUDENT="$STUDENT" \
      TEACHER_DATA="$TEACHER_DATA" \
        bash job_title.sh "$TARGET" "$job_kind"
    )
  else
    output=$(
      TEACHER="$TEACHER" \
      STUDENT="$STUDENT" \
      TEACHER_DATA="$TEACHER_DATA" \
        bash job_title.sh "$TARGET" "$job_kind"
    )
  fi

  printf '%s\n' "$output"
  LAST_JOB_ID="$(printf '%s\n' "$output" | extract_job_id)"

  if [[ -z "$LAST_JOB_ID" ]]; then
    echo "[ERROR] Could not parse Slurm job id for '$job_kind'." >&2
    exit 1
  fi

  echo "[JOB] $job_kind=$LAST_JOB_ID"
}

afterok() {
  local IFS=:
  echo "afterok:$*"
}

usage() {
  cat <<USAGE
Usage:
  bash submit_h100.sh
  sbatch submit_h100.sh

Optional model overrides:
  TEACHER="meta-llama/Meta-Llama-3.1-70B" STUDENT="meta-llama/Meta-Llama-3.1-8B" bash submit_h100.sh

Pipeline dependencies:
  env -> shards
  shards -> teacher caches and traditional student training
  teacher caches -> feature/relation/response distillation
  distillations + traditional student -> EPT jobs
  EPT jobs -> harness submitter
USAGE
}

case "${1:-all}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  all)
    ;;
  *)
    echo "[ERROR] This launcher currently supports only the full 'all' pipeline." >&2
    usage >&2
    exit 2
    ;;
esac

echo "[PIPELINE] target=$TARGET"
echo "[PIPELINE] teacher=$TEACHER"
echo "[PIPELINE] student=$STUDENT"
echo "[PIPELINE] teacher_data=$TEACHER_DATA"

submit_job env
ENV_ID="$LAST_JOB_ID"

submit_job build_shards "$(afterok "$ENV_ID")"
SHARDS_ID="$LAST_JOB_ID"

submit_job build_caches "$(afterok "$SHARDS_ID")"
CACHES_ID="$LAST_JOB_ID"

submit_job traditional "$(afterok "$SHARDS_ID")"
TRADITIONAL_ID="$LAST_JOB_ID"

submit_job feature "$(afterok "$CACHES_ID")"
FEATURE_ID="$LAST_JOB_ID"

submit_job relation "$(afterok "$CACHES_ID")"
RELATION_ID="$LAST_JOB_ID"

submit_job response "$(afterok "$CACHES_ID")"
RESPONSE_ID="$LAST_JOB_ID"

DISTILL_AND_TRAD_DEP="$(afterok "$FEATURE_ID" "$RELATION_ID" "$RESPONSE_ID" "$TRADITIONAL_ID")"

submit_job ept_feature "$DISTILL_AND_TRAD_DEP"
EPT_FEATURE_ID="$LAST_JOB_ID"

submit_job ept_relation "$DISTILL_AND_TRAD_DEP"
EPT_RELATION_ID="$LAST_JOB_ID"

submit_job ept_response "$DISTILL_AND_TRAD_DEP"
EPT_RESPONSE_ID="$LAST_JOB_ID"

submit_job ept_llama8b_student "$DISTILL_AND_TRAD_DEP"
EPT_LLAMA8B_ID="$LAST_JOB_ID"

submit_job ept_llama70b_teacher "$DISTILL_AND_TRAD_DEP"
EPT_LLAMA70B_ID="$LAST_JOB_ID"

submit_job ept_trad_student "$DISTILL_AND_TRAD_DEP"
EPT_TRAD_ID="$LAST_JOB_ID"

submit_job hardness_submitter "$(afterok "$EPT_FEATURE_ID" "$EPT_RELATION_ID" "$EPT_RESPONSE_ID" "$EPT_LLAMA8B_ID" "$EPT_LLAMA70B_ID" "$EPT_TRAD_ID")"
HARNESS_ID="$LAST_JOB_ID"

cat <<SUMMARY
[SUMMARY]
  env=$ENV_ID
  shards=$SHARDS_ID
  caches=$CACHES_ID
  traditional=$TRADITIONAL_ID
  feature=$FEATURE_ID
  relation=$RELATION_ID
  response=$RESPONSE_ID
  ept_feature=$EPT_FEATURE_ID
  ept_relation=$EPT_RELATION_ID
  ept_response=$EPT_RESPONSE_ID
  ept_llama8b_student=$EPT_LLAMA8B_ID
  ept_llama70b_teacher=$EPT_LLAMA70B_ID
  ept_trad_student=$EPT_TRAD_ID
  harness=$HARNESS_ID
SUMMARY
