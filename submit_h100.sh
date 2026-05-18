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
SAFE_STUDENT_NAME="${STUDENT//\//_}"
SAFE_TEACHER_NAME="${TEACHER//\//_}"
TEACHER_DATA="${TEACHER_DATA:-$SAFE_TEACHER_NAME}"
SHARD_HF_DATASETS="${HF_DATASETS:-allenai/tulu-3-sft-mixture,HuggingFaceFW/fineweb-edu}"
SHARD_WEIGHTS="${WEIGHTS:-3,1}"
SHARD_MAX_SAMPLES="${MAX_SAMPLES:-300000}"
SHARD_SPLIT="${SPLIT:-train}"
SHARD_STREAMING="${STREAMING:-1}"
SHARD_OUT="${OUT:-data/shards.jsonl}"

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
  local env_args=(
    "TEACHER=$TEACHER"
    "STUDENT=$STUDENT"
    "TEACHER_DATA=$TEACHER_DATA"
  )

  if [[ "$job_kind" == "build_shards" ]]; then
    env_args+=(
      "HF_DATASETS=$SHARD_HF_DATASETS"
      "WEIGHTS=$SHARD_WEIGHTS"
      "MAX_SAMPLES=$SHARD_MAX_SAMPLES"
      "SPLIT=$SHARD_SPLIT"
      "STREAMING=$SHARD_STREAMING"
      "OUT=$SHARD_OUT"
    )
  fi

  if [[ "$job_kind" == "traditional" ]]; then
    env_args+=(
      "MODEL_NAME=traditional_student"
      "MODEL=$STUDENT"
    )
  fi

  echo "[SUBMIT] kind=$job_kind target=$TARGET dependency=${dependency:-none}"

  if [[ -n "$dependency" ]]; then
    output=$(
      env "SBATCH_DEPENDENCY=$dependency" "${env_args[@]}" \
        bash job_title.sh "$TARGET" "$job_kind"
    )
  else
    output=$(
      env "${env_args[@]}" \
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

Shard build defaults:
  HF_DATASETS="$SHARD_HF_DATASETS"
  WEIGHTS="$SHARD_WEIGHTS"
  MAX_SAMPLES="$SHARD_MAX_SAMPLES"
  SPLIT="$SHARD_SPLIT"
  STREAMING="$SHARD_STREAMING"
  OUT="$SHARD_OUT"

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
echo "[PIPELINE] shard_datasets=$SHARD_HF_DATASETS"
echo "[PIPELINE] shard_weights=$SHARD_WEIGHTS"
echo "[PIPELINE] shard_max_samples=$SHARD_MAX_SAMPLES"
echo "[PIPELINE] shard_out=$SHARD_OUT"

submit_job env
ENV_ID="$LAST_JOB_ID"

submit_job build_shards "$(afterok "$ENV_ID")"
SHARDS_ID="$LAST_JOB_ID"

submit_job build_feature_cache "$(afterok "$SHARDS_ID")"
FEATURE_CACHE_ID="$LAST_JOB_ID"

submit_job build_relation_cache "$(afterok "$SHARDS_ID")"
RELATION_CACHE_ID="$LAST_JOB_ID"

submit_job build_response_cache "$(afterok "$SHARDS_ID")"
RESPONSE_CACHE_ID="$LAST_JOB_ID"

submit_job traditional "$(afterok "$SHARDS_ID")"
TRADITIONAL_ID="$LAST_JOB_ID"

submit_job feature "$(afterok "$FEATURE_CACHE_ID")"
FEATURE_ID="$LAST_JOB_ID"

submit_job relation "$(afterok "$RELATION_CACHE_ID")"
RELATION_ID="$LAST_JOB_ID"

submit_job response "$(afterok "$RESPONSE_CACHE_ID")"
RESPONSE_ID="$LAST_JOB_ID"

# DISTILL_AND_TRAD_DEP="$(afterok "$FEATURE_ID" "$RELATION_ID" "$RESPONSE_ID" "$TRADITIONAL_ID")"

# submit_job ept_feature "$DISTILL_AND_TRAD_DEP"
# EPT_FEATURE_ID="$LAST_JOB_ID"

# submit_job ept_relation "$DISTILL_AND_TRAD_DEP"
# EPT_RELATION_ID="$LAST_JOB_ID"

# submit_job ept_response "$DISTILL_AND_TRAD_DEP"
# EPT_RESPONSE_ID="$LAST_JOB_ID"

# submit_job ept_llama8b_student "$DISTILL_AND_TRAD_DEP"
# EPT_LLAMA8B_ID="$LAST_JOB_ID"

# submit_job ept_llama70b_teacher "$DISTILL_AND_TRAD_DEP"
# EPT_LLAMA70B_ID="$LAST_JOB_ID"

# submit_job ept_trad_student "$DISTILL_AND_TRAD_DEP"
# EPT_TRAD_ID="$LAST_JOB_ID"

# submit_job hardness_submitter "$(afterok "$EPT_FEATURE_ID" "$EPT_RELATION_ID" "$EPT_RESPONSE_ID" "$EPT_LLAMA8B_ID" "$EPT_LLAMA70B_ID" "$EPT_TRAD_ID")"
# HARNESS_ID="$LAST_JOB_ID"

cat <<SUMMARY
[SUMMARY]
  env=$ENV_ID
  shards=$SHARDS_ID
  feature_cache=$FEATURE_CACHE_ID
  relation_cache=$RELATION_CACHE_ID
  response_cache=$RESPONSE_CACHE_ID
  traditional=$TRADITIONAL_ID
  feature=$FEATURE_ID
  relation=$RELATION_ID
  response=$RESPONSE_ID
SUMMARY

  # ept_feature=$EPT_FEATURE_ID
  # ept_relation=$EPT_RELATION_ID
  # ept_response=$EPT_RESPONSE_ID
  # ept_llama8b_student=$EPT_LLAMA8B_ID
  # ept_llama70b_teacher=$EPT_LLAMA70B_ID
  # ept_trad_student=$EPT_TRAD_ID
  # harness=$HARNESS_ID

  #sbatch --export=ALL,TARGET=repacss,TEACHER=meta-llama/Meta-Llama-3.1-70B,STUDENT=meta-llama/Meta-Llama-3.1-8B submit_h100.sh
