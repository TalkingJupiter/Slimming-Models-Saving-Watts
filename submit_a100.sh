#!/usr/bin/env bash
#SBATCH --job-name=pipeline_launcher
#SBATCH --partition=nocona            # CPU partition to run the launcher itself
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/a100/%x_%j.out
#SBATCH --error=logs/a100/%x_%j.err

# This job only SUBMITS other jobs using sbatch (very light). Those jobs do the real work.

set -euo pipefail

TARGET="${TARGET:-hpcc_a100}"
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
EPT_REPEATS="${EPT_REPEATS:-5}"

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
    "EPT_REPEATS=$EPT_REPEATS"
    "HARNESS_REPEATS=${HARNESS_REPEATS:-$EPT_REPEATS}"
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
  bash submit_a100.sh [all]
  sbatch submit_a100.sh [all]

Optional model overrides:
  TEACHER="meta-llama/Meta-Llama-3.1-70B" STUDENT="meta-llama/Meta-Llama-3.1-8B" bash submit_a100.sh

Optional EPT override:
  EPT_REPEATS="$EPT_REPEATS" bash submit_a100.sh

Shard build defaults:
  HF_DATASETS="$SHARD_HF_DATASETS"
  WEIGHTS="$SHARD_WEIGHTS"
  MAX_SAMPLES="$SHARD_MAX_SAMPLES"
  SPLIT="$SHARD_SPLIT"
  STREAMING="$SHARD_STREAMING"
  OUT="$SHARD_OUT"

Pipeline dependencies:
  env -> build_shards -> warm_hf_cache
  warm_hf_cache -> feature cache -> relation cache -> response cache
  response cache -> feature KD -> relation KD -> response KD -> traditional SFT
  traditional SFT -> base student EPT -> base teacher EPT
  base teacher EPT -> feature EPT -> relation EPT -> response EPT -> traditional EPT
  traditional EPT -> harness submitter -> traditional eval -> teacher harness -> student harness

Log output layout:
  EPT:
    logs/eval/ept/<safe_model_name>/<method>/raw_or_slurm_file.{out,err}
    logs/eval/ept/${SAFE_STUDENT_NAME}/{feature,relation,response,traditional}/<model_index>/repeat<repeat>_<array_job_id>_<array_task_id>.{out,err}
  Harness:
    logs/eval/harness/${SAFE_STUDENT_NAME}/{feature,response,relation}/raw/<job_name>_<array_job_id>_<array_task_id>.{out,err}
    logs/eval/harness/${SAFE_STUDENT_NAME}/traditional/raw/<job_name>_<array_job_id>_<array_task_id>.{out,err}
    logs/eval/harness/<safe_model_name>/BASE/<job_name>_<job_id>.{out,err}

Harness output layout:
  Base student/teacher:
    results/<safe_model_name>/BASE/harness/eval_<safe_model_name>.json
  KD feature/relation/response:
    results/${SAFE_STUDENT_NAME}/{feature,relation,response}/<model_index>/harness/eval_repeat<repeat>.json
  Traditional SFT:
    results/${SAFE_STUDENT_NAME}/traditional/<model_index>/harness/eval_repeat<repeat>.json

EPT output layout:
  Base student:
    results/${SAFE_STUDENT_NAME}/BASE/EPT/ept_base<array_task>.json
  Base teacher:
    results/${SAFE_TEACHER_NAME}/BASE/EPT/ept_base<array_task>.json
  KD feature/relation/response:
    results/${SAFE_STUDENT_NAME}/{feature,relation,response}/<model_index>/EPT/ept_repeat<repeat>.json
    model_index  = SLURM_ARRAY_TASK_ID / EPT_REPEATS
    repeat       = SLURM_ARRAY_TASK_ID % EPT_REPEATS + 1
  Traditional SFT:
    results/${SAFE_STUDENT_NAME}/traditional/<model_index>/EPT/ept_repeat<repeat>.json
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
echo "[PIPELINE] shard_split=$SHARD_SPLIT"
echo "[PIPELINE] shard_streaming=$SHARD_STREAMING"
echo "[PIPELINE] shard_out=$SHARD_OUT"
echo "[PIPELINE] ept_repeats=$EPT_REPEATS"
echo "[PIPELINE] hf_cache_root=${HF_CACHE_ROOT:-.hf_cache}"
echo "[PIPELINE] ept_kd_layout=results/${SAFE_STUDENT_NAME}/{feature,relation,response}/<model_index>/EPT/ept_repeat<repeat>.json"

submit_job env
ENV_ID="$LAST_JOB_ID"

submit_job build_shards "$(afterok "$ENV_ID")"
SHARDS_ID="$LAST_JOB_ID"

submit_job warm_hf_cache "$(afterok "$SHARDS_ID")"
WARM_HF_CACHE_ID="$LAST_JOB_ID"

submit_job build_feature_cache "$(afterok "$WARM_HF_CACHE_ID")"
FEATURE_CACHE_ID="$LAST_JOB_ID"

submit_job build_relation_cache "$(afterok "$FEATURE_CACHE_ID")"
RELATION_CACHE_ID="$LAST_JOB_ID"

submit_job build_response_cache "$(afterok "$RELATION_CACHE_ID")"
RESPONSE_CACHE_ID="$LAST_JOB_ID"

submit_job feature "$(afterok "$RESPONSE_CACHE_ID")"
FEATURE_ID="$LAST_JOB_ID"

submit_job relation "$(afterok "$FEATURE_ID")"
RELATION_ID="$LAST_JOB_ID"

submit_job response "$(afterok "$RELATION_ID")"
RESPONSE_ID="$LAST_JOB_ID"

submit_job traditional "$(afterok "$RESPONSE_ID")"
TRADITIONAL_ID="$LAST_JOB_ID"

submit_job ept_student "$(afterok "$TRADITIONAL_ID")"
EPT_STUDENT_ID="$LAST_JOB_ID"

submit_job ept_teacher "$(afterok "$EPT_STUDENT_ID")"
EPT_TEACHER_ID="$LAST_JOB_ID"

submit_job ept_feature "$(afterok "$EPT_TEACHER_ID")"
EPT_FEATURE_ID="$LAST_JOB_ID"

submit_job ept_relation "$(afterok "$EPT_FEATURE_ID")"
EPT_RELATION_ID="$LAST_JOB_ID"

submit_job ept_response "$(afterok "$EPT_RELATION_ID")"
EPT_RESPONSE_ID="$LAST_JOB_ID"

submit_job ept_trad_student "$(afterok "$EPT_RESPONSE_ID")"
EPT_TRAD_ID="$LAST_JOB_ID"

submit_job hardness_submitter "$(afterok "$EPT_TRAD_ID")"
HARNESS_ID="$LAST_JOB_ID"

submit_job traditional_eval "$(afterok "$HARNESS_ID")"
TRADITIONAL_EVAL="$LAST_JOB_ID"

submit_job teacher_harness "$(afterok "$TRADITIONAL_EVAL")"
TEACHER_HARNESS="$LAST_JOB_ID"

submit_job student_harness "$(afterok "$TEACHER_HARNESS")"
STUDENT_HARNESS="$LAST_JOB_ID"

cat <<SUMMARY
[SUMMARY]
  env=${ENV_ID:-skipped}
  build_shards=${SHARDS_ID:-skipped}
  warm_hf_cache=${WARM_HF_CACHE_ID:-skipped}
  feature_cache=${FEATURE_CACHE_ID:-skipped}
  relation_cache=${RELATION_CACHE_ID:-skipped}
  response_cache=${RESPONSE_CACHE_ID:-skipped}
  feature_kd=${FEATURE_ID:-skipped}
  relation_kd=${RELATION_ID:-skipped}
  response_kd=${RESPONSE_ID:-skipped}
  traditional_sft=${TRADITIONAL_ID:-skipped}
  ept_student=${EPT_STUDENT_ID:-skipped}
  ept_teacher=${EPT_TEACHER_ID:-skipped}
  ept_feature=${EPT_FEATURE_ID:-skipped}
  ept_relation=${EPT_RELATION_ID:-skipped}
  ept_response=${EPT_RESPONSE_ID:-skipped}
  ept_traditional=${EPT_TRAD_ID:-skipped}
  harness_submitter=${HARNESS_ID:-skipped}
  traditional_eval=${TRADITIONAL_EVAL:-skipped}
  teacher_harness=${TEACHER_HARNESS:-skipped}
  student_harness=${STUDENT_HARNESS:-skipped}

[OUTPUTS]
  ept_student=results/${SAFE_STUDENT_NAME}/BASE/EPT/ept_base<array_task>.json
  ept_teacher=results/${SAFE_TEACHER_NAME}/BASE/EPT/ept_base<array_task>.json
  ept_feature=results/${SAFE_STUDENT_NAME}/feature/<model_index>/EPT/ept_repeat<repeat>.json
  ept_relation=results/${SAFE_STUDENT_NAME}/relation/<model_index>/EPT/ept_repeat<repeat>.json
  ept_response=results/${SAFE_STUDENT_NAME}/response/<model_index>/EPT/ept_repeat<repeat>.json
  ept_traditional=results/${SAFE_STUDENT_NAME}/traditional/<model_index>/EPT/ept_repeat<repeat>.json
SUMMARY
