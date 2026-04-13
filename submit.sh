#!/usr/bin/env bash
#SBATCH --job-name=pipeline_launcher
#SBATCH --partition=quanah              # CPU partition to run the launcher itself
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This job only SUBMITS other jobs using sbatch (very light). Those jobs do the real work.

#!/usr/bin/env bash
set -Eeuo pipefail

# =======================
# Config
# =======================
LOGDIR="logs"

# Stage scripts
ENV_JOB="scripts/_env_single_node.sh"

SHARDS_JOB="scripts/run_build_shards.sh"
CACHE_TOPK_JOB="scripts/run_build_topk_cache.sh"
CACHE_FB_JOB="scripts/run_build_fb_cache.sh"
CACHE_RELB_JOB="scripts/run_build_relb_cache.sh"

# KD_JOB="scripts/submit_all_kd_single_node.sh"
KD_FEATURE="scripts/run_feature_based_single_node.sh"
KD_RELATION="scripts/run_relation_based_single_node.sh"
KD_RESPONSE="scripts/run_response_based_single_node.sh"

EPT_TEACHER="eval/ept/ept_Llama70B_teacher.sh"
EPT_STUDENT="eval/ept/ept_Llama8B_student.sh"
EPT_FEATURE="eval/ept/ept_feature.sh"
EPT_RELATION="eval/ept/ept_relation.sh"
EPT_RESPONSE="eval/ept/ept_response.sh"

EVAL_TEACHER="Base/LLama-3.1-70B-Ins_harness.sh"
EVAL_STUDENT="Base/LLama-3.1-8B-Ins_harness.sh"
EVAL_KD="eval/hardness_submitter.sh"

TRAD_MODEL_TRAIN="traditional-model/slurm/run_sft.sh"
TRAD_MODEL_EVAL="traditional-model/slurm/eval_8B_submitter.sh"
TRAD_MODEL_EPT="eval/ept/ept_Traditional_student.sh"

# Partition for the tiny cleanup/disarm jobs (CPU ok)
PARTITION_CPU="${PARTITION_CPU:-quanah}"

mkdir -p "$LOGDIR"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# =======================
# Helpers
# =======================
die() { echo "[FATAL] $*" >&2; exit 1; }

need_file() {
  [[ -f "$1" ]] || die "[ERROR] Missing required file: $1"
}

# submit: sbatch <opts...> <script> ; echo JobID
submit() {
  local script="$1"; shift || true
  local out jid
  out=$(sbatch "$@" "$script" 2>&1) || die "sbatch failed: $script $* :: $out"
  jid=$(awk '/Submitted batch job/ {print $4}' <<<"$out")
  [[ -n "${jid:-}" ]] || die "Could not parse JobID from: $out"
  echo "$jid"
}

# submit a cleanup job that cancels downstream targets if <dep_expr> fails
submit_cleanup() {
  local dep="$1"; shift
  [[ $# -gt 0 ]] || { echo ""; return 0; }
  submit --job-name="pipeline_cleanup" \
         --partition="${PARTITION_CPU}" \
         --time=00:05:00 \
         --output="${LOGDIR}/%x_%j.out" \
         --error="${LOGDIR}/%x_%j.err" \
         --dependency="${dep}" \
         --wrap "$(printf 'scancel %s || true\n' "$@")"
}

# submit a disarm job that cancels the given cleanup job if upstream succeeds
submit_disarm() {
  local dep_ok="$1"   # e.g., afterok:<upstream_jid>
  local cleanup_jid="$2"
  [[ -n "${cleanup_jid:-}" ]] || return 0
  sbatch --job-name="cleanup_disarm" \
         --partition="${PARTITION_CPU}" \
         --time=00:02:00 \
         --output="${LOGDIR}/%x_%j.out" \
         --error="${LOGDIR}/%x_%j.err" \
         --dependency="${dep_ok}" \
         --wrap "scancel ${cleanup_jid} || true" >/dev/null
}

# =======================
# Presence checks
# =======================
need_file "$ENV_JOB"

need_file "$SHARDS_JOB"
need_file "$CACHE_TOPK_JOB"
need_file "$CACHE_FB_JOB"
need_file "$CACHE_RELB_JOB"

# need_file "$KD_JOB"

need_file "$KD_FEATURE"
need_file "$KD_RELATION"
need_file "$KD_RESPONSE"

need_file "$EPT_TEACHER"
need_file "$EPT_STUDENT"
need_file "$EPT_FEATURE"
need_file "$EPT_RELATION"
need_file "$EPT_RESPONSE"
need_file "$EVAL_TEACHER"
need_file "$EVAL_STUDENT"
need_file "$EVAL_KD"

need_file "$TRAD_MODEL_TRAIN"
need_file "$TRAD_MODEL_EVAL"
need_file "$TRAD_MODEL_EPT"

# =======================
# Submit chain
# =======================
echo "[INFO] Submitting pipeline…"

jid_env=$(submit "$ENV_JOB")
echo "[SUBMIT] env             -> $jid_env"

# Shards with fixed --export values (your request)
jid_shards=$(submit "$SHARDS_JOB" \
  --dependency="afterok:${jid_env}" \
  --export=HF_DATASETS="tatsu-lab/alpaca,cerebras/SlimPajama-627B",WEIGHTS="1,1",SPLIT=train,STREAMING=1,OUT=data/shards.jsonl)
echo "[SUBMIT] build_shards    -> $jid_shards (afterok:$jid_env)"

jid_cache_topk=$(submit "$CACHE_TOPK_JOB" --dependency="afterok:${jid_shards}")
echo "[SUBMIT] cache_topk      -> $jid_cache_topk (afterok:$jid_shards)"

jid_cache_fb=$(submit "$CACHE_FB_JOB" --dependency="afterok:${jid_cache_topk}")
echo "[SUBMIT] cache_fb        -> $jid_cache_fb (afterok:$jid_cache_topk)"

jid_cache_relb=$(submit "$CACHE_RELB_JOB" --dependency="afterok:${jid_cache_fb}")
echo "[SUBMIT] cache_relb      -> $jid_cache_relb (afterok:$jid_cache_fb)"

# jid_kd=$(submit "$KD_JOB" --dependency="afterok:${jid_cache_relb}")
# echo "[SUBMIT] kd_pipeline     -> $jid_kd (afterok:$jid_cache_relb)"

jid_feature_kd=$(submit "$KD_FEATURE" --dependency="afterok:${jid_cache_fb}")
echo "[SUBMIT] Feature Distillation      -> $jid_feature_kd (afterok:$jid_cache_fb)"

jid_relation_kd=$(submit "$KD_RELATION" --dependency="afterok:${jid_cache_relb}")
echo "[SUBMIT] Relation Distillation      -> $jid_relation_kd (afterok:$jid_cache_relb)"

jid_response_kd=$(submit "$KD_RESPONSE" --dependency="afterok:${jid_cache_topk}")
echo "[SUBMIT] Response Distillation      -> $jid_response_kd (afterok:$jid_cache_topk)"

jid_kd_eval=$(submit "$EVAL_KD" --dependency="afterok:${jid_feature_kd}:${jid_relation_kd}:${jid_response_kd}")
echo "[SUBMIT] KD Model Evaluation     -> $jid_kd_eval (afterok:${jid_feature_kd}:${jid_relation_kd}:${jid_response_kd})"

jid_teacher_eval=$(submit "$EVAL_TEACHER" --dependency="afterok:${jid_env}")
echo "[SUBMIT] Teacher Evaluation -> (afterok:$jid_env)"

jid_student_eval=$(submit "$EVAL_STUDENT" --dependency="afterok:${jid_env}")
echo "[SUBMIT] Student Evaluation -> (afterok:$jid_env)"

jid_teacher_ept=$(submit "$EPT_TEACHER" --dependency="afterok:${jid_env}")
echo "[SUBMIT] TEACHER EPT -> (afterok:$jid_env)"

jid_student_ept=$(submit "$EPT_TEACHER" --dependency="afterok:${jid_env}")
echo "[SUBMIT] STUDENT EPT-> (afterok:$jid_env)"

jid_feature_ept=$(submit "$EPT_FEATURE" --dependency="afterok:${jid_feature_kd}")
echo "[SUBMIT] FEATURE EPT -> (afterok:$jid_feature_kd)"

jid_relation_ept=$(submit "$EPT_RELATION" --dependency="afterok:${jid_relation_kd}")
echo "[SUBMIT] RELATION EPT-> (afterok:$jid_relation_kd)"

jid_response_ept=$(submit "$EPT_RESPONSE" --dependency="afterok:${jid_response_kd}")
echo "[SUBMIT] RESPONSE EPT -> (afterok:$jid_response_kd)"

jid_trad_train=$(submit "$TRAD_MODEL_TRAIN" --dependency="afterok:${jid_env}")
echo "[SUBMIT] TRADITIONAL Student Training Submitted (afterok:${jid_env})"

jid_trad_eval=$(submit "$TRAD_MODEL_EVAL" --dependency="afterok:${jid_trad_train}")
echo "[SUBMIT] TRADITIONAL Student EVAL -> $jid_trad_eval (afterok:$jid_trad_train)"

jid_trad_ept=$(submit "$TRAD_MODEL_EPT" --dependency="afterok:${jid_train_train}")
echo "[SUBMIT] TRADITIONAL Student EPT -> $jid_trad_ept (afterok:$jid_trad_train)"

# =======================
# Cleanup on failure + Disarm on success
# =======================
# If env fails -> cancel shards, caches, kd
# jid_clean_env=$(submit_cleanup "afternotok:${jid_env}" \
#   "$jid_shards" "$jid_cache_topk" "$jid_cache_fb" "$jid_cache_relb" "$jid_kd")
# submit_disarm "afterok:${jid_env}"    "$jid_clean_env"

# # If shards fail -> cancel caches, kd
# jid_clean_shr=$(submit_cleanup "afternotok:${jid_shards}" \
#   "$jid_cache_topk" "$jid_cache_fb" "$jid_cache_relb" "$jid_kd")
# submit_disarm "afterok:${jid_shards}" "$jid_clean_shr"

# # If top-k cache fails -> cancel downstream caches and kd
# jid_clean_topk=$(submit_cleanup "afternotok:${jid_cache_topk}" \
#   "$jid_cache_fb" "$jid_cache_relb" "$jid_kd")
# submit_disarm "afterok:${jid_cache_topk}" "$jid_clean_topk"

# # If FB cache fails -> cancel downstream caches and kd
# jid_clean_fb=$(submit_cleanup "afternotok:${jid_cache_fb}" \
#   "$jid_cache_relb" "$jid_kd")
# submit_disarm "afterok:${jid_cache_fb}" "$jid_clean_fb"

# # If RelB cache fails -> cancel kd
# jid_clean_relb=$(submit_cleanup "afternotok:${jid_cache_relb}" "$jid_kd")
# submit_disarm "afterok:${jid_cache_relb}" "$jid_clean_relb"

# echo "[INFO] All jobs submitted:"
# printf "  env:        %s\n  shards:     %s\n  cache_topk: %s\n  cache_fb:   %s\n  cache_relb: %s\n  kd:         %s\n" \
#   "$jid_env" "$jid_shards" "$jid_cache_topk" "$jid_cache_fb" "$jid_cache_relb" "$jid_kd"
