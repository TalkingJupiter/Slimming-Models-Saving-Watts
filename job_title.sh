#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-repacss}"
JOB_KIND="${2:-env}"
TEACHER="${TEACHER:-meta-llama/Meta-Llama-3.1-70B}"
STUDENT="${STUDENT:-meta-llama/Meta-Llama-3.1-8B}"
FAMILY=${FAMILY:-llama}
shift 2 || true

CPU=""
GPU=""

SAFE_TEACHER_NAME="${TEACHER//\//_}"
SAFE_STUDENT_NAME="${STUDENT//\//_}"
TEACHER_DATA="${TEACHER_DATA:-$SAFE_TEACHER_NAME}"

# AUTOMATED PARTITION SELECTION
if [[ "$TARGET" == "repacss" ]]; then
  echo "Target entered as repacss"
  CPU="zen4"
  GPU="h100"
  PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
elif [[ "$TARGET" == "hpcc" || "$TARGET" == "hpcc_a100" ]]; then
  echo "Target entered as hpcc_a100"
  TARGET="hpcc"
  CPU="cpu"
  GPU="a100"
  PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
elif [[ "$TARGET" == "hpcc_v100" ]]; then
  echo "Target entered as hpcc_v100"
  TARGET="hpcc"
  CPU="cpu"
  GPU="v100"
  PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
fi

GRES_ARGS=()
ARRAY_ARGS=()
EXTRA_SBATCH_ARGS=()
DEPENDENCY_ARGS=()

if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  DEPENDENCY_ARGS=(--dependency="$SBATCH_DEPENDENCY")
fi

case "$TARGET:$JOB_KIND" in

  repacss:env)
    JOB_NAME="env_setup"
    PARTITION=$CPU
    NODES="1"
    GPUS_PER_NODE="0"
    CPUS_PER_TASK="1"
    MEM="1G"
    TIME="00:20:00"
    JOB_SCRIPT="scripts/_env_single_node.sh"

    OUTPUT="logs/env_setup/%x_%j.out"
    ERROR="logs/env_setup/%x_%j.err"
    ;;


  repacss:warm_hf_cache)
    JOB_NAME="warm_hf_cache_${SAFE_TEACHER_NAME}"
    PARTITION=$CPU
    NODES="1"
    GPUS_PER_NODE="0"
    CPUS_PER_TASK="4"
    MEM="64G"
    TIME="12:00:00"
    JOB_SCRIPT="scripts/warm_hf_cache.sh"

    OUTPUT="logs/hf_cache/%x_%j.out"
    ERROR="logs/hf_cache/%x_%j.err"
    ;;

  repacss:build_shards)
    JOB_NAME="build_shards"
    PARTITION=$CPU
    NODES="1"
    GPUS_PER_NODE="0"
    CPUS_PER_TASK="8"
    MEM="50G"
    TIME="06:00:00"
    JOB_SCRIPT="scripts/run_build_shards.sh"

    OUTPUT="logs/shards/%x_%j.out"
    ERROR="logs/shards/%x_%j.err"
    ;;

  repacss:build_feature_cache)
  JOB_NAME="kd_build_feature_cache_${SAFE_TEACHER_NAME}"
  PARTITION=$GPU
  NODES="1"
  GPUS_PER_NODE="4"
  CPUS_PER_TASK="64"
  MEM="256G"
  TIME="48:00:00"
  JOB_SCRIPT="scripts/build_feature_cache.sh"
  GRES_ARGS=(--gpus-per-node=4)
  ARRAY_ARGS=(--array=0-4)
  EXTRA_SBATCH_ARGS=(
    --exclusive
    --signal=B:SIGUSR1@300
    --requeue
  )
  
  SCRIPT_ARGS=()

  OUTPUT="logs/build_cache/feature/%x_%A_%a.out"
  ERROR="logs/build_cache/feature/%x_%A_%a.err"
  ;;

  repacss:build_relation_cache)
    JOB_NAME="kd_build_relation_cache_${SAFE_TEACHER_NAME}"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="64"
    MEM="256G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/build_relation_cache.sh"
    GRES_ARGS=(--gpus-per-node=4)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )
    
    SCRIPT_ARGS=()

    OUTPUT="logs/build_cache/relation/%x_%A_%a.out"
    ERROR="logs/build_cache/relation/%x_%A_%a.err"
    ;;
  
  repacss:build_response_cache)
    JOB_NAME="kd_build_response_cache_${SAFE_TEACHER_NAME}"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="64"
    MEM="256G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/build_response_cache.sh"
    GRES_ARGS=(--gpus-per-node=4)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )
    
    SCRIPT_ARGS=()

    OUTPUT="logs/build_cache/response/%x_%A_%a.out"
    ERROR="logs/build_cache/response/%x_%A_%a.err"
    ;;  

  repacss:feature)
    JOB_NAME="kd_feature_${SAFE_STUDENT_NAME}_from_${SAFE_TEACHER_NAME}"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="256G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/run_feature_based_single_node.sh"

    GRES_ARGS=(--gpus-per-node=4)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )

    OUTPUT="logs/distillation/feature/%x_%A_%a.out"
    ERROR="logs/distillation/feature/%x_%A_%a.err"
    ;;

  repacss:relation)
    JOB_NAME="kd_relation_${SAFE_STUDENT_NAME}_from_${SAFE_TEACHER_NAME}"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="256G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/run_relation_based_single_node.sh"

    GRES_ARGS=(--gpus-per-node=4)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )

    OUTPUT="logs/distillation/relation/%x_%A_%a.out"
    ERROR="logs/distillation/relation/%x_%A_%a.err"
    ;;

  repacss:response)
    JOB_NAME="kd_response_${SAFE_STUDENT_NAME}_from_${SAFE_TEACHER_NAME}"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="256G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/run_response_based_single_node.sh"

    GRES_ARGS=(--gpus-per-node=4)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )

    OUTPUT="logs/distillation/response/%x_%A_%a.out"
    ERROR="logs/distillation/response/%x_%A_%a.err"
    ;;

  repacss:traditional)
    JOB_NAME="Traditional_student_training_h100"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="1"
    CPUS_PER_TASK="6"
    MEM="256G"
    TIME="48:00:00"
    JOB_SCRIPT="traditional-model/slurm/run_sft.sh"

    GRES_ARGS=(--gpus-per-node=1)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=()
    OUTPUT="logs/distillation/traditional/%x_%A_%a.out"
    ERROR="logs/distillation/traditional/%x_%A_%a.err"
    ;;

  repacss:ept_feature)
    JOB_NAME="ept_feature_h100"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="24G"
    TIME="48:00:00"
    JOB_SCRIPT="eval/ept/ept_feature.sh"

    GRES_ARGS=(--gpus-per-node=1)
    EPT_ARRAY_MAX=$((${EPT_REPEATS:-5} * 5 - 1))
    ARRAY_ARGS=(--array=0-${EPT_ARRAY_MAX})
    EXTRA_SBATCH_ARGS=()

    OUTPUT="logs/eval/ept/${SAFE_STUDENT_NAME}/feature/%x_%A_%a.out"
    ERROR="logs/eval/ept/${SAFE_STUDENT_NAME}/feature/%x_%A_%a.err"
    ;;

  repacss:ept_relation)
    JOB_NAME="ept_relation_h100"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="24G"
    TIME="48:00:00"
    JOB_SCRIPT="eval/ept/ept_relation.sh"

    GRES_ARGS=(--gpus-per-node=1)
    EPT_ARRAY_MAX=$((${EPT_REPEATS:-5} * 5 - 1))
    ARRAY_ARGS=(--array=0-${EPT_ARRAY_MAX})
    EXTRA_SBATCH_ARGS=()

    OUTPUT="logs/eval/ept/${SAFE_STUDENT_NAME}/relation/%x_%A_%a.out"
    ERROR="logs/eval/ept/${SAFE_STUDENT_NAME}/relation/%x_%A_%a.err"
    ;; 

  repacss:ept_response)
    JOB_NAME="ept_response_h100"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="24G"
    TIME="48:00:00"
    JOB_SCRIPT="eval/ept/ept_response.sh"

    GRES_ARGS=(--gpus-per-node=1)
    EPT_ARRAY_MAX=$((${EPT_REPEATS:-5} * 5 - 1))
    ARRAY_ARGS=(--array=0-${EPT_ARRAY_MAX})
    EXTRA_SBATCH_ARGS=()

    OUTPUT="logs/eval/ept/${SAFE_STUDENT_NAME}/response/%x_%A_%a.out"
    ERROR="logs/eval/ept/${SAFE_STUDENT_NAME}/response/%x_%A_%a.err"
    ;; 

  repacss:ept_student)
    JOB_NAME="ept_${STUDENT}_base_h100"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="24G"
    TIME="48:00:00"
    JOB_SCRIPT="eval/ept/ept_student.sh"

    GRES_ARGS=(--gpus-per-node=1)
    EPT_ARRAY_MAX=$((${EPT_REPEATS:-5} - 1))
    ARRAY_ARGS=(--array=0-${EPT_ARRAY_MAX})
    EXTRA_SBATCH_ARGS=()

    OUTPUT="logs/eval/ept/${SAFE_STUDENT_NAME}/BASE/%x_%A_%a.out"
    ERROR="logs/eval/ept/${SAFE_STUDENT_NAME}/BASE/%x_%A_%a.err"
    ;; 

  repacss:ept_teacher)
    JOB_NAME="ept_${TEACHER}_base_h100"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="24G"
    TIME="48:00:00"
    JOB_SCRIPT="eval/ept/ept_teacher.sh"

    GRES_ARGS=(--gpus-per-node=1)
    EPT_ARRAY_MAX=$((${EPT_REPEATS:-5} - 1))
    ARRAY_ARGS=(--array=0-${EPT_ARRAY_MAX})
    EXTRA_SBATCH_ARGS=()

    OUTPUT="logs/eval/ept/${SAFE_TEACHER_NAME}/BASE/%x_%A_%a.out"
    ERROR="logs/eval/ept/${SAFE_TEACHER_NAME}/BASE/%x_%A_%a.err"
    ;; 

  repacss:ept_trad_student)
    JOB_NAME="ept_trad_student_sft_h100"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="4"
    CPUS_PER_TASK="16"
    MEM="24G"
    TIME="48:00:00"
    JOB_SCRIPT="eval/ept/ept_Traditional_student.sh"

    GRES_ARGS=(--gpus-per-node=1)
    EPT_ARRAY_MAX=$((${EPT_REPEATS:-5} * 5 - 1))
    ARRAY_ARGS=(--array=0-${EPT_ARRAY_MAX})
    EXTRA_SBATCH_ARGS=()

    OUTPUT="logs/eval/ept/${SAFE_STUDENT_NAME}/traditional/raw/%x_%A_%a.out"
    ERROR="logs/eval/ept/${SAFE_STUDENT_NAME}/traditional/raw/%x_%A_%a.err"
    ;; 

  repacss:hardness_submitter)
    JOB_NAME="harness_submitter"
    PARTITION=$CPU
    NODES="1"
    GPUS_PER_NODE="0"
    CPUS_PER_TASK="1"
    MEM="4G"
    TIME="00:04:00"
    JOB_SCRIPT="eval/hardness_submitter.sh"

    EXTRA_SBATCH_ARGS=()

    OUTPUT="logs/eval/harness/${SAFE_STUDENT_NAME}/submission/%x_%j.out"
    ERROR="logs/eval/harness/${SAFE_STUDENT_NAME}/submission/%x_%j.err"
    ;;
  
  repacss:traditional_eval)
    JOB_NAME="Traditional_${SAFE_STUDENT_NAME}_harness"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="1"
    CPUS_PER_TASK="6"
    MEM="32G"
    TIME="48:00:00"
    JOB_SCRIPT="traditional-model/slurm/eval_8B_submitter.sh"
    GRES_ARGS=(--gpus-per-node=1)
    HARNESS_REPEATS="${HARNESS_REPEATS:-${EPT_REPEATS:-5}}"
    HARNESS_ARRAY_MAX=$((HARNESS_REPEATS * 5 - 1))
    ARRAY_ARGS=(--array=0-${HARNESS_ARRAY_MAX})
    OUTPUT="logs/eval/harness/${SAFE_STUDENT_NAME}/traditional/raw/%x_%A_%a.out"
    ERROR="logs/eval/harness/${SAFE_STUDENT_NAME}/traditional/raw/%x_%A_%a.err"
  ;;

  repacss:teacher_harness)
    JOB_NAME="base_${SAFE_TEACHER_NAME}_harness"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="1"
    CPUS_PER_TASK="6"
    MEM="32G"
    TIME="48:00:00"
    JOB_SCRIPT="Base/Teacher_harness.sh"
    GRES_ARGS=(--gpus-per-node=1)
    OUTPUT="logs/eval/harness/${SAFE_TEACHER_NAME}/BASE/%x_%j.out"
    ERROR="logs/eval/harness/${SAFE_TEACHER_NAME}/BASE/%x_%j.err"
  ;;

  repacss:student_harness)
    JOB_NAME="base_${SAFE_STUDENT_NAME}_harness"
    PARTITION=$GPU
    NODES="1"
    GPUS_PER_NODE="1"
    CPUS_PER_TASK="6"
    MEM="32G"
    TIME="48:00:00"
    JOB_SCRIPT="Base/Student_harness.sh"
    GRES_ARGS=(--gpus-per-node=1)
    OUTPUT="logs/eval/harness/${SAFE_STUDENT_NAME}/BASE/%x_%j.out"
    ERROR="logs/eval/harness/${SAFE_STUDENT_NAME}/BASE/%x_%j.err"
  ;;


# ---------------------------------
# TEXAS TECH HPCC A100(Quanah);;
# ---------------------------------
  hpcc:env)
    JOB_NAME="env_setup"
    PARTITION="cpu"   # TODO: change if HPCC CPU partition has another name
    NODES="1"
    GPUS_PER_NODE="0"
    CPUS_PER_TASK="1"
    MEM="1G"
    TIME="00:20:00"
    JOB_SCRIPT="scripts/_env_single_node.sh"

    OUTPUT="logs/env_setup/%x_%j.out"
    ERROR="logs/env_setup/%x_%j.err"
    ;;

  hpcc:build_shards)
    JOB_NAME="build_shards"
    PARTITION="cpu"
    NODES="1"
    GPUS_PER_NODE="0"
    CPUS_PER_TASK="8"
    MEM="1G"
    TIME="06:00:00"
    JOB_SCRIPT="scripts/_env_single_node.sh"

    OUTPUT="logs/shards/%x_%j.out"
    ERROR="logs/shards/%x_%j.err"
    ;;


  hpcc:build_caches)
    JOB_NAME="kd_build_caches_${SAFE_TEACHER_NAME}"
    PARTITION="$GPU"
    NODES="1"
    GPUS_PER_NODE="3"
    CPUS_PER_TASK="16"
    MEM="128G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/build_caches.sh"

    GRES_ARGS=(--gres=gpu:a100:3)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )

    OUTPUT="logs/build_cache/%x_%A_%a.out"
    ERROR="logs/build_cache/%x_%A_%a.err"
    ;;

  hpcc:feature)
    JOB_NAME="kd_feature_${SAFE_STUDENT_NAME}_from_${SAFE_TEACHER_NAME}"
    PARTITION="$GPU"
    NODES="1"
    GPUS_PER_NODE="3"
    CPUS_PER_TASK="16"
    MEM="128G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/run_feature_based_single_node.sh"

    GRES_ARGS=(--gres=gpu:a100:3)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )

    OUTPUT="logs/feature/%x_%A_%a.out"
    ERROR="logs/feature/%x_%A_%a.err"
    ;;

  hpcc:relation)
    JOB_NAME="kd_relation_${SAFE_STUDENT_NAME}_from_${SAFE_TEACHER_NAME}"
    PARTITION="$GPU"
    NODES="1"
    GPUS_PER_NODE="3"
    CPUS_PER_TASK="16"
    MEM="128G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/run_relation_based_single_node.sh"

    GRES_ARGS=(--gpus-per-node=3)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )

    OUTPUT="logs/relation/%x_%A_%a.out"
    ERROR="logs/relation/%x_%A_%a.err"
    ;;

  hpcc:response)
    JOB_NAME="kd_response_${SAFE_STUDENT_NAME}_from_${SAFE_TEACHER_NAME}"
    PARTITION="$GPU"
    NODES="1"
    GPUS_PER_NODE="3"
    CPUS_PER_TASK="16"
    MEM="128G"
    TIME="48:00:00"
    JOB_SCRIPT="scripts/run_response_based_single_node.sh"

    GRES_ARGS=(--gpus-per-node=3)
    ARRAY_ARGS=(--array=0-4)
    EXTRA_SBATCH_ARGS=(
      --exclusive
      --signal=B:SIGUSR1@300
      --requeue
    )

    OUTPUT="logs/response/%x_%A_%a.out"
    ERROR="logs/response/%x_%A_%a.err"
    ;;

  *)
    echo "[ERROR] Unknown target/job kind: $TARGET:$JOB_KIND"
    echo "Examples:"
    echo "  bash submit.sh repacss env"
    echo "  bash submit.sh repacss build_shards"
    echo "  bash submit.sh repacss build_caches"
    echo "  bash submit.sh repacss feature"
    echo "  bash submit.sh repacss relation"
    echo "  bash submit.sh repacss response"
    echo "  bash submit.sh hpcc env"
    echo "  bash submit.sh hpcc build_shards"
    echo "  bash submit.sh hpcc build_caches"
    echo "  bash submit.sh hpcc feature"
    echo "  bash submit.sh hpcc relation"
    echo "  bash submit.sh hpcc response"
    exit 1
    ;;
esac

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$ERROR")"

sbatch \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --nodes="$NODES" \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --time="$TIME" \
  --output="$OUTPUT" \
  --error="$ERROR" \
  "${GRES_ARGS[@]}" \
  "${ARRAY_ARGS[@]}" \
  "${EXTRA_SBATCH_ARGS[@]}" \
  "${DEPENDENCY_ARGS[@]}" \
  --export=ALL,TARGET="$TARGET",JOB_KIND="$JOB_KIND",GPUS_PER_NODE="$GPUS_PER_NODE",PROCS_PER_GPU="$PROCS_PER_GPU",ENV_NAME="kd",TEACHER="$TEACHER",STUDENT="$STUDENT",STUDENT_MODEL="$STUDENT",TEACHER_DATA="$TEACHER_DATA",SAFE_STUDENT_NAME="$SAFE_STUDENT_NAME",SAFE_TEACHER_NAME="$SAFE_TEACHER_NAME" \
  "$JOB_SCRIPT" "$@"
