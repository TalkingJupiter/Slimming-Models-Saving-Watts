#!/usr/bin/env bash
set -euo pipefail

STUDENT_MODEL=${STUDENT:-"meta-llama/Llama-3.1-8B"}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-${STUDENT_MODEL//\//_}}

FEATURE_MODELS=()
if [[ -d "serialization_dir/${SAFE_STUDENT_NAME}/feature" ]]; then
    mapfile -t FEATURE_MODELS < <(
        find "serialization_dir/${SAFE_STUDENT_NAME}/feature" -mindepth 1 -maxdepth 1 -type d | sort
    )
fi

if [[ ${#FEATURE_MODELS[@]} -eq 0 ]]; then
    echo "[WARN] No feature KD models found under serialization_dir/${SAFE_STUDENT_NAME}/feature"
    exit 0
fi

NUM_PROMPTS=${EPT_NUM_PROMPTS:-100}        # number of prompts
BATCH_SIZE=${EPT_BATCH_SIZE:-4}           # batch size for students
REPEATS=${EPT_REPEATS:-5}
GPU_INDEX=${EPT_GPU_INDICES:-0}            # GPU index or comma-separated indices to monitor
MAX_NEW_TOKENS_LIST=${EPT_MAX_NEW_TOKENS_LIST:-32,64,128,256,512,1024}
WARMUP_BATCHES=${EPT_WARMUP_BATCHES:-2}
SAMPLE_INTERVAL=${EPT_SAMPLE_INTERVAL:-1.0}
EPT_PROMPTS=${EPT_PROMPTS:-}

JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
# OUTDIR="results/eval/harness/${SAFE_STUDENT_NAME}/feature_${JOB_GROUP_ID}"
OUTDIR="results/${SAFE_STUDENT_NAME}/feature/"

mkdir -p "$OUTDIR" logs || true

echo "===================================================="
echo "   EPT-Bench: Energy-Per-Token — FEATURE STUDENTS"
echo "===================================================="
echo "[EPT] Prompt count   : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] Repeats/model  : $REPEATS"
echo "[EPT] GPU indices    : $GPU_INDEX"
echo "[EPT] Token sweep    : $MAX_NEW_TOKENS_LIST"
echo "[EPT] Warmup batches : $WARMUP_BATCHES"
echo "[EPT] Sample interval: $SAMPLE_INTERVAL"
if [[ -n "$EPT_PROMPTS" ]]; then
    echo "[EPT] Prompt file    : $EPT_PROMPTS"
else
    echo "[EPT] Prompt source  : Dolly seed 42"
fi
echo "[EPT] Output dir     : $OUTDIR"
echo "===================================================="

# -----------------------------------------------------
# Activate Environment
# -----------------------------------------------------
source ~/.bashrc || true
conda activate kd || true
source scripts/_env_single_node.sh
PROMPT_ARGS=()
if [[ -n "$EPT_PROMPTS" ]]; then
    PROMPT_ARGS+=(--prompts "$EPT_PROMPTS")
else
    PROMPT_ARGS+=(--use-dolly)
fi
STUDENT_MODEL_SOURCE=$(resolve_hf_model "$STUDENT_MODEL")
echo "[EPT] Student source : $STUDENT_MODEL_SOURCE"

# -----------------------------------------------------
# Helper: run one model with a KD label and index
# -----------------------------------------------------
run_ept_for_model () {
    local KD_LABEL="$1"     # e.g., FB
    local INDEX="$2"        # 1..5
    local REPEAT="$3"       # 1..REPEATS
    local MODEL="$4"

    local OUTFILE="${OUTDIR}/${INDEX}/EPT/ept_repeat${REPEAT}.json"

    echo "----------------------------------------------------"
    echo "[EPT] KD Type       : ${KD_LABEL}"
    echo "[EPT] Model Index   : ${INDEX}"
    echo "[EPT] Repeat        : ${REPEAT}/${REPEATS}"
    echo "[EPT] Model Path    : ${MODEL}"
    echo "[EPT] Output File   : ${OUTFILE}"
    echo "----------------------------------------------------"

    python eval/ept/benchmark/run_ept_benchmark.py \
        --model "$MODEL" \
        --base-model "$STUDENT_MODEL_SOURCE" \
        --adapter "$MODEL" \
        --method "feature" \
        --checkpoint "$MODEL" \
        "${PROMPT_ARGS[@]}" \
        --num-prompts "$NUM_PROMPTS" \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens-list "$MAX_NEW_TOKENS_LIST" \
        --warmup-batches "$WARMUP_BATCHES" \
        --sample-interval "$SAMPLE_INTERVAL" \
        --gpu-indices "$GPU_INDEX" \
        --out "$OUTFILE"

    echo "[EPT] DONE: ${KD_LABEL} model #${INDEX}, repeat ${REPEAT}/${REPEATS} (${MODEL})"
}

# -----------------------------------------------------
# Run one Feature-based KD model/repeat per Slurm array task
# -----------------------------------------------------
echo "================ FEATURE-BASED KD (FB) =============="
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_INDEX=$((TASK_ID / REPEATS))
REPEAT_INDEX=$((TASK_ID % REPEATS + 1))

if [[ "$MODEL_INDEX" -ge "${#FEATURE_MODELS[@]}" ]]; then
    echo "[WARN] No feature KD model for SLURM_ARRAY_TASK_ID=$TASK_ID"
    exit 0
fi

JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
STRUCTURED_LOG_DIR="logs/eval/ept/${SAFE_STUDENT_NAME}/feature/$MODEL_INDEX"
STRUCTURED_LOG_BASE="${STRUCTURED_LOG_DIR}/repeat${REPEAT_INDEX}_${JOB_GROUP_ID}_${TASK_ID}"
mkdir -p "$STRUCTURED_LOG_DIR"
exec > >(tee -a "${STRUCTURED_LOG_BASE}.out") 2> >(tee -a "${STRUCTURED_LOG_BASE}.err" >&2)
echo "[INFO] Structured log: ${STRUCTURED_LOG_BASE}.{out,err}"

run_ept_for_model "FB" "$MODEL_INDEX" "$REPEAT_INDEX" "${FEATURE_MODELS[$MODEL_INDEX]}"

echo "===================================================="
echo "[EPT] FEATURE KD array task complete."
echo "[EPT] Results directory: ${OUTDIR}"
echo "===================================================="
