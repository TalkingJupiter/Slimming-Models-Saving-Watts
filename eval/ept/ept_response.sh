#!/usr/bin/env bash

set -euo pipefail

STUDENT_MODEL=${STUDENT:-"meta-llama/Llama-3.1-8B"}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-${STUDENT_MODEL//\//_}}

RESPONSE_MODELS=()
if [[ -d "serialization_dir/${SAFE_STUDENT_NAME}/response" ]]; then
    mapfile -t RESPONSE_MODELS < <(
        find "serialization_dir/${SAFE_STUDENT_NAME}/response" -mindepth 1 -maxdepth 1 -type d | sort
    )
fi

if [[ ${#RESPONSE_MODELS[@]} -eq 0 ]]; then
    echo "[WARN] No response KD models found under serialization_dir/${SAFE_STUDENT_NAME}/response"
    exit 0
fi

NUM_PROMPTS=100
BATCH_SIZE=4
REPEATS=${EPT_REPEATS:-5}
GPU_INDEX=0

JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUTDIR="results/${SAFE_STUDENT_NAME}/response/"

mkdir -p "$OUTDIR" logs || true

echo "===================================================="
echo "   EPT-Bench: Energy-Per-Token — RESPONSE STUDENTS"
echo "===================================================="
echo "[EPT] Prompt count   : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] Repeats/model  : $REPEATS"
echo "[EPT] GPU index      : $GPU_INDEX"
echo "[EPT] Output dir     : $OUTDIR"
echo "===================================================="

# -----------------------------------------------------
# Activate Environment
# -----------------------------------------------------
source ~/.bashrc || true
conda activate kd || true
source scripts/_env_single_node.sh
STUDENT_MODEL_SOURCE=$(resolve_hf_model "$STUDENT_MODEL")
echo "[EPT] Student source : $STUDENT_MODEL_SOURCE"

# -----------------------------------------------------
# Helper
# -----------------------------------------------------
run_ept_for_model () {
    local KD_LABEL="$1"     # RESP
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
        --use-dolly \
        --num-prompts "$NUM_PROMPTS" \
        --batch-size "$BATCH_SIZE" \
        --gpu-indices "$GPU_INDEX" \
        --out "$OUTFILE"

    echo "[EPT] DONE: ${KD_LABEL} model #${INDEX}, repeat ${REPEAT}/${REPEATS} (${MODEL})"
}

# -----------------------------------------------------
# Run one Response-based KD model/repeat per Slurm array task
# -----------------------------------------------------
echo "================ RESPONSE-BASED KD (RESP) ============"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_INDEX=$((TASK_ID / REPEATS))
REPEAT_INDEX=$((TASK_ID % REPEATS + 1))

if [[ "$MODEL_INDEX" -ge "${#RESPONSE_MODELS[@]}" ]]; then
    echo "[WARN] No response KD model for SLURM_ARRAY_TASK_ID=$TASK_ID"
    exit 0
fi

run_ept_for_model "RESP" "$((MODEL_INDEX + 1))" "$REPEAT_INDEX" "${RESPONSE_MODELS[$MODEL_INDEX]}"

echo "===================================================="
echo "[EPT] RESPONSE KD array task complete."
echo "[EPT] Results directory: ${OUTDIR}"
echo "===================================================="
