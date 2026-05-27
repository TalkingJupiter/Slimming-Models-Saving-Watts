#!/usr/bin/env bash

set -euo pipefail

STUDENT_MODEL=${STUDENT:-"meta-llama/Llama-3.1-8B"}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-${STUDENT_MODEL//\//_}}

RELATION_MODELS=()
if [[ -d "serialization_dir/${SAFE_STUDENT_NAME}/relation" ]]; then
    mapfile -t RELATION_MODELS < <(
        find "serialization_dir/${SAFE_STUDENT_NAME}/relation" -mindepth 1 -maxdepth 1 -type d | sort
    )
fi

if [[ ${#RELATION_MODELS[@]} -eq 0 ]]; then
    echo "[WARN] No relation KD models found under serialization_dir/${SAFE_STUDENT_NAME}/relation"
    exit 0
fi

NUM_PROMPTS=100
BATCH_SIZE=4
REPEATS=${EPT_REPEATS:-5}
GPU_INDEX=0

JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUTDIR="results/${SAFE_STUDENT_NAME}/relation/"

mkdir -p "$OUTDIR" logs || true

echo "===================================================="
echo "   EPT-Bench: Energy-Per-Token — RELATION STUDENTS"
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
    local KD_LABEL="$1"     # REL
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
# Run one Relation-based KD model/repeat per Slurm array task
# -----------------------------------------------------
echo "================ RELATION-BASED KD (REL) =============="
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_INDEX=$((TASK_ID / REPEATS))
REPEAT_INDEX=$((TASK_ID % REPEATS + 1))

if [[ "$MODEL_INDEX" -ge "${#RELATION_MODELS[@]}" ]]; then
    echo "[WARN] No relation KD model for SLURM_ARRAY_TASK_ID=$TASK_ID"
    exit 0
fi

run_ept_for_model "REL" "$((MODEL_INDEX + 1))" "$REPEAT_INDEX" "${RELATION_MODELS[$MODEL_INDEX]}"

echo "===================================================="
echo "[EPT] RELATION KD array task complete."
echo "[EPT] Results directory: ${OUTDIR}"
echo "===================================================="
