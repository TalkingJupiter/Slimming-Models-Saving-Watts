#!/usr/bin/env bash

set -euo pipefail

STUDENT_MODEL=${STUDENT:-"meta-llama/Llama-3.1-8B"}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-${STUDENT_MODEL//\//_}}

TRAD_ROOT="traditional_student/${SAFE_STUDENT_NAME}"
TRAD_MODELS=()
if [[ -d "$TRAD_ROOT" ]]; then
    mapfile -t TRAD_MODELS < <(
        find "$TRAD_ROOT" -mindepth 2 -maxdepth 2 -type d -name final | sort
    )
fi

if [[ ${#TRAD_MODELS[@]} -eq 0 ]]; then
    echo "[WARN] No traditional SFT final models found under $TRAD_ROOT"
    exit 0
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$TASK_ID" -ge "${#TRAD_MODELS[@]}" ]]; then
    echo "[WARN] No traditional SFT model for SLURM_ARRAY_TASK_ID=$TASK_ID"
    exit 0
fi

MODEL="${TRAD_MODELS[$TASK_ID]}"
RUN_ID="$(basename "$(dirname "$MODEL")")"
NUM_PROMPTS=100
BATCH_SIZE=4
GPU_INDEX=0

JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
# OUTDIR="eval/ept/benchmark/results/traditional_${JOB_GROUP_ID}"
OUTDIR="results/${SAFE_STUDENT_NAME}/traditional/${TASK_ID}/EPT"
OUTFILE="${OUTDIR}/ept_traditional_${SLURM_ARRAY_TASK_ID}.json"

mkdir -p eval/ept/benchmark "$OUTDIR" logs results || true

echo "===================================================="
echo "           EPT-Bench: Energy-Per-Token"
echo "===================================================="
echo "[EPT] Student        : $STUDENT_MODEL"
echo "[EPT] Run ID         : $RUN_ID"
echo "[EPT] Model          : $MODEL"
echo "[EPT] Prompts        : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] GPU index      : $GPU_INDEX"
echo "[EPT] Output file    : $OUTFILE"
echo "[EPT] SLURM Job ID   : ${SLURM_JOB_ID:-manual}"
echo "[EPT] Array task     : ${SLURM_ARRAY_TASK_ID:-none}"
echo "----------------------------------------------------"

source ~/.bashrc || true
conda activate kd || true

python eval/ept/benchmark/run_ept_benchmark.py \
    --model "$MODEL" \
    --use-dolly \
    --num-prompts "$NUM_PROMPTS" \
    --batch-size "$BATCH_SIZE" \
    --gpu-indices "$GPU_INDEX" \
    --out "$OUTFILE"

echo "[EPT] DONE: traditional SFT run $RUN_ID"
