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
REPEATS=${EPT_REPEATS:-5}

if [[ "$REPEATS" -lt 1 ]]; then
    echo "[ERROR] EPT_REPEATS must be >= 1, got: $REPEATS" >&2
    exit 2
fi

MODEL_INDEX=$((TASK_ID / REPEATS))
REPEAT_INDEX=$((TASK_ID % REPEATS + 1))

if [[ "$MODEL_INDEX" -ge "${#TRAD_MODELS[@]}" ]]; then
    echo "[WARN] No traditional SFT model for SLURM_ARRAY_TASK_ID=$TASK_ID MODEL_INDEX=$MODEL_INDEX"
    exit 0
fi

MODEL="${TRAD_MODELS[$MODEL_INDEX]}"
RUN_ID="$(basename "$(dirname "$MODEL")")"
NUM_PROMPTS=${EPT_NUM_PROMPTS:-100}
BATCH_SIZE=${EPT_BATCH_SIZE:-4}
GPU_INDEX=${EPT_GPU_INDICES:-0}
MAX_NEW_TOKENS_LIST=${EPT_MAX_NEW_TOKENS_LIST:-32,64,128,256,512,1024}
WARMUP_BATCHES=${EPT_WARMUP_BATCHES:-2}
SAMPLE_INTERVAL=${EPT_SAMPLE_INTERVAL:-1.0}
EPT_PROMPTS=${EPT_PROMPTS:-}

JOB_GROUP_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
# OUTDIR="eval/ept/benchmark/results/traditional_${JOB_GROUP_ID}"
OUTDIR="results/${SAFE_STUDENT_NAME}/traditional/$MODEL_INDEX/EPT"
OUTFILE="${OUTDIR}/ept_repeat${REPEAT_INDEX}.json"

mkdir -p eval/ept/benchmark "$OUTDIR" logs results || true
STRUCTURED_LOG_DIR="logs/eval/ept/${SAFE_STUDENT_NAME}/traditional/$MODEL_INDEX"
STRUCTURED_LOG_BASE="${STRUCTURED_LOG_DIR}/repeat${REPEAT_INDEX}_${JOB_GROUP_ID}_${TASK_ID}"
mkdir -p "$STRUCTURED_LOG_DIR"
exec > >(tee -a "${STRUCTURED_LOG_BASE}.out") 2> >(tee -a "${STRUCTURED_LOG_BASE}.err" >&2)
echo "[INFO] Structured log: ${STRUCTURED_LOG_BASE}.{out,err}"

echo "===================================================="
echo "           EPT-Bench: Energy-Per-Token"
echo "===================================================="
echo "[EPT] Student        : $STUDENT_MODEL"
echo "[EPT] Run ID         : $RUN_ID"
echo "[EPT] Model index    : $MODEL_INDEX/${#TRAD_MODELS[@]}"
echo "[EPT] Repeat         : $REPEAT_INDEX/$REPEATS"
echo "[EPT] Model          : $MODEL"
echo "[EPT] Prompts        : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] GPU indices    : $GPU_INDEX"
echo "[EPT] Token sweep    : $MAX_NEW_TOKENS_LIST"
echo "[EPT] Warmup batches : $WARMUP_BATCHES"
echo "[EPT] Sample interval: $SAMPLE_INTERVAL"
if [[ -n "$EPT_PROMPTS" ]]; then
    echo "[EPT] Prompt file    : $EPT_PROMPTS"
else
    echo "[EPT] Prompt source  : Dolly seed 42"
fi
echo "[EPT] Output file    : $OUTFILE"
echo "[EPT] SLURM Job ID   : ${SLURM_JOB_ID:-manual}"
echo "[EPT] Array task     : ${SLURM_ARRAY_TASK_ID:-none}"
echo "----------------------------------------------------"

source ~/.bashrc || true
conda activate kd || true
PROMPT_ARGS=()
if [[ -n "$EPT_PROMPTS" ]]; then
    PROMPT_ARGS+=(--prompts "$EPT_PROMPTS")
else
    PROMPT_ARGS+=(--use-dolly)
fi

python eval/ept/benchmark/run_ept_benchmark.py \
    --model "$MODEL" \
    --method "traditional" \
    --checkpoint "$MODEL" \
    "${PROMPT_ARGS[@]}" \
    --num-prompts "$NUM_PROMPTS" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens-list "$MAX_NEW_TOKENS_LIST" \
    --warmup-batches "$WARMUP_BATCHES" \
    --sample-interval "$SAMPLE_INTERVAL" \
    --gpu-indices "$GPU_INDEX" \
    --out "$OUTFILE"

echo "[EPT] DONE: traditional SFT run $RUN_ID repeat $REPEAT_INDEX/$REPEATS"
