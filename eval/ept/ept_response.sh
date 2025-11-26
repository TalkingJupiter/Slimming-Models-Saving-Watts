#!/usr/bin/env bash
#SBATCH --job-name=ept_response_students
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=96:00:00
#SBATCH --output=eval/ept/benchmark/logs/%x_%j.out
#SBATCH --error=eval/ept/benchmark/logs/%x_%j.err

set -euo pipefail

# -----------------------------------------------------
# Manual Configuration (EDIT THESE DIRECTLY)
# -----------------------------------------------------

RESPONSE_MODELS=(
    "serialization_dir/response/20251114_0121_RB_1n"
    "serialization_dir/response/20251114_0130_RB_1n"
    "serialization_dir/response/20251114_0141_RB_1n"
    "serialization_dir/response/20251114_0151_RB_1n"
    "serialization_dir/response/20251114_0202_RB_1n"
)

NUM_PROMPTS=100
BATCH_SIZE=4
GPU_INDEX=0

OUTDIR_ROOT="eval/ept/benchmark"
OUTDIR="${OUTDIR_ROOT}/results/response_${SLURM_JOB_ID}"

mkdir -p "$OUTDIR_ROOT" "$OUTDIR" logs results || true

echo "===================================================="
echo "   EPT-Bench: Energy-Per-Token — RESPONSE STUDENTS"
echo "===================================================="
echo "[EPT] Prompt count   : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] GPU index      : $GPU_INDEX"
echo "[EPT] Output dir     : $OUTDIR"
echo "===================================================="

# -----------------------------------------------------
# Activate Environment
# -----------------------------------------------------
source ~/.bashrc || true
conda activate kd || true

# -----------------------------------------------------
# Helper
# -----------------------------------------------------
run_ept_for_model () {
    local KD_LABEL="$1"     # RESP
    local INDEX="$2"        # 1..5
    local MODEL="$3"

    local BASENAME
    BASENAME=$(basename "$MODEL")
    local OUTFILE="${OUTDIR}/ept_${KD_LABEL}_${INDEX}_${BASENAME}_${SLURM_JOB_ID}.json"

    echo "----------------------------------------------------"
    echo "[EPT] KD Type       : ${KD_LABEL}"
    echo "[EPT] Model Index   : ${INDEX}"
    echo "[EPT] Model Path    : ${MODEL}"
    echo "[EPT] Output File   : ${OUTFILE}"
    echo "----------------------------------------------------"

    python eval/ept/benchmark/run_ept_benchmark.py \
        --model "$MODEL" \
        --use-dolly \
        --num-prompts "$NUM_PROMPTS" \
        --batch-size "$BATCH_SIZE" \
        --gpu-indices "$GPU_INDEX" \
        --out "$OUTFILE"

    echo "[EPT] DONE: ${KD_LABEL} model #${INDEX} (${MODEL})"
}

# -----------------------------------------------------
# Run Response-based KD models
# -----------------------------------------------------
echo "================ RESPONSE-BASED KD (RESP) ============"
i=1
for MODEL in "${RESPONSE_MODELS[@]}"; do
    run_ept_for_model "RESP" "$i" "$MODEL"
    i=$((i + 1))
done

echo "===================================================="
echo "[EPT] All RESPONSE KD models have been processed."
echo "[EPT] Results directory: ${OUTDIR}"
echo "===================================================="
