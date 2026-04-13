#!/usr/bin/env bash
#SBATCH --job-name=ept_response_students
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=96:00:00
#SBATCH --output=eval/ept/benchmark/logs/%x_%j.out
#SBATCH --error=eval/ept/benchmark/logs/%x_%j.err

set -euo pipefail

# -----------------------------------------------------
# Manual Configuration (EDIT THESE DIRECTLY)
# -----------------------------------------------------

RESPONSE_MODELS=(
    "serialization_dir/response/Model1"
    "serialization_dir/response/Model2"
    "serialization_dir/response/Model3"
    "serialization_dir/response/Model4"
    "serialization_dir/response/Model5"
)

NUM_PROMPTS=100
BATCH_SIZE=2

OUTDIR_ROOT="eval/ept/benchmark"
LOGDIR="${OUTDIR_ROOT}/logs"
RESULTSDIR="${OUTDIR_ROOT}/results"
OUTDIR="${RESULTSDIR}/response_${SLURM_JOB_ID}"

mkdir -p "$LOGDIR" "$RESULTSDIR" "$OUTDIR"

echo "===================================================="
echo "   EPT-Bench: Energy-Per-Token — RESPONSE STUDENTS"
echo "===================================================="
echo "[EPT] Prompt count   : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] Output dir     : $OUTDIR"
echo "===================================================="

# -----------------------------------------------------
# Activate Environment
# -----------------------------------------------------
source ~/.bashrc || true
conda activate kd || true

# -----------------------------------------------------
# Helper: run one model on one GPU
# -----------------------------------------------------
run_ept_for_model () {
    local GPU_SLOT="$1"     # physical GPU slot on the node: 0..3
    local KD_LABEL="$2"     # RESP
    local INDEX="$3"        # 1..5
    local MODEL="$4"

    local BASENAME
    BASENAME=$(basename "$MODEL")
    local OUTFILE="${OUTDIR}/ept_${KD_LABEL}_${INDEX}_${BASENAME}_${SLURM_JOB_ID}.json"

    echo "----------------------------------------------------"
    echo "[EPT] GPU slot      : ${GPU_SLOT}"
    echo "[EPT] KD Type       : ${KD_LABEL}"
    echo "[EPT] Model Index   : ${INDEX}"
    echo "[EPT] Model Path    : ${MODEL}"
    echo "[EPT] Output File   : ${OUTFILE}"
    echo "----------------------------------------------------"

    CUDA_VISIBLE_DEVICES="${GPU_SLOT}" \
    python eval/ept/benchmark/run_ept_benchmark.py \
        --model "$MODEL" \
        --use-dolly \
        --num-prompts "$NUM_PROMPTS" \
        --batch-size "$BATCH_SIZE" \
        --gpu-indices "0" \
        --out "$OUTFILE"

    echo "[EPT] DONE: ${KD_LABEL} model #${INDEX} (${MODEL})"
}

echo "================ RESPONSE-BASED KD (RESP) ============"

running_pids=()
running_gpus=()

start_job () {
    local gpu="$1"
    local idx="$2"
    local model="$3"

    run_ept_for_model "$gpu" "RESP" "$idx" "$model" &
    running_pids+=("$!")
    running_gpus+=("$gpu")
}

wait_for_one () {
    local finished_pid=""
    while true; do
        for i in "${!running_pids[@]}"; do
            local pid="${running_pids[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                FREE_GPU="${running_gpus[$i]}"
                unset 'running_pids[$i]'
                unset 'running_gpus[$i]'
                running_pids=("${running_pids[@]}")
                running_gpus=("${running_gpus[@]}")
                return 0
            fi
        done
        sleep 2
    done
}

FREE_GPU=0
i=1
for MODEL in "${RESPONSE_MODELS[@]}"; do
    if [ "${#running_pids[@]}" -lt 4 ]; then
        gpu_to_use="${#running_pids[@]}"
    else
        wait_for_one
        gpu_to_use="$FREE_GPU"
    fi

    start_job "$gpu_to_use" "$i" "$MODEL"
    i=$((i + 1))
done

for pid in "${running_pids[@]}"; do
    wait "$pid"
done

echo "===================================================="
echo "[EPT] All RESPONSE KD models have been processed."
echo "[EPT] Results directory: ${OUTDIR}"
echo "===================================================="