#!/usr/bin/env bash
#SBATCH --job-name=ept_feature_students
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=eval/ept/benchmark/logs/%x_%j.out
#SBATCH --error=eval/ept/benchmark/logs/%x_%j.err

set -euo pipefail

FEATURE_MODELS=(
    "serialization_dir/feature/Model1"
    "serialization_dir/feature/Model2"
    "serialization_dir/feature/Model3"
    "serialization_dir/feature/Model4"
    "serialization_dir/feature/Model5"
)

NUM_PROMPTS=100
BATCH_SIZE=2

OUTDIR_ROOT="eval/ept/benchmark"
LOGDIR="${OUTDIR_ROOT}/logs"
RESULTSDIR="${OUTDIR_ROOT}/results"
OUTDIR="${RESULTSDIR}/feature_${SLURM_JOB_ID}"

mkdir -p "$LOGDIR" "$RESULTSDIR" "$OUTDIR"

echo "===================================================="
echo "   EPT-Bench: Energy-Per-Token — FEATURE STUDENTS"
echo "===================================================="
echo "[EPT] Prompt count   : $NUM_PROMPTS"
echo "[EPT] Batch size     : $BATCH_SIZE"
echo "[EPT] Output dir     : $OUTDIR"
echo "===================================================="

source ~/.bashrc || true
conda activate kd || true

run_ept_for_model () {
    local GPU_SLOT="$1"     # 0..3 physical GPU slot
    local KD_LABEL="$2"
    local INDEX="$3"
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
}

echo "================ FEATURE-BASED KD (FB) =============="

running_pids=()
running_gpus=()

start_job () {
    local gpu="$1"
    local idx="$2"
    local model="$3"

    run_ept_for_model "$gpu" "FB" "$idx" "$model" &
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
for MODEL in "${FEATURE_MODELS[@]}"; do
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
echo "[EPT] All FEATURE KD models have been processed."
echo "[EPT] Results directory: ${OUTDIR}"
echo "===================================================="