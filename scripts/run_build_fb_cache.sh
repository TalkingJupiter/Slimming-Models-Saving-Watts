#!/usr/bin/env bash
#SBATCH --job-name=kd_build_fb_cache
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=toreador
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/fb-cache/%x_%j.out
#SBATCH --error=logs/fb-cache/%x_%j.err

set -euo pipefail
source scripts/_env_single_node.sh

IN=${IN:-data/shards.jsonl}
TEACHER=${TEACHER:-meta-llama/Llama-3.1-70B-Instruct}

echo "[INFO] Stage:   FB hidden-state cache"
echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Input:   $IN"
echo "[INFO] GPUs visible: ${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -s "$IN" ]]; then
  echo "[ERROR] Input jsonl '$IN' not found or empty." >&2
  exit 2
fi

mkdir -p data/fb_hints_L22

mkdir -p logs/telemetry/FB_CACHE/$SLURM_JOB_ID
python monitor.py --output logs/telemetry/FB_CACHE/$SLURM_JOB_ID/${HOSTNAME}.jsonl --interval 1 &
MON_PID=$!

python teacher_farm/make_hidden_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/fb_hints_L22/ \
  --layers 22 \
  --batch_size 1 \
  --max_length 2048 \
  --dtype bfloat16 \
  --flush_every 256

echo "[INFO] FB hidden-state cache build complete"
