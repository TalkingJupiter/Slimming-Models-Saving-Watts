#!/usr/bin/env bash
#SBATCH --job-name=baseline_8B_sft
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=traditional-model/logs/sft_%j.out
#SBATCH --error=traditional-model/logs/sft_%j.err

set -euo pipefail
cd ${SLURM_SUBMIT_DIR:-$PWD}

source ~/.bashrc

export CUDA_HOME=/opt/apps/nfs/spack-1.0.1/opt/spack/linux-sapphirerapids/cuda-12.9.0-u2ppthmoi4r6ddyxdqesrj5oqja3byph
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "CUDA_HOME in job: $CUDA_HOME"
which nvcc || echo "nvcc not found in PATH"

conda activate kd

python monitor.py --output traditional-model/telemetry/telemetry_sft.jsonl --interval 1 &
MONITOR_PID=$!

accelerate launch traditional-model/train_sft.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --shards_file "data/shards.jsonl" \
  --output_dir "traditional-model/checkpoints/" \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 1e-5 \
  --num_epochs 1 \
  --max_length 2048

kill $MONITOR_PID
