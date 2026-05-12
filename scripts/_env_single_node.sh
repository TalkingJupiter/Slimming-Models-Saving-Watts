#!/usr/bin/env bash
#SBATCH --job-name=env_setup
#SBATCH --partition=zen4              # CPU partition to run the launcher itself
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=logs/env_setup/%x_%j.out
#SBATCH --error=logs/env_setup/%x_%j.err


set -euo pipefail

# -------------------------------
# Conda environment setup
# -------------------------------
ENV_NAME="kd"
REQ_FILE="requirements.txt"

# Ensure conda is initialized
source ~/.bashrc

if ! conda env list | grep -q "^$ENV_NAME "; then
  echo "[INFO] Conda env '$ENV_NAME' not found. Creating..."
  conda create -y -n $ENV_NAME python=3.10
  conda activate $ENV_NAME
  if [[ -f "$REQ_FILE" ]]; then
    echo "[INFO] Installing requirements..."
    pip install -r $REQ_FILE
  else
    echo "[WARN] $REQ_FILE not found, skipping requirements install"
  fi
else
  echo "[INFO] Conda env '$ENV_NAME' exists. Activating..."
  conda activate $ENV_NAME
fi

# -------------------------------
# Node/GPU topology
# -------------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-4}      # each REPACSS node has 4x H100
PROCS_PER_GPU=${PROCS_PER_GPU:-1}      # set >1 if you want multiple processes per GPU
NUM_PROCESSES=$(( GPUS_PER_NODE * PROCS_PER_GPU ))

# -------------------------------
# NCCL tuning for NVLink on H100
# -------------------------------
export NCCL_P2P_LEVEL=NVL
export NCCL_MIN_NCHANNELS=8
export NCCL_DEBUG=WARN

# -------------------------------
# Project dirs
# -------------------------------
mkdir -p logs results serialization_dir
