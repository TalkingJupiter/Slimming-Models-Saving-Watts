#!/usr/bin/env bash

set -euo pipefail

# -------------------------------
# Conda environment setup
# -------------------------------
ENV_NAME="kd"
REQ_FILE="requirements.txt"

# Ensure conda is initialized
source ~/.bashrc

export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}

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
# Hugging Face cache
# -------------------------------
# Some shared filesystems reject the chmod/lock operations used by
# huggingface_hub, which shows up as os.fchmod(...): Invalid argument.
# Use a persistent project-local cache by default so array jobs share one
# warmed model snapshot. Override HF_HOME/HF_CACHE_ROOT if your cluster has a
# better persistent scratch filesystem.
HF_CACHE_ROOT=${HF_CACHE_ROOT:-$PROJECT_ROOT/.hf_cache}
export HF_HOME=${HF_HOME:-$HF_CACHE_ROOT}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_TRANSFER_CONCURRENCY=${HF_TRANSFER_CONCURRENCY:-4}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
echo "[INFO] HF_HOME: $HF_HOME"
echo "[INFO] HF_HUB_CACHE: $HF_HUB_CACHE"
echo "[INFO] HF_TRANSFER_CONCURRENCY: $HF_TRANSFER_CONCURRENCY"


safe_hf_model_name() {
  local model="$1"
  printf '%s' "${model//\//_}"
}

resolve_hf_model() {
  local model="$1"
  if [[ -z "$model" || -d "$model" ]]; then
    printf '%s
' "$model"
    return
  fi
  local safe
  safe=$(safe_hf_model_name "$model")
  local local_dir="$PROJECT_ROOT/.hf_models/$safe"
  if [[ -d "$local_dir" ]]; then
    printf '%s
' "$local_dir"
  else
    printf '%s
' "$model"
  fi
}

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
