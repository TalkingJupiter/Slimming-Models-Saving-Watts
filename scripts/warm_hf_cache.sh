#!/usr/bin/env bash
set -euo pipefail

source scripts/_env_single_node.sh

TEACHER=${TEACHER:-meta-llama/Meta-Llama-3.1-70B-Instruct}
STUDENT=${STUDENT:-meta-llama/Meta-Llama-3.1-8B-Instruct}
HF_WARM_MAX_WORKERS=${HF_WARM_MAX_WORKERS:-1}
HF_DOWNLOAD_CACHE_ROOT=${HF_DOWNLOAD_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp}/huggingface-warm-${USER:-user}}

warm_one() {
  local model="$1"
  local safe_name
  safe_name=$(safe_hf_model_name "$model")
  local model_dir="${HF_MODEL_ROOT:-$PROJECT_ROOT/.hf_models}/$safe_name"

  export MODEL_TO_WARM="$model" HF_DOWNLOAD_CACHE_ROOT HF_WARM_MAX_WORKERS
  mkdir -p "$HF_DOWNLOAD_CACHE_ROOT" "$(dirname "$model_dir")"

  echo "[INFO] Warming Hugging Face model directory"
  echo "[INFO] Model: $model"
  echo "[INFO] Download cache: $HF_DOWNLOAD_CACHE_ROOT/hub"
  echo "[INFO] Local model dir: $model_dir"
  echo "[INFO] HF_WARM_MAX_WORKERS: $HF_WARM_MAX_WORKERS"

  local snapshot_path
  snapshot_path=$(python - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ["MODEL_TO_WARM"]
cache_dir = os.path.join(os.environ["HF_DOWNLOAD_CACHE_ROOT"], "hub")
max_workers = int(os.environ.get("HF_WARM_MAX_WORKERS", "1"))

path = snapshot_download(
    repo_id=model,
    cache_dir=cache_dir,
    max_workers=max_workers,
    resume_download=True,
)
print(path)
PY
)

  echo "[INFO] Snapshot downloaded: $snapshot_path"
  echo "[INFO] Copying dereferenced snapshot into $model_dir"
  rm -rf "$model_dir.tmp" "$model_dir.new"
  mkdir -p "$model_dir.tmp"
  cp -aL "$snapshot_path"/. "$model_dir.tmp"/
  mv "$model_dir.tmp" "$model_dir.new"
  rm -rf "$model_dir"
  mv "$model_dir.new" "$model_dir"
  echo "[COMPLETED] Hugging Face model warm complete: $model_dir"
}

warm_one "$TEACHER"
warm_one "$STUDENT"
