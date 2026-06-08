#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_harness_dataset_warmup
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

source ~/.bashrc || true
conda activate kd || true
source scripts/_env_single_node.sh

HARNESS_TASKS="${HARNESS_TASKS:-mmlu,hellaswag,bbh,arc_challenge}"
HARNESS_DATASETS_CACHE="${HARNESS_DATASETS_CACHE:-$HF_HOME/datasets_eval/harness_shared}"
LOCK_FILE="${HARNESS_DATASETS_CACHE}.lock"
MARKER_FILE="${HARNESS_DATASETS_CACHE}/.prepared_${HARNESS_TASKS//,/__}"

mkdir -p "$HARNESS_DATASETS_CACHE" "$(dirname "$LOCK_FILE")"

echo "[INFO] Harness tasks: $HARNESS_TASKS"
echo "[INFO] HF_DATASETS_CACHE: $HARNESS_DATASETS_CACHE"
echo "[INFO] Marker: $MARKER_FILE"

(
  flock 9
  if [[ -f "$MARKER_FILE" ]]; then
    echo "[INFO] Harness dataset cache already prepared."
    exit 0
  fi

  export HF_DATASETS_CACHE="$HARNESS_DATASETS_CACHE"
  unset HF_DATASETS_OFFLINE
  unset HF_HUB_OFFLINE

  python eval/harness_prepare_datasets.py --tasks "$HARNESS_TASKS"
  touch "$MARKER_FILE"
  echo "[INFO] Harness dataset cache prepared."
) 9>"$LOCK_FILE"
