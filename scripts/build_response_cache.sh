#!/usr/bin/env bash
set -euo pipefail

source scripts/_env_single_node.sh

IN=${IN:-data/shards.jsonl}
TEACHER=${TEACHER:-}
SAFE_TEACHER_NAME=${SAFE_TEACHER_NAME:-${TEACHER//\//_}}
TEACHER_DATA=${TEACHER_DATA:-$SAFE_TEACHER_NAME}
TELEMETRY_OUTPUT="${TELEMETRY_OUTPUT:-results/cache/telemetry/$TEACHER_DATA/response.jsonl}"

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Teacher data dir: data/$TEACHER_DATA"
echo "[INFO] INPUT: $IN"
echo "[INFO] GPUs visible: ${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -s "$IN" ]]; then
  echo "[ERROR] Input jsonl '$IN' not found or empty" >&2
  exit 2
fi

mkdir -p data/$TEACHER_DATA/topk_k16

python teacher_farm/make_topk_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/$TEACHER_DATA/topk_k16/ \
  --k 16 \
  --dtype bfloat16 \
  --telemetry \
  --telemetry_output "$TELEMETRY_OUTPUT" \
  --telemetry_interval 1

echo "[COMPLETED] $TEACHER response cache build complete: data/$TEACHER_DATA"