#!/usr/bin/env bash
set -euo pipefail

source scripts/_env_single_node.sh

IN=${IN:-data/shards.jsonl}
TEACHER=${TEACHER:-}
SAFE_TEACHER_NAME=${SAFE_TEACHER_NAME:-${TEACHER//\//_}}
TEACHER_DATA=${TEACHER_DATA:-$SAFE_TEACHER_NAME}
RESPONSE_CACHE_SHARD_INDEX=${RESPONSE_CACHE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
RESPONSE_CACHE_NUM_SHARDS=${RESPONSE_CACHE_NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
TELEMETRY_OUTPUT="${TELEMETRY_OUTPUT:-results/cache/telemetry/$TEACHER_DATA/response_${RESPONSE_CACHE_SHARD_INDEX}.jsonl}"

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Teacher data dir: data/$TEACHER_DATA"
echo "[INFO] INPUT: $IN"
echo "[INFO] Response cache shard: $RESPONSE_CACHE_SHARD_INDEX / $RESPONSE_CACHE_NUM_SHARDS"
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
  --shard_index "$RESPONSE_CACHE_SHARD_INDEX" \
  --num_shards "$RESPONSE_CACHE_NUM_SHARDS" \
  --telemetry \
  --telemetry_output "$TELEMETRY_OUTPUT" \
  --telemetry_interval 1

echo "[COMPLETED] $TEACHER response cache build complete: data/$TEACHER_DATA"
