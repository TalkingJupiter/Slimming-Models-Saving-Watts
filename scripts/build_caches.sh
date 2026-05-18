#!/usr/bin/env bash
set -euo pipefail

source scripts/_env_single_node.sh

IN=${IN:-data/shards.jsonl}
TEACHER=${TEACHER:-}
SAFE_TEACHER_NAME=${SAFE_TEACHER_NAME:-${TEACHER//\//_}}
TEACHER_DATA=${TEACHER_DATA:-$SAFE_TEACHER_NAME}

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Teacher data dir: data/$TEACHER_DATA"
echo "[INFO] Input:   $IN"
echo "[INFO] GPUs visible: ${CUDA_VISIBLE_DEVICES:-unset}"

# Fail fast if input is missing/empty
if [[ ! -s "$IN" ]]; then
  echo "[ERROR] Input jsonl '$IN' not found or empty." >&2
  exit 2
fi

# Make sure output dirs exist
mkdir -p data/$TEACHER_DATA/topk_k16 data/$TEACHER_DATA/fb_hints_L22 data/$TEACHER_DATA/relb_embeds

# ---- RB top-k caches
python teacher_farm/make_topk_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/$TEACHER_DATA/topk_k16/ \
  --k 16 \
  --dtype bfloat16

# ---- FB hidden-state caches (e.g., teacher layer 22)
python teacher_farm/make_hidden_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/$TEACHER_DATA/fb_hints_L22/ \
  --layers 22 \
  --batch_size 1 \
  --max_length 2048 \
  --dtype bfloat16 \
  --flush_every 256


# ---- RelB pooled embedding caches
python teacher_farm/make_embed_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/$TEACHER_DATA/relb_embeds/

echo "[INFO] $TEACHER Cache build complete: data/$TEACHER_DATA"
