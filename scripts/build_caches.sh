#!/usr/bin/env bash
set -euo pipefail

source scripts/_env_single_node.sh

IN=${IN:-data/shards.jsonl}
TEACHER=${TEACHER:-}
SAFE_TEACHER_NAME=${SAFE_TEACHER_NAME:-${TEACHER//\//_}}
TEACHER_DATA=${TEACHER_DATA:-$SAFE_TEACHER_NAME}
HF_MODEL_DIR=${HF_MODEL_DIR:-$PROJECT_ROOT/.hf_models/$SAFE_TEACHER_NAME}
if [[ -d "$HF_MODEL_DIR" ]]; then
  MODEL_SOURCE="$HF_MODEL_DIR"
else
  MODEL_SOURCE="$TEACHER"
fi
HF_LOCAL_FILES_ONLY=${HF_LOCAL_FILES_ONLY:-1}
HF_LOCAL_ARGS=()
if [[ "$HF_LOCAL_FILES_ONLY" == "1" ]]; then
  HF_LOCAL_ARGS+=(--local_files_only)
fi

if [[ "$HF_LOCAL_FILES_ONLY" == "1" && ! -d "$HF_MODEL_DIR" ]]; then
  echo "[ERROR] Local-only HF loading requested, but warmed model dir is missing: $HF_MODEL_DIR" >&2
  echo "[ERROR] Run the warm_hf_cache job first, or set HF_LOCAL_FILES_ONLY=0 to allow downloads." >&2
  exit 2
fi

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Teacher data dir: data/$TEACHER_DATA"
echo "[INFO] Model source: $MODEL_SOURCE"
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
  --model "$MODEL_SOURCE" \
  --input_jsonl "$IN" \
  --out_dir data/$TEACHER_DATA/topk_k16/ \
  --k 16 \
  --dtype bfloat16 \
  "${HF_LOCAL_ARGS[@]}"

# ---- FB hidden-state caches (e.g., teacher layer 22)
python teacher_farm/make_hidden_cache.py \
  --model "$MODEL_SOURCE" \
  --input_jsonl "$IN" \
  --out_dir data/$TEACHER_DATA/fb_hints_L22/ \
  --layers 22 \
  --batch_size 1 \
  --max_length 2048 \
  --dtype bfloat16 \
  --flush_every 256 \
  "${HF_LOCAL_ARGS[@]}"


# ---- RelB pooled embedding caches
python teacher_farm/make_embed_cache.py \
  --model "$MODEL_SOURCE" \
  --input_jsonl "$IN" \
  --out_dir data/$TEACHER_DATA/relb_embeds/ \
  "${HF_LOCAL_ARGS[@]}"

echo "[INFO] $TEACHER Cache build complete: data/$TEACHER_DATA"
