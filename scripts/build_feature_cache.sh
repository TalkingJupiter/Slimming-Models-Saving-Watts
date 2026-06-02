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
FEATURE_CACHE_SHARD_INDEX=${FEATURE_CACHE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
FEATURE_CACHE_NUM_SHARDS=${FEATURE_CACHE_NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
STUDENT_MODEL=${STUDENT_MODEL:-${STUDENT:-}}
STUDENT_MODEL_SOURCE=$(resolve_hf_model "$STUDENT_MODEL")
FEATURE_LAYER_RATIO=${FEATURE_LAYER_RATIO:-0.60}
FEATURE_LAYER_ENV=$(python scripts/resolve_feature_layers.py \
  --teacher "$MODEL_SOURCE" \
  --student "$STUDENT_MODEL_SOURCE" \
  --ratio "$FEATURE_LAYER_RATIO" \
  --teacher-layer "${FEATURE_TEACHER_LAYER:-}" \
  --student-layer "${FEATURE_STUDENT_LAYER:-}" \
  --project-root "$PROJECT_ROOT")
eval "$FEATURE_LAYER_ENV"
FEATURE_STUDENT_DEPTH=${FEATURE_STUDENT_DEPTH:-unknown}
FEATURE_STUDENT_LAYER=${FEATURE_STUDENT_LAYER:-unknown}
FEATURE_CACHE_DIR="data/$TEACHER_DATA/fb_hints_L$FEATURE_TEACHER_LAYER"
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

TELEMETRY_OUTPUT="${TELEMETRY_OUTPUT:-results/$TEACHER_DATA/cache/feature/feature_${FEATURE_CACHE_SHARD_INDEX}.jsonl}"

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Teacher data dir: data/$TEACHER_DATA"
echo "[INFO] Model source: $MODEL_SOURCE"
echo "[INFO] INPUT: $IN"
echo "[INFO] Feature cache shard: $FEATURE_CACHE_SHARD_INDEX / $FEATURE_CACHE_NUM_SHARDS"
echo "[FEATURE_LAYER_MAP] ratio=$FEATURE_LAYER_RATIO teacher_depth=$FEATURE_TEACHER_DEPTH student_depth=$FEATURE_STUDENT_DEPTH teacher_layer=$FEATURE_TEACHER_LAYER student_layer=$FEATURE_STUDENT_LAYER"
echo "[INFO] Feature cache dir: $FEATURE_CACHE_DIR"
echo "[INFO] GPUs visible: ${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -s "$IN" ]]; then
  echo "[ERROR] Input jsonl '$IN' not found or empty" >&2
  exit 2
fi

mkdir -p "$FEATURE_CACHE_DIR"
cat > "$FEATURE_CACHE_DIR/feature_layer_map.json" <<EOF
{
  "ratio": "$FEATURE_LAYER_RATIO",
  "teacher_model": "$TEACHER",
  "student_model": "$STUDENT_MODEL",
  "teacher_depth": "$FEATURE_TEACHER_DEPTH",
  "student_depth": "$FEATURE_STUDENT_DEPTH",
  "teacher_layer": "$FEATURE_TEACHER_LAYER",
  "student_layer": "$FEATURE_STUDENT_LAYER",
  "cache_dir": "$FEATURE_CACHE_DIR"
}
EOF

python teacher_farm/make_hidden_cache.py \
    --model "$MODEL_SOURCE" \
    --input_jsonl "$IN" \
    --out_dir "$FEATURE_CACHE_DIR" \
    --layers "$FEATURE_TEACHER_LAYER" \
    --batch_size 2 \
    --max_length 2048 \
    --dtype bfloat16 \
    --flush_every 16 \
    --shard_index "$FEATURE_CACHE_SHARD_INDEX" \
    --num_shards "$FEATURE_CACHE_NUM_SHARDS" \
    --telemetry \
    --telemetry_output "$TELEMETRY_OUTPUT" \
    --telemetry_interval 1 \
    "${HF_LOCAL_ARGS[@]}"



echo "[COMPLETED] $TEACHER feature cache build complete: data/$TEACHER_DATA"
