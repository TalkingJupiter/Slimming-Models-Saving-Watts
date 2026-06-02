#!/usr/bin/env bash

set -euo pipefail
source scripts/_env_single_node.sh

STUDENT_MODEL=${STUDENT_MODEL:-}
TEACHER_DATA=${TEACHER_DATA:-}
SAFE_STUDENT_NAME=${SAFE_STUDENT_NAME:-}
STUDENT_MODEL_SOURCE=$(resolve_hf_model "$STUDENT_MODEL")
TEACHER_MODEL=${TEACHER:-}
TEACHER_MODEL_SOURCE=$(resolve_hf_model "$TEACHER_MODEL")
FEATURE_LAYER_RATIO=${FEATURE_LAYER_RATIO:-0.60}
FEATURE_LAYER_ENV=$(python scripts/resolve_feature_layers.py \
  --teacher "$TEACHER_MODEL_SOURCE" \
  --student "$STUDENT_MODEL_SOURCE" \
  --ratio "$FEATURE_LAYER_RATIO" \
  --teacher-layer "${FEATURE_TEACHER_LAYER:-}" \
  --student-layer "${FEATURE_STUDENT_LAYER:-}" \
  --project-root "$PROJECT_ROOT")
eval "$FEATURE_LAYER_ENV"
FEATURE_CACHE_DIR="data/$TEACHER_DATA/fb_hints_L$FEATURE_TEACHER_LAYER"

echo "[INFO] $STUDENT_MODEL Feature-Based KD | node=1 | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES"
echo "[INFO] Student model source: $STUDENT_MODEL_SOURCE"
echo "[INFO] Teacher model source: $TEACHER_MODEL_SOURCE"
echo "[FEATURE_LAYER_MAP] ratio=$FEATURE_LAYER_RATIO teacher_depth=$FEATURE_TEACHER_DEPTH student_depth=$FEATURE_STUDENT_DEPTH teacher_layer=$FEATURE_TEACHER_LAYER student_layer=$FEATURE_STUDENT_LAYER"
echo "[INFO] Feature cache dir: $FEATURE_CACHE_DIR"

RUN_DIR="serialization_dir/$SAFE_STUDENT_NAME/feature/$SLURM_ARRAY_TASK_ID"
# TELEMETRY_OUT="logs/telemetry/$SAFE_STUDENT_NAME/feature/${SLURM_ARRAY_TASK_ID}/${SLURM_JOB_ID}_telemetry.jsonl"
TELEMETRY_OUT="results/${SAFE_STUDENT_NAME}/feature/${SLURM_ARRAY_TASK_ID}/telemetry.jsonl"
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/feature_layer_map.json" <<EOF
{
  "ratio": "$FEATURE_LAYER_RATIO",
  "teacher_model": "$TEACHER_MODEL",
  "student_model": "$STUDENT_MODEL",
  "teacher_depth": "$FEATURE_TEACHER_DEPTH",
  "student_depth": "$FEATURE_STUDENT_DEPTH",
  "teacher_layer": "$FEATURE_TEACHER_LAYER",
  "student_layer": "$FEATURE_STUDENT_LAYER",
  "cache_dir": "$FEATURE_CACHE_DIR"
}
EOF

accelerate launch \
  --num_machines 1 \
  --num_processes ${NUM_PROCESSES} \
  --deepspeed_config_file configs/ds_zero3.json \
  --module kd.train \
    --kd.mode fb \
    --student "$STUDENT_MODEL_SOURCE" \
    --data "$FEATURE_CACHE_DIR/*.parquet" \
    --fb.teacher_layer "$FEATURE_TEACHER_LAYER" \
    --fb.student_layer "$FEATURE_STUDENT_LAYER" \
    --fb.token_subset_ratio 0.25 \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr 1e-4 \
    --bash_size 2 \
    --save-dir "$RUN_DIR" \
    --save_every 200 \
    --max_steps 5000 \
    --telemetry \
    --telemetry_output "$TELEMETRY_OUT" \
    --telemetry_interval 1.0 \
    --resume auto 

echo "[INFO] $STUDENT_MODEL FB KD complete"
