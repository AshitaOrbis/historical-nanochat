#!/usr/bin/env bash
# Single-GPU CORE-metric eval for a 3090-trained checkpoint.
#
# Usage:
#   MODEL_TAG=d16_3090 bash historical_3090_eval.sh
#   MODEL_TAG=d16_3090 MODEL_STEP=20000 MAX_PER_TASK=500 bash historical_3090_eval.sh

set -euo pipefail

MODEL_TAG="${MODEL_TAG:?set MODEL_TAG to the checkpoint dir name}"
MODEL_STEP="${MODEL_STEP:-}"
MAX_PER_TASK="${MAX_PER_TASK:-500}"    # 500 is a good single-GPU default

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cat <<EOF
===== historical_3090_eval (single GPU) =====
  MODEL_TAG    = ${MODEL_TAG}
  MODEL_STEP   = ${MODEL_STEP:-<latest>}
  MAX_PER_TASK = ${MAX_PER_TASK}
==============================================
EOF

STEP_FLAG=""
[[ -n "${MODEL_STEP}" ]] && STEP_FLAG="--step=${MODEL_STEP}"

exec python -m scripts.base_eval \
  --model-tag="${MODEL_TAG}" \
  ${STEP_FLAG} \
  --max-per-task="${MAX_PER_TASK}"
