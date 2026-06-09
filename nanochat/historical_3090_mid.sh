#!/usr/bin/env bash
# Single-GPU midtraining on an RTX 3090 (24 GB).
#
# Runs after historical_3090_base.sh. Conservative defaults: smaller microbatch,
# T=1024. If you hit OOM bump up DEVICE_BATCH_SIZE=1 or drop MAX_SEQ_LEN to 768.
#
# Usage:
#   MODEL_TAG=d16_3090 bash historical_3090_mid.sh
#   MODEL_TAG=d16_3090 MODEL_STEP=10000 bash historical_3090_mid.sh

set -euo pipefail

MODEL_TAG="${MODEL_TAG:?set MODEL_TAG to the base checkpoint dir name (e.g. d16_3090)}"
MODEL_STEP="${MODEL_STEP:-}"                 # empty = latest
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"  # mid loss uses structured tasks, bump down
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"       # -1 = full epoch through the mixture
RUN_NAME="${RUN_NAME:-mid_${MODEL_TAG}}"
EVAL_EVERY="${EVAL_EVERY:-150}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

TOKENS_PER_STEP=$(( DEVICE_BATCH_SIZE * MAX_SEQ_LEN ))
if (( TOTAL_BATCH_SIZE % TOKENS_PER_STEP != 0 )); then
  echo "ERROR: TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} must be divisible by ${TOKENS_PER_STEP}" >&2
  exit 1
fi
GRAD_ACCUM=$(( TOTAL_BATCH_SIZE / TOKENS_PER_STEP ))

cat <<EOF
===== historical_3090_mid (single GPU) =====
  MODEL_TAG         = ${MODEL_TAG}
  MODEL_STEP        = ${MODEL_STEP:-<latest>}
  MAX_SEQ_LEN       = ${MAX_SEQ_LEN}
  DEVICE_BATCH_SIZE = ${DEVICE_BATCH_SIZE}
  TOTAL_BATCH_SIZE  = ${TOTAL_BATCH_SIZE}
  grad_accum_steps  = ${GRAD_ACCUM}
  RUN_NAME          = ${RUN_NAME}
=============================================
EOF

STEP_FLAG=""
[[ -n "${MODEL_STEP}" ]] && STEP_FLAG="--model_step=${MODEL_STEP}"
HORIZON_FLAG=""
[[ "${NUM_ITERATIONS}" != "-1" ]] && HORIZON_FLAG="--num_iterations=${NUM_ITERATIONS}"

exec python -m scripts.mid_train \
  --run="${RUN_NAME}" \
  --model_tag="${MODEL_TAG}" \
  ${STEP_FLAG} \
  --max_seq_len="${MAX_SEQ_LEN}" \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --total_batch_size="${TOTAL_BATCH_SIZE}" \
  --eval_every="${EVAL_EVERY}" \
  ${HORIZON_FLAG}
