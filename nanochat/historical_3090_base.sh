#!/usr/bin/env bash
# Single-GPU base pretraining on an RTX 3090 (24 GB).
#
# Prints grad-accum + token estimates up front, defaults to d16 at T=1024 with
# activation checkpointing + chunked loss so the model fits with headroom. Override
# any of the knobs via env vars; see the top of the script for all of them.
#
# Usage:
#   NANOCHAT_PARQUET_DIR=/path/to/shards_1913 bash historical_3090_base.sh
#   DEPTH=14 MAX_SEQ_LEN=1024 bash historical_3090_base.sh

set -euo pipefail

# ----- user-overridable knobs ------------------------------------------------
DEPTH="${DEPTH:-16}"                             # d12/14/16/18 recommended for 3090
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"               # base-stage context window
SEQ_LEN_LATE="${SEQ_LEN_LATE:--1}"               # -1 disables; try 1536 or 2048 late
SEQ_LEN_LATE_FRAC="${SEQ_LEN_LATE_FRAC:-0.85}"   # switch point for curriculum
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"      # microbatch; 4 fits d16@T=1024 w/ ckpt
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-262144}"   # tokens / optimizer step
TARGET_RATIO="${TARGET_RATIO:-8}"                # tokens:params ratio (Chinchilla=20)
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"           # override ratio with explicit step count
KV_HEAD_RATIO="${KV_HEAD_RATIO:-1.0}"            # 0.5 or 0.25 to enable GQA
ACTIVATION_CKPT="${ACTIVATION_CKPT:-1}"          # 1 = on, 0 = off
CKPT_EVERY_N="${CKPT_EVERY_N:-1}"                # checkpoint every Nth block
CHUNKED_LOSS="${CHUNKED_LOSS:-1}"                # 1 = on, 0 = off
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-1024}"
COMPILE_MODE="${COMPILE_MODE:-none}"             # torch.compile: default|reduce-overhead|max-autotune|none
SDPA_BACKEND="${SDPA_BACKEND:-auto}"             # auto|flash|efficient|math
SAVE_EVERY="${SAVE_EVERY:-500}"                  # 3090 runs are long; save often
EVAL_EVERY="${EVAL_EVERY:-500}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"     # off by default (expensive on 3090)
SAMPLE_EVERY="${SAMPLE_EVERY:-2000}"
RUN_NAME="${RUN_NAME:-historical_3090_d${DEPTH}}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}_3090}"
RESUME_FROM="${RESUME_FROM:--1}"
BENCHMARK_CSV="${BENCHMARK_CSV:-${HOME}/.cache/nanochat/benchmarks/${MODEL_TAG}.csv}"
# -----------------------------------------------------------------------------

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# torch.compile is brittle on some setups; let the caller opt in via COMPILE_MODE.
if [[ "${COMPILE_MODE}" == "none" ]]; then
  export TORCH_COMPILE_DISABLE=1
fi

mkdir -p "$(dirname "${BENCHMARK_CSV}")"

# ---- print setup + estimates ------------------------------------------------
cat <<EOF
===== historical_3090_base (single GPU) =====
  DEPTH               = ${DEPTH}
  MAX_SEQ_LEN         = ${MAX_SEQ_LEN}
  SEQ_LEN_LATE        = ${SEQ_LEN_LATE} (frac=${SEQ_LEN_LATE_FRAC})
  DEVICE_BATCH_SIZE   = ${DEVICE_BATCH_SIZE}
  TOTAL_BATCH_SIZE    = ${TOTAL_BATCH_SIZE}
  KV_HEAD_RATIO       = ${KV_HEAD_RATIO}
  ACTIVATION_CKPT     = ${ACTIVATION_CKPT} (every ${CKPT_EVERY_N} block(s))
  CHUNKED_LOSS        = ${CHUNKED_LOSS} (chunk=${LOSS_CHUNK_SIZE})
  COMPILE_MODE        = ${COMPILE_MODE}
  SDPA_BACKEND        = ${SDPA_BACKEND}
  RUN / MODEL_TAG     = ${RUN_NAME} / ${MODEL_TAG}
  PARQUET_DIR         = ${NANOCHAT_PARQUET_DIR:-<default base_data>}
  BENCHMARK_CSV       = ${BENCHMARK_CSV}
EOF

TOKENS_PER_STEP_RANK=$(( DEVICE_BATCH_SIZE * MAX_SEQ_LEN ))
if (( TOTAL_BATCH_SIZE % TOKENS_PER_STEP_RANK != 0 )); then
  echo "ERROR: TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} must be divisible by DEVICE_BATCH_SIZE*MAX_SEQ_LEN=${TOKENS_PER_STEP_RANK}" >&2
  exit 1
fi
GRAD_ACCUM=$(( TOTAL_BATCH_SIZE / TOKENS_PER_STEP_RANK ))
echo "  grad_accum_steps    = ${GRAD_ACCUM}"
echo "  tokens / step       = ${TOTAL_BATCH_SIZE}"
if (( NUM_ITERATIONS > 0 )); then
  EST_TOKENS=$(( NUM_ITERATIONS * TOTAL_BATCH_SIZE ))
  echo "  total tokens (est)  = ${EST_TOKENS}"
fi
echo "==============================================="

ACTIVATION_FLAG=""
[[ "${ACTIVATION_CKPT}" == "1" ]] && ACTIVATION_FLAG="--activation_checkpoint --ckpt_every_n_blocks=${CKPT_EVERY_N}"
CHUNKED_FLAG=""
[[ "${CHUNKED_LOSS}" == "1" ]] && CHUNKED_FLAG="--chunked_loss --loss_chunk_size=${LOSS_CHUNK_SIZE}"
HORIZON_FLAG="--target_param_data_ratio=${TARGET_RATIO}"
[[ "${NUM_ITERATIONS}" != "-1" ]] && HORIZON_FLAG="--num_iterations=${NUM_ITERATIONS}"
CURRICULUM_FLAG=""
if (( SEQ_LEN_LATE > 0 )); then
  CURRICULUM_FLAG="--seq_len_late=${SEQ_LEN_LATE} --seq_len_late_frac=${SEQ_LEN_LATE_FRAC}"
fi
RESUME_FLAG=""
[[ "${RESUME_FROM}" != "-1" ]] && RESUME_FLAG="--resume_from_step=${RESUME_FROM}"

exec python -m scripts.base_train \
  --run="${RUN_NAME}" \
  --model_tag="${MODEL_TAG}" \
  --depth="${DEPTH}" \
  --max_seq_len="${MAX_SEQ_LEN}" \
  ${CURRICULUM_FLAG} \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --total_batch_size="${TOTAL_BATCH_SIZE}" \
  --kv_head_ratio="${KV_HEAD_RATIO}" \
  ${ACTIVATION_FLAG} ${CHUNKED_FLAG} \
  --compile_mode="${COMPILE_MODE}" \
  --sdpa_backend="${SDPA_BACKEND}" \
  --save_every="${SAVE_EVERY}" \
  --eval_every="${EVAL_EVERY}" \
  --core_metric_every="${CORE_METRIC_EVERY}" \
  --sample_every="${SAMPLE_EVERY}" \
  --benchmark_csv="${BENCHMARK_CSV}" \
  ${HORIZON_FLAG} \
  ${RESUME_FLAG}
