#!/bin/bash
# Sequential scaling law runs for 3090

set -Eeuo pipefail

export OMP_NUM_THREADS=1
export TORCH_COMPILE_DISABLE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

cd "$(dirname "$0")/nanochat"
source ../.venv/bin/activate

DEPTHS=(8 12 16)
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
RESULTS_DIR="$BASE_DIR/scaling_3090_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"
STATUS_FILE="$RESULTS_DIR/run_status.csv"

# CSV header
if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,num_params,num_iterations,tokens_trained,val_bpb,train_time_sec" > "$RESULTS_FILE"
fi
if [ ! -f "$STATUS_FILE" ]; then
    echo "depth,status,detail,exit_code" > "$STATUS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

CURRENT_DEPTH=""
STATUS_RECORDED=0

record_status() {
    local depth="$1"
    local status="$2"
    local detail="$3"
    local exit_code="$4"
    printf '%s,%s,%s,%s\n' "$depth" "$status" "$detail" "$exit_code" >> "$STATUS_FILE"
    STATUS_RECORDED=1
}

on_error() {
    local exit_code=$?
    if [[ -n "$CURRENT_DEPTH" && "$STATUS_RECORDED" -eq 0 ]]; then
        record_status "$CURRENT_DEPTH" "FAILED" "unexpected_exit_${exit_code}" "$exit_code"
        log "FAILED d=$CURRENT_DEPTH: unexpected command exit $exit_code"
    fi
    exit "$exit_code"
}

trap on_error ERR

log "=============================================="
log "3090 Scaling Law Runs"
log "=============================================="

for d in "${DEPTHS[@]}"; do
    CURRENT_DEPTH="$d"
    STATUS_RECORDED=0
    log "Training d=$d..."
    TAG="scaling_d${d}"
    START_TIME=$(date +%s)

    if python -m scripts.base_train \
            --depth="$d" \
            --device_batch_size=2 \
            --total_batch_size=262144 \
            --max_seq_len=2048 \
            --target_param_data_ratio=8 \
            --run="scaling_3090_d${d}" \
            --model_tag="$TAG" \
            --eval_every=500 \
            --core_metric_every=-1 \
            --sample_every=-1 \
            --save_every=-1 \
            2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"; then
        :
    else
        # pipefail already told us SOMETHING in this pipeline failed; PIPESTATUS[0]
        # alone only tells us about the trainer. A tee failure (e.g. log disk full)
        # with a successful trainer must not be reported as trainer_exit_0/exit 0 --
        # that discards the one nonzero status that actually explains the failure.
        # Snapshot the WHOLE array in one assignment: PIPESTATUS is reset after
        # every simple command (including a plain assignment), so reading it in
        # two separate statements would silently zero out the second read.
        PIPE_STATUSES=("${PIPESTATUS[@]}")
        TRAIN_STATUS=${PIPE_STATUSES[0]}
        TEE_STATUS=${PIPE_STATUSES[1]:-0}
        if [[ "$TRAIN_STATUS" -ne 0 ]]; then
            record_status "$d" "FAILED" "trainer_exit_${TRAIN_STATUS}" "$TRAIN_STATUS"
            log "FAILED d=$d: trainer exited with status $TRAIN_STATUS; failure recorded"
            exit "$TRAIN_STATUS"
        else
            record_status "$d" "FAILED" "tee_exit_${TEE_STATUS}" "$TEE_STATUS"
            log "FAILED d=$d: trainer succeeded but log tee exited with status $TEE_STATUS (no observation appended, this depth was not fully run); failure recorded"
            exit "$TEE_STATUS"
        fi
    fi

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract stats
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
    NUM_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | head -1 | tr -d ',' || true)
    NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',' || true)
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$' || true)
    if [[ ! "$NUM_PARAMS" =~ ^[0-9]+$ || ! "$NUM_ITERS" =~ ^[0-9]+$ ||
          ! "$VAL_BPB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        record_status "$d" "FAILED" "invalid_metrics" "1"
        log "FAILED d=$d: missing or invalid numeric metrics; failure recorded"
        exit 1
    fi

    CHECKPOINT_DIR="$BASE_DIR/base_checkpoints/$TAG"
    shopt -s nullglob
    COMPLETE_SENTINELS=("$CHECKPOINT_DIR"/complete_*.json)
    shopt -u nullglob
    if (( ${#COMPLETE_SENTINELS[@]} == 0 )); then
        record_status "$d" "FAILED" "missing_completion_sentinel" "1"
        log "FAILED d=$d: no complete_*.json checkpoint sentinel; failure recorded"
        exit 1
    fi

    TOKENS_TRAINED=$((NUM_ITERS * 262144))

    log "  d=$d: params=$NUM_PARAMS, iters=$NUM_ITERS, bpb=$VAL_BPB, time=${TRAIN_TIME}s"
    echo "$d,$NUM_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$TRAIN_TIME" >> "$RESULTS_FILE"
    record_status "$d" "COMPLETED" "ok" "0"
done

CURRENT_DEPTH=""

log "=============================================="
log "Scaling Runs Complete!"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
column -t -s',' "$RESULTS_FILE"
