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

# CSV header
if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,num_params,num_iterations,tokens_trained,val_bpb,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "3090 Scaling Law Runs"
log "=============================================="

for d in "${DEPTHS[@]}"; do
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
        TRAIN_STATUS=${PIPESTATUS[0]}
        log "FAILED d=$d: trainer exited with status $TRAIN_STATUS; no observation recorded"
        exit "$TRAIN_STATUS"
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
        log "FAILED d=$d: missing or invalid numeric metrics; no observation recorded"
        exit 1
    fi

    CHECKPOINT_DIR="$BASE_DIR/base_checkpoints/$TAG"
    shopt -s nullglob
    COMPLETE_SENTINELS=("$CHECKPOINT_DIR"/complete_*.json)
    shopt -u nullglob
    if (( ${#COMPLETE_SENTINELS[@]} == 0 )); then
        log "FAILED d=$d: no complete_*.json checkpoint sentinel; no observation recorded"
        exit 1
    fi

    TOKENS_TRAINED=$((NUM_ITERS * 262144))

    log "  d=$d: params=$NUM_PARAMS, iters=$NUM_ITERS, bpb=$VAL_BPB, time=${TRAIN_TIME}s"
    echo "$d,$NUM_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$TRAIN_TIME" >> "$RESULTS_FILE"
done

log "=============================================="
log "Scaling Runs Complete!"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
column -t -s',' "$RESULTS_FILE"
