# Single RTX 3090 Historical Training Guide

Practical guide to running historical-nanochat end-to-end on a single RTX 3090
(24 GB). Default target: **d16 @ T=1024** with activation checkpointing and
chunked LM-head loss, finishing in ~1–2 weeks of wall clock.

## Quick start

```bash
# 1. Point training at pre-packaged historical shards. No base_data/ wrapper required.
export NANOCHAT_PARQUET_DIR=data/shards

# 2. Base pretraining
cd nanochat
bash historical_3090_base.sh          # default: d16, T=1024, ckpt+chunked on

# 3. Midtraining (structured tasks: GSM8K / MMLU / SmolTalk / etc.)
MODEL_TAG=d16_3090 bash historical_3090_mid.sh

# 4. CORE-metric evaluation
MODEL_TAG=d16_3090 bash historical_3090_eval.sh
```

## Knob reference

| Env var                 | Default         | Notes |
|-------------------------|-----------------|-------|
| `NANOCHAT_PARQUET_DIR`  | unset           | If unset, loader falls back to FineWeb auto-download at `<base_dir>/base_data`. |
| `DEPTH`                 | `16`            | `12/14/16/18` supported. `18+` only fits with GQA+ckpt+chunked. |
| `MAX_SEQ_LEN`           | `1024`          | Start-of-training context. Rotary cache is sized for `max(MAX_SEQ_LEN, SEQ_LEN_LATE) * 10`. |
| `SEQ_LEN_LATE`          | `-1`            | Optional late-stage bump (e.g. `1536` or `2048`). See curriculum below. |
| `SEQ_LEN_LATE_FRAC`     | `0.85`          | When to switch (fraction of total iters). |
| `DEVICE_BATCH_SIZE`     | `4`             | Microbatch per step. Drop to 2 on OOM. |
| `TOTAL_BATCH_SIZE`      | `262144`        | Tokens per optimizer step (must be divisible by `DEVICE_BATCH_SIZE*MAX_SEQ_LEN`). |
| `ACTIVATION_CKPT`       | `1`             | Enables `torch.utils.checkpoint` on every block. |
| `CKPT_EVERY_N`          | `1`             | Checkpoint every Nth block. `2` trades some VRAM for less recompute. |
| `CHUNKED_LOSS`          | `1`             | Chunked LM-head CE. Biggest single VRAM win on large vocab. |
| `LOSS_CHUNK_SIZE`       | `1024`          | Tokens per CE chunk. Smaller = less peak memory, more overhead. |
| `COMPILE_MODE`          | `none`          | `default`/`reduce-overhead`/`max-autotune`. Disable on brittle stacks. |
| `SDPA_BACKEND`          | `auto`          | `flash`/`efficient`/`math`. Logs effective set on startup. |
| `KV_HEAD_RATIO`         | `1.0`           | `0.5` or `0.25` enables GQA. Second-pass experiment — benchmark first. |
| `SAVE_EVERY`            | `500`           | More frequent checkpoints so long local runs survive crashes. |
| `BENCHMARK_CSV`         | auto            | Per-step throughput/VRAM metrics appended here (rank 0 only). |

All knobs are env vars read by `historical_3090_base.sh`; the script translates
them into CLI flags for `scripts/base_train.py`.

## Memory budget (approximate)

A d16 model at T=1024 with `DEVICE_BATCH_SIZE=4` on a 3090:

| Config                                           | Peak VRAM | Notes |
|--------------------------------------------------|-----------|-------|
| Baseline (no tricks)                             | OOM       | Full `(B,T,V)` logits alone exhausts memory. |
| Chunked loss only                                | ~20 GB    | Largest single win. |
| Chunked loss + activation checkpoint             | ~11–13 GB | Recommended default. |
| ckpt+chunked + `KV_HEAD_RATIO=0.5`               | ~9–11 GB  | Needs retraining-from-scratch; don't mix with non-GQA checkpoints. |

Numbers are indicative; run `scripts/benchmark_3090.py --sweep` to measure on
your own hardware. Results append to CSV if `--csv` is passed.

## Sequence-length curriculum

If you want longer-context capability without eating 4x the VRAM early on:

```bash
SEQ_LEN_LATE=1536 SEQ_LEN_LATE_FRAC=0.85 bash historical_3090_base.sh
```

The base_train loop rebuilds the dataloader at the switch point (still using the
same parquet state so no data is repeated). The rotary cache is pre-sized for
`SEQ_LEN_LATE` so there's no recompute cost at the transition.

## Recommended presets

These are starting points — benchmark to confirm on your box before committing
to a multi-day run.

| Preset          | Depth | T start | T late | DBS | Notes |
|-----------------|-------|---------|--------|-----|-------|
| **1 week**      | d14   | 1024    | —      | 4   | Safe path. Finishes quickly; leaves time for midtraining+eval. |
| **2 week**      | d16   | 1024    | —      | 4   | Default. Best quality/time tradeoff for a 3090. |
| **2 week+ctx**  | d16   | 1024    | 1536   | 4   | As above with late-stage context bump. Requires clean eval of the transition. |
| **Aggressive**  | d18   | 1024    | —      | 2   | Needs ckpt+chunked and probably GQA. Not recommended without benchmarking. |

Populate the table below as you run:

| Date | Preset | Steps | Tokens | Val BPB | tok/sec | Peak VRAM |
|------|--------|-------|--------|---------|---------|-----------|
| _TBD_ | d14 / T=1024 | | | | | |
| _TBD_ | d16 / T=1024 | | | | | |
| _TBD_ | d16 / T=1024→1536 | | | | | |

## Benchmarking

Microbenchmark a single config:

```bash
python -m scripts.benchmark_3090 \
    --depth 16 --seq_len 1024 --device_batch_size 4 \
    --activation_checkpoint --chunked_loss \
    --compile_mode none \
    --csv ~/.cache/nanochat/benchmarks/d16_sweep.csv
```

Preset grid (d12/14/16 × ckpt on/off × chunked on/off):

```bash
python -m scripts.benchmark_3090 --sweep --csv ~/.cache/nanochat/benchmarks/sweep.csv
```

Check whether tokenization is the bottleneck:

```bash
python -m scripts.bench_dataloader --parquet-dir $NANOCHAT_PARQUET_DIR --batches 50
```

If the dataloader's CPU tokens/sec is within ~1.5x of base_train's per-step
tokens/sec, run `scripts/build_token_cache.py` to pre-tokenize your shards.

## Resuming after a crash

Every `SAVE_EVERY` steps a full checkpoint is written to
`<base_dir>/base_checkpoints/<MODEL_TAG>/`. To resume:

```bash
RESUME_FROM=6000 MODEL_TAG=d16_3090 bash historical_3090_base.sh
```

The dataloader state (parquet/row-group indices) round-trips through the
checkpoint, so you resume near where you left off without repeating documents.

## Troubleshooting

**`dxgkio_make_resident: Ioctl failed: -12`** — This is a WSL-specific VRAM
exhaustion symptom from pre-March-2026 runs. The default 3090 config here
(activation ckpt + chunked loss + T=1024 + DBS=4) uses <60% of VRAM, leaving
headroom. If you still see it, drop `DEVICE_BATCH_SIZE=2`.

**`torch.compile` errors on WSL/older CUDA** — Set `COMPILE_MODE=none` (default)
or export `TORCH_COMPILE_DISABLE=1`. Eager mode is about 5–15% slower but
bulletproof.

**"No dataset parquet files found"** — Either `NANOCHAT_PARQUET_DIR` is wrong, or
the shards don't end in `.parquet`. The loader now prints the directory it
searched in the assertion message.

**Loss explodes on curriculum switch** — The LR schedule is unaware of the T
switch. If the longer context hits a loss spike, bump `SEQ_LEN_LATE_FRAC` later
(e.g. 0.9) so fewer steps run at the new T, or drop the long-stage LR via a
separate resume run with `--final_lr_frac=0.1`.
