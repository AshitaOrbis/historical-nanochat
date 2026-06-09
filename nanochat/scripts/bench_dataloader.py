"""
Detect whether tokenization is bottlenecking the dataloader.

Runs the parquet dataloader for N batches and reports:
  - wall time per batch
  - tokens/sec produced by the loader (CPU-side)

Compare against the per-step tok/sec reported by base_train.py. If the loader's
CPU tokens/sec is close to training's tok/sec, tokenization is the bottleneck
and an offline token cache (scripts/build_token_cache.py) is worth trying.

Usage:
    python -m scripts.bench_dataloader --parquet-dir /path/to/shards --batches 50
"""

import argparse
import time

import torch

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=str, default=None)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--device-batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    loader = tokenizing_distributed_data_loader_with_state(
        args.device_batch_size, args.seq_len, split="train",
        device=args.device, parquet_dir=args.parquet_dir,
    )

    # warmup
    for _ in range(args.warmup):
        next(loader)

    tokens_per_batch = args.device_batch_size * args.seq_len
    t0 = time.time()
    for _ in range(args.batches):
        x, y, _ = next(loader)
    dt = time.time() - t0
    total_tokens = tokens_per_batch * args.batches
    print(f"Batches: {args.batches}, batch_shape=({args.device_batch_size},{args.seq_len}+1)")
    print(f"Wall time: {dt:.2f}s, avg batch: {dt/args.batches*1000:.1f}ms")
    print(f"Loader throughput: {total_tokens/dt:,.0f} tokens/sec (CPU+H2D)")
    print()
    print("Rule of thumb: if the training step rate is <= 1.5x this loader rate,")
    print("tokenization is likely the bottleneck. Consider scripts/build_token_cache.py.")


if __name__ == "__main__":
    main()
