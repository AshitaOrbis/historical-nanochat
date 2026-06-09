"""
Single-GPU microbenchmark for base-training throughput on an RTX 3090.

Runs a short synthetic loop (random tokens, no dataloader) and records:
  - step_time_ms, tokens/sec, peak VRAM
  - compile_mode, sequence_length, device_batch_size
  - activation checkpointing + chunked loss toggles

Writes a CSV row per configuration, so you can sweep and diff. The benchmark
avoids the parquet dataloader on purpose so data-pipeline cost doesn't show up
in step-time numbers; that is measured separately by base_train.py's own
--benchmark_csv.

Usage:
  python -m scripts.benchmark_3090 --depth 14 --seq_len 1024 --device_batch_size 4
  python -m scripts.benchmark_3090 --depth 16 --device_batch_size 2 --activation_checkpoint --chunked_loss
  python -m scripts.benchmark_3090 --sweep  # runs a small preset grid
"""

import argparse
import csv
import os
import time
from contextlib import nullcontext

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from nanochat.common import autodetect_device_type, print0
from nanochat.gpt import GPT, GPTConfig


def configure_sdpa(backend: str):
    if not torch.cuda.is_available():
        return
    b = torch.backends.cuda
    if backend == "flash":
        b.enable_flash_sdp(True); b.enable_mem_efficient_sdp(False); b.enable_math_sdp(False)
    elif backend == "efficient":
        b.enable_flash_sdp(False); b.enable_mem_efficient_sdp(True); b.enable_math_sdp(False)
    elif backend == "math":
        b.enable_flash_sdp(False); b.enable_mem_efficient_sdp(False); b.enable_math_sdp(True)


def run_one(depth, device_batch_size, seq_len, activation_ckpt, chunked_loss,
            compile_mode, sdpa_backend, warmup_steps, measure_steps,
            vocab_size, aspect_ratio, head_dim, kv_head_ratio, device):
    """Run a single config and return a metrics dict. Raises OOM if it doesn't fit."""
    configure_sdpa(sdpa_backend)

    model_dim = depth * aspect_ratio
    # Pick num_heads closest to target head_dim that divides model_dim.
    def find_num_heads(md, target_hd):
        ideal = max(1, round(md / target_hd))
        for offset in range(md):
            for c in [ideal + offset, ideal - offset]:
                if c > 0 and md % c == 0:
                    return c
        return 1
    n_head = find_num_heads(model_dim, head_dim)
    n_kv_target = max(1, round(n_head * kv_head_ratio))
    n_kv = next((c for c in range(min(n_kv_target, n_head), 0, -1) if n_head % c == 0), 1)

    cfg = GPTConfig(sequence_len=seq_len, vocab_size=vocab_size,
                    n_layer=depth, n_head=n_head, n_kv_head=n_kv, n_embd=model_dim)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    model.use_activation_checkpoint = activation_ckpt
    model.use_chunked_loss = chunked_loss

    if compile_mode != "none" and not os.environ.get("TORCH_COMPILE_DISABLE"):
        try:
            model = torch.compile(model, mode=compile_mode, dynamic=False)
        except Exception as e:
            print0(f"torch.compile failed ({e}), running eager")

    # Synthetic inputs: random tokens in [0, vocab_size)
    x = torch.randint(0, vocab_size, (device_batch_size, seq_len), device=device, dtype=torch.long)
    y = torch.randint(0, vocab_size, (device_batch_size, seq_len), device=device, dtype=torch.long)

    params = list(model.parameters())
    # Cheap optimizer; step-time dominated by model fwd/bwd, not optimizer.
    optim = torch.optim.SGD(params, lr=0.0, momentum=0.0)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup (captures compile/autotune cost too so it doesn't pollute the measurement).
    model.train()
    for _ in range(warmup_steps):
        with autocast_ctx:
            loss = model(x, y)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    synchronize()

    t0 = time.time()
    for _ in range(measure_steps):
        with autocast_ctx:
            loss = model(x, y)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    synchronize()
    dt = (time.time() - t0) / measure_steps

    peak_mib = (torch.cuda.max_memory_allocated() / 1024 / 1024) if device.type == "cuda" else 0.0
    tokens_per_step = device_batch_size * seq_len

    return {
        "depth": depth,
        "seq_len": seq_len,
        "device_batch_size": device_batch_size,
        "activation_ckpt": int(activation_ckpt),
        "chunked_loss": int(chunked_loss),
        "compile_mode": compile_mode,
        "sdpa_backend": sdpa_backend,
        "n_head": n_head,
        "n_kv_head": n_kv,
        "step_time_ms": round(dt * 1000, 3),
        "tokens_per_sec": int(tokens_per_step / dt),
        "peak_mem_mib": round(peak_mib, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


FIELDS = [
    "depth", "seq_len", "device_batch_size", "activation_ckpt", "chunked_loss",
    "compile_mode", "sdpa_backend", "n_head", "n_kv_head",
    "step_time_ms", "tokens_per_sec", "peak_mem_mib", "timestamp",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--device_batch_size", type=int, default=4)
    ap.add_argument("--aspect_ratio", type=int, default=64)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--kv_head_ratio", type=float, default=1.0)
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--activation_checkpoint", action="store_true")
    ap.add_argument("--chunked_loss", action="store_true")
    ap.add_argument("--compile_mode", type=str, default="none",
                    choices=["default", "reduce-overhead", "max-autotune", "none"])
    ap.add_argument("--sdpa_backend", type=str, default="auto",
                    choices=["auto", "flash", "efficient", "math"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--csv", type=str, default=None, help="Append results to this CSV")
    ap.add_argument("--sweep", action="store_true",
                    help="Run a small preset sweep (d12/14/16 x {ckpt,loss on/off})")
    args = ap.parse_args()

    device_type = autodetect_device_type()
    device = torch.device(device_type)

    configs = []
    if args.sweep:
        for d in (12, 14, 16):
            for ac in (False, True):
                for cl in (False, True):
                    configs.append(dict(depth=d, activation_checkpoint=ac, chunked_loss=cl))
    else:
        configs.append(dict(depth=args.depth,
                            activation_checkpoint=args.activation_checkpoint,
                            chunked_loss=args.chunked_loss))

    writer = None
    fp = None
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        new_file = not os.path.exists(args.csv)
        fp = open(args.csv, "a", newline="")
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

    for cfg in configs:
        depth = cfg["depth"]
        ac = cfg["activation_checkpoint"]
        cl = cfg["chunked_loss"]
        print0(f"\n=== depth={depth} T={args.seq_len} B={args.device_batch_size} "
               f"ckpt={ac} chunked_loss={cl} compile={args.compile_mode} sdpa={args.sdpa_backend} ===")
        try:
            result = run_one(
                depth=depth,
                device_batch_size=args.device_batch_size,
                seq_len=args.seq_len,
                activation_ckpt=ac,
                chunked_loss=cl,
                compile_mode=args.compile_mode,
                sdpa_backend=args.sdpa_backend,
                warmup_steps=args.warmup,
                measure_steps=args.steps,
                vocab_size=args.vocab_size,
                aspect_ratio=args.aspect_ratio,
                head_dim=args.head_dim,
                kv_head_ratio=args.kv_head_ratio,
                device=device,
            )
        except torch.cuda.OutOfMemoryError as e:
            print0(f"  OOM: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            result = {
                "depth": depth, "seq_len": args.seq_len,
                "device_batch_size": args.device_batch_size,
                "activation_ckpt": int(ac), "chunked_loss": int(cl),
                "compile_mode": args.compile_mode, "sdpa_backend": args.sdpa_backend,
                "n_head": -1, "n_kv_head": -1,
                "step_time_ms": -1, "tokens_per_sec": 0, "peak_mem_mib": -1,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        print0(f"  tokens/sec: {result['tokens_per_sec']:,}  "
               f"step: {result['step_time_ms']} ms  peak: {result['peak_mem_mib']} MiB")
        if writer:
            writer.writerow(result)
            fp.flush()

        # Free everything between configs so OOMs don't bleed across runs.
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if fp:
        fp.close()


if __name__ == "__main__":
    main()
