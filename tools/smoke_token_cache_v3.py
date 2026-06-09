"""Dataloader + forward-pass smoke on token_cache_v3.

Verifies end-to-end compatibility between the governed token cache and
the existing training stack WITHOUT touching the GPU (training is live).

What it checks:
  - cache_manifest.json present + well-formed
  - At least one .bin shard readable + mmap-able
  - cached_distributed_data_loader_with_state yields correctly-shaped
    (inputs, targets, state) tuples for both 'train' and 'val' splits
  - Tokens are in vocab range [0, vocab_size)
  - BOS token appears (smoke for BOS prefixing)
  - A tiny GPTConfig model runs forward() on CPU without NaN/inf

Output:
  report/token_cache_v3_smoke.md

Usage:
  python tools/smoke_token_cache_v3.py --cache-dir /home/user/historical-nanochat/data/token_cache_v3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nanochat"))
os.environ.setdefault("NANOCHAT_BASE_DIR", str(REPO))

# Force single-rank environment for the test (no DDP)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir",
        default="/home/user/historical-nanochat/data/token_cache_v3",
    )
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument(
        "--out",
        default=str(REPO / "report" / "token_cache_v3_smoke.md"),
    )
    args = ap.parse_args()

    import numpy as np
    import torch

    results = []

    cache_dir = Path(args.cache_dir)
    # v4 layout: train/ and val/ subdirs each with their own manifest.
    v4_train_mf = cache_dir / "train" / "cache_manifest.json"
    v4_val_mf = cache_dir / "val" / "cache_manifest.json"
    is_v4_root = v4_train_mf.exists() and v4_val_mf.exists()
    manifest_path = cache_dir / "cache_manifest.json"

    if is_v4_root:
        train_m = json.loads(v4_train_mf.read_text())
        val_m = json.loads(v4_val_mf.read_text())
        required_keys = {"dtype", "shards"}
        missing_train = required_keys - set(train_m.keys())
        missing_val = required_keys - set(val_m.keys())
        results.append(
            (
                "cache_manifest well-formed (v4 train+val)",
                not missing_train and not missing_val,
                f"train: dtype={train_m['dtype']}, shards={len(train_m['shards'])}; "
                f"val: dtype={val_m['dtype']}, shards={len(val_m['shards'])}",
            )
        )
        train_bins = sorted((cache_dir / "train").glob("shard_*.bin"))
        val_bins = sorted((cache_dir / "val").glob("shard_*.bin"))
        shards = train_bins  # for downstream checks below
        results.append(
            (
                "at least one .bin shard (v4 train+val)",
                bool(train_bins) and bool(val_bins),
                f"train={len(train_bins)} bins, val={len(val_bins)} bins",
            )
        )
    elif manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        required_keys = {"dtype", "shards"}
        missing = required_keys - set(manifest.keys())
        results.append(
            (
                "cache_manifest well-formed",
                not missing,
                f"missing keys: {missing}" if missing else f"dtype={manifest['dtype']}, shards={len(manifest['shards'])}",
            )
        )
        shards = sorted(cache_dir.glob("shard_*.bin"))
        results.append(
            ("at least one .bin shard", bool(shards), f"found {len(shards)} shards")
        )
    else:
        results.append(("cache_manifest present", False, f"not found: {manifest_path}"))
        shards = sorted(cache_dir.glob("shard_*.bin"))
        results.append(
            ("at least one .bin shard", bool(shards), f"found {len(shards)} shards")
        )

    # Dataloader check
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state  # type: ignore

    # Detect v4 layout (cache_dir has train/ + val/ subdirs with manifests) vs
    # legacy v3 layout (cache_dir has one flat manifest).
    train_manifest = cache_dir / "train" / "cache_manifest.json"
    val_manifest = cache_dir / "val" / "cache_manifest.json"
    v4_layout = train_manifest.exists() and val_manifest.exists()
    if v4_layout:
        train_cache = str(cache_dir / "train")
        val_cache = str(cache_dir / "val")
        train_split = "all"
        val_split = "all"
    else:
        train_cache = str(cache_dir)
        val_cache = str(cache_dir)
        train_split = "train"
        val_split = "val"

    try:
        loader = cached_distributed_data_loader_with_state(
            B=args.batch,
            T=args.seq_len,
            split=train_split,
            device="cpu",
            cache_dir=train_cache,
        )
        x, y, state = next(loader)
        shape_ok = (x.shape == (args.batch, args.seq_len)) and (
            y.shape == (args.batch, args.seq_len)
        )
        results.append(
            (
                "dataloader yields (B,T) tensors (train split)",
                shape_ok,
                f"x={tuple(x.shape)} y={tuple(y.shape)} state_keys={list(state.keys())} "
                f"layout={'v4_train' if v4_layout else 'v3'}",
            )
        )

        # Vocab range
        xmax, xmin = int(x.max().item()), int(x.min().item())
        results.append(
            (
                "tokens in vocab range [0, 32768)",
                0 <= xmin and xmax < 32768,
                f"min={xmin} max={xmax}",
            )
        )

        # BOS present check (BOS id should be near top of vocab; nanochat uses 32759-32767)
        bos_candidates = set(range(32759, 32768))
        has_bos = bool(set(x.flatten().tolist()) & bos_candidates)
        results.append(
            (
                "at least one special token visible in first batch",
                has_bos,
                f"found specials: {sorted(set(x.flatten().tolist()) & bos_candidates)}",
            )
        )
    except Exception as e:
        import traceback
        results.append(
            ("dataloader yields (B,T) tensors (train split)", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[:400]}")
        )
        x = None

    # Val split check
    val_shards_found = v4_layout or len(shards) >= 2
    if val_shards_found:
        try:
            val_loader = cached_distributed_data_loader_with_state(
                B=args.batch, T=args.seq_len, split=val_split,
                device="cpu", cache_dir=val_cache,
            )
            xv, yv, _ = next(val_loader)
            results.append(
                ("dataloader yields (B,T) tensors (val split)", True,
                 f"x={tuple(xv.shape)} y={tuple(yv.shape)} layout={'v4_val' if v4_layout else 'v3'}")
            )
        except Exception as e:
            results.append(
                ("dataloader yields (B,T) tensors (val split)", False, f"{type(e).__name__}: {e}")
            )
    else:
        results.append(
            ("dataloader yields (B,T) tensors (val split)", False, "skipped; <2 shards")
        )

    # Tiny model forward (CPU)
    if not args.skip_forward and x is not None:
        try:
            from nanochat.gpt import GPT, GPTConfig  # type: ignore

            cfg = GPTConfig(
                sequence_len=args.seq_len,
                vocab_size=32768,
                n_layer=2,
                n_head=4,
                n_kv_head=4,
                n_embd=128,
            )
            with torch.device("cpu"):
                model = GPT(cfg)
            model.init_weights()
            model.eval()
            with torch.no_grad():
                logits = model(x.to("cpu"))
            ok = (
                torch.isfinite(logits).all().item()
                and logits.shape[-1] == 32768
                and logits.shape[0] == args.batch
            )
            results.append(
                ("tiny model forward on CPU (no NaN/inf)", bool(ok),
                 f"logits shape={tuple(logits.shape)}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}"
                )
            )
        except Exception as e:
            import traceback
            results.append(
                ("tiny model forward on CPU", False,
                 f"{type(e).__name__}: {e}\n{traceback.format_exc()[:600]}")
            )
    else:
        results.append(("tiny model forward on CPU", False, "skipped (flag or no batch)"))

    # Write report
    overall = all(ok for _, ok, _ in results)
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    lines = [
        "# token_cache_v3 Smoke Report",
        "",
        f"- cache_dir: `{cache_dir}`",
        f"- batch: {args.batch}",
        f"- seq_len: {args.seq_len}",
        f"- timestamp: {ts}",
        f"- overall: **{'PASS' if overall else 'FAIL'}**",
        "",
        "| # | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for i, (name, ok, detail) in enumerate(results, 1):
        mark = "PASS" if ok else "FAIL"
        detail_short = detail.replace("\n", " ").replace("|", "\\|")[:400]
        lines.append(f"| {i} | {name} | {mark} | {detail_short} |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\noverall: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
