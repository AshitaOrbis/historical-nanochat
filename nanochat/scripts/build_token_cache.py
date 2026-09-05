"""
Build an offline token cache for a directory of parquet shards.

Optional optimization: on-the-fly tokenization in the dataloader is fine when
the GPU is the bottleneck, but on a 3090 with activation checkpointing + a
heavy CPU-bound BPE it can become the limiter. This script precomputes a
binary token stream (uint16 or uint32) per shard so the dataloader can mmap
and feed tokens without doing BPE during training.

Output layout:
    <cache_dir>/
        shard_00000.bin       # raw contiguous token ids (uint16 if vocab < 65536 else uint32)
        shard_00000.meta.json # { "tokens": N, "docs": D, "source_file": "<path>" }
        cache_manifest.json   # global summary + dtype info

Consumed by `nanochat.dataloader_cached` (pass `--token_cache_dir` to base_train.py).
Deliberately simple: doesn't replicate the row-group semantics of the parquet
dataloader — it's a flat token stream per shard, which is what cached training
needs.

Usage:
    python -m scripts.build_token_cache --input-dir path/to/shards --output-dir path/to/cache
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from nanochat.artifact_guard import validate_tokenizer_artifacts
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_token_bytes, get_tokenizer


def tokenize_shard(parquet_path: str, tokenizer, bos_token: int,
                   row_group_size: int, tokenizer_threads: int) -> tuple:
    """Tokenize one parquet file. Returns (np.ndarray of token ids, n_docs)."""
    pf = pq.ParquetFile(parquet_path)
    all_tokens: list[int] = []
    n_docs = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column('text').to_pylist()
        n_docs += len(texts)
        for i in range(0, len(texts), row_group_size):
            batch = texts[i:i + row_group_size]
            token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=tokenizer_threads)
            for toks in token_lists:
                all_tokens.extend(toks)
    return np.asarray(all_tokens, dtype=np.int64), n_docs


def pick_dtype(vocab_size: int):
    return np.dtype("<u2") if vocab_size < 65536 else np.dtype("<u4")


def resolve_input_shards(input_dir: Path) -> list:
    """
    Resolve the ordered list of parquet shards to tokenize from input_dir.

    If input_dir holds a manifest.json (written by
    data/process/shard_packager.py), that manifest is the source of truth for
    which shards belong to the current corpus. A shard_*.parquet file on disk
    but absent from the manifest is an obsolete leftover from a prior
    packaging run into this same directory and must not be silently tokenized
    into the cache alongside the current corpus; a manifest-listed shard
    missing from disk means the corpus is incompletely materialized. Both
    raise SystemExit rather than proceeding on a partial/contaminated shard
    set. Directories with no manifest.json fall back to plain glob discovery.
    """
    on_disk = sorted(p for p in input_dir.glob("*.parquet"))
    packager_manifest_path = input_dir / "manifest.json"
    if not packager_manifest_path.exists():
        return on_disk
    with packager_manifest_path.open(encoding="utf-8") as manifest_file:
        packager_manifest = json.load(manifest_file)
    manifest_names = sorted(
        shard["filename"] for shard in packager_manifest.get("shards", [])
    )
    on_disk_names = [p.name for p in on_disk]
    missing = [name for name in manifest_names if name not in on_disk_names]
    if missing:
        raise SystemExit(
            f"{packager_manifest_path} lists shard(s) missing from {input_dir}: {missing}"
        )
    unexpected = [name for name in on_disk_names if name not in manifest_names]
    if unexpected:
        raise SystemExit(
            f"{input_dir} has parquet file(s) not listed in {packager_manifest_path}: "
            f"{unexpected}. These look like obsolete shards left behind by a prior "
            "packaging run into this same directory; repackage with --replace or remove "
            "the stale files before building a token cache from this data."
        )
    return [input_dir / name for name in manifest_names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory of parquet shards")
    ap.add_argument("--output-dir", required=True, help="Directory to write .bin files + manifest")
    ap.add_argument("--tokenizer-batch-size", type=int, default=128)
    ap.add_argument("--tokenizer-threads", type=int, default=4)
    ap.add_argument("--max-shards", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_dir = Path(args.input_dir)
    shards = resolve_input_shards(input_dir)
    if not shards:
        raise SystemExit(f"No parquet shards in {args.input_dir}")
    if args.max_shards:
        shards = shards[:args.max_shards]

    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    vocab = tokenizer.get_vocab_size()
    artifact_identity = validate_tokenizer_artifacts(
        get_base_dir(), tokenizer, get_token_bytes(device="cpu")
    )
    dtype = pick_dtype(vocab)
    print(f"Tokenizer vocab: {vocab}, cache dtype: {dtype}")

    manifest = {
        "format_version": 1,
        "byte_order": "little",
        "input_dir": str(input_dir),
        "vocab_size": vocab,
        "dtype": dtype.name,
        "tokenizer_identity": artifact_identity,
        "shards": [],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    total_tokens = 0
    total_docs = 0
    t_start = time.time()
    for shard_idx, shard_path in enumerate(shards):
        t0 = time.time()
        tokens, n_docs = tokenize_shard(str(shard_path), tokenizer, bos,
                                        args.tokenizer_batch_size, args.tokenizer_threads)
        # Cast down to the cache dtype; overflow should be impossible given pick_dtype.
        tokens_cast = tokens.astype(dtype)
        out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
        tokens_cast.tofile(out_path)

        meta = {
            "shard_index": shard_idx,
            "source_file": str(shard_path),
            "docs": n_docs,
            "tokens": int(tokens.shape[0]),
            "bytes": os.path.getsize(out_path),
        }
        with open(out_path.replace(".bin", ".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        manifest["shards"].append(meta)

        total_tokens += tokens.shape[0]
        total_docs += n_docs
        dt = time.time() - t0
        print(f"[{shard_idx+1}/{len(shards)}] {shard_path.name}: {n_docs:,} docs, "
              f"{tokens.shape[0]:,} tokens, {dt:.1f}s  ({tokens.shape[0]/max(dt,1e-6):,.0f} tok/s)")

    manifest["total_tokens"] = total_tokens
    manifest["total_docs"] = total_docs
    manifest["elapsed_sec"] = time.time() - t_start
    with open(os.path.join(args.output_dir, "cache_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(shards)} shards to {args.output_dir}")
    print(f"Total: {total_docs:,} docs, {total_tokens:,} tokens in {manifest['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
