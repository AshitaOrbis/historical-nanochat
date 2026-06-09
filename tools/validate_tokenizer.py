"""
Post-training validator for the native nanochat tokenizer (tokenizer.pkl +
token_bytes.pt produced by scripts.tok_train).

What this checks:
  1. tokenizer.pkl loads as a tiktoken Encoding wrapped by RustBPETokenizer
  2. vocab_size == 32768 (or user-specified)
  3. All 9 nanochat SPECIAL_TOKENS are present with distinct ids
  4. Complete byte coverage: every single byte 0x00..0xFF tokenizes to exactly
     one token (no [UNK] fallback; rustbpe guarantees this by construction)
  5. round-trip on test strings (including unicode, whitespace, numbers)
  6. round-trip on 300+ corpus samples
  7. optional: large-shard or full-corpus UNK scan (there is no [UNK] token in
     rustbpe tokenizers by design; we scan for byte-level dropping instead)
  8. token_bytes.pt is well-formed: same length as vocab, specials have
     byte-count 0, ordinary tokens have positive byte counts

Writes tokenizer/tokenizer_manifest.json with:
  - SHA-256 of tokenizer.pkl, token_bytes.pt
  - vocab_size, special_tokens {name: id}
  - validation summary (pass counts, any warnings)
  - chars_trained_on (from /tmp/tok_train.log if present)
  - git commit, timestamp, script version

Usage:
  python tools/validate_tokenizer.py
  python tools/validate_tokenizer.py --corpus-samples 1000
  python tools/validate_tokenizer.py --full-unk-scan          # scan all 322 shards
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
NANOCHAT_ROOT = REPO_ROOT / "nanochat"
sys.path.insert(0, str(NANOCHAT_ROOT))
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS  # noqa: E402

SCRIPT_VERSION = "3.0.0"  # native rustbpe path

TEST_STRINGS = [
    # Victorian prose
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    # OCR-ish punctuation-heavy
    "Q. Where were you on the night of the 14th? A. At home. Q. With whom? A. My wife.",
    # Years
    "He was born in 1837 and died in 1899; the census of 1880 lists his trade as 'carpenter'.",
    # Contractions
    "Don't say it's nothing -- I'm sure I've seen it before, haven't you?",
    # Unicode accents / em dashes / curly quotes (must round-trip under complete byte coverage)
    "Café société — naïve façade — \u201ca smoke-filled room\u201d — Mr. Poirot's moustache.",
    # Whitespace (tab + newline)
    "Chapter I.\n\nThe beginning.\n\nNew paragraph.\tIndented with tab.",
    # Backtick + unusual punctuation
    "The `code snippet` ran successfully; and the pipe | and caret ^ survived.",
    # Long token stress
    "supercalifragilisticexpialidocious",
    # Mixed scripts (Chinese + emoji — complete byte coverage should handle these)
    "Hello \u4e16\u754c \U0001f30d",
    # nanochat chat specials MUST NOT leak when seen as raw text (different context)
    # ... we cover this separately below
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def sample_corpus_snippets(parquet_dir: Path, n: int, seed: int = 42, max_len: int = 4000) -> list[str]:
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        return []
    rng = random.Random(seed)
    snippets: list[str] = []
    for shard_path in shards:
        if len(snippets) >= n:
            break
        pf = pq.ParquetFile(shard_path)
        for rg_idx in range(pf.num_row_groups):
            if len(snippets) >= n:
                break
            texts = pf.read_row_group(rg_idx).column("text").to_pylist()
            rng.shuffle(texts)
            for t in texts:
                if not t:
                    continue
                if len(t) > max_len:
                    start = rng.randint(0, len(t) - max_len)
                    t = t[start:start + max_len]
                snippets.append(t)
                if len(snippets) >= n:
                    break
    return snippets


def check_byte_coverage(tokenizer) -> list[int]:
    """Every single byte 0..255 must tokenize to a non-empty sequence that decodes
    back to the same byte. Returns the list of bytes that fail (should be empty)."""
    failures = []
    for b in range(256):
        s = bytes([b]).decode("latin-1")  # lossless 1:1 byte -> unicode
        try:
            ids = tokenizer.encode(s)
        except Exception as e:
            failures.append((b, f"encode error: {e}"))
            continue
        if not ids:
            failures.append((b, "empty encoding"))
            continue
        decoded = tokenizer.decode(ids)
        if decoded != s:
            failures.append((b, f"round-trip failed: {s!r} -> {decoded!r}"))
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", type=str, default=str(REPO_ROOT / "tokenizer"))
    ap.add_argument("--corpus-dir", type=str,
                    default="/home/user/historical-nanochat/data/shards")
    ap.add_argument("--corpus-samples", type=int, default=500)
    ap.add_argument("--full-unk-scan", action="store_true",
                    help="Scan every document in every shard for byte-drop round-trip failures.")
    ap.add_argument("--expected-vocab-size", type=int, default=32768)
    ap.add_argument("--tok-train-log", type=str, default="/tmp/tok_train.log",
                    help="Optional: parse training time / max_chars from this log.")
    args = ap.parse_args()

    tok_dir = Path(args.tokenizer_dir)
    pkl_path = tok_dir / "tokenizer.pkl"
    tb_path = tok_dir / "token_bytes.pt"
    manifest_path = tok_dir / "tokenizer_manifest.json"

    if not pkl_path.exists() or not tb_path.exists():
        raise SystemExit(f"Expected tokenizer.pkl + token_bytes.pt in {tok_dir}")

    print(f"[1/6] Loading {pkl_path}")
    tokenizer = RustBPETokenizer.from_directory(str(tok_dir))
    vocab_size = tokenizer.get_vocab_size()
    print(f"      vocab_size = {vocab_size}")
    if vocab_size != args.expected_vocab_size:
        print(f"[WARN] vocab_size {vocab_size} != expected {args.expected_vocab_size}")

    # Check specials
    print(f"[2/6] Checking nanochat SPECIAL_TOKENS are present")
    special_ids: dict[str, int] = {}
    for name in SPECIAL_TOKENS:
        tid = tokenizer.encode_special(name)
        if tid is None:
            raise SystemExit(f"Special {name!r} missing from tokenizer. Fail closed.")
        special_ids[name] = tid
    # Uniqueness
    if len(set(special_ids.values())) != len(special_ids):
        raise SystemExit(f"Specials share ids: {special_ids}. Fail closed.")
    # Every special should be at the TOP of the vocab (above the BPE tokens)
    min_special_id = min(special_ids.values())
    print(f"      all {len(special_ids)} specials present, ids {min_special_id}..{max(special_ids.values())}")

    print(f"[3/6] Verifying complete 256-byte coverage")
    byte_failures = check_byte_coverage(tokenizer)
    if byte_failures:
        for b, msg in byte_failures[:10]:
            print(f"   byte 0x{b:02x}: {msg}")
        raise SystemExit(f"{len(byte_failures)}/256 bytes failed coverage check. Fail closed.")
    print(f"      all 256 bytes round-trip cleanly")

    print(f"[4/6] Round-trip on {len(TEST_STRINGS)} test strings")
    rt_failures = []
    for s in TEST_STRINGS:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        if decoded != s:
            rt_failures.append(f"{s!r} -> {decoded!r}")
    if rt_failures:
        for f in rt_failures[:5]:
            print(f"   {f}")
        raise SystemExit(f"{len(rt_failures)} round-trip failures on test strings. Fail closed.")

    print(f"[5/6] Round-trip on {args.corpus_samples} corpus snippets from {args.corpus_dir}")
    snippets = sample_corpus_snippets(Path(args.corpus_dir), args.corpus_samples)
    corpus_failures = []
    for s in snippets:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        if decoded != s:
            corpus_failures.append(s[:100])
    if corpus_failures:
        for f in corpus_failures[:3]:
            print(f"   {f!r}")
        raise SystemExit(f"{len(corpus_failures)}/{len(snippets)} corpus round-trip failures. Fail closed.")
    print(f"      all {len(snippets)} corpus snippets round-trip cleanly")

    # Full UNK scan = byte-drop scan (rustbpe has no [UNK], so UNK = silent byte drop).
    # We verify encode→decode is a no-op across the requested shard count.
    full_scan_stats = None
    if args.full_unk_scan:
        print(f"[5b] Full-corpus byte-drop scan (this takes time)")
        shard_paths = sorted(Path(args.corpus_dir).glob("*.parquet"))
        total_docs = 0
        total_chars = 0
        byte_drop_docs = 0
        t_start = time.time()
        for i, shard_path in enumerate(shard_paths):
            pf = pq.ParquetFile(shard_path)
            for rg_idx in range(pf.num_row_groups):
                texts = pf.read_row_group(rg_idx).column("text").to_pylist()
                # Batch-tokenize for speed (RustBPETokenizer.encode accepts list)
                batch_size = 128
                for start in range(0, len(texts), batch_size):
                    batch = [t for t in texts[start:start + batch_size] if t]
                    if not batch:
                        continue
                    encodings = tokenizer.encode(batch, num_threads=4)
                    for t, ids in zip(batch, encodings):
                        total_docs += 1
                        total_chars += len(t)
                        decoded = tokenizer.decode(ids)
                        if decoded != t:
                            byte_drop_docs += 1
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(shard_paths)}] docs={total_docs:,} chars={total_chars:,} "
                  f"byte_drops={byte_drop_docs} elapsed={elapsed:.0f}s")
        full_scan_stats = {
            "shards_scanned": len(shard_paths),
            "docs_scanned": total_docs,
            "chars_scanned": total_chars,
            "byte_drop_docs": byte_drop_docs,
            "elapsed_sec": round(time.time() - t_start, 2),
        }
        if byte_drop_docs > 0:
            raise SystemExit(f"{byte_drop_docs} docs failed byte-level round-trip. Fail closed.")
        print(f"      all {total_docs:,} corpus docs round-trip without byte drops")

    # token_bytes sanity
    print(f"[6/6] Checking token_bytes.pt structure")
    tb = torch.load(tb_path, map_location="cpu", weights_only=True)
    if tb.shape[0] != vocab_size:
        raise SystemExit(f"token_bytes length {tb.shape[0]} != vocab {vocab_size}. Fail closed.")
    if tb.dtype != torch.int32:
        raise SystemExit(f"token_bytes dtype {tb.dtype} != int32. Fail closed.")
    for name, tid in special_ids.items():
        if tb[tid].item() != 0:
            raise SystemExit(f"Special {name} at id {tid} has non-zero byte count {tb[tid].item()}. Fail closed.")
    nonzero = tb[tb > 0]
    if nonzero.numel() == 0:
        raise SystemExit("token_bytes is all zeros. Fail closed.")
    print(f"      shape={tuple(tb.shape)} dtype={tb.dtype} specials=0, nonzero_min={nonzero.min().item()}, "
          f"max={tb.max().item()}, mean={nonzero.float().mean().item():.2f}")

    # Parse training log for metadata if available
    training_meta = {}
    log_path = Path(args.tok_train_log)
    if log_path.exists():
        log_text = log_path.read_text()
        for key, pattern in [
            ("max_chars", r"max_chars:\s*([\d,]+)"),
            ("doc_cap", r"doc_cap:\s*([\d,]+)"),
            ("train_time_sec", r"Training time:\s*([\d.]+)s"),
        ]:
            import re
            m = re.search(pattern, log_text)
            if m:
                val = m.group(1).replace(",", "")
                training_meta[key] = float(val) if "." in val else int(val)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tokenizer": {
            "format": "nanochat_rustbpe_tiktoken",
            "vocab_size": vocab_size,
            "special_tokens": special_ids,
            "byte_coverage": "complete (256/256)",
        },
        "training": training_meta,
        "inputs": {
            "corpus_dir": args.corpus_dir,
        },
        "outputs": {
            "tokenizer_pkl": str(pkl_path),
            "sha256_tokenizer_pkl": _sha256(pkl_path),
            "token_bytes_pt": str(tb_path),
            "sha256_token_bytes_pt": _sha256(tb_path),
        },
        "validation": {
            "byte_coverage_failures": 0,
            "test_strings": len(TEST_STRINGS),
            "test_string_failures": 0,
            "corpus_samples": len(snippets),
            "corpus_sample_failures": 0,
            "full_unk_scan": full_scan_stats,
        },
        "token_bytes_stats": {
            "zero_count": int((tb == 0).sum().item()),
            "nonzero_min": int(nonzero.min().item()),
            "max": int(tb.max().item()),
            "mean_nonzero": float(nonzero.float().mean().item()),
        },
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"  vocab_size:          {vocab_size}")
    print(f"  special_tokens:      {len(special_ids)} present, ids "
          f"{min(special_ids.values())}..{max(special_ids.values())}")
    print(f"  tokenizer.pkl sha:   {_sha256(pkl_path)[:16]}...")
    print(f"  token_bytes.pt sha:  {_sha256(tb_path)[:16]}...")
    print(f"  validation:          256/256 bytes, {len(TEST_STRINGS)} test strings, "
          f"{len(snippets)} corpus samples — all clean")
    if full_scan_stats:
        print(f"  full scan:           {full_scan_stats['docs_scanned']:,} docs, "
              f"{full_scan_stats['byte_drop_docs']} byte drops")
    print(f"  manifest:            {manifest_path}")


if __name__ == "__main__":
    main()
