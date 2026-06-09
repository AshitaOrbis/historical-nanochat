"""
One-time recovery of the per-token byte-length tensor (`token_bytes.pt`) for the
historical-nanochat tokenizer. The tokenizer pickle (`tokenizer.pkl`) is
intentionally NOT produced — see "Why we don't write a tiktoken pickle" below.

Why we're here
--------------
The step-3000 base-training run used `tokenizer/tokenizer.json` (HuggingFace
BPE + ByteLevel pre-tokenizer, vocab=32000, specials `[UNK] [PAD] [BOS] [EOS]`).
The `tokenizer.pkl` and `token_bytes.pt` that nanochat's loader expects were
lost in the Feb 2026 cleanup. We need to regenerate just enough to make the
existing checkpoint resumable without changing a single token id.

Why we don't write a tiktoken pickle
------------------------------------
The project's tokenizer.json was trained with HF's BPE-with-byte-fallback and
only saw 151 of the 256 possible single-byte tokens (0x00-0x09 and others that
never appeared in the pre-1913 corpus are absent from the vocab). tiktoken's
BPE requires complete byte coverage in `mergeable_ranks` — it panics at
inference when it encounters an unseen byte. Synthesizing the missing 105
bytes would grow `vocab_size` beyond 32000 and break id parity with the
step-3000 checkpoint's embedding rows.

The clean solution is the one the nanochat code already supports natively:
use `HuggingFaceTokenizer.from_directory(...)` when a json is present and
there's no pickle. `nanochat.tokenizer.get_tokenizer()` has been updated to
auto-detect this (see the companion change in nanochat/tokenizer.py). The
token ids, merges, and regex pre-tokenization all remain identical to what
produced the step-3000 checkpoint — because we're using the same tokenizer
object that trained it.

What this script does
---------------------
1. Loads `tokenizer/tokenizer.json` via HuggingFace `tokenizers`.
2. Verifies structural invariants (vocab size, id contiguity, BPE + ByteLevel).
3. Builds `token_bytes.pt`: `int32[vocab_size]` where each entry is the UTF-8
   byte-length of the corresponding token's decoded string; specials get 0.
   This matches `scripts/tok_train.py` exactly so the bpb metric is unchanged.
4. Validates `decode(encode(x)) == x` on a test suite and optional corpus
   samples. Fails closed on any mismatch.
5. Writes `tokenizer/tokenizer_manifest.json` with SHA-256 hashes, specials,
   vocab size, conversion-script version + git commit, timestamp, and
   documented deviations (so future debugging has a chain of custody).

Usage:
    python tools/recover_tokenizer_artifacts.py
    python tools/recover_tokenizer_artifacts.py --corpus-samples 500
    python tools/recover_tokenizer_artifacts.py --force
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
from tokenizers import Tokenizer as HFTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
NANOCHAT_ROOT = REPO_ROOT / "nanochat"
sys.path.insert(0, str(NANOCHAT_ROOT))
from nanochat.tokenizer import SPECIAL_TOKENS as NANOCHAT_SPECIAL_TOKENS  # noqa: E402

SCRIPT_VERSION = "2.0.0"  # v2: HF-native path, no tiktoken pickle


# ---------------------------------------------------------------------------
# Test suite (per user spec #9)
# ---------------------------------------------------------------------------

# Test strings chosen to exercise the tokenizer's actual byte coverage. This
# tokenizer was trained on the pre-1913 corpus only, so its vocab covers 151
# of 256 possible bytes — notably MISSING: tab (0x09), CR (0x0d), backtick,
# most 0x80+ bytes (curly quotes, non-breaking hyphen, em dash '—' etc.).
# Round-trip on unseen bytes is expected to substitute [UNK]; that's correct
# behavior for this tokenizer, not corruption. We separately verify the UNK
# fallback below. The strings here use ASCII + the specific non-ASCII bytes
# actually present in the corpus.
TEST_STRINGS = [
    # Victorian prose
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    # OCR-ish punctuation-heavy
    'Q. Where were you on the night of the 14th? A. At home. Q. With whom? A. My wife.',
    # Years (common in historical corpus)
    "He was born in 1837 and died in 1899; the census of 1880 lists his trade as 'carpenter'.",
    "Proclaimed in 1913: one hundred forty-seven pounds, three shillings, sixpence.",
    # Contractions + straight quotes
    "Don't say it's nothing -- I'm sure I've seen it before, haven't you?",
    # Whitespace (newline is in-vocab; tab is NOT — tested separately)
    "Chapter I.\n\nThe beginning.\n\nThis is a new paragraph.  Indented with spaces.",
    # Strings with token-like sequences that are NOT in this tokenizer's vocab.
    # nanochat specials like <|bos|>, <|user_start|> don't exist here, so their
    # literal text must round-trip cleanly. Do NOT add strings containing literal
    # `[BOS]` / `[UNK]` here — HF treats those as special tokens in any context,
    # which is the intended training behavior (see validate_specials_not_merged).
    "Someone wrote <|bos|> in the margin, but it was just a note.",
    "The user said <|user_start|> but this is ordinary text.",
    # Long token stress
    "supercalifragilisticexpialidocious",
    # Mixed / digits
    "The year A.D. 1666 -- the Great Fire of London, 347 years before 2013.",
]

# Strings containing at least one byte/codepoint that this tokenizer can't
# represent via any BPE merge. These MUST encode to a sequence containing
# [UNK]. NB: curly quotes / em-dashes are actually in-vocab via multi-byte
# merges (the pre-1913 corpus contained them), so they don't belong here.
# Tab (0x09) and backtick (0x60) are genuinely missing from every merge path.
UNSEEN_BYTE_STRINGS = [
    "Paragraph\twith\ttabs.",               # tab (0x09) is not in any merge
    "A `backtick` mark in the text.",       # backtick (0x60) is absent
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def verify_tokenizer_json_structure(tokenizer_json_path: Path) -> dict:
    """Per user spec #6/#9: fail closed on structural surprises."""
    raw = json.loads(tokenizer_json_path.read_text())
    model = raw.get("model", {})
    if model.get("type") != "BPE":
        raise SystemExit(f"Expected model.type == 'BPE', got {model.get('type')!r}. Fail closed.")

    pre = raw.get("pre_tokenizer") or {}
    if pre.get("type") != "ByteLevel":
        raise SystemExit(
            f"Expected pre_tokenizer.type == 'ByteLevel', got {pre.get('type')!r}. Fail closed."
        )
    if not pre.get("use_regex", True):
        raise SystemExit(
            "pre_tokenizer.use_regex is False; this is an unexpected configuration. Fail closed."
        )

    vocab = model["vocab"]
    added = {at["id"] for at in raw.get("added_tokens", []) or []}
    all_ids = sorted(set(vocab.values()) | added)
    if all_ids[0] != 0 or all_ids[-1] != len(all_ids) - 1:
        raise SystemExit(
            f"Token ids not contiguous 0..{len(all_ids)-1}: min={all_ids[0]}, max={all_ids[-1]}, "
            f"count={len(all_ids)}. Fail closed."
        )

    return {
        "vocab_size": len(all_ids),
        "special_tokens": {at["content"]: at["id"] for at in (raw.get("added_tokens", []) or [])
                          if at.get("special")},
        "raw": raw,
    }


def build_token_bytes(hf_tok: HFTokenizer, vocab_size: int, special_ids: set[int]) -> torch.Tensor:
    """Mirror `scripts/tok_train.py` lines 78-87 exactly, but for the HF path.

    For each token id in 0..vocab_size-1:
      - if the id is a special: byte count = 0 (excluded from bpb)
      - else: byte count = len(decoded_string.encode('utf-8'))

    This preserves the bpb metric across this recovery — no change in the
    definition of bits-per-byte versus tok_train.
    """
    byts = torch.zeros(vocab_size, dtype=torch.int32)
    for tid in range(vocab_size):
        if tid in special_ids:
            continue
        tok_str = hf_tok.decode([tid], skip_special_tokens=False)
        byts[tid] = len(tok_str.encode("utf-8"))
    return byts


def validate_roundtrip(hf_tok: HFTokenizer, strings: list[str]) -> list[str]:
    failures = []
    for s in strings:
        ids = hf_tok.encode(s, add_special_tokens=False).ids
        decoded = hf_tok.decode(ids, skip_special_tokens=False)
        if decoded != s:
            failures.append(f"{s!r} -> {decoded!r}")
    return failures


def validate_unk_fallback(hf_tok: HFTokenizer, strings: list[str], unk_id: int) -> list[str]:
    """Strings containing bytes outside the vocab MUST encode to a sequence that
    contains at least one [UNK] token. This proves the tokenizer's fallback is
    active; silent byte loss (no [UNK] in output) would mean we're corrupting
    text without a warning."""
    failures = []
    for s in strings:
        ids = hf_tok.encode(s, add_special_tokens=False).ids
        if unk_id not in ids:
            failures.append(
                f"Expected [UNK] in encoding of text with unseen bytes but got none: {s!r} -> ids[:20]={ids[:20]}"
            )
    return failures


def validate_specials_not_merged(hf_tok: HFTokenizer) -> list[str]:
    """Ensure strings containing token-like sequences that AREN'T in this tokenizer's
    special_tokens set encode as ordinary characters, not as rogue special ids.

    NB: strings containing literal in-vocab specials (like `[BOS]`) WILL be encoded
    as the matching special id — that's HF's documented behavior for added_tokens,
    consistent with how the training corpus was tokenized. We only fail here if a
    token name that doesn't exist in the vocab somehow encodes to a nonzero ID
    (which would indicate pattern-level contamination)."""
    failures = []
    for probe, spoofed in [
        # nanochat specials are NOT in this vocab; their literal text should
        # tokenize as ordinary bytes, never as a special id.
        ("Someone wrote <|bos|> in the margin, but it was just a note.", "<|bos|>"),
        ("The user said <|user_start|> but this is ordinary text.", "<|user_start|>"),
    ]:
        special_id = hf_tok.token_to_id(spoofed)
        if special_id is None:
            continue  # expected — spoofed token not in vocab
        ids = hf_tok.encode(probe, add_special_tokens=False).ids
        if special_id in ids:
            failures.append(f"Special {spoofed!r} leaked into ordinary encoding of: {probe!r}")
    return failures


def sample_corpus_snippets(parquet_dir: Path, n: int, seed: int = 42) -> list[str]:
    if not parquet_dir.exists():
        return []
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
                if len(t) > 1024:
                    start = rng.randint(0, len(t) - 1024)
                    t = t[start:start + rng.randint(100, 1024)]
                snippets.append(t)
                if len(snippets) >= n:
                    break
    return snippets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", type=str, default=str(REPO_ROOT / "tokenizer"))
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--corpus-dir", type=str,
                    default="/home/user/historical-nanochat/data/shards")
    ap.add_argument("--corpus-samples", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tok_dir = Path(args.tokenizer_dir)
    out_dir = Path(args.output_dir) if args.output_dir else tok_dir
    tok_json = tok_dir / "tokenizer.json"
    if not tok_json.exists():
        raise SystemExit(f"tokenizer.json not found at {tok_json}")

    tb_out = out_dir / "token_bytes.pt"
    manifest_out = out_dir / "tokenizer_manifest.json"

    if tb_out.exists() and not args.force:
        raise SystemExit(
            f"{tb_out} already exists. Pass --force to overwrite, or delete the file first."
        )

    print(f"[1/5] Reading {tok_json}")
    structure = verify_tokenizer_json_structure(tok_json)
    vocab_size = structure["vocab_size"]
    specials_in_json = structure["special_tokens"]
    print(f"      vocab_size={vocab_size}")
    print(f"      specials:   {specials_in_json}")

    # Warn about missing nanochat specials (expected for this tokenizer).
    missing = [t for t in NANOCHAT_SPECIAL_TOKENS if t not in specials_in_json]
    if missing:
        print(
            f"[WARN] nanochat specials not present in this tokenizer: {missing}\n"
            "       Base training/eval works because BOS falls back to [BOS] (see\n"
            "       HuggingFaceTokenizer.get_bos_token_id). Midtraining/SFT would need\n"
            "       tokenizer retraining or vocab extension — out of scope for reviving\n"
            "       the step-3000 base run."
        )

    print(f"[2/5] Loading HF tokenizer")
    hf_tok = HFTokenizer.from_file(str(tok_json))
    if hf_tok.get_vocab_size() != vocab_size:
        raise SystemExit(
            f"vocab size mismatch: structure={vocab_size}, HFTokenizer={hf_tok.get_vocab_size()}"
        )

    print(f"[3/5] Validating HF tokenizer on {len(TEST_STRINGS)} test strings")
    rt_failures = validate_roundtrip(hf_tok, TEST_STRINGS)
    if rt_failures:
        for f in rt_failures[:5]:
            print("  " + f)
        raise SystemExit(f"{len(rt_failures)} round-trip failures. Fail closed.")

    leak_failures = validate_specials_not_merged(hf_tok)
    if leak_failures:
        for f in leak_failures[:5]:
            print("  " + f)
        raise SystemExit(f"{len(leak_failures)} special-token leakage failures. Fail closed.")

    unk_id = specials_in_json.get("[UNK]")
    if unk_id is None:
        print("[WARN] No [UNK] token found; skipping UNK-fallback verification.")
        unk_failures = []
    else:
        print(f"      Verifying UNK fallback ({len(UNSEEN_BYTE_STRINGS)} strings with unseen bytes)")
        unk_failures = validate_unk_fallback(hf_tok, UNSEEN_BYTE_STRINGS, unk_id)
        if unk_failures:
            for f in unk_failures:
                print("  " + f)
            raise SystemExit(f"{len(unk_failures)} UNK-fallback failures. Fail closed.")

    if args.corpus_samples > 0:
        print(f"[4/5] Validating round-trip on {args.corpus_samples} corpus snippets from {args.corpus_dir}")
        snippets = sample_corpus_snippets(Path(args.corpus_dir), args.corpus_samples)
        if len(snippets) < args.corpus_samples:
            print(f"      (only got {len(snippets)} snippets — corpus smaller than requested)")
        corpus_failures = validate_roundtrip(hf_tok, snippets)
        if corpus_failures:
            for f in corpus_failures[:3]:
                print("  " + f[:200])
            raise SystemExit(f"{len(corpus_failures)}/{len(snippets)} corpus round-trip failures. Fail closed.")
    else:
        print(f"[4/5] Skipping corpus validation (--corpus-samples=0)")
        snippets = []

    print(f"[5/5] Writing {tb_out}")
    special_ids = set(specials_in_json.values())
    token_bytes = build_token_bytes(hf_tok, vocab_size, special_ids)
    with tb_out.open("wb") as f:
        torch.save(token_bytes, f)

    print(f"      Writing manifest {manifest_out}")
    manifest = {
        "script_version": SCRIPT_VERSION,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {
            "tokenizer_json": str(tok_json),
            "sha256_tokenizer_json": _sha256(tok_json),
        },
        "outputs": {
            "token_bytes_pt": str(tb_out),
            "sha256_token_bytes_pt": _sha256(tb_out),
            # No tokenizer.pkl — see 'pickle_skipped_reason' below.
        },
        "tokenizer": {
            "format": "huggingface_bytelevel_bpe",
            "vocab_size": vocab_size,
            "special_tokens": specials_in_json,
            "missing_nanochat_specials": missing,
        },
        "validation": {
            "test_strings": len(TEST_STRINGS),
            "test_string_roundtrip_failures": 0,
            "special_leak_failures": 0,
            "unseen_byte_strings": len(UNSEEN_BYTE_STRINGS),
            "unk_fallback_failures": 0,
            "corpus_samples": len(snippets),
            "corpus_roundtrip_failures": 0,
        },
        "token_bytes_stats": {
            "nonzero_count": int((token_bytes > 0).sum().item()),
            "zero_count": int((token_bytes == 0).sum().item()),
            "min_nonzero": int(token_bytes[token_bytes > 0].min().item()),
            "max": int(token_bytes.max().item()),
            "mean_nonzero": float(token_bytes[token_bytes > 0].float().mean().item()),
        },
        "pickle_skipped_reason": (
            "This tokenizer's vocab covers only 151 of 256 possible single-byte tokens; "
            "the remaining 105 never appeared in the pre-1913 corpus. tiktoken's BPE "
            "requires complete byte coverage or it panics at inference. Synthesizing "
            "the missing bytes would grow vocab_size beyond 32000 and break id parity "
            "with the step-3000 checkpoint. We use HuggingFaceTokenizer directly "
            "(get_tokenizer() auto-detects tokenizer.json when tokenizer.pkl is absent)."
        ),
        "notes": [
            "Token IDs are preserved exactly: we use the same HuggingFaceTokenizer "
            "instance that produced the step-3000 checkpoint. No conversion, no renaming, "
            "no merges regenerated.",
            "[BOS] (id=2) is used as the BOS token via HuggingFaceTokenizer.get_bos_token_id's "
            "fallback chain (<|bos|> -> <|endoftext|> -> [BOS]). Unchanged by this recovery.",
            "token_bytes.pt is regenerated from scratch, mirroring scripts/tok_train.py "
            "lines 78-87 exactly: specials get 0, ordinary tokens get UTF-8 byte length of "
            "decode([id]).",
        ],
    }
    with manifest_out.open("w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"  vocab_size:     {vocab_size}")
    print(f"  token_bytes:    {tb_out}")
    print(f"  manifest:       {manifest_out}")
    print(f"  BOS token:      [BOS] (id={specials_in_json.get('[BOS]')})")
    print(f"  specials found: {dict(sorted(specials_in_json.items(), key=lambda x: x[1]))}")
    if missing:
        print(f"  missing specials (OK for base): {missing}")
    print(f"  validation: {len(TEST_STRINGS)} test strings + {len(snippets)} corpus samples, "
          f"all round-trip + no special leaks")
    print()
    print("Next: nanochat.tokenizer.get_tokenizer() now auto-detects tokenizer.json and will")
    print("load via HuggingFaceTokenizer. No tokenizer.pkl is written (see manifest for why).")


if __name__ == "__main__":
    main()
