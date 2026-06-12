"""
Streaming UNK scan across all historical shards.

Why: the recovered HF tokenizer covers 151 of 256 possible single-byte tokens.
Bytes it can't tokenize (tab, CR, backtick, some 0x80+) fall back to [UNK] and
lose information on the return trip. Acceptable iff the actual training corpus
rarely triggers this. This script measures that rate, surfaces the worst
offenders, and gives you the data to decide whether to add a normalization
pre-filter.

Output: JSON report at --output. Shape:
  {
    "config": { ... },
    "totals": {
      "docs_scanned": N,
      "chars_scanned": C,
      "tokens_produced": T,
      "unk_tokens": U,
      "unk_rate_per_token": U/T,
      "docs_with_any_unk": D,
      "docs_with_high_unk": Dh,  # > threshold
    },
    "per_source": { source: {docs, chars, tokens, unk, rate, docs_with_unk} },
    "top_offenders": [ { source, file, rg_idx, doc_idx, unk_count, unk_rate, preview } ... ],
    "unk_byte_examples": [ { byte, count_as_unk, example_contexts } ... ],
  }

Usage:
  python tools/unk_scan.py --shards data/shards \
                           --output tokenizer/unk_scan.json \
                           --sample-every 1       # scan every doc
                           --top-n 20
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer as HFTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_source(shard_path: Path) -> str:
    """Shards are flat here (no subdirs); 'source' isn't known unless we peek
    at record metadata. Fall back to shard filename for grouping."""
    return shard_path.stem


def source_from_record(text: str) -> str | None:
    """Return None: this tokenizer's parquets store only the 'text' column
    (JSONL source metadata is upstream). Keep the signature for future use
    if the parquet schema grows a 'source' column."""
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=str, default="data/shards",
                    help="Directory of parquet shards.")
    ap.add_argument("--tokenizer-json", type=str, default=str(REPO_ROOT / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--output", type=str, default=str(REPO_ROOT / "tokenizer" / "unk_scan.json"))
    ap.add_argument("--sample-every", type=int, default=1,
                    help="Scan every Nth document (1 = full scan). Use >1 for a fast preview.")
    ap.add_argument("--max-docs", type=int, default=-1,
                    help="Stop after this many docs (-1 = no limit).")
    ap.add_argument("--top-n", type=int, default=20, help="Top offending docs to report.")
    ap.add_argument("--high-unk-threshold", type=float, default=0.01,
                    help="Flag docs whose UNK rate exceeds this (default 1%).")
    ap.add_argument("--max-shards", type=int, default=-1,
                    help="Process at most N shards (-1 = all).")
    args = ap.parse_args()

    shard_dir = Path(args.shards)
    shards = sorted(shard_dir.glob("*.parquet"))
    if args.max_shards > 0:
        shards = shards[:args.max_shards]
    if not shards:
        raise SystemExit(f"No parquet shards found in {shard_dir}")
    print(f"Scanning {len(shards)} shards...")

    hf_tok = HFTokenizer.from_file(args.tokenizer_json)
    unk_id = hf_tok.token_to_id("[UNK]")
    if unk_id is None:
        raise SystemExit("Tokenizer has no [UNK] token; UNK scan is meaningless.")
    bos_id = hf_tok.token_to_id("[BOS]")  # exclude BOS from counts even though we don't prepend here

    # Counters
    total_docs = 0
    total_chars = 0
    total_tokens = 0
    total_unk = 0
    docs_with_any_unk = 0
    docs_with_high_unk = 0

    # Per-shard breakdown (since we don't have source metadata in the parquet schema,
    # shard filename is our proxy — shards packed from the same source cluster together).
    per_shard = collections.defaultdict(
        lambda: {"docs": 0, "chars": 0, "tokens": 0, "unk": 0, "docs_with_unk": 0}
    )

    # Top offenders (heap-style: keep max top_n by unk_count, tie-break by rate)
    top_offenders: list[dict] = []
    # Rolling log of bytes/chars surrounding [UNK] tokens, to guess which bytes are failing.
    unk_context_samples: list[dict] = []
    UNK_CONTEXT_MAX = 50  # cap to keep memory bounded
    # Byte-level tally of characters in UNK-producing texts. We can't map tokens→bytes
    # cleanly without re-tokenization, so we count codepoints in docs that produced UNK.
    # This isn't as clean as bytes-that-failed, but it's fast and gives a good directional signal.

    t_start = time.time()
    batch_size = 256  # HF encode_batch is fastest with small-to-moderate batches
    doc_global_idx = 0

    for shard_idx, shard_path in enumerate(shards):
        pf = pq.ParquetFile(shard_path)
        pending_texts: list[str] = []
        pending_meta: list[tuple[int, int]] = []  # (rg_idx, doc_idx_within_rg)

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            for doc_idx_in_rg, text in enumerate(texts):
                if args.sample_every > 1 and (doc_global_idx % args.sample_every) != 0:
                    doc_global_idx += 1
                    continue
                if args.max_docs > 0 and total_docs >= args.max_docs:
                    break
                pending_texts.append(text or "")
                pending_meta.append((rg_idx, doc_idx_in_rg))
                doc_global_idx += 1

                if len(pending_texts) >= batch_size:
                    encodings = hf_tok.encode_batch(pending_texts, add_special_tokens=False)
                    for (rgi, didx), enc, txt in zip(pending_meta, encodings, pending_texts):
                        ids = enc.ids
                        if not ids:
                            continue
                        n_tokens = len(ids)
                        n_unk = sum(1 for i in ids if i == unk_id)
                        n_chars = len(txt)

                        total_docs += 1
                        total_chars += n_chars
                        total_tokens += n_tokens
                        total_unk += n_unk

                        shard_key = shard_path.name
                        per_shard[shard_key]["docs"] += 1
                        per_shard[shard_key]["chars"] += n_chars
                        per_shard[shard_key]["tokens"] += n_tokens
                        per_shard[shard_key]["unk"] += n_unk

                        if n_unk > 0:
                            docs_with_any_unk += 1
                            per_shard[shard_key]["docs_with_unk"] += 1
                            rate = n_unk / max(1, n_tokens)
                            if rate >= args.high_unk_threshold:
                                docs_with_high_unk += 1
                            # Keep a top-N list of worst offenders.
                            offender = {
                                "shard": shard_key,
                                "rg_idx": rgi,
                                "doc_idx": didx,
                                "tokens": n_tokens,
                                "unk_count": n_unk,
                                "unk_rate": rate,
                                "preview": txt[:200].replace("\n", "\\n"),
                            }
                            top_offenders.append(offender)
                            top_offenders.sort(key=lambda o: (o["unk_count"], o["unk_rate"]),
                                               reverse=True)
                            del top_offenders[args.top_n:]
                            # Sample unk contexts (up to UNK_CONTEXT_MAX globally).
                            if len(unk_context_samples) < UNK_CONTEXT_MAX:
                                tokens = enc.tokens
                                for k, tok in enumerate(tokens):
                                    if ids[k] == unk_id:
                                        left = "".join(tokens[max(0, k-3):k])
                                        right = "".join(tokens[k+1:k+4])
                                        unk_context_samples.append({
                                            "shard": shard_key,
                                            "left_tokens": left,
                                            "right_tokens": right,
                                        })
                                        if len(unk_context_samples) >= UNK_CONTEXT_MAX:
                                            break
                    pending_texts.clear()
                    pending_meta.clear()
            if args.max_docs > 0 and total_docs >= args.max_docs:
                break

        # Drain any remaining pending batch at end of shard.
        if pending_texts:
            encodings = hf_tok.encode_batch(pending_texts, add_special_tokens=False)
            for (rgi, didx), enc, txt in zip(pending_meta, encodings, pending_texts):
                ids = enc.ids
                if not ids:
                    continue
                n_tokens = len(ids)
                n_unk = sum(1 for i in ids if i == unk_id)
                n_chars = len(txt)
                total_docs += 1
                total_chars += n_chars
                total_tokens += n_tokens
                total_unk += n_unk
                shard_key = shard_path.name
                per_shard[shard_key]["docs"] += 1
                per_shard[shard_key]["chars"] += n_chars
                per_shard[shard_key]["tokens"] += n_tokens
                per_shard[shard_key]["unk"] += n_unk
                if n_unk > 0:
                    docs_with_any_unk += 1
                    per_shard[shard_key]["docs_with_unk"] += 1
                    if (n_unk / max(1, n_tokens)) >= args.high_unk_threshold:
                        docs_with_high_unk += 1
            pending_texts.clear()

        elapsed = time.time() - t_start
        print(f"  [{shard_idx+1}/{len(shards)}] {shard_path.name}: "
              f"docs={total_docs:,}, tokens={total_tokens:,}, unk={total_unk:,} "
              f"({total_unk / max(1, total_tokens) * 100:.4f}%), "
              f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - t_start
    for k in per_shard:
        bucket = per_shard[k]
        bucket["rate"] = bucket["unk"] / max(1, bucket["tokens"])

    report = {
        "config": {
            "shards_dir": str(shard_dir),
            "tokenizer_json": args.tokenizer_json,
            "sample_every": args.sample_every,
            "max_docs": args.max_docs,
            "max_shards": args.max_shards,
            "high_unk_threshold": args.high_unk_threshold,
        },
        "totals": {
            "shards_scanned": len(shards),
            "docs_scanned": total_docs,
            "chars_scanned": total_chars,
            "tokens_produced": total_tokens,
            "unk_tokens": total_unk,
            "unk_rate_per_token": total_unk / max(1, total_tokens),
            "docs_with_any_unk": docs_with_any_unk,
            "docs_with_high_unk_rate": docs_with_high_unk,
            "elapsed_sec": round(elapsed, 2),
        },
        "per_shard": dict(sorted(per_shard.items())),
        "top_offenders": top_offenders,
        "unk_context_samples": unk_context_samples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print()
    print("=" * 60)
    print(f"UNK SCAN SUMMARY  (elapsed: {elapsed:.0f}s)")
    print("=" * 60)
    print(f"  Shards scanned:        {len(shards)}")
    print(f"  Documents scanned:     {total_docs:,}")
    print(f"  Total chars:           {total_chars:,}")
    print(f"  Total tokens produced: {total_tokens:,}")
    print(f"  Total [UNK] tokens:    {total_unk:,}")
    unk_pct = (total_unk / max(1, total_tokens) * 100)
    print(f"  UNK rate per token:    {unk_pct:.6f}%")
    print(f"  Docs with any UNK:     {docs_with_any_unk:,} ({docs_with_any_unk / max(1, total_docs) * 100:.2f}%)")
    print(f"  Docs with UNK > {args.high_unk_threshold*100:.1f}%: {docs_with_high_unk:,}")
    print(f"  Report written to:     {args.output}")


if __name__ == "__main__":
    main()
