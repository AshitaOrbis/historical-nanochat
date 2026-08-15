"""
Package historical texts into nanochat-compatible parquet shards.

Matches the format expected by nanochat/dataset.py:
- Parquet files with 'text' column
- ~250M characters per shard
- Row group size of 1024
- zstd compression

Streaming-first (bounded shuffle buffer) so large corpora don't blow out RAM.
Emits a manifest.json alongside the shards with per-shard document counts,
character counts, and per-source distributions, plus a rejection report so
operators can see *why* documents were dropped.

Legacy behavior (full in-memory shuffle) is gone; the old `package_shards`
function is retained as a thin wrapper that forwards to the streaming path.
"""

import os
import json
import argparse
import hashlib
import random
from typing import Iterator, List, Optional, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from data.process.contamination_check import (
        check_contamination,
        clean_gutenberg_headers,
    )
except ImportError:
    check_contamination = None
    clean_gutenberg_headers = None


# Default sampling rates for balanced corpus.
# Values < 1.0 downsample, = 1.0 keep all. Keys match detect_source_from_path().
DEFAULT_SAMPLE_RATES = {
    'american_stories': 0.33,  # 54.5B -> ~18B
    'bhl': 0.60,               # 29.9B -> ~18B
    'bl_newspapers': 1.0,      # Keep all 2.8B
    'gutenberg': 1.0,          # Keep all 1.7B
    'eebo': 1.0,               # Keep all 1.6B
    'tcp': 1.0,                # Keep all 0.25B
    'oldbailey': 1.0,          # Keep all 0.11B
    'default': 1.0,            # Keep unknown sources
}


# Sources where the text comes from OCR'd scans; more aggressive quality gating helps.
# Oldbailey is intentionally excluded: it's court transcripts with short Q/A lines that
# would fail the short-line heuristic. Gutenberg is also excluded because Project Gutenberg
# texts are hand-corrected, not OCR (and include poetry/plays with short lines).
OCR_HEAVY_SOURCES = {"american_stories", "bhl", "bl_newspapers", "chronicling", "eebo", "tcp"}

# Sources where prose structure is atypical (verse, transcripts, reference works) and
# the default OCR-quality thresholds would false-positive. These bypass the OCR filter
# entirely; rely on length + contamination + dedup for gating instead.
OCR_FILTER_BYPASS = {"oldbailey", "gutenberg"}


def ocr_quality_ok(
    text: str,
    min_alpha_ratio: float = 0.65,
    max_short_line_ratio: float = 0.80,
    min_avg_word_len: float = 2.5,
) -> bool:
    """
    Cheap OCR-quality heuristic. Returns False when the text looks like OCR garbage.

    Signals we check (all tunable by caller):
      - Alpha ratio: fraction of alphabetic characters. OCR garbage is dominated by
        punctuation, numbers, and symbols. 65% is a loose floor for English prose.
      - Short-line ratio: fraction of lines shorter than 20 chars. Column-broken OCR
        produces hundreds of fragmentary lines.
      - Avg word length: OCR of text with frequent rec errors fragments into
        single-character 'words'.

    This is a fast pre-filter, not a classifier; rely on it for coarse rejection
    before more expensive checks (contamination, dedup).
    """
    if not text:
        return False
    n = len(text)
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / n < min_alpha_ratio:
        return False
    lines = text.splitlines()
    if lines:
        short = sum(1 for line in lines if len(line.strip()) < 20)
        if short / len(lines) > max_short_line_ratio:
            return False
    words = text.split()
    if words:
        avg_wl = sum(len(w) for w in words) / len(words)
        if avg_wl < min_avg_word_len:
            return False
    return True


def content_fingerprint(text: str, head_tail_chars: int = 512, mid_chars: int = 128) -> str:
    """
    Build a near-dedup fingerprint: sha1 of (len, first N, middle M, last N chars).

    Conservative: catches exact duplicates and re-ingests that share prefix/suffix
    (same newspaper page scanned twice). The middle slice drops collision probability
    on boilerplate-heavy sources like newspaper archives where many pages share an
    identical masthead and footer ≥512 chars long. Doesn't do semantic near-dup —
    that would need MinHash/SimHash and is out of scope for a pre-training filter.
    """
    normalized = " ".join(text.split())  # collapse whitespace so trivial reformats match
    n = len(normalized)
    # Middle slice: safe even when n < mid_chars (Python handles overrun by clamping).
    mid_start = max(0, n // 2 - mid_chars // 2)
    mid = normalized[mid_start:mid_start + mid_chars]
    key = f"{n}|{normalized[:head_tail_chars]}|{mid}|{normalized[-head_tail_chars:]}"
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()


def detect_source_from_path(filepath: str) -> str:
    """Detect source name from file path (parent dir > filename > 'default')."""
    path = Path(filepath)
    parent = path.parent.name.lower()
    for source in DEFAULT_SAMPLE_RATES:
        if source != 'default' and source in parent:
            return source
    name = path.stem.lower()
    for source in DEFAULT_SAMPLE_RATES:
        if source != 'default' and source in name:
            return source
    return 'default'


def iter_jsonl_texts_streaming(
    input_files: List[str],
    sample_rates: Dict[str, float],
    seed: int = 42,
    min_length: int = 100,
    cutoff_year: Optional[int] = None,
    run_contamination: bool = False,
    progress: bool = True,
    rejection_report: Optional[Dict] = None,
    run_ocr_quality: bool = True,
    run_dedup: bool = True,
    seen_fingerprints: Optional[set] = None,
) -> Iterator[Tuple[str, str]]:
    """
    Stream (text, source) tuples from JSONL files.

    Memory efficient - processes one line at a time. Yields source label
    so downstream code can track per-source stats.

    If `run_contamination` is True (and cutoff_year set and the checker is
    importable) contaminated docs are dropped. Gutenberg records get header
    cleanup inline so headers don't look like contamination.

    If `rejection_report` dict is passed, per-file counts (accepted, sampling,
    length, contamination, json_error) are appended under rejection_report['per_file'].
    Callers typically hand this to the packager to persist alongside the manifest.
    """
    rng = random.Random(seed)
    per_file = rejection_report['per_file'] if rejection_report is not None else None
    if seen_fingerprints is None and run_dedup:
        seen_fingerprints = set()

    for filepath in input_files:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            if per_file is not None:
                per_file.append({"file": filepath, "error": "not_found"})
            continue

        source = detect_source_from_path(filepath)
        rate = sample_rates.get(source, sample_rates.get('default', 1.0))

        file_size = os.path.getsize(filepath)
        print(f"Processing: {filepath}")
        print(f"  Source: {source}, Sample rate: {rate:.2f}")

        counts = {
            "file": filepath,
            "source": source,
            "sample_rate": rate,
            "accepted": 0,
            "rejected_sampling": 0,
            "rejected_length": 0,
            "rejected_contamination": 0,
            "rejected_ocr_quality": 0,
            "rejected_duplicate": 0,
            "json_errors": 0,
        }

        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"  {source}") if progress else None
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if pbar:
                    pbar.update(len(line.encode('utf-8')))

                line = line.strip()
                if not line:
                    continue

                # Per-source downsampling happens first — cheapest reject.
                if rate < 1.0 and rng.random() > rate:
                    counts["rejected_sampling"] += 1
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    counts["json_errors"] += 1
                    continue

                text = record.get('text', '')
                record_source = record.get('source', source)

                # Gutenberg headers/footers look like contamination, so strip them first.
                if record_source == 'gutenberg' and clean_gutenberg_headers is not None:
                    text = clean_gutenberg_headers(text)

                if not text or len(text) < min_length:
                    counts["rejected_length"] += 1
                    continue

                # Cheap quality gate for OCR-heavy sources (runs before contamination).
                # Bypass sources that have atypical prose structure (transcripts, verse).
                if (run_ocr_quality
                        and record_source in OCR_HEAVY_SOURCES
                        and record_source not in OCR_FILTER_BYPASS):
                    if not ocr_quality_ok(text):
                        counts["rejected_ocr_quality"] += 1
                        continue

                if run_contamination and check_contamination is not None and cutoff_year is not None:
                    result = check_contamination(text, cutoff_year)
                    if result.is_contaminated:
                        counts["rejected_contamination"] += 1
                        continue

                # Near-dedup via content fingerprint. Safe under streaming: fingerprint
                # set is small (40 bytes per doc) compared to the text itself.
                if run_dedup and seen_fingerprints is not None:
                    fp = content_fingerprint(text)
                    if fp in seen_fingerprints:
                        counts["rejected_duplicate"] += 1
                        continue
                    seen_fingerprints.add(fp)

                counts["accepted"] += 1
                yield text, record_source

        if pbar:
            pbar.close()
        print(f"  Accepted: {counts['accepted']:,}  "
              f"Rejected: sampling={counts['rejected_sampling']:,} "
              f"length={counts['rejected_length']:,} "
              f"ocr_quality={counts['rejected_ocr_quality']:,} "
              f"contamination={counts['rejected_contamination']:,} "
              f"duplicate={counts['rejected_duplicate']:,}")
        if per_file is not None:
            per_file.append(counts)


def package_shards_streaming(
    input_files: List[str],
    output_dir: str,
    sample_rates: Optional[Dict[str, float]] = None,
    chars_per_shard: int = 250_000_000,
    row_group_size: int = 1024,
    seed: int = 42,
    max_shards: Optional[int] = None,
    max_tokens: Optional[int] = None,
    buffer_size: int = 50000,
    cutoff_year: Optional[int] = None,
    run_contamination: bool = False,
    run_ocr_quality: bool = True,
    run_dedup: bool = True,
) -> dict:
    """
    Package texts into parquet shards using a bounded shuffle buffer.

    Keeps memory ~O(buffer_size) documents regardless of corpus size. Writes
    a manifest.json next to the shards with:
      - per-shard doc/char counts and source distribution
      - per-source totals across all shards
      - sample_rates used and seed
      - contamination rejection counts (when enabled)
    """
    os.makedirs(output_dir, exist_ok=True)
    if sample_rates is None:
        sample_rates = DEFAULT_SAMPLE_RATES

    print("=" * 60)
    print("STREAMING SHARD PACKAGER")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Characters per shard: {chars_per_shard:,}")
    print(f"Shuffle buffer size: {buffer_size:,}")
    print(f"Contamination check: {'on' if run_contamination else 'off'}"
          f" (cutoff_year={cutoff_year})")
    if max_tokens:
        print(f"Max tokens: {max_tokens:,}")
    if max_shards:
        print(f"Max shards: {max_shards}")

    rng = random.Random(seed)
    stats = {
        "total_docs": 0,
        "total_chars": 0,
        "num_shards": 0,
        "source_counts": {},          # {source: {"docs": N, "chars": M}}
        "shards": [],                 # list of per-shard dicts
        "rejections": {               # populated by iter_jsonl_texts_streaming
            "per_file": [],           # list of per-input-file counters
            "totals": {},             # aggregated across all input files
        },
        "config": {
            "chars_per_shard": chars_per_shard,
            "row_group_size": row_group_size,
            "buffer_size": buffer_size,
            "seed": seed,
            "sample_rates": sample_rates,
            "cutoff_year": cutoff_year,
            "contamination_enabled": run_contamination,
        },
    }

    # per-shard accumulator
    shard_docs: List[str] = []
    shard_sources: List[str] = []
    shard_chars = 0
    shard_index = 0
    # bounded shuffle buffer
    shuffle_buffer: List[Tuple[str, str]] = []

    def write_shard():
        nonlocal shard_docs, shard_sources, shard_chars, shard_index
        if not shard_docs:
            return
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table, shard_path,
            row_group_size=min(row_group_size, len(shard_docs)),
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        # per-shard source distribution
        dist: Dict[str, int] = {}
        for s in shard_sources:
            dist[s] = dist.get(s, 0) + 1
        stats["shards"].append({
            "index": shard_index,
            "filename": os.path.basename(shard_path),
            "docs": len(shard_docs),
            "chars": shard_chars,
            "source_distribution": dist,
        })
        stats["num_shards"] += 1
        print(f"  Wrote shard {shard_index}: {len(shard_docs):,} docs, {shard_chars:,} chars")
        shard_docs, shard_sources = [], []
        shard_chars = 0
        shard_index += 1

    def emit_from_buffer(force_all: bool = False):
        """Pop documents from shuffle buffer into the current shard."""
        nonlocal shuffle_buffer, shard_docs, shard_sources, shard_chars
        stop = False
        # When not forcing, we drain half the buffer to keep some mixing depth.
        target_drain = len(shuffle_buffer) if force_all else buffer_size // 2
        for _ in range(target_drain):
            if not shuffle_buffer:
                break
            if max_shards is not None and shard_index >= max_shards:
                stop = True
                break
            if max_tokens is not None and stats["total_chars"] // 4 >= max_tokens:
                stop = True
                break
            idx = rng.randint(0, len(shuffle_buffer) - 1)
            text, src = shuffle_buffer.pop(idx)
            shard_docs.append(text)
            shard_sources.append(src)
            shard_chars += len(text)
            stats["total_docs"] += 1
            stats["total_chars"] += len(text)
            bucket = stats["source_counts"].setdefault(src, {"docs": 0, "chars": 0})
            bucket["docs"] += 1
            bucket["chars"] += len(text)
            if shard_chars >= chars_per_shard and len(shard_docs) % row_group_size == 0:
                write_shard()
        return stop

    stop_signal = False
    for text, src in iter_jsonl_texts_streaming(
        input_files, sample_rates, seed=seed,
        cutoff_year=cutoff_year, run_contamination=run_contamination,
        rejection_report=stats["rejections"],
        run_ocr_quality=run_ocr_quality, run_dedup=run_dedup,
    ):
        shuffle_buffer.append((text, src))
        if len(shuffle_buffer) >= buffer_size:
            if emit_from_buffer(force_all=False):
                stop_signal = True
                break

    # drain remainder
    if not stop_signal:
        emit_from_buffer(force_all=True)

    # final (possibly undersized) shard
    if shard_docs and (max_shards is None or shard_index < max_shards):
        write_shard()

    # Aggregate rejection totals across per-file counters.
    totals: Dict[str, int] = {}
    for per_file in stats["rejections"]["per_file"]:
        for key, val in per_file.items():
            if isinstance(val, int):
                totals[key] = totals.get(key, 0) + val
    stats["rejections"]["totals"] = totals

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(stats, f, indent=2)
    # Legacy filename kept for compatibility with existing tooling.
    with open(os.path.join(output_dir, "stats.json"), 'w') as f:
        json.dump({k: v for k, v in stats.items() if k != "shards"}, f, indent=2)

    print()
    print("=" * 60)
    print("PACKAGING COMPLETE")
    print("=" * 60)
    print(f"Total documents: {stats['total_docs']:,}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Est. tokens (chars/4): {stats['total_chars'] // 4:,}")
    print(f"Number of shards: {stats['num_shards']}")
    if stats['num_shards'] > 0:
        print(f"Average chars/shard: {stats['total_chars'] // stats['num_shards']:,}")
    print()
    print("Per-source totals:")
    for src, bucket in sorted(stats["source_counts"].items()):
        print(f"  {src}: {bucket['docs']:,} docs, {bucket['chars']:,} chars")
    return stats


# Retained for backwards compatibility; forwards to the streaming path.
def package_shards(input_files, output_dir, cutoff_year, chars_per_shard=250_000_000,
                   row_group_size=1024, shuffle_seed=42, max_shards=None):
    return package_shards_streaming(
        input_files=input_files,
        output_dir=output_dir,
        sample_rates=None,
        chars_per_shard=chars_per_shard,
        row_group_size=row_group_size,
        seed=shuffle_seed,
        max_shards=max_shards,
        cutoff_year=cutoff_year,
        run_contamination=True,
    )


def estimate_training_capacity(stats: dict) -> dict:
    """Estimate training capacity based on corpus size (Chinchilla ratios)."""
    total_chars = stats.get('total_chars', 0)
    chars_per_token = 4.8
    total_tokens = total_chars / chars_per_token
    optimal_params = total_tokens / 20

    estimates = {
        "total_tokens": int(total_tokens),
        "optimal_params": int(optimal_params),
        "recommended_depth": None,
        "training_hours_8xh100": None,
    }

    depth_params = {
        20: 561_000_000,
        26: 1_100_000_000,
        32: 1_800_000_000,
        34: 2_200_000_000,
    }
    for depth, params in depth_params.items():
        if optimal_params >= params:
            estimates["recommended_depth"] = depth

    if estimates["recommended_depth"]:
        tokens_per_hour = 11_000_000_000 / 3
        estimates["training_hours_8xh100"] = total_tokens / tokens_per_hour
    return estimates


def find_corpus_files(data_dir: str) -> List[str]:
    """Find all JSONL corpus files in a data directory."""
    files = []
    data_path = Path(data_dir)
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir() or 'test' in subdir.name:
            continue
        for jsonl in subdir.glob("*.jsonl"):
            files.append(str(jsonl))
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package historical texts into nanochat shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default balanced sampling (streaming, bounded memory)
  python -m data.process.shard_packager --data-dir data/raw --output-dir shards/ --cutoff 1913 --check-contamination

  # Custom token budget
  python -m data.process.shard_packager --data-dir data/raw --output-dir shards/ --max-tokens 44000000000

  # No sampling (keep every document)
  python -m data.process.shard_packager --input data/raw/gutenberg/*.jsonl --output-dir shards/ --no-sample
        """)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", nargs="+", help="Input JSONL files")
    input_group.add_argument("--data-dir", type=str, help="Data directory to scan for JSONL files")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--no-sample", action="store_true", help="Disable per-source sampling")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--chars-per-shard", type=int, default=250_000_000)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff", type=int, default=1913, help="Temporal cutoff year")
    parser.add_argument("--check-contamination", action="store_true",
                        help="Run contamination filter during packaging (slower, safer).")
    parser.add_argument("--no-ocr-quality", action="store_true",
                        help="Disable OCR-quality pre-filter for OCR-heavy sources.")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable near-dedup via content fingerprint.")
    args = parser.parse_args()

    if args.data_dir:
        input_files = find_corpus_files(args.data_dir)
        print(f"Found {len(input_files)} corpus files in {args.data_dir}")
        for f in input_files:
            print(f"  - {f}")
    else:
        input_files = args.input
    if not input_files:
        print("No input files found!")
        exit(1)

    stats = package_shards_streaming(
        input_files=input_files,
        output_dir=args.output_dir,
        sample_rates=None if args.no_sample else DEFAULT_SAMPLE_RATES,
        chars_per_shard=args.chars_per_shard,
        seed=args.seed,
        max_shards=args.max_shards,
        max_tokens=args.max_tokens,
        buffer_size=args.buffer_size,
        cutoff_year=args.cutoff,
        run_contamination=args.check_contamination,
        run_ocr_quality=not args.no_ocr_quality,
        run_dedup=not args.no_dedup,
    )

    estimates = estimate_training_capacity(stats)
    print(f"\n=== Training Capacity Estimates ===")
    print(f"Total tokens: {estimates['total_tokens']:,}")
    print(f"Optimal params (Chinchilla): {estimates['optimal_params']:,}")
    if estimates['recommended_depth']:
        print(f"Recommended depth: d{estimates['recommended_depth']}")
    if estimates['training_hours_8xh100']:
        print(f"Estimated training time (8xH100): {estimates['training_hours_8xh100']:.1f} hours")
