"""
Package historical texts into nanochat-compatible parquet shards.

Matches the format expected by nanochat/dataset.py:
- Parquet files with 'text' column
- ~250M characters per shard
- Row group size of 1024
- zstd compression
"""
import os
import json
import argparse
import random
from typing import Iterator, List, Optional
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from data.process.contamination_check import (
    check_contamination,
    clean_gutenberg_headers,
    ContaminationResult,
)


def iter_jsonl_texts(
    input_files: List[str],
    cutoff_year: int,
    clean_headers: bool = True,
    check_contam: bool = True,
) -> Iterator[str]:
    """
    Iterate over texts from JSONL files with filtering.

    Args:
        input_files: List of JSONL file paths
        cutoff_year: Temporal cutoff for contamination check
        clean_headers: Remove Gutenberg headers/footers
        check_contam: Run contamination check

    Yields:
        Clean text strings
    """
    for filepath in input_files:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        print(f"Processing: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    text = record.get('text', '')

                    if not text or len(text) < 100:
                        continue

                    # Clean Gutenberg headers if applicable
                    if clean_headers and record.get('source') == 'gutenberg':
                        text = clean_gutenberg_headers(text)

                    # Run contamination check
                    if check_contam:
                        result = check_contamination(text, cutoff_year)
                        if result.is_contaminated:
                            continue

                    yield text

                except json.JSONDecodeError:
                    continue


def package_shards(
    input_files: List[str],
    output_dir: str,
    cutoff_year: int,
    chars_per_shard: int = 250_000_000,
    row_group_size: int = 1024,
    shuffle_seed: int = 42,
    max_shards: Optional[int] = None,
) -> dict:
    """
    Package texts into nanochat-compatible parquet shards.

    Args:
        input_files: List of input JSONL files
        output_dir: Output directory for shards
        cutoff_year: Temporal cutoff year
        chars_per_shard: Characters per shard (~250M default)
        row_group_size: Parquet row group size (1024 default)
        shuffle_seed: Random seed for shuffling
        max_shards: Maximum number of shards to create

    Returns:
        Statistics dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Packaging shards for cutoff year {cutoff_year}")
    print(f"Output directory: {output_dir}")
    print(f"Characters per shard: {chars_per_shard:,}")

    stats = {
        "total_docs": 0,
        "total_chars": 0,
        "num_shards": 0,
        "rejected_contamination": 0,
    }

    # Collect all texts first (for shuffling)
    print("\nCollecting texts...")
    all_texts = list(iter_jsonl_texts(
        input_files,
        cutoff_year,
        clean_headers=True,
        check_contam=True,
    ))

    print(f"Collected {len(all_texts):,} documents")

    if not all_texts:
        print("No texts to process!")
        return stats

    # Shuffle
    print(f"Shuffling with seed {shuffle_seed}...")
    random.seed(shuffle_seed)
    random.shuffle(all_texts)

    # Package into shards
    print("\nPackaging into shards...")
    shard_docs = []
    shard_chars = 0
    shard_index = 0

    for text in tqdm(all_texts, desc="Processing documents"):
        if max_shards and shard_index >= max_shards:
            break

        shard_docs.append(text)
        shard_chars += len(text)
        stats["total_docs"] += 1
        stats["total_chars"] += len(text)

        # Check if we have enough for a shard
        collected_enough = shard_chars >= chars_per_shard
        docs_aligned = len(shard_docs) % row_group_size == 0

        if collected_enough and docs_aligned:
            # Write shard
            shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=row_group_size,
                use_dictionary=False,
                compression="zstd",
                compression_level=3,
                write_statistics=False,
            )

            print(f"Wrote {shard_path}: {len(shard_docs):,} docs, {shard_chars:,} chars")
            stats["num_shards"] += 1

            shard_docs = []
            shard_chars = 0
            shard_index += 1

    # Write remaining documents as final shard (even if not full)
    if shard_docs and (max_shards is None or shard_index < max_shards):
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=min(row_group_size, len(shard_docs)),
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )

        print(f"Wrote final shard {shard_path}: {len(shard_docs):,} docs, {shard_chars:,} chars")
        stats["num_shards"] += 1

    # Save stats
    stats_file = os.path.join(output_dir, "stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total documents: {stats['total_docs']:,}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Number of shards: {stats['num_shards']}")
    print(f"Average chars/shard: {stats['total_chars'] // max(1, stats['num_shards']):,}")

    return stats


def estimate_training_capacity(stats: dict) -> dict:
    """
    Estimate training capacity based on corpus size.
    Uses nanochat's Chinchilla-optimal ratios.
    """
    total_chars = stats.get('total_chars', 0)

    # Assume ~4.8 chars/token (nanochat's typical ratio)
    chars_per_token = 4.8
    total_tokens = total_chars / chars_per_token

    # Chinchilla optimal: tokens = 20 * params
    # So params = tokens / 20
    optimal_params = total_tokens / 20

    estimates = {
        "total_tokens": int(total_tokens),
        "optimal_params": int(optimal_params),
        "recommended_depth": None,
        "training_hours_8xh100": None,
    }

    # Nanochat depth-to-params mapping (approximate)
    depth_params = {
        20: 561_000_000,   # 561M
        26: 1_100_000_000,  # 1.1B
        32: 1_800_000_000,  # 1.8B
        34: 2_200_000_000,  # 2.2B
    }

    for depth, params in depth_params.items():
        if optimal_params >= params:
            estimates["recommended_depth"] = depth

    # Rough training time estimate (based on nanochat benchmarks)
    # d20 on 8xH100: ~3 hours for 11B tokens
    if estimates["recommended_depth"]:
        tokens_per_hour = 11_000_000_000 / 3  # ~3.7B tokens/hour on 8xH100
        estimates["training_hours_8xh100"] = total_tokens / tokens_per_hour

    return estimates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package historical texts into nanochat shards")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input JSONL files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for shards")
    parser.add_argument("--cutoff", type=int, default=1913,
                        help="Cutoff year for contamination check")
    parser.add_argument("--chars-per-shard", type=int, default=250_000_000,
                        help="Characters per shard")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Maximum shards to create")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")

    args = parser.parse_args()

    stats = package_shards(
        input_files=args.input,
        output_dir=args.output_dir,
        cutoff_year=args.cutoff,
        chars_per_shard=args.chars_per_shard,
        max_shards=args.max_shards,
        shuffle_seed=args.seed,
    )

    # Show training estimates
    estimates = estimate_training_capacity(stats)
    print(f"\n=== Training Capacity Estimates ===")
    print(f"Total tokens: {estimates['total_tokens']:,}")
    print(f"Optimal params (Chinchilla): {estimates['optimal_params']:,}")
    print(f"Recommended depth: d{estimates['recommended_depth']}")
    if estimates['training_hours_8xh100']:
        print(f"Estimated training time (8xH100): {estimates['training_hours_8xh100']:.1f} hours")
