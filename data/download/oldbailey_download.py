"""
Process a locally acquired Old Bailey Proceedings corpus for nanochat training.

The Old Bailey Corpus contains trial proceedings from 1674-1913.
All content is pre-WWI by definition.
The corpus must be downloaded separately from CLARIN-D; this entry point never
pretends that writing instructions constitutes acquisition.
"""
import os
import json
import argparse
import requests
from typing import Optional, Dict, Any
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import zipfile
import io

# Old Bailey Online provides XML exports
OLD_BAILEY_API_BASE = "https://www.oldbaileyonline.org/api"

# CLARIN-D corpus download (if available)
CLARIN_CORPUS_URL = "https://fedora.clarin-d.uni-saarland.de/oldbailey/"


def extract_text_from_xml(xml_content: str) -> str:
    """
    Extract plain text from Old Bailey XML format.
    """
    soup = BeautifulSoup(xml_content, 'lxml-xml')

    # Find all text content, excluding metadata
    text_parts = []

    # Get trial text
    for element in soup.find_all(['p', 'persName', 'placeName', 'rs']):
        if element.string:
            text_parts.append(element.string.strip())
        else:
            text_parts.append(element.get_text(separator=' ', strip=True))

    return ' '.join(text_parts)


def parse_trial_date(trial_id: str) -> Optional[int]:
    """
    Extract year from Old Bailey trial ID.
    Format is typically like 't17800112-1' (t + YYYYMMDD + -)
    """
    match = re.match(r't(\d{4})\d{4}', trial_id)
    if match:
        return int(match.group(1))
    return None


def process_oldbailey_corpus(
    corpus_dir: str,
    output_dir: str = "data/raw/oldbailey",
    cutoff: str = "1913",
    max_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process locally downloaded Old Bailey XML proceedings.
    """
    from data.download.gutenberg_download import CUTOFF_CONFIGS

    if cutoff not in CUTOFF_CONFIGS:
        raise ValueError(f"Unknown cutoff: {cutoff}")

    cutoff_year = CUTOFF_CONFIGS[cutoff]["year"]
    corpus_path = os.path.abspath(corpus_dir)
    if not os.path.isdir(corpus_path):
        raise FileNotFoundError(f"Old Bailey corpus directory not found: {corpus_dir}")

    xml_files = sorted(f for f in os.listdir(corpus_path) if f.endswith('.xml'))
    if max_trials is not None:
        xml_files = xml_files[:max_trials]
    if not xml_files:
        raise RuntimeError(
            "Old Bailey acquired no records "
            "(requested=0, fetched=0, written=0; corpus directory has no XML files)"
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Old Bailey Corpus (1674-1913)")
    print(f"All content is pre-1913 by definition")
    print(f"Cutoff year: {cutoff_year}")

    stats = {
        "records_requested": len(xml_files),
        "records_fetched": 0,
        "records_written": 0,
        "total_processed": 0,
        "accepted": 0,
        "rejected": 0,
        "total_chars": 0,
        "years_distribution": {},
    }

    output_file = os.path.join(output_dir, f"oldbailey_{cutoff}.jsonl")

    print(f"\nProcessing local corpus at {corpus_path}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for xml_file in tqdm(xml_files, desc="Processing XML files"):
            filepath = os.path.join(corpus_path, xml_file)
            try:
                with open(filepath, 'r', encoding='utf-8') as xf:
                    xml_content = xf.read()
                stats["records_fetched"] += 1

                text = extract_text_from_xml(xml_content)

                if len(text) < 100:
                    stats["rejected"] += 1
                    continue

                year = parse_trial_date(xml_file)
                if year and year > cutoff_year:
                    stats["rejected"] += 1
                    continue

                stats["accepted"] += 1
                stats["records_written"] += 1
                stats["total_chars"] += len(text)

                if year:
                    decade = (year // 10) * 10
                    stats["years_distribution"][decade] = stats["years_distribution"].get(decade, 0) + 1

                record = {
                    "text": text,
                    "source": "oldbailey",
                    "filename": xml_file,
                    "year": year,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                stats["rejected"] += 1

            stats["total_processed"] += 1

    stats_file = os.path.join(output_dir, f"oldbailey_{cutoff}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults:")
    print(f"  Processed: {stats['total_processed']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(
        "  Acquisition counts: "
        f"requested={stats['records_requested']} "
        f"fetched={stats['records_fetched']} written={stats['records_written']}"
    )

    if stats["records_written"] == 0:
        raise RuntimeError(
            "Old Bailey acquired no records "
            f"(requested={stats['records_requested']}, "
            f"fetched={stats['records_fetched']}, written=0)"
        )

    return stats


def download_oldbailey_sample(
    output_dir: str = "data/raw/oldbailey",
    cutoff: str = "1913",
    max_trials: Optional[int] = None,
    corpus_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper; acquisition is still explicit and local-only."""
    if corpus_dir is None:
        raise FileNotFoundError(
            "--corpus-dir is required; download the Old Bailey XML corpus separately"
        )
    return process_oldbailey_corpus(corpus_dir, output_dir, cutoff, max_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a locally downloaded Old Bailey XML corpus")
    parser.add_argument("--corpus-dir", type=str, required=True,
                        help="Directory containing Old Bailey XML files (required)")
    parser.add_argument("--cutoff", type=str, default="1913",
                        help="Temporal cutoff year")
    parser.add_argument("--output-dir", type=str, default="data/raw/oldbailey",
                        help="Output directory")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Maximum trials to process")

    args = parser.parse_args()

    process_oldbailey_corpus(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        cutoff=args.cutoff,
        max_trials=args.max_trials,
    )
