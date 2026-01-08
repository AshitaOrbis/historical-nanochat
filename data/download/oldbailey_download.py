"""
Download Old Bailey Proceedings corpus for historical nanochat training.

The Old Bailey Corpus contains trial proceedings from 1674-1913.
All content is pre-WWI by definition.
Download from CLARIN-D or process from Old Bailey Online XML.
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


def download_oldbailey_sample(
    output_dir: str = "data/raw/oldbailey",
    cutoff: str = "1913",
    max_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Download Old Bailey proceedings.

    Note: The full corpus requires manual download from CLARIN-D.
    This function provides a sample via the API for testing.
    """
    from data.download.gutenberg_download import CUTOFF_CONFIGS

    if cutoff not in CUTOFF_CONFIGS:
        raise ValueError(f"Unknown cutoff: {cutoff}")

    cutoff_year = CUTOFF_CONFIGS[cutoff]["year"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Old Bailey Corpus (1674-1913)")
    print(f"All content is pre-1913 by definition")
    print(f"Cutoff year: {cutoff_year}")

    stats = {
        "total_processed": 0,
        "accepted": 0,
        "rejected": 0,
        "total_chars": 0,
        "years_distribution": {},
    }

    output_file = os.path.join(output_dir, f"oldbailey_{cutoff}.jsonl")

    # Note: For full corpus, user should download from CLARIN-D
    print("\nNOTE: For the full corpus, download from CLARIN-D:")
    print("  https://fedora.clarin-d.uni-saarland.de/oldbailey/")
    print("\nThis script provides guidance on processing the corpus.")

    # Create a placeholder with instructions
    instructions = {
        "instructions": "Download the Old Bailey Corpus from CLARIN-D",
        "url": CLARIN_CORPUS_URL,
        "format": "XML files with trial transcripts",
        "date_range": "1674-1913",
        "total_words": "127 million words",
        "processing_notes": [
            "All content is pre-1913 - no date filtering needed for 1913 cutoff",
            "For earlier cutoffs, filter by trial date in filename",
            "Use BeautifulSoup with lxml-xml parser for extraction",
            "Preserve speaker annotations if desired for sociolinguistic research",
        ],
    }

    instructions_file = os.path.join(output_dir, "README.json")
    with open(instructions_file, 'w') as f:
        json.dump(instructions, f, indent=2)

    print(f"\nInstructions saved to: {instructions_file}")

    # If corpus files exist locally, process them
    corpus_dir = os.path.join(output_dir, "corpus")
    if os.path.exists(corpus_dir):
        print(f"\nFound local corpus at {corpus_dir}, processing...")

        xml_files = [f for f in os.listdir(corpus_dir) if f.endswith('.xml')]

        with open(output_file, 'w', encoding='utf-8') as f:
            for xml_file in tqdm(xml_files, desc="Processing XML files"):
                if max_trials and stats["accepted"] >= max_trials:
                    break

                filepath = os.path.join(corpus_dir, xml_file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as xf:
                        xml_content = xf.read()

                    text = extract_text_from_xml(xml_content)

                    if len(text) < 100:
                        stats["rejected"] += 1
                        continue

                    # Extract year from filename
                    year = parse_trial_date(xml_file)
                    if year and year > cutoff_year:
                        stats["rejected"] += 1
                        continue

                    stats["accepted"] += 1
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

        # Save stats
        stats_file = os.path.join(output_dir, f"oldbailey_{cutoff}_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nResults:")
        print(f"  Processed: {stats['total_processed']}")
        print(f"  Accepted: {stats['accepted']}")
        print(f"  Total chars: {stats['total_chars']:,}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Old Bailey Corpus")
    parser.add_argument("--cutoff", type=str, default="1913",
                        help="Temporal cutoff year")
    parser.add_argument("--output-dir", type=str, default="data/raw/oldbailey",
                        help="Output directory")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Maximum trials to process")

    args = parser.parse_args()

    download_oldbailey_sample(
        output_dir=args.output_dir,
        cutoff=args.cutoff,
        max_trials=args.max_trials,
    )
