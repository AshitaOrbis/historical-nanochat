"""
Download historical case law from the Caselaw Access Project (Harvard Law).

Covers US court decisions from 1658-2020.
Available on HuggingFace: free-law/Caselaw_Access_Project
"""
import os
import json
import argparse
from typing import Optional, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
import re

# Date extraction patterns for case law
DATE_PATTERNS = [
    r'(\d{4})',  # Just year
    r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # Month Day, Year
    r'(\d{1,2})\s+(\w+)\s+(\d{4})',  # Day Month Year
]


def extract_case_year(case_data: Dict[str, Any]) -> Optional[int]:
    """
    Extract the decision year from case metadata.
    """
    # Try decision_date field first
    decision_date = case_data.get('decision_date', '')
    if decision_date:
        match = re.search(r'(\d{4})', str(decision_date))
        if match:
            return int(match.group(1))

    # Try date_filed
    date_filed = case_data.get('date_filed', '')
    if date_filed:
        match = re.search(r'(\d{4})', str(date_filed))
        if match:
            return int(match.group(1))

    # Try to extract from citation
    citation = case_data.get('citation', '') or case_data.get('citations', '')
    if citation:
        # Citations often end with year
        match = re.search(r'\((\d{4})\)', str(citation))
        if match:
            return int(match.group(1))

    return None


def is_case_suitable(
    text: str,
    case_data: Dict[str, Any],
    cutoff_year: int,
    min_length: int = 500,
) -> tuple[bool, str]:
    """
    Determine if a case is suitable for the historical corpus.
    """
    if len(text) < min_length:
        return False, f"Too short ({len(text)} chars)"

    year = extract_case_year(case_data)
    if year is None:
        return False, "Unknown decision year"
    if year > cutoff_year:
        return False, f"Post-cutoff: {year} > {cutoff_year}"

    return True, "OK"


def download_caselaw(
    cutoff: str = "1913",
    output_dir: str = "data/raw/caselaw",
    max_cases: Optional[int] = None,
    courts: Optional[list] = None,
    streaming: bool = True,
) -> Dict[str, Any]:
    """
    Download and filter case law from Caselaw Access Project.

    Args:
        cutoff: Temporal cutoff year
        output_dir: Output directory
        max_cases: Maximum cases to process
        courts: Filter by specific courts (e.g., ["scotus", "federal"])
        streaming: Use streaming mode
    """
    from data.download.gutenberg_download import CUTOFF_CONFIGS

    if cutoff not in CUTOFF_CONFIGS:
        raise ValueError(f"Unknown cutoff: {cutoff}")

    cutoff_year = CUTOFF_CONFIGS[cutoff]["year"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Caselaw Access Project - Historical US Court Decisions")
    print(f"Cutoff year: {cutoff_year}")
    print(f"Output directory: {output_dir}")

    stats = {
        "total_processed": 0,
        "accepted": 0,
        "rejected": 0,
        "rejection_reasons": {},
        "total_chars": 0,
        "years_distribution": {},
        "courts_distribution": {},
    }

    output_file = os.path.join(output_dir, f"caselaw_{cutoff}.jsonl")

    print("\nLoading dataset from HuggingFace...")
    print("Note: This is a large dataset (~50GB). Streaming mode is recommended.")

    try:
        # The dataset structure may vary - try different configurations
        ds = load_dataset("free-law/Caselaw_Access_Project", streaming=streaming, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nAlternative: Download directly from case.law")
        print("  API: https://case.law/docs/")
        print("  HuggingFace: https://huggingface.co/datasets/free-law/Caselaw_Access_Project")

        # Create instructions file
        instructions = {
            "instructions": "Download Caselaw Access Project data",
            "huggingface": "https://huggingface.co/datasets/free-law/Caselaw_Access_Project",
            "api": "https://case.law/",
            "notes": [
                "Large dataset - use streaming mode",
                "Filter by decision_date for temporal cutoff",
                "500 case/day limit without API key",
            ],
        }

        with open(os.path.join(output_dir, "README.json"), 'w') as f:
            json.dump(instructions, f, indent=2)

        return stats

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, case in enumerate(tqdm(ds, desc="Processing cases")):
            if max_cases and stats["accepted"] >= max_cases:
                break

            stats["total_processed"] += 1

            # Extract text - field name varies by dataset version
            text = case.get('casebody', {}).get('data', {}).get('opinions', [{}])[0].get('text', '')
            if not text:
                text = case.get('text', '')
            if not text:
                text = str(case.get('casebody', ''))

            suitable, reason = is_case_suitable(text, case, cutoff_year)

            if suitable:
                stats["accepted"] += 1
                stats["total_chars"] += len(text)

                year = extract_case_year(case)
                if year:
                    decade = (year // 10) * 10
                    stats["years_distribution"][decade] = stats["years_distribution"].get(decade, 0) + 1

                court = case.get('court', {}).get('name', 'Unknown')
                stats["courts_distribution"][court] = stats["courts_distribution"].get(court, 0) + 1

                record = {
                    "text": text,
                    "source": "caselaw",
                    "year": year,
                    "court": court,
                    "name_abbreviation": case.get('name_abbreviation', ''),
                    "citation": case.get('citations', []),
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            else:
                stats["rejected"] += 1
                stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1

    # Save stats
    stats_file = os.path.join(output_dir, f"caselaw_{cutoff}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults:")
    print(f"  Processed: {stats['total_processed']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Total chars: {stats['total_chars']:,}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Caselaw Access Project data")
    parser.add_argument("--cutoff", type=str, default="1913",
                        help="Temporal cutoff year")
    parser.add_argument("--output-dir", type=str, default="data/raw/caselaw",
                        help="Output directory")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Maximum cases to process")

    args = parser.parse_args()

    download_caselaw(
        cutoff=args.cutoff,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
    )
