"""
Download and filter Project Gutenberg texts for historical nanochat training.

Uses the HuggingFace dataset: manu/project_gutenberg
Filters by publication date to ensure temporal cutoff compliance.
"""
import os
import json
import argparse
from datetime import datetime
from typing import Optional, Iterator, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
import re

# Temporal cutoff configurations
CUTOFF_CONFIGS = {
    "1850": {"year": 1850, "description": "Pre-industrial/early Victorian"},
    "1900": {"year": 1900, "description": "Victorian/pre-Edwardian"},
    "1913": {"year": 1913, "description": "Pre-WWI"},
    "1950": {"year": 1950, "description": "Pre-Cold War"},
}

# Known anachronistic terms to detect contamination
# Expanded list for each cutoff
ANACHRONISM_TERMS = {
    1850: [
        "telephone", "electric light", "automobile", "airplane", "radio",
        "photograph", "typewriter", "gramophone", "phonograph",
    ],
    1900: [
        "airplane", "radio", "television", "computer", "atomic",
        "world war", "nazi", "fascist", "soviet", "bolshevik",
    ],
    1913: [
        "world war", "nazi", "fascist", "soviet", "bolshevik",
        "atomic bomb", "nuclear", "television", "computer", "hitler",
        "mussolini", "stalin", "holocaust", "concentration camp",
    ],
    1950: [
        "internet", "computer", "smartphone", "email", "website",
        "vietnam war", "moon landing", "apollo", "sputnik",
    ],
}


def extract_year_from_metadata(metadata: Dict[str, Any]) -> Optional[int]:
    """
    Extract publication year from Gutenberg metadata.
    Returns None if year cannot be determined.
    """
    # Try different metadata fields
    for field in ['authoryearofbirth', 'authoryearofdeath', 'issued', 'downloads']:
        if field in metadata and metadata[field]:
            try:
                # Try to extract a 4-digit year
                year_match = re.search(r'(\d{4})', str(metadata[field]))
                if year_match:
                    year = int(year_match.group(1))
                    if 1000 <= year <= 2100:
                        return year
            except (ValueError, TypeError):
                continue

    # Check 'language' and other fields for date hints
    # Many Gutenberg texts have date info in various places
    text_to_search = json.dumps(metadata)
    year_matches = re.findall(r'\b(1[0-9]{3})\b', text_to_search)
    if year_matches:
        # Return the most plausible publication year (often the earliest non-birth year)
        years = [int(y) for y in year_matches if 1400 <= int(y) <= 2025]
        if years:
            return min(years)

    return None


def estimate_year_from_text(text: str, max_chars: int = 5000) -> Optional[int]:
    """
    Try to estimate publication year from the text itself.
    Checks the beginning of the text for copyright notices, dates, etc.
    """
    sample = text[:max_chars].lower()

    # Look for copyright notices
    copyright_match = re.search(r'copyright[^\d]*(\d{4})', sample)
    if copyright_match:
        return int(copyright_match.group(1))

    # Look for "published in YYYY" or similar
    published_match = re.search(r'published[^\d]*(\d{4})', sample)
    if published_match:
        return int(published_match.group(1))

    # Look for "printed in YYYY"
    printed_match = re.search(r'printed[^\d]*(\d{4})', sample)
    if printed_match:
        return int(printed_match.group(1))

    return None


def check_for_anachronisms(text: str, cutoff_year: int) -> list:
    """
    Check text for anachronistic terms that would indicate post-cutoff content.
    Returns list of found anachronistic terms.
    """
    text_lower = text.lower()
    found = []

    # Get terms for this cutoff (and all earlier cutoffs)
    terms_to_check = set()
    for year, terms in ANACHRONISM_TERMS.items():
        if year >= cutoff_year:
            terms_to_check.update(terms)

    for term in terms_to_check:
        if term in text_lower:
            found.append(term)

    return found


def is_text_suitable(
    text: str,
    metadata: Dict[str, Any],
    cutoff_year: int,
    strict: bool = True,
    min_length: int = 1000,
    max_length: int = 10_000_000,
) -> tuple[bool, str]:
    """
    Determine if a text is suitable for the historical corpus.
    Returns (is_suitable, reason).
    """
    # Length checks
    if len(text) < min_length:
        return False, f"Too short ({len(text)} chars)"
    if len(text) > max_length:
        return False, f"Too long ({len(text)} chars)"

    # Language check (prefer English)
    lang = metadata.get('language', '').lower()
    if lang and 'english' not in lang and 'en' != lang:
        return False, f"Non-English language: {lang}"

    # Try to determine year
    year = extract_year_from_metadata(metadata)
    if year is None:
        year = estimate_year_from_text(text)

    if year is not None:
        if year > cutoff_year:
            return False, f"Post-cutoff year: {year} > {cutoff_year}"
    elif strict:
        # In strict mode, reject texts with unknown dates
        return False, "Unknown publication date"

    # Check for anachronisms
    if strict:
        anachronisms = check_for_anachronisms(text, cutoff_year)
        if anachronisms:
            return False, f"Anachronistic terms found: {anachronisms[:3]}"

    return True, "OK"


def download_gutenberg(
    cutoff: str = "1913",
    output_dir: str = "data/raw/gutenberg",
    strict: bool = True,
    max_docs: Optional[int] = None,
    streaming: bool = True,
) -> Dict[str, Any]:
    """
    Download and filter Project Gutenberg texts.

    Args:
        cutoff: Temporal cutoff ("1850", "1900", "1913", "1950")
        output_dir: Directory to save filtered texts
        strict: If True, reject texts with unknown dates or anachronisms
        max_docs: Maximum number of documents to process (None for all)
        streaming: Use streaming mode for memory efficiency

    Returns:
        Statistics dictionary
    """
    if cutoff not in CUTOFF_CONFIGS:
        raise ValueError(f"Unknown cutoff: {cutoff}. Choose from {list(CUTOFF_CONFIGS.keys())}")

    cutoff_year = CUTOFF_CONFIGS[cutoff]["year"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading Project Gutenberg with cutoff year {cutoff_year}")
    print(f"Output directory: {output_dir}")
    print(f"Strict mode: {strict}")

    # Load dataset
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("manu/project_gutenberg", split="en", streaming=streaming)

    stats = {
        "total_processed": 0,
        "accepted": 0,
        "rejected": 0,
        "rejection_reasons": {},
        "total_chars": 0,
        "years_distribution": {},
    }

    output_file = os.path.join(output_dir, f"gutenberg_{cutoff}.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(tqdm(ds, desc="Processing Gutenberg")):
            if max_docs and i >= max_docs:
                break

            stats["total_processed"] += 1

            text = doc.get('text', '')
            metadata = {k: v for k, v in doc.items() if k != 'text'}

            suitable, reason = is_text_suitable(
                text, metadata, cutoff_year, strict=strict
            )

            if suitable:
                stats["accepted"] += 1
                stats["total_chars"] += len(text)

                # Record year if known
                year = extract_year_from_metadata(metadata)
                if year is None:
                    year = estimate_year_from_text(text)
                if year:
                    decade = (year // 10) * 10
                    stats["years_distribution"][decade] = stats["years_distribution"].get(decade, 0) + 1

                # Write to output
                record = {
                    "text": text,
                    "source": "gutenberg",
                    "metadata": metadata,
                    "estimated_year": year,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            else:
                stats["rejected"] += 1
                stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1

    # Save stats
    stats_file = os.path.join(output_dir, f"gutenberg_{cutoff}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults:")
    print(f"  Processed: {stats['total_processed']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  Output: {output_file}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Project Gutenberg for historical nanochat")
    parser.add_argument("--cutoff", type=str, default="1913",
                        choices=list(CUTOFF_CONFIGS.keys()),
                        help="Temporal cutoff year")
    parser.add_argument("--output-dir", type=str, default="data/raw/gutenberg",
                        help="Output directory")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Strict mode (reject unknown dates, check anachronisms)")
    parser.add_argument("--no-strict", action="store_false", dest="strict",
                        help="Disable strict mode")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum documents to process (for testing)")

    args = parser.parse_args()

    download_gutenberg(
        cutoff=args.cutoff,
        output_dir=args.output_dir,
        strict=args.strict,
        max_docs=args.max_docs,
    )
