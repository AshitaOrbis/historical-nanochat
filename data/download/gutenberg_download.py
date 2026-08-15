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

from data.process.contamination_check import clean_gutenberg_headers, check_contamination

# Temporal cutoff configurations
CUTOFF_CONFIGS = {
    "1850": {"year": 1850, "description": "Pre-industrial/early Victorian"},
    "1900": {"year": 1900, "description": "Victorian/pre-Edwardian"},
    "1913": {"year": 1913, "description": "Pre-WWI"},
    "1950": {"year": 1950, "description": "Pre-Cold War"},
}

# Note: Anachronism detection now uses the improved check_contamination() function
# from contamination_check.py which has:
# - Word boundary matching (no partial matches)
# - Context-aware detection (e.g., "atomic" only flags with "bomb" context)
# - Safe historical terms list (e.g., "apollo" the god is OK)


def extract_year_from_metadata(metadata: Dict[str, Any]) -> Optional[int]:
    """
    Extract a publication year from Gutenberg metadata, in precedence order.

    Precedence:
      1. 'issued' or 'publicationyear' or 'published' — closest to publication date.
      2. 'authoryearofdeath' — upper bound on when the author could have written it.
         (Books are usually published before the author dies; safer than birth year.)
      3. 'authoryearofbirth' + 20 as a floor (author likely not publishing before ~age 20).
      4. Last-resort scan of the metadata JSON for 4-digit years in a plausible range.

    Notable change vs the previous implementation: we no longer treat the
    `downloads` field as year-bearing — it is a raw count, not a date, and the
    old code would mis-read "1972" download counts as a 1972 publication year.
    """
    # 1. Explicit publication fields.
    for field in ('issued', 'publicationyear', 'published', 'publication_year', 'date'):
        value = metadata.get(field)
        if value:
            match = re.search(r'(1[0-9]{3}|20[0-2]\d)', str(value))
            if match:
                year = int(match.group(1))
                if 1000 <= year <= 2100:
                    return year

    # 2. Author year of death bounds publication.
    death = metadata.get('authoryearofdeath')
    if death:
        match = re.search(r'(\d{4})', str(death))
        if match:
            year = int(match.group(1))
            if 1000 <= year <= 2100:
                return year

    # 3. Author year of birth → rough floor.
    birth = metadata.get('authoryearofbirth')
    if birth:
        match = re.search(r'(\d{4})', str(birth))
        if match:
            year = int(match.group(1))
            if 1000 <= year <= 2100:
                return year + 20

    # 4. Last resort: scan a JSON dump for a plausible year. Avoid the 'downloads'
    # field because raw download counts frequently contain 4-digit numbers.
    safe_metadata = {k: v for k, v in metadata.items() if k != 'downloads'}
    text_to_search = json.dumps(safe_metadata)
    year_matches = re.findall(r'\b(1[0-9]{3})\b', text_to_search)
    if year_matches:
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




def is_text_suitable(
    text: str,
    metadata: Dict[str, Any],
    cutoff_year: int,
    strict: bool = True,
    min_length: int = 1000,
    max_length: int = 10_000_000,
) -> tuple[bool, str, str]:
    """
    Determine if a text is suitable for the historical corpus.
    Returns (is_suitable, reason, cleaned_text).

    The cleaned_text has Gutenberg headers/footers removed, which is used
    for anachronism checking and should be used for the final output.
    """
    # Clean Gutenberg headers/footers first (they contain modern terms like 'email')
    cleaned_text = clean_gutenberg_headers(text)

    # Length checks on cleaned text
    if len(cleaned_text) < min_length:
        return False, f"Too short ({len(cleaned_text)} chars)", cleaned_text
    if len(cleaned_text) > max_length:
        return False, f"Too long ({len(cleaned_text)} chars)", cleaned_text

    # Language check (prefer English)
    lang = metadata.get('language', '').lower()
    if lang and 'english' not in lang and 'en' != lang:
        return False, f"Non-English language: {lang}", cleaned_text

    # Try to determine year (use original text for header info)
    year = extract_year_from_metadata(metadata)
    if year is None:
        year = estimate_year_from_text(text)  # Use original for date extraction

    if year is not None:
        if year > cutoff_year:
            return False, f"Post-cutoff year: {year} > {cutoff_year}", cleaned_text
    elif strict:
        # In strict mode, reject texts with unknown dates
        return False, "Unknown publication date", cleaned_text

    # Check for anachronisms on CLEANED text using improved detection
    if strict:
        result = check_contamination(cleaned_text, cutoff_year)
        if result.is_contaminated:
            # Report first few matched terms for clarity
            terms = result.matched_terms[:3] if result.matched_terms else ["(see reasons)"]
            return False, f"Contaminated: {terms}", cleaned_text

    return True, "OK", cleaned_text


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

            suitable, reason, cleaned_text = is_text_suitable(
                text, metadata, cutoff_year, strict=strict
            )

            if suitable:
                stats["accepted"] += 1
                stats["total_chars"] += len(cleaned_text)

                # Record year + provenance (so downstream audits can trace which field gave us the year)
                year_from_metadata = extract_year_from_metadata(metadata)
                year_from_text = None if year_from_metadata is not None else estimate_year_from_text(text)
                year = year_from_metadata if year_from_metadata is not None else year_from_text
                year_source = ("metadata" if year_from_metadata is not None else
                               "text" if year_from_text is not None else "unknown")
                if year:
                    decade = (year // 10) * 10
                    stats["years_distribution"][decade] = stats["years_distribution"].get(decade, 0) + 1

                # Write to output (use cleaned text without Gutenberg headers)
                record = {
                    "text": cleaned_text,
                    "source": "gutenberg",
                    "metadata": metadata,
                    "estimated_year": year,
                    "year_source": year_source,
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
