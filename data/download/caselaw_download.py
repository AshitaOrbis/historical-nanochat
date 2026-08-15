"""
Download historical case law from the Caselaw Access Project (Harvard Law).

Covers US court decisions from 1658-2020.
Available on HuggingFace: common-pile/caselaw_access_project
(Note: Previous location free-law/Caselaw_Access_Project is deprecated)

Dataset structure (common-pile version):
- id: filename identifier
- text: full case text (includes case name, court, date, opinion)
- metadata: {author, license, url}
- source: "Caselaw Access Project"

Date extraction: The decision date is embedded in the text, typically:
- "United States Court of Appeals, Ninth Circuit.\n    Feb. 12, 1973."
- "Decided Jan. 26, 1973."
- "Submitted Jan. 29, 1973.\n    Decided Feb. 15, 1973."
"""
import os
import json
import argparse
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from datasets import load_dataset
import re

# Dataset configuration
DATASET_NAME = "common-pile/caselaw_access_project"
DATASET_NAME_OLD = "free-law/Caselaw_Access_Project"  # Deprecated

# Month patterns for date extraction
MONTHS = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'


def extract_case_year(case_data: Dict[str, Any]) -> Optional[int]:
    """
    Extract the decision year from case data.

    For the common-pile dataset, the date is in the text body.
    We look for patterns like "Month Day, Year" or "Decided Month Day, Year".

    IMPORTANT: Only look at the header (first ~500 chars before judge name)
    to avoid picking up dates from the case body.
    """
    text = case_data.get('text', '')
    if not text:
        return None

    # Only search in the header area - stop at first judge name or "J.\n"
    # This prevents picking up dates from the case body
    header = text[:1500]
    # Find where the actual opinion starts (usually after "J.\n" or "J.:")
    judge_marker = re.search(r'\b[A-Z][a-z]+,?\s+J\.[\s:\n]', header)
    if judge_marker:
        header = header[:judge_marker.start()]

    # Pattern 1: "Decided Month Day, Year" or "Submitted Month Day, Year"
    # Allow optional space after comma: "February 13,1940" or "Feb. 12, 1973"
    decided_pattern = rf'(?:Decided|Submitted|Argued|Filed)\s+{MONTHS}\.?\s+\d{{1,2}},?\s*(\d{{4}})'
    match = re.search(decided_pattern, header, re.IGNORECASE)
    if match:
        year = int(match.group(1))
        if 1600 <= year <= 2025:
            return year

    # Pattern 2: Court/County line followed by date
    # "Supreme Court, Oneida County,\n    February 13,1940."
    # Allow optional space after comma before year
    court_date_pattern = rf'(?:Circuit|Court|County|Division)[,.\s]*\n\s*{MONTHS}\.?\s+\d{{1,2}},?\s*(\d{{4}})'
    match = re.search(court_date_pattern, header, re.IGNORECASE)
    if match:
        year = int(match.group(1))
        if 1600 <= year <= 2025:
            return year

    # Pattern 3: Standard "Month Day,Year" or "Month Day, Year" in header
    # Must be in first 500 chars of header to avoid body dates
    early_header = header[:500]
    date_pattern = rf'{MONTHS}\.?\s+\d{{1,2}},?\s*(\d{{4}})'
    matches = re.findall(date_pattern, early_header, re.IGNORECASE)
    if matches:
        # Take the first valid year found
        for year_str in matches:
            year = int(year_str)
            if 1600 <= year <= 2025:
                return year

    # Pattern 4: Year in parentheses after citation (e.g., "123 U.S. 456 (1889)")
    citation_pattern = r'\d+\s+[A-Z][a-z]*\.?\s*(?:\d+|[A-Z]+)\s*\.?\s*\d+\s*\((\d{4})\)'
    match = re.search(citation_pattern, header)
    if match:
        year = int(match.group(1))
        if 1600 <= year <= 2025:
            return year

    return None


def extract_court_from_text(text: str) -> str:
    """
    Extract court name from case text.

    The court name typically appears in the first few lines:
    "United States Court of Appeals, Ninth Circuit."
    "Supreme Court of the United States."
    """
    if not text:
        return "Unknown"

    header = text[:1000]

    # Common court patterns
    court_patterns = [
        r'(Supreme Court of the United States)',
        r'(United States (?:Court of Appeals|District Court)[^.\n]*)',
        r'(Circuit Court[^.\n]*)',
        r'(Court of (?:Appeals|Claims|Customs)[^.\n]*)',
        r'((?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|D\.C\.|Federal) Circuit)',
    ]

    for pattern in court_patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "Unknown"


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


def get_reporter_era(case_id: str) -> str:
    """
    Determine the era of a case based on its reporter series ID.

    Reporter series (federal):
    - us_*: US Reports (Supreme Court, 1790-present)
    - f_*: Federal Reporter 1st (1880-1924) - includes pre-1913 cases!
    - f2d_*: Federal Reporter 2d (1924-1993)
    - f3d_*: Federal Reporter 3d (1993-present)
    - f-supp_*: Federal Supplement (1932-1998)
    - f-supp-2d_*: Federal Supplement 2d (1998-2014)
    - f-supp-3d_*: Federal Supplement 3d (2014-present)
    - fed-appx_*: Federal Appendix (2001-present)

    State reporters vary but generally:
    - Older state reporters predate 1913
    """
    if not case_id:
        return 'unknown'

    # Extract the reporter prefix (everything before the volume number)
    # Examples: f2d_474 -> f2d, f-supp_471 -> f-supp, us_100 -> us
    first_part = case_id.split('/')[0] if '/' in case_id else case_id

    # Split by underscore, the last part should be the volume number
    # f2d_474 -> ['f2d', '474'] -> prefix = 'f2d'
    # f-supp_471 -> ['f-supp', '471'] -> prefix = 'fsupp'
    parts = first_part.split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        prefix = '_'.join(parts[:-1]).lower().replace('-', '')
    else:
        prefix = parts[0].lower().replace('-', '')

    # Modern reporters (definitely post-cutoff for 1913)
    # f2d (1924-1993), f3d (1993+), fsupp (1932+), fedappx (2001+), bcd (bankruptcy)
    modern_prefixes = {
        'f2d', 'f3d',
        'fsupp', 'fsupp2d', 'fsupp3d',
        'fedappx', 'fedapp',
        'bcd', 'br',  # Bankruptcy
    }

    # Historical reporters that may contain pre-1913 cases
    # f (Federal Reporter 1st, 1880-1924), us (US Reports, 1790+)
    historical_prefixes = {'f', 'us'}

    if prefix in modern_prefixes:
        return 'modern'
    elif prefix in historical_prefixes:
        return 'historical'
    else:
        return 'unknown'  # State reporters, mixed dates


def download_caselaw(
    cutoff: str = "1913",
    output_dir: str = "data/raw/caselaw",
    max_cases: Optional[int] = None,
    courts: Optional[list] = None,
    streaming: bool = True,
    skip_modern: bool = True,
    max_total: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Download and filter case law from Caselaw Access Project.

    Args:
        cutoff: Temporal cutoff year
        output_dir: Output directory
        max_cases: Maximum ACCEPTED cases to include
        courts: Filter by specific courts (e.g., ["scotus", "federal"])
        streaming: Use streaming mode
        skip_modern: Skip cases from known modern reporter series (f2d, f3d)
        max_total: Maximum total cases to PROCESS (for testing)

    Note: The dataset has ~6.7M cases. Pre-1913 cases are in older reporter
    series (us_*, f_*) which may be later in the stream. With skip_modern=True,
    modern reporters are skipped to find older cases faster.
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
        "skipped_modern": 0,
        "accepted": 0,
        "rejected": 0,
        "rejection_reasons": {},
        "total_chars": 0,
        "years_distribution": {},
        "courts_distribution": {},
        "reporter_distribution": {},
    }

    output_file = os.path.join(output_dir, f"caselaw_{cutoff}.jsonl")

    print(f"\nLoading dataset from HuggingFace: {DATASET_NAME}")
    print("Note: This is a large dataset (~6.7M cases). Streaming mode is recommended.")
    if skip_modern:
        print(f"Skipping modern reporters (f2d, f3d, etc.) to find pre-{cutoff_year} cases faster.")

    try:
        ds = load_dataset(
            DATASET_NAME,
            streaming=streaming,
            split="train",
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"\nDataset URL: https://huggingface.co/datasets/{DATASET_NAME}")
        print("Alternative: Download directly from case.law API: https://case.law/docs/")

        # Create instructions file
        instructions = {
            "instructions": "Download Caselaw Access Project data",
            "huggingface": f"https://huggingface.co/datasets/{DATASET_NAME}",
            "api": "https://case.law/",
            "error": str(e),
            "notes": [
                "Large dataset - use streaming mode",
                "Filter by decision_date for temporal cutoff",
            ],
        }

        with open(os.path.join(output_dir, "README.json"), 'w') as f:
            json.dump(instructions, f, indent=2)

        return stats

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, case in enumerate(tqdm(ds, desc="Processing cases")):
            if max_cases and stats["accepted"] >= max_cases:
                print(f"\nReached max_cases limit ({max_cases} accepted)")
                break

            if max_total and i >= max_total:
                print(f"\nReached max_total limit ({max_total} processed)")
                break

            case_id = case.get('id', '')

            # Skip modern reporters if requested
            # Only skip known modern federal reporters (f2d, f3d, fsupp, etc.)
            # Keep unknown (state) and historical reporters for year checking
            if skip_modern:
                era = get_reporter_era(case_id)
                if era == 'modern':
                    stats["skipped_modern"] += 1
                    # Progress update every 100k skipped
                    if stats["skipped_modern"] % 100000 == 0:
                        print(f"  Skipped {stats['skipped_modern']:,} modern cases...")
                    continue
                # Track era for non-skipped cases
                stats["era_distribution"] = stats.get("era_distribution", {})
                stats["era_distribution"][era] = stats["era_distribution"].get(era, 0) + 1

            stats["total_processed"] += 1

            # Track reporter distribution
            reporter = case_id.split('/')[0] if '/' in case_id else case_id[:10]
            stats["reporter_distribution"][reporter] = stats["reporter_distribution"].get(reporter, 0) + 1

            # Extract text - common-pile format has text directly
            text = case.get('text', '')

            suitable, reason = is_case_suitable(text, case, cutoff_year)

            if suitable:
                stats["accepted"] += 1
                stats["total_chars"] += len(text)

                year = extract_case_year(case)
                if year:
                    decade = (year // 10) * 10
                    stats["years_distribution"][decade] = stats["years_distribution"].get(decade, 0) + 1

                # Extract court from text (common-pile format embeds it)
                court = extract_court_from_text(text)
                stats["courts_distribution"][court] = stats["courts_distribution"].get(court, 0) + 1

                # Extract case name from first line of text
                first_line = text.strip().split('\n')[0].strip() if text else ''
                case_name = first_line[:200] if first_line else case.get('id', 'Unknown')

                record = {
                    "text": text,
                    "source": "caselaw",
                    "year": year,
                    "court": court,
                    "case_name": case_name,
                    "id": case.get('id', ''),
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
    print(f"  Processed: {stats['total_processed']:,}")
    if skip_modern:
        print(f"  Skipped (modern): {stats['skipped_modern']:,}")
    print(f"  Accepted: {stats['accepted']:,}")
    print(f"  Rejected: {stats['rejected']:,}")
    print(f"  Total chars: {stats['total_chars']:,}")

    if stats['years_distribution']:
        print(f"\nYears distribution:")
        for decade in sorted(stats['years_distribution'].keys()):
            print(f"  {decade}s: {stats['years_distribution'][decade]:,}")

    if stats.get('era_distribution'):
        print(f"\nEra distribution (non-skipped):")
        for era, count in sorted(stats['era_distribution'].items()):
            print(f"  {era}: {count:,}")

    if stats.get('reporter_distribution'):
        print(f"\nTop reporters processed:")
        top_reporters = sorted(stats['reporter_distribution'].items(), key=lambda x: -x[1])[:10]
        for reporter, count in top_reporters:
            print(f"  {reporter}: {count:,}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Caselaw Access Project data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download up to 100 pre-1913 cases (skip modern reporters)
  python -m data.download.caselaw_download --cutoff 1913 --max-cases 100

  # Process all cases without skipping (very slow)
  python -m data.download.caselaw_download --no-skip-modern

  # Test with limited total processing
  python -m data.download.caselaw_download --max-total 10000
        """
    )
    parser.add_argument("--cutoff", type=str, default="1913",
                        help="Temporal cutoff year")
    parser.add_argument("--output-dir", type=str, default="data/raw/caselaw",
                        help="Output directory")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Maximum ACCEPTED cases to include")
    parser.add_argument("--max-total", type=int, default=None,
                        help="Maximum total cases to PROCESS (for testing)")
    parser.add_argument("--skip-modern", action="store_true", default=True,
                        help="Skip modern reporters (f2d, f3d, etc.) [default]")
    parser.add_argument("--no-skip-modern", action="store_false", dest="skip_modern",
                        help="Process all cases including modern reporters")

    args = parser.parse_args()

    download_caselaw(
        cutoff=args.cutoff,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        max_total=args.max_total,
        skip_modern=args.skip_modern,
    )
