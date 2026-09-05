"""
Download and filter Project Gutenberg texts for historical nanochat training.

Uses the HuggingFace dataset: manu/project_gutenberg
Filters by publication date to ensure temporal cutoff compliance.
"""
import os
import json
import argparse
from dataclasses import asdict, dataclass
from typing import Optional, Iterator, Dict, Any
import warnings
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


PUBLICATION_DATE_FIELDS = (
    "issued",
    "publicationyear",
    "published",
    "publication_year",
)


@dataclass(frozen=True)
class PublicationDateEvidence:
    year: int
    source_field: str
    source_value: str
    confidence: str


def _year_from_value(value: Any) -> Optional[int]:
    match = re.search(r"\b(1\d{3}|20\d{2})\b", str(value))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1000 <= year <= 2099 else None


def _years_from_value(value: Any) -> list[int]:
    return [
        int(year)
        for year in re.findall(r"\b(1\d{3}|20\d{2})\b", str(value))
        if 1000 <= int(year) <= 2099
    ]


def extract_publication_date(
    metadata: Dict[str, Any],
    text: str = "",
) -> Optional[PublicationDateEvidence]:
    """
    Return affirmative publication/issue evidence, never an author-life proxy.

    Only allowlisted metadata fields and explicit "published/printed in YEAR"
    text phrases qualify. Author dates and free-form metadata years are not
    publication evidence and are handled only as non-strict diagnostics.
    """
    for field in PUBLICATION_DATE_FIELDS:
        value = metadata.get(field)
        year = _year_from_value(value) if value not in (None, "") else None
        if year is not None:
            return PublicationDateEvidence(
                year=year,
                source_field=field,
                source_value=str(value),
                confidence="validated_publication_field",
            )

    sample = text[:5000]
    for label, pattern in (
        ("text:published", r"\bpublished(?:\s+in)?[^\d]{0,20}(1\d{3}|20\d{2})\b"),
        ("text:printed", r"\bprinted(?:\s+in)?[^\d]{0,20}(1\d{3}|20\d{2})\b"),
    ):
        match = re.search(pattern, sample, re.IGNORECASE)
        if match:
            return PublicationDateEvidence(
                year=int(match.group(1)),
                source_field=label,
                source_value=match.group(0),
                confidence="validated_publication_phrase",
            )

    return None


def extract_year_from_metadata(metadata: Dict[str, Any]) -> Optional[int]:
    """Legacy diagnostic helper; never use its hints for strict admission."""
    evidence = extract_publication_date(metadata)
    if evidence:
        return evidence.year

    warnings.warn(
        "extract_year_from_metadata may return non-binding author/free-form hints; "
        "use extract_publication_date for admission decisions",
        DeprecationWarning,
        stacklevel=2,
    )
    death = _year_from_value(metadata.get("authoryearofdeath"))
    if death is not None:
        return death
    birth = _year_from_value(metadata.get("authoryearofbirth"))
    if birth is not None:
        return birth + 20
    candidates = []
    for field, value in metadata.items():
        if field == "downloads" or value in (None, ""):
            continue
        year = _year_from_value(value)
        if year is not None:
            candidates.append(year)
    return min(candidates) if candidates else None


def estimate_year_from_text(text: str, max_chars: int = 5000) -> Optional[int]:
    """Compatibility helper for explicit publication/printing phrases only."""
    evidence = extract_publication_date({}, text[:max_chars])
    return evidence.year if evidence else None


def extract_non_strict_date_hint(metadata: Dict[str, Any]) -> Optional[PublicationDateEvidence]:
    """Compatibility helper returning the first non-binding diagnostic hint."""
    hints = extract_non_strict_date_hints(metadata)
    return hints[0] if hints else None


def extract_non_strict_date_hints(metadata: Dict[str, Any]) -> list[PublicationDateEvidence]:
    """Return all diagnostic hints; callers may use them only to reject."""
    hints: list[PublicationDateEvidence] = []
    for field in ("authoryearofdeath", "authoryearofbirth"):
        value = metadata.get(field)
        for year in _years_from_value(value) if value not in (None, "") else []:
            hints.append(PublicationDateEvidence(
                    year=year,
                    source_field=field,
                    source_value=str(value),
                    confidence="author_life_hint_non_binding",
                ))

    for field, value in metadata.items():
        if (
            field == "downloads"
            or field in PUBLICATION_DATE_FIELDS
            or field in {"authoryearofdeath", "authoryearofbirth"}
            or value in (None, "")
        ):
            continue
        for year in _years_from_value(value):
            hints.append(PublicationDateEvidence(
                year=year,
                source_field=field,
                source_value=str(value),
                confidence="free_form_metadata_hint_non_binding",
            ))
    return hints




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

    date_evidence = extract_publication_date(metadata, text)

    if date_evidence is not None:
        if date_evidence.year > cutoff_year:
            return False, (
                f"Post-cutoff publication year: {date_evidence.year} > {cutoff_year}"
            ), cleaned_text
    elif strict:
        # In strict mode, reject texts with unknown dates
        return False, "Unknown publication date", cleaned_text

    if not strict:
        post_cutoff_hints = [
            hint for hint in extract_non_strict_date_hints(metadata)
            if hint.year > cutoff_year
        ]
        if post_cutoff_hints:
            year = max(hint.year for hint in post_cutoff_hints)
            return False, f"Post-cutoff diagnostic year: {year} > {cutoff_year}", cleaned_text

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
        "records_requested": 0,
        "records_fetched": 0,
        "records_written": 0,
        "total_processed": 0,
        "accepted": 0,
        "rejected": 0,
        "rejection_reasons": {},
        "total_chars": 0,
        "years_distribution": {},
        "config": {
            "cutoff_year": cutoff_year,
            "temporal_filter_mode": "strict" if strict else "non_strict",
        },
    }

    output_file = os.path.join(output_dir, f"gutenberg_{cutoff}.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(tqdm(ds, desc="Processing Gutenberg")):
            if max_docs and i >= max_docs:
                break

            stats["records_requested"] += 1
            stats["records_fetched"] += 1
            stats["total_processed"] += 1

            text = doc.get('text', '')
            metadata = {k: v for k, v in doc.items() if k != 'text'}

            suitable, reason, cleaned_text = is_text_suitable(
                text, metadata, cutoff_year, strict=strict
            )

            if suitable:
                stats["accepted"] += 1
                stats["records_written"] += 1
                stats["total_chars"] += len(cleaned_text)

                # Record year + provenance (so downstream audits can trace which field gave us the year)
                publication_date = extract_publication_date(metadata, text)
                date_diagnostics = (
                    extract_non_strict_date_hints(metadata) if not strict else []
                )
                year_evidence = publication_date or (
                    date_diagnostics[0] if date_diagnostics else None
                )
                year = year_evidence.year if year_evidence else None
                if year:
                    decade = (year // 10) * 10
                    stats["years_distribution"][decade] = stats["years_distribution"].get(decade, 0) + 1

                # Write to output (use cleaned text without Gutenberg headers)
                record = {
                    "text": cleaned_text,
                    "source": "gutenberg",
                    "metadata": metadata,
                    "publication_date": asdict(publication_date) if publication_date else None,
                    "date_diagnostics": [asdict(item) for item in date_diagnostics],
                    "temporal_filter_status": (
                        "STRICT_PUBLICATION_DATE" if strict else "NON_STRICT"
                    ),
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
    print(
        "  Acquisition counts: "
        f"requested={stats['records_requested']} "
        f"fetched={stats['records_fetched']} "
        f"written={stats['records_written']}"
    )

    if stats["records_written"] == 0:
        raise RuntimeError(
            "Project Gutenberg acquired no records "
            f"(requested={stats['records_requested']}, "
            f"fetched={stats['records_fetched']}, written=0)"
        )

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
