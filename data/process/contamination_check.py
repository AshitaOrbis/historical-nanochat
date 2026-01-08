"""
Contamination detection utilities for historical nanochat.

Detects anachronistic content that would contaminate the temporal boundaries.
"""
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ContaminationResult:
    """Result of contamination check."""
    is_contaminated: bool
    confidence: float  # 0.0 to 1.0
    reasons: List[str]
    matched_terms: List[str]


# Comprehensive anachronism dictionaries by cutoff year
# Terms that should NOT appear in text before the cutoff

ANACHRONISM_ENTITIES = {
    # Events
    1913: {
        "world war i", "world war 1", "the great war", "wwi",
        "world war ii", "world war 2", "wwii", "second world war",
        "russian revolution", "bolshevik revolution",
        "treaty of versailles", "league of nations",
        "great depression", "stock market crash 1929",
        "pearl harbor", "d-day", "hiroshima", "nagasaki",
        "holocaust", "concentration camp", "auschwitz",
        "cold war", "korean war", "vietnam war",
        "cuban missile crisis", "bay of pigs",
        "moon landing", "apollo 11",
    },
    1900: {
        "wright brothers", "kitty hawk", "first flight",
        "world war", "wwi", "wwii",
        "russian revolution", "bolshevik",
        "relativity", "quantum mechanics",
        "radio broadcast", "television",
    },
    1850: {
        "civil war" if "american" in "context" else "",  # Be careful with this one
        "darwin's origin of species",  # Published 1859
        "telephone", "electric light", "light bulb",
        "automobile", "motor car", "internal combustion",
        "airplane", "aeroplane",
        "world war", "wwi", "wwii",
    },
}

ANACHRONISM_PEOPLE = {
    1913: {
        "adolf hitler", "hitler",
        "benito mussolini", "mussolini",
        "joseph stalin", "stalin",
        "franklin roosevelt", "fdr",
        "winston churchill" if "prime minister" in "context" else "",
        "albert einstein" if "relativity" in "context" else "",
        "mao zedong", "mao tse-tung",
        "fidel castro",
        "john f kennedy", "jfk",
        "martin luther king",
        "nelson mandela",
    },
    1900: {
        "wright brothers",
        "albert einstein",
        "marie curie" if "radioactivity" in "context" else "",
    },
}

ANACHRONISM_TECHNOLOGY = {
    1913: {
        "television", "tv set",
        "radio broadcast", "wireless radio",
        "computer", "computing machine",
        "atomic bomb", "nuclear weapon", "hydrogen bomb",
        "jet engine", "jet aircraft",
        "helicopter",
        "penicillin", "antibiotic",
        "plastic",
        "nylon",
    },
    1900: {
        "airplane", "aeroplane", "aircraft",
        "radio", "wireless telegraphy",
        "automobile", "motor car",
        "moving picture", "cinema",
    },
    1850: {
        "telephone",
        "phonograph", "gramophone",
        "electric light", "incandescent",
        "typewriter",
        "photograph" if "daguerreotype" not in "context" else "",
    },
}

ANACHRONISM_CONCEPTS = {
    1913: {
        "nazi", "nazism", "fascism", "fascist",
        "soviet union", "ussr", "soviet",
        "communist party",
        "united nations",
        "human rights" if "declaration" in "context" else "",
        "genocide",
        "existentialism",
        "psychoanalysis" if "widespread" in "context" else "",  # Freud started earlier
    },
    1900: {
        "radioactivity", "radiation",
        "x-ray",
        "electron",
        "relativity",
        "quantum",
    },
}


def get_all_anachronisms(cutoff_year: int) -> Set[str]:
    """
    Get all anachronistic terms for a given cutoff year.
    Returns terms that should not appear in pre-cutoff texts.
    """
    terms = set()

    for category in [ANACHRONISM_ENTITIES, ANACHRONISM_PEOPLE,
                     ANACHRONISM_TECHNOLOGY, ANACHRONISM_CONCEPTS]:
        for year, year_terms in category.items():
            if year >= cutoff_year:
                # Terms for this year and later are anachronistic
                for term in year_terms:
                    if term:  # Skip empty strings
                        terms.add(term.lower())

    return terms


def check_for_modern_references(text: str, cutoff_year: int) -> List[str]:
    """
    Check for modern references that indicate post-cutoff content.
    Returns list of suspicious patterns found.
    """
    text_lower = text.lower()
    found = []

    # Check for year references after cutoff
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    for match in re.finditer(year_pattern, text):
        year = int(match.group(1))
        if year > cutoff_year:
            found.append(f"Year reference: {year}")

    # Check for modern date formats (less common in historical texts)
    modern_date_pattern = r'\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b'
    if re.search(modern_date_pattern, text):
        found.append("Modern date format")

    # Check for URLs and email
    if re.search(r'https?://|www\.|\.com|\.org|@\w+\.\w+', text_lower):
        found.append("URL or email address")

    # Check for modern currency symbols and amounts
    if re.search(r'\$\d{1,3}(,\d{3})+', text):  # Large dollar amounts
        found.append("Modern currency format")

    return found


def check_contamination(
    text: str,
    cutoff_year: int,
    context_window: int = 100,
    threshold: float = 0.3,
) -> ContaminationResult:
    """
    Comprehensive contamination check for a text.

    Args:
        text: The text to check
        cutoff_year: The temporal cutoff year
        context_window: Characters around match to check for context
        threshold: Confidence threshold for flagging (0.0-1.0)

    Returns:
        ContaminationResult with details
    """
    text_lower = text.lower()
    reasons = []
    matched_terms = []
    confidence = 0.0

    # Get anachronistic terms for this cutoff
    anachronisms = get_all_anachronisms(cutoff_year)

    # Check each term
    for term in anachronisms:
        if term in text_lower:
            # Get context around the match
            idx = text_lower.find(term)
            context = text_lower[max(0, idx - context_window):idx + len(term) + context_window]

            matched_terms.append(term)
            reasons.append(f"Anachronistic term '{term}' found")
            confidence = min(1.0, confidence + 0.2)

    # Check for modern references
    modern_refs = check_for_modern_references(text, cutoff_year)
    for ref in modern_refs:
        reasons.append(ref)
        confidence = min(1.0, confidence + 0.3)

    # Check for Gutenberg headers that might indicate modern edition
    if "project gutenberg" in text_lower[:2000]:
        # This is fine for the actual book content, but we should check for
        # modern introductions or annotations
        intro_markers = [
            "introduction by",
            "edited by",
            "annotated by",
            "foreword by",
            "notes by",
        ]
        for marker in intro_markers:
            if marker in text_lower[:5000]:
                reasons.append(f"Possible modern annotation: '{marker}'")
                confidence = min(1.0, confidence + 0.1)

    is_contaminated = confidence >= threshold

    return ContaminationResult(
        is_contaminated=is_contaminated,
        confidence=confidence,
        reasons=reasons,
        matched_terms=matched_terms,
    )


def clean_gutenberg_headers(text: str) -> str:
    """
    Remove Project Gutenberg headers and footers.
    These contain modern metadata that isn't part of the original text.
    """
    # Common Gutenberg header markers
    header_end_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
        "END OF THE PROJECT GUTENBERG HEADER",
    ]

    footer_start_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]

    # Remove header
    for marker in header_end_markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            # Find the end of the line with the marker
            end_idx = text.find('\n', idx)
            if end_idx != -1:
                text = text[end_idx + 1:]
            break

    # Remove footer
    for marker in footer_start_markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


def batch_check_contamination(
    texts: List[str],
    cutoff_year: int,
    return_clean: bool = True,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Check multiple texts for contamination.

    Args:
        texts: List of texts to check
        cutoff_year: The temporal cutoff year
        return_clean: If True, return only clean texts

    Returns:
        Tuple of (clean texts, statistics dict)
    """
    clean_texts = []
    stats = {
        "total": len(texts),
        "clean": 0,
        "contaminated": 0,
        "contamination_reasons": {},
    }

    for text in texts:
        result = check_contamination(text, cutoff_year)

        if not result.is_contaminated:
            stats["clean"] += 1
            if return_clean:
                clean_texts.append(text)
        else:
            stats["contaminated"] += 1
            for reason in result.reasons:
                stats["contamination_reasons"][reason] = stats["contamination_reasons"].get(reason, 0) + 1

    return clean_texts, stats


if __name__ == "__main__":
    # Test the contamination checker
    test_texts = [
        "The king rode his horse to the castle in 1812.",  # Clean
        "Hitler invaded Poland in 1939.",  # Contaminated - post-1913
        "The airplane flew over the city.",  # Contaminated for 1900, OK for 1913
        "Visit our website at www.example.com for more info.",  # Modern reference
        "The telegraph was a remarkable invention.",  # Clean for all cutoffs
    ]

    print("Testing contamination detection:\n")

    for cutoff in [1850, 1900, 1913]:
        print(f"\n=== Cutoff: {cutoff} ===")
        for text in test_texts:
            result = check_contamination(text, cutoff)
            status = "CONTAMINATED" if result.is_contaminated else "CLEAN"
            print(f"\n[{status}] (confidence: {result.confidence:.2f})")
            print(f"  Text: {text[:50]}...")
            if result.reasons:
                print(f"  Reasons: {result.reasons}")
