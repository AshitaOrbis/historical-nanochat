"""
Contamination detection utilities for historical nanochat.

Detects anachronistic content that would contaminate temporal cutoffs. Uses
word-boundary matching plus nearby-context rules to reduce false positives.

Design notes
------------
- "Definite" anachronisms: phrases that can't plausibly appear pre-cutoff
  (e.g. "nuclear reactor", "world war ii"). Matched with word boundaries.
- "Contextual" anachronisms: single tokens that *can* appear in historical
  texts with a non-modern meaning (e.g. "atomic" in atomic theory, "radio"
  as the Latin root). These are flagged ONLY if a modern context word
  appears within a small window. The older code incorrectly short-circuited
  these terms via SAFE_HISTORICAL_TERMS, effectively disabling the
  contextual check. That behavior is fixed here.
- Metadata-contamination (headers, producer notes, modern URLs inside
  bibliographic records) is separated from body-contamination so we can
  strip the former and only treat the latter as a reject signal.
"""
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


CHECKER_VERSION = "3.0.0"


@dataclass
class ContaminationResult:
    """Result of contamination check."""
    is_contaminated: bool
    confidence: float  # 0.0 to 1.0
    reasons: List[str]
    matched_terms: List[str]
    # New: allow callers to see which class of signal fired the verdict.
    signal_categories: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# ANACHRONISM DETECTION CONFIGURATION
# ============================================================================
# Terms are grouped by their earliest valid year. A term is anachronistic when
# that year is later than the requested cutoff. Keeping actual first-valid-year
# semantics prevents a later cutoff (notably 1950) from disabling genuinely
# later concepts merely because they once lived in a coarse "1913" bucket.
# ============================================================================

ANACHRONISM_DEFINITE = {
    1914: {
        "world war i", "world war 1", "wwi", "first world war",
        "the great war", "world war",
    },
    1919: {
        "benito mussolini", "fascism", "fascist",
    },
    1920: {
        "hitler", "stalin", "adolf hitler", "nazi", "nazism",
        "radio broadcast", "broadcast station",
    },
    1925: {
        "ss troops", "quantum mechanics", "quantum physics",
    },
    1933: {
        "gestapo", "third reich", "fuhrer", "führer",
        "concentration camp", "death camp",
    },
    1939: {
        "world war ii", "world war 2", "wwii", "second world war",
        "holocaust", "auschwitz", "concentration camp", "death camp",
        "pearl harbor", "d-day", "hiroshima", "nagasaki",
        "blitzkrieg", "luftwaffe",
    },
    1942: {"nuclear reactor"},
    1944: {"genocide"},
    1945: {"atomic bomb", "nuclear weapon"},
    1947: {"cold war", "iron curtain"},
    1950: {"korean war"},
    1952: {"hydrogen bomb"},
    1953: {"software"},
    1955: {"vietnam war"},
    1957: {"sputnik", "cosmonaut"},
    1958: {"nasa"},
    1959: {"astronaut"},
    1961: {"apollo mission", "bay of pigs"},
    1962: {"cuban missile crisis"},
    1969: {"moon landing", "apollo 11", "internet"},
    1971: {"email"},
    1973: {"cell phone", "mobile phone"},
    1990: {"website"},
    1994: {"smartphone"},
    1917: {"soviet union", "ussr"},
    1903: {
        "wright brothers", "kitty hawk",
        "airplane flight", "aeroplane flight",
        "airplane", "aeroplane", "aircraft",
    },
    1905: {
        "theory of relativity", "special relativity", "general relativity",
    },
    1876: {"telephone call"},
    1879: {"electric light bulb", "light bulb"},
    1860: {"internal combustion engine", "darwinism"},
    1858: {"natural selection"},
    1859: {"origin of species"},
    1928: {"television broadcast", "tv broadcast"},
    1946: {"computer program"},
}

# Context-dependent terms: require at least one nearby context word to fire.
# NOTE: context words were tightened after both Opus and GPT-5.4 reviews flagged
# over-firing on legitimate 19th-century scientific prose (e.g. "atomic energy"
# appears in pre-1913 chemistry discussing bond energies; "radio broadcast" appears
# in pre-1913 discussions of Marconi/Hertz). We keep only context words that are
# specifically post-cutoff in usage, so a single contextual hit crossing the 0.3
# threshold is a strong signal rather than a false positive.
#
# Dead-code note: 'hitler' and 'stalin' also appear in ANACHRONISM_DEFINITE[1920]
# above, so the definite pass will fire before the contextual pass ever sees them.
# The contextual entries below for those names are kept for clarity only.
ANACHRONISM_CONTEXTUAL = {
    1920: {
        "stalin": ["joseph", "soviet", "purge", "gulag"],  # covered by definite; kept for documentation
        "hitler": ["adolf", "nazi", "reich", "fuhrer"],     # covered by definite
        # Removed "station", "wireless" from radio context words: Marconi/Hertz had
        # pre-1913 wireless stations. Keep only clearly post-cutoff terms.
        "radio": ["broadcast station", "radio program", "radio channel"],
    },
    1940: {
        "churchill": ["prime minister", "world war", "britain at war"],
    },
    1946: {
        "computer": ["electronic", "digital", "software", "program code"],  # 'machine' was too generic
    },
    1928: {
        "television": ["broadcast", "tv channel", "tv screen", "tv program"],
    },
    1945: {
        # Removed "energy", "reactor", "power plant" from atomic/nuclear context words:
        # "atomic energy" is legit pre-1913 scientific discourse (bond energies).
        # The definite-list entry "atomic bomb" and "nuclear reactor" still catches
        # explicit post-1913 phrases.
        "atomic": ["bomb", "weapon", "fission"],
        "nuclear": ["bomb", "weapon", "fission", "warhead"],
    },
    1917: {
        "soviet": ["union", "ussr", "politburo", "kremlin"],  # 'russia' was too generic
    },
    1896: {
        "radioactiv": ["curie", "uranium", "radium", "decay"],
    },
    1895: {
        "x-ray": ["roentgen", "medical", "photograph"],
    },
}

# STANDALONE_SAFE_TERMS: terms that, ON THEIR OWN (not in a contextual entry),
# look like anachronisms but have legitimate historical uses. These are only
# filtered in the DEFINITE pass. Terms that have contextual rules must NOT be
# listed here — the whole point of the contextual check is to flag them when
# they *do* appear with modern context.
STANDALONE_SAFE_TERMS = {
    # Greek/Roman gods that name later technology
    "apollo", "mercury", "mars", "saturn", "jupiter", "neptune", "venus",
    "titan", "atlas",
    # Latin / etymological overlaps with modern tech but with no contextual rule
    "plastic",      # "plastic arts" pre-1860
    "broadcast",    # agricultural term (scattering seed)
    "network",      # textile/fishing term
    "satellite",    # astronomical moons
    "electron",     # Greek "elektron" (amber), used pre-1900
    "quantum",      # Latin for "how much", philosophical
}


def get_definite_anachronisms(cutoff_year: int) -> Set[str]:
    """High-confidence anachronistic phrases for a given cutoff year."""
    terms: Set[str] = set()
    for year, year_terms in ANACHRONISM_DEFINITE.items():
        if year > cutoff_year:
            for term in year_terms:
                if term:
                    terms.add(term.lower())
    return terms


def get_contextual_anachronisms(cutoff_year: int) -> Dict[str, List[str]]:
    """Context-dependent anachronistic terms for a given cutoff year."""
    terms: Dict[str, List[str]] = {}
    for year, year_terms in ANACHRONISM_CONTEXTUAL.items():
        if year > cutoff_year:
            for term, contexts in year_terms.items():
                terms[term.lower()] = [c.lower() for c in contexts]
    return terms


def match_with_word_boundary(text_lower: str, term: str) -> List[int]:
    """Find all occurrences of term with word boundaries. Handles simple plurals."""
    escaped = re.escape(term)
    if ' ' not in term:
        if term.endswith(('s', 'x', 'z', 'ch', 'sh')):
            pattern = r'\b' + escaped + r'(?:es)?\b'
        else:
            pattern = r'\b' + escaped + r's?\b'
    else:
        pattern = r'\b' + escaped + r'\b'
    return [m.start() for m in re.finditer(pattern, text_lower, re.IGNORECASE)]


def check_context(text_lower: str, term_idx: int, term: str,
                  context_words: List[str], window: int = 200) -> bool:
    """True if any context word appears within ±window chars of term."""
    start = max(0, term_idx - window)
    end = min(len(text_lower), term_idx + len(term) + window)
    snippet = text_lower[start:end]
    return any(ctx in snippet for ctx in context_words)


def check_for_modern_references(text: str, cutoff_year: int) -> List[str]:
    """
    Detect modern-era signals (URLs, emails, post-cutoff years) in the body.
    Filters obvious false positives (currency, page/MS numbers).
    """
    text_lower = text.lower()
    found: List[str] = []

    year_pattern = r'\b(1\d{3}|20\d{2})\b'
    post_cutoff_years: Set[int] = set()
    for match in re.finditer(year_pattern, text):
        year = int(match.group(1))
        if year > cutoff_year:
            idx = match.start()
            before = text[max(0, idx - 30):idx].lower()
            after = text[match.end():min(len(text), match.end() + 30)].lower()
            if re.search(r'[£$]\s*$', before):
                continue
            if re.search(r'(?:no|vol|pp?|mss?|egerton|ms)\.*\s*$', before):
                continue
            if re.search(r'\d[,.]$', before) or re.search(r'^(?:\d|[,.]\d)', after):
                continue
            post_cutoff_years.add(year)

    if post_cutoff_years:
        examples = sorted(post_cutoff_years)[:3]
        found.append(f"Post-cutoff years: {examples}")

    if re.search(r'https?://', text_lower):
        found.append("URL found (https/http)")
    elif re.search(r'www\.\w+\.\w+', text_lower):
        found.append("URL found (www.)")
    elif re.search(r'\S+@\S+\.\S+', text_lower):
        found.append("Email address found")

    return found


def check_contamination(
    text: str,
    cutoff_year: int,
    context_window: int = 200,
    threshold: float = 0.3,
    check_contextual: bool = True,
) -> ContaminationResult:
    """
    Body-level contamination check.

    IMPORTANT: pass a text that already has metadata/headers stripped (e.g.
    Gutenberg headers). Header/producer notes commonly contain modern URLs
    or dates that would falsely flag an otherwise legitimate historical work.
    Use clean_gutenberg_headers / strip_metadata_preamble first.
    """
    text_lower = text.lower()
    reasons: List[str] = []
    matched_terms: List[str] = []
    categories: Dict[str, int] = {}
    confidence = 0.0

    # 1) definite anachronisms
    definite_terms = get_definite_anachronisms(cutoff_year)
    for term in definite_terms:
        # Single-token terms (no space): skip if term is standalone-safe.
        if ' ' not in term and term in STANDALONE_SAFE_TERMS:
            continue
        matches = match_with_word_boundary(text_lower, term)
        if matches:
            matched_terms.append(term)
            reasons.append(f"Definite anachronism: '{term}'")
            confidence = min(1.0, confidence + 0.3)
            categories["definite"] = categories.get("definite", 0) + 1

    # 2) contextual anachronisms — these are INTENTIONALLY not filtered via
    # STANDALONE_SAFE_TERMS; the whole point is to allow the benign use while
    # catching the modern-context use.
    if check_contextual:
        contextual_terms = get_contextual_anachronisms(cutoff_year)
        for term, context_words in contextual_terms.items():
            matches = match_with_word_boundary(text_lower, term)
            for idx in matches:
                if check_context(text_lower, idx, term, context_words, context_window):
                    matched_terms.append(term)
                    reasons.append(f"Contextual anachronism: '{term}' (with modern context)")
                    # Bumped from 0.25 to 0.3 so a single contextual hit alone crosses
                    # the default threshold (matches operator intent: if "radio broadcast"
                    # appears, flag it — don't require a second signal).
                    confidence = min(1.0, confidence + 0.3)
                    categories["contextual"] = categories.get("contextual", 0) + 1
                    break  # one hit per term is enough

    # 3) modern references in body text
    modern_refs = check_for_modern_references(text, cutoff_year)
    for ref in modern_refs:
        reasons.append(ref)
        confidence = min(1.0, confidence + 0.4)
        categories["modern_ref"] = categories.get("modern_ref", 0) + 1

    is_contaminated = confidence >= threshold
    return ContaminationResult(
        is_contaminated=is_contaminated,
        confidence=confidence,
        reasons=reasons,
        matched_terms=matched_terms,
        signal_categories=categories,
    )


def check_metadata_contamination(metadata: Dict, cutoff_year: int) -> ContaminationResult:
    """
    Metadata-only contamination check.

    Different signal profile: we care about e.g. catalog numbers, download
    counts, scanner notes that shouldn't gate the body text. For now this
    simply flags post-cutoff year references found in metadata values so
    operators can track how often bad metadata slips through, without
    necessarily rejecting the document.
    """
    reasons: List[str] = []
    confidence = 0.0
    flat = " ".join(str(v) for v in metadata.values() if v is not None)
    modern = check_for_modern_references(flat, cutoff_year)
    for m in modern:
        reasons.append(f"Metadata: {m}")
        confidence = min(1.0, confidence + 0.2)
    return ContaminationResult(
        is_contaminated=False,  # metadata alone never rejects
        confidence=confidence,
        reasons=reasons,
        matched_terms=[],
        signal_categories={"metadata": len(modern)} if modern else {},
    )


def clean_gutenberg_headers(text: str) -> str:
    """
    Strip Project Gutenberg headers, footers, and modern-era metadata from
    the body of a document. Produces a cleaned body suitable for training.
    """
    header_end_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
        "END OF THE PROJECT GUTENBERG HEADER",
    ]
    footer_start_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]
    for marker in header_end_markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            end_idx = text.find('\n', idx)
            if end_idx != -1:
                text = text[end_idx + 1:]
            break
    for marker in footer_start_markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            text = text[:idx]
            break

    metadata_markers = [
        "Distributed Proofreading", "Internet Archive",
        "produced by", "Produced by", "prepared by", "Prepared by",
        "scanned by", "Scanned by", "This eBook", "This ebook",
        "transcriber's note", "Transcriber's Note",
        "note: images of the original", "Note: Images of the original",
    ]
    first_section = text[:3000].lower()
    latest_metadata_end = 0
    for marker in metadata_markers:
        idx = first_section.find(marker.lower())
        if idx != -1:
            para_break = text.find('\n\n', idx)
            if para_break != -1 and para_break < 3000:
                if para_break + 2 > latest_metadata_end:
                    latest_metadata_end = para_break + 2
    if latest_metadata_end > 0:
        text = text[latest_metadata_end:]

    first_part = text[:500]
    rest = text[500:] if len(text) > 500 else ""
    first_part = re.sub(r'https?://[^\s\)]+', '', first_part)
    first_part = re.sub(r'www\.[^\s\)]+', '', first_part)
    first_part = re.sub(r'\S+@\S+\.\S+', '', first_part)
    return (first_part + rest).strip()


def batch_check_contamination(
    texts: List[str],
    cutoff_year: int,
    return_clean: bool = True,
) -> Tuple[List[str], Dict]:
    """Batch helper. Returns (clean_texts, stats with reasons + categories)."""
    clean_texts: List[str] = []
    stats = {
        "total": len(texts),
        "clean": 0,
        "contaminated": 0,
        "contamination_reasons": {},
        "signal_categories": {},
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
            for cat, count in result.signal_categories.items():
                stats["signal_categories"][cat] = stats["signal_categories"].get(cat, 0) + count
    return clean_texts, stats


# Backwards-compat alias for external code that imported SAFE_HISTORICAL_TERMS.
SAFE_HISTORICAL_TERMS = STANDALONE_SAFE_TERMS


if __name__ == "__main__":
    # Smoke tests: if these fail the contamination checker regressed.
    test_texts = [
        # Clean historical
        ("The king rode his horse to the castle in 1812.", False, "Clean historical"),
        ("The telegraph was a remarkable invention.", False, "Clean - telegraph"),
        ("Apollo was worshipped by the ancient Greeks.", False, "Clean - Apollo the god"),
        ("The computer calculated the astronomical tables.", False, "Clean - human computer"),
        ("The atomic theory of matter was debated.", False, "Clean - atomic theory"),
        ("Mercury and Venus are the inner planets.", False, "Clean - planets"),
        # Definite contamination
        ("Hitler invaded Poland in 1939.", True, "Nazi era reference"),
        ("The Holocaust killed millions.", True, "Holocaust reference"),
        ("During World War II, the allies...", True, "WWII reference"),
        ("Visit our website at www.example.com", True, "URL reference"),
        ("The astronauts landed on the moon.", True, "Moon landing"),
        # Contextual: these are the critical regression tests for the bug we fixed.
        ("The atomic bomb was dropped on Hiroshima.", True, "Atomic with bomb context"),
        ("The Soviet Union collapsed in 1991.", True, "Soviet Union with Russia context"),
        ("The radio broadcast reached thousands of listeners.", True, "Radio with broadcast context"),
        # Contextual without modern context
        ("The atomic structure of elements varies.", False, "Atomic without weapon context"),
        ("Television of distant objects was imagined.", False, "Television without broadcast context"),
    ]

    print("Testing contamination detection:\n")
    fails = 0
    for text, expected_contaminated, description in test_texts:
        result = check_contamination(text, 1913)
        status = "CONTAMINATED" if result.is_contaminated else "CLEAN"
        ok = result.is_contaminated == expected_contaminated
        if not ok:
            fails += 1
        marker = "OK  " if ok else "FAIL"
        print(f"[{marker}] [{status}] {description}")
        print(f"        conf={result.confidence:.2f} terms={result.matched_terms}")
    print(f"\n{len(test_texts) - fails}/{len(test_texts)} tests passed")
    if fails:
        raise SystemExit(1)
