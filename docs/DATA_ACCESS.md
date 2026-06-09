# Historical Nanochat - Data Source Access Guide

## 1. Project Gutenberg
**Status**: ✅ Automated (HuggingFace dataset)

```bash
# Already implemented in our downloader
python -m data.download.gutenberg_download --cutoff 1913
```

- **Source**: HuggingFace `manu/project_gutenberg`
- **Size**: ~70,000 books, expect ~15,000 pre-1913 accepted
- **Tokens**: ~1.4B estimated


## 2. Caselaw Access Project
**Status**: ✅ Automated (HuggingFace dataset)

```bash
# Already implemented in our downloader
python -m data.download.caselaw_download --cutoff 1913
```

- **Source**: HuggingFace `common-pile/caselaw_access_project`
- **Size**: 6.7M cases, ~100K pre-1913
- **Tokens**: ~200-500M estimated


## 3. Chronicling America (Newspapers)
**Status**: ✅ Automated (LOC API)

```bash
# Already implemented in our downloader
python -m data.download.chronicling_download --cutoff 1913 --max-pages 10000
```

- **Source**: Library of Congress API
- **Size**: 946K+ pages pre-1913
- **Tokens**: 500M-2B estimated (depends on pages downloaded)

### Image Access for Gemini OCR (Optional)
High-resolution images available via IIIF:
```
https://tile.loc.gov/image-services/iiif/service:{batch_id}/full/full/0/default.jpg
```
- Full images: ~6000x7500 pixels, 6-8MB JPEG
- Can use Gemini Flash for higher-quality OCR than LOC's automated OCR


## 4. Old Bailey Corpus
**Status**: ⚠️ Manual download (no login required)

### Direct Download
```bash
# Download the corpus (182MB zip)
mkdir -p data/raw/oldbailey
cd data/raw/oldbailey
wget https://fedora.clarin-d.uni-saarland.de/oldbailey/downloads/OldBaileyCorpus2.zip
unzip OldBaileyCorpus2.zip
```

- **Source**: CLARIN-D (Saarland University)
- **Size**: 182MB compressed, 127M words
- **Date Range**: 1720-1913
- **Format**: XML with linguistic annotations
- **License**: CC BY-NC-SA 4.0
- **Tokens**: ~150M estimated

### Documentation
- [Downloads Page](https://fedora.clarin-d.uni-saarland.de/oldbailey/downloads.html)
- [Manual (PDF)](https://fedora.clarin-d.uni-saarland.de/oldbailey/downloads/OBC_2.0_Manual%202016-07-13.pdf)
- [CLARIN Showcase](https://www.clarin.eu/showcase/old-bailey-corpus-20-1720-1913)


## 5. EEBO-TCP (Early English Books Online)
**Status**: ⚠️ Manual download (no login required)

### Direct Download
```bash
# Download via Dropbox (several GB)
mkdir -p data/raw/eebo
cd data/raw/eebo

# Option 1: Dropbox (primary)
# Visit: https://www.dropbox.com/sh/pfx619wnjdck2lj/AAAeQjd_dv29oPymNoKJWfEYa?dl=0
# Download the ZIP files you need

# Option 2: Box.com (backup mirror)
# Visit: https://app.box.com/s/jjzmnrx98dkvanipopz3nxkvymnjccht
```

- **Source**: Text Creation Partnership
- **Size**: 60,000+ works, several GB
- **Date Range**: 1475-1700 (very early modern English)
- **Format**: TEI P5 XML (recommended) or P4 XML
- **License**: Public domain (as of Aug 2020)
- **Tokens**: ~500M estimated

### Available Formats
1. **TCP (P4) XML** - UTF-8 with TEI headers (recommended)
2. **SGML** - Original 7-bit encoding
3. **P5 XML** - TEI P5 conformant

### Documentation
- [TCP Official Site](https://textcreationpartnership.org/)
- [FAQ with Downloads](https://textcreationpartnership.org/faq/)
- [EarlyPrint Introduction](https://earlyprint.org/intros/intro-to-eebo-and-eebo-tcp.html)

### Note on EEBO-TCP for 1913 Cutoff
EEBO-TCP covers 1475-1700, which is very early modern English. The language
is significantly different from 1800s-1900s texts. Consider whether this
helps or hurts a 1913-cutoff model:
- **Pro**: Adds ~500M tokens, unique historical content
- **Con**: Very archaic language may not generalize well
- **Recommendation**: Test with a small sample first


## 6. HathiTrust (Future Option)
**Status**: ❌ Requires institutional access

HathiTrust has 6.6M volumes but requires:
1. Institutional affiliation
2. Research proposal
3. Approval process

See: https://www.hathitrust.org/htrc-access-policy


## Token Budget Summary

| Source | Est. Tokens | Status |
|--------|-------------|--------|
| Project Gutenberg | 1.4B | ✅ Automated |
| Chronicling America | 500M-2B | ✅ Automated |
| Old Bailey | 150M | ⚠️ Manual (easy) |
| EEBO-TCP | 500M | ⚠️ Manual (easy) |
| Caselaw | 200-500M | ✅ Automated |
| **TOTAL** | **2.75-4.55B** | |

This should be sufficient for **d16 model** (needs ~1.4B tokens with ratio=8).


## Processing Notes

### Old Bailey XML → JSONL
The Old Bailey corpus is in XML format. You'll need to:
1. Parse XML to extract trial proceedings
2. Convert to our JSONL format
3. Filter by date (1720-1913, all pre-cutoff)

### EEBO-TCP XML → JSONL
Similar process for EEBO:
1. Parse TEI XML
2. Extract text content
3. Convert to JSONL format
4. Note: Very old spelling (e.g., "ye" for "the")
