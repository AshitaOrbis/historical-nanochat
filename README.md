# Historical Nanochat

Build "time-locked" language models using Karpathy's nanochat pipeline, trained exclusively on texts from before specific historical cutoff dates. The resulting models genuinely don't know about events after their cutoff - creating authentic historical worldview simulation.

## Project Overview

This project extends [nanochat](https://github.com/karpathy/nanochat) to train language models on historical text corpora. Unlike fine-tuning modern models for historical roleplay, these models are trained from scratch on pre-cutoff texts, ensuring genuine temporal ignorance.

### Key Features

- **Temporal cutoffs**: Pre-1850, Pre-1900, Pre-1913 (WWI), Pre-1950
- **Multiple data sources**: Project Gutenberg, Old Bailey, Chronicling America, Caselaw Access Project
- **Contamination detection**: Automated detection of anachronistic content
- **Nanochat-compatible**: Produces shards in the exact format nanochat expects

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/historical-nanochat.git
cd historical-nanochat

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Or with uv (faster)
uv pip install -e .
```

## Quick Start

### 1. Download Historical Data

```bash
# Download Project Gutenberg with 1913 cutoff
python -m data.download.gutenberg_download --cutoff 1913 --max-docs 1000

# Download Old Bailey proceedings
python -m data.download.oldbailey_download --cutoff 1913

# Download historical newspapers
python -m data.download.chronicling_download --cutoff 1913 --max-pages 500

# Download historical case law
python -m data.download.caselaw_download --cutoff 1913 --max-cases 500
```

### 2. Package into Shards

```bash
# Combine sources and package into nanochat shards
python -m data.process.shard_packager \
    --input data/raw/gutenberg/gutenberg_1913.jsonl \
            data/raw/oldbailey/oldbailey_1913.jsonl \
            data/raw/chronicling_america/newspapers_1913.jsonl \
    --output-dir data/processed/shards_1913 \
    --cutoff 1913
```

### 3. Train with Nanochat

```bash
# Point nanochat to historical shards
export NANOCHAT_BASE_DIR="./data/processed/shards_1913"

# Run nanochat training
cd nanochat
bash speedrun.sh
```

## Data Sources

| Source | Size | Date Range | Access |
|--------|------|------------|--------|
| **Project Gutenberg** | ~3B tokens, 50K+ books | Pre-1924 | HuggingFace |
| **Old Bailey Corpus** | 127M words | 1674-1913 | CLARIN-D |
| **Chronicling America** | Newspapers | 1756-1963 | LOC API |
| **Caselaw Access Project** | 6.7M cases | 1658-2020 | HuggingFace |

## Temporal Cutoffs

| Cutoff | Model Name | What It Doesn't Know |
|--------|-----------|---------------------|
| **1850** | `nanochat-1850` | Telephone, electric light, Darwin's Origin |
| **1900** | `nanochat-1900` | Airplanes, radio, relativity |
| **1913** | `nanochat-1913` | WWI, Russian Revolution, Hitler, atomic bomb |
| **1950** | `nanochat-1950` | Cold War, computers, space race |

## Contamination Detection

The contamination checker detects:
- Anachronistic terms (e.g., "atomic bomb" in pre-1913 text)
- Post-cutoff year references
- Modern date formats, URLs, emails
- Modern annotations in digitized texts

```python
from data.process.contamination_check import check_contamination

result = check_contamination("Hitler invaded Poland in 1939.", cutoff_year=1913)
print(result.is_contaminated)  # True
print(result.reasons)  # ['Anachronistic term 'hitler' found', 'Year reference: 1939']
```

## Project Structure

```
historical-nanochat/
├── nanochat/              # Karpathy's nanochat (cloned)
├── data/
│   ├── download/          # Data download scripts
│   │   ├── gutenberg_download.py
│   │   ├── oldbailey_download.py
│   │   ├── chronicling_download.py
│   │   └── caselaw_download.py
│   ├── process/           # Data processing
│   │   ├── contamination_check.py
│   │   └── shard_packager.py
│   ├── raw/               # Downloaded data
│   └── processed/         # Processed shards
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks
└── scripts/               # Utility scripts
```

## Training Requirements

| Hardware | Config | Time | Cost |
|----------|--------|------|------|
| RTX 3090 | d20, reduced batch | ~2-3 weeks | Electricity |
| 8xH100 (Lambda) | d20-d26 | 4-12 hours | ~$100-300 |
| 8xH100 (Lambda) | d34 | ~40 hours | ~$1000+ |

## Research Questions

1. How does expressed optimism change across temporal cutoffs?
2. What beliefs shift between 1900 → 1913 → 1950?
3. Can the model "predict" future events from its knowledge state?
4. How do scientific explanations differ by era?

## Related Work

- [Ranke-4B](https://github.com/DGoettlich/history-llms) - Historical LLMs from Zurich (pre-release)
- [Vintage LLMs](https://owainevans.github.io/talk-transcript.html) - Concept exploration
- [nanochat](https://github.com/karpathy/nanochat) - Base training pipeline

## License

MIT License (same as nanochat)

## Contributing

Contributions welcome! See the plan file at `.claude/plans/purring-moseying-cocoa.md` for the full implementation roadmap.
