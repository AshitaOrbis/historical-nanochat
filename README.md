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
git clone https://github.com/AshitaOrbis/historical-nanochat.git
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
# Streaming packager with bounded-memory shuffle + per-shard manifest
python -m data.process.shard_packager \
    --data-dir data/raw \
    --output-dir data/processed/shards_1913 \
    --cutoff 1913 \
    --check-contamination
```

The output directory gets a `manifest.json` with per-shard doc/char counts and
per-source distributions. Use `--input <files...>` instead of `--data-dir` to
target specific JSONL files, `--max-tokens` to cap corpus size, or `--no-sample`
to disable per-source downsampling.

### 3. Train on a single RTX 3090

```bash
# Point training at the historical shards directly — no base_data/ wrapper needed.
export NANOCHAT_PARQUET_DIR="$(pwd)/data/processed/shards_1913"

# Base pretraining (defaults: d16, T=1024, activation ckpt + chunked loss on)
cd nanochat
bash historical_3090_base.sh

# Midtraining (structured tasks mixture)
MODEL_TAG=d16_3090 bash historical_3090_mid.sh

# Evaluation (CORE metric)
MODEL_TAG=d16_3090 bash historical_3090_eval.sh
```

See [`docs/TRAINING_3090.md`](docs/TRAINING_3090.md) for the full knob reference,
benchmark methodology, and recommended 1-week vs 2-week presets.

### Original 8xH100 path

```bash
# Legacy path: FineWeb auto-download + speedrun. Still works when NANOCHAT_PARQUET_DIR is unset.
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
| RTX 3090 (24 GB) | **d16, T=1024, ckpt+chunked** (recommended) | ~1–2 weeks | Electricity |
| RTX 3090 (24 GB) | d20, T=2048 (tight, not recommended) | ~2-3 weeks | Electricity |
| 8xH100 (Lambda) | d20-d26 | 4-12 hours | ~$100-300 |
| 8xH100 (Lambda) | d34 | ~40 hours | ~$1000+ |

**3090 notes**: pass `--activation_checkpoint --chunked_loss --max_seq_len=1024`
(the 3090 scripts do this automatically). If you OOM, drop `--device_batch_size`
to 2, or set `KV_HEAD_RATIO=0.5` to enable GQA. See `docs/TRAINING_3090.md`.

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

Contributions welcome! Open an issue or submit a pull request.
