# Historical Nanochat

Build experimental "time-locked" language models using Karpathy's nanochat pipeline and corpora filtered toward specific historical cutoff dates. The resulting models are intended to approximate historical information states, not to guarantee ignorance of later events.

## Project Overview

This project extends [nanochat](https://github.com/karpathy/nanochat) to train language models from scratch on historical text corpora. Sources are selected with publication-date metadata and contamination filters to reduce post-cutoff exposure, but effective temporal ignorance must be measured. Annotations, reprints, OCR metadata, semantic anachronisms, and memorized overlap remain documented residual contamination risks.

**Load-bearing claim:** these controls reduce known post-cutoff exposure; they do
not establish that a trained model is temporally ignorant. Later editorial
annotations, reprints, OCR metadata, semantic anachronisms, memorized overlap,
and incorrect or missing source metadata can still introduce later text.

### What the cutoff controls establish

- Strict Gutenberg acquisition admits records only with publication/issue metadata
  or an explicit publication/printing phrase at or before the cutoff.
- Acquisition commands report requested, fetched, and written counts;
  zero-record acquisitions fail instead of looking complete.
- The contamination checker flags explicit post-cutoff years (after narrow
  currency/page-number exclusions), anachronistic terms, URLs, emails, and
  modern references in document text and metadata.
- Checked shard packaging fails if the checker is unavailable, an input is
  missing, no records/shards are produced, or not every packaged record was
  examined. Its manifest records the checker version and SHA-256 plus the
  examined-record count. Runs without the check are explicitly `UNCHECKED`.

These are selection, refusal, and attestation controls—not proof over the
meaning of every training token. Any stronger temporal-ignorance claim requires
published corpus/artifact hashes, audit coverage, leakage probes, known
false-negative classes, and uncertainty bounds for the evaluated model.

### Key Features

- **Temporal cutoffs**: Pre-1850, Pre-1900, Pre-1913 (WWI), Pre-1950
- **Multiple data sources**: Project Gutenberg, Old Bailey, Chronicling America, Caselaw Access Project
- **Contamination detection**: Automated detection of anachronistic content
- **Nanochat-compatible**: Produces shards in the exact format nanochat expects

## Network and external-data prerequisites

A fresh clone includes the `data` download/processing package and the validated
`tokenizer/tokenizer.pkl`, `tokenizer/token_bytes.npy`, and
`tokenizer/tokenizer_manifest.json` bundle. This checkout does not include the historical corpus,
generated shards/token caches, or trained checkpoint weights, and it
cannot acquire a corpus offline unless you stage the source data separately.

- Installation fetches the root dependencies and `hatchling` build backend from
  PyPI. The nested nanochat environment also fetches its locked dependencies and
  PyTorch wheel from the index selected in `nanochat/pyproject.toml`. For an
  offline install, pre-populate both package caches and use the installers'
  offline modes.
- Gutenberg acquisition reads Hugging Face dataset `manu/project_gutenberg`.
- Chronicling America acquisition uses the Library of Congress API and OCR
  endpoints.
- Old Bailey is not downloaded by this repository: obtain the XML corpus from
  CLARIN-D, then pass its directory with `--corpus-dir`.
- Caselaw acquisition reads Hugging Face dataset
  `common-pile/caselaw_access_project`; the documented alternative is the
  `case.law` API.

## Installation

```bash
# Clone this repository
git clone https://github.com/AshitaOrbis/historical-nanochat.git
cd historical-nanochat

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python3 -m pip install -e .

# Or with uv (faster)
uv pip install -e .

# Install the nested nanochat training environment (choose cpu or gpu).
# Keep the root environment active for the data commands below.
(cd nanochat && uv sync --extra gpu)
```

## Quick Start

### 1. Acquire Historical Data

```bash
# Download Project Gutenberg with a strict 1913 publication-date cutoff
python3 -m data.download.gutenberg_download --cutoff 1913 --max-docs 1000

# Old Bailey is a local-corpus processor, not a downloader. First acquire the
# XML corpus from CLARIN-D, then supply its directory explicitly.
python3 -m data.download.oldbailey_download --corpus-dir /path/to/old-bailey-xml --cutoff 1913

# Download historical newspapers
python3 -m data.download.chronicling_download --cutoff 1913 --max-pages 500

# Download historical case law
python3 -m data.download.caselaw_download --cutoff 1913 --max-cases 500
```

All acquisition entry points report records requested, fetched, and written and
exit non-zero when they write zero records. Gutenberg strict mode admits
only an actual publication/issue field or explicit publication/printing phrase;
`--no-strict` is exploratory and stamps its records and stats as non-strict.

### 2. Package into Shards

```bash
# Streaming packager with bounded-memory shuffle + per-shard manifest
python3 -m data.process.shard_packager \
    --data-dir data/raw \
    --output-dir data/processed/shards_1913 \
    --cutoff 1913 \
    --check-contamination
```

The output directory gets a `manifest.json` with per-shard doc/char counts and
per-source distributions. A checked artifact records the checker module,
version, SHA-256, and examined-record count at both manifest and shard level.
Omitting `--check-contamination` explicitly stamps the artifact `UNCHECKED`;
it cannot be represented as checked downstream. Missing inputs, an unavailable
checker, zero examined records, or zero output shards are hard failures. Use
`--input <files...>` instead of `--data-dir` to target specific JSONL files,
`--max-tokens` to cap corpus size, or `--no-sample` to disable per-source
downsampling.

### 3. Verify artifacts and train on a single RTX 3090

```bash
# The repository includes tokenizer/tokenizer.pkl, tokenizer/token_bytes.npy,
# and tokenizer/tokenizer_manifest.json. The trainer verifies their hashes,
# dtype, shape, vocabulary, and BOS identity before allocating the model.

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

| Cutoff | Model Name | Target excluded knowledge (must be evaluated) |
|--------|-----------|-----------------------------------------------|
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
└── docs/                  # Documentation
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

## Security

**Read [`SECURITY.md`](SECURITY.md) before running this on anything you don't
fully control.** This is a single-user research pipeline built on upstream
nanochat. It assumes you trust the model-generated code it executes, every
checkpoint and tokenizer artifact you load, and the local filesystem it operates
on. In particular:

- The HumanEval `execute_code` path (`nanochat/nanochat/execution.py`) is **not a
  security sandbox** — untrusted model-generated code can achieve host code
  execution. Run it inside a real container/microVM if the model is untrusted.
- Loading checkpoints (`torch.load`) or `tokenizer.pkl` (`pickle.load`) from an
  untrusted source is **arbitrary code execution**. Only load artifacts you
  produced or trust; prefer `safetensors` / `tokenizer.json`.

`SECURITY.md` documents each issue, its disposition, and the operational
mitigations honestly — these are inherent properties of the design, not bugs a
code patch can remove without changing what the tool is.

`nanochat/scripts/chat_web.py` is unauthenticated. Its cross-origin grant is
limited to exact `127.0.0.1` and `localhost` origins on the configured port, but
that is not authentication: **do not expose this service on a LAN, public
interface, tunnel, or shared host.**

## License

MIT License (same as nanochat)

## Contributing

Contributions welcome! Open an issue or submit a pull request.
