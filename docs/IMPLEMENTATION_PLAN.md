# Historical Nanochat Implementation Plan

## Project Goal
Build a "vintage" or "time-locked" language model using Karpathy's nanochat pipeline, trained exclusively on texts from before specific historical cutoff dates (1850-1900, pre-1913, 1914-1950). The model should genuinely not know about events after its cutoff - creating authentic historical worldview simulation for research purposes.

## Background Research Summary

### Nanochat Overview
- Full ChatGPT training pipeline: Tokenizer → Pretraining → Midtraining → SFT → RL
- ~$100 on 8xH100 for 4 hours (560M params, depth=20)
- ~$300 for d26 model, ~$2500 for d34 (2.2B params)
- Single RTX 4090: ~13 days for full pipeline
- Uses FineWeb-Edu dataset (shuffled into 1822 shards)
- Key files: `speedrun.sh`, `nanochat/dataset.py`, `nanochat/model.py`

### Ranke-4B Reference (Zurich Team)
- 4B params trained on 80B tokens from pre-cutoff texts
- Cutoffs: 1913, 1929, 1933, 1939, 1946
- Uses Qwen3 architecture (not nanochat's GPT-style)
- Key insight: "uncontaminated bootstrapping" - no modern knowledge leakage
- Not yet publicly available (pre-release)

### Key Difference: Options 1 vs 2
- **Option 1 (Our approach)**: Train from scratch on historical data only. Model genuinely doesn't know WWI because it never saw text mentioning it.
- **Option 2 (Rejected)**: Fine-tune modern model. Faster but modern knowledge "baked in" - just a veneer of historical roleplay.

---

## Phase 1: Environment Setup & Nanochat Understanding

### 1.1 Clone and Explore Nanochat
```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

### 1.2 Key Files to Study
| File | Purpose |
|------|---------|
| `speedrun.sh` | Full training pipeline script |
| `nanochat/dataset.py` | Data loading, shard management |
| `nanochat/tokenizer.py` | BPE tokenizer (Rust-based) |
| `nanochat/model.py` | Transformer architecture |
| `nanochat/engine.py` | Distributed training engine |
| `scripts/train_*.py` | Training scripts per stage |
| `dev/gen_synthetic_data.py` | Custom data generation |
| `dev/repackage_data_reference.py` | Data repackaging example |

### 1.3 Understand the Pipeline
1. **Tokenizer training**: BPE on text corpus → 65,536 vocab
2. **Pretraining**: Base model on large corpus (~11B tokens for d20)
3. **Midtraining**: Conversation format, SmolTalk/MMLU/GSM8K
4. **SFT**: Cherry-picked high-quality conversations
5. **RL (optional)**: GRPO on math problems

---

## Phase 2: Historical Corpus Assembly

### 2.1 Primary Data Sources (Freely Accessible)

| Source | Size | Date Range | Access |
|--------|------|------------|--------|
| **Project Gutenberg** | ~3B tokens, 50K+ books | Pre-1924 | HuggingFace: `manu/project_gutenberg` |
| **EEBO-TCP** | 60K+ works | 1475-1700 | textcreationpartnership.org (free since 2020) |
| **Old Bailey Corpus** | 127M words | 1674-1913 | CLARIN-D download |
| **Caselaw Access Project** | 6.7M cases | 1658-2020 | HuggingFace: `free-law/Caselaw_Access_Project` |
| **Chronicling America** | Newspapers | 1756-1963 | LOC API, bulk OCR |

### 2.2 Secondary Sources (Require Application)

| Source | Size | Access |
|--------|------|--------|
| **HathiTrust** | 6.6M volumes, 5.4TB | Request via institution |
| **COHA** | 475M words, 1820s-2010s | Access restrictions |

### 2.3 Data Collection Scripts to Build

```
data/
├── download/
│   ├── gutenberg_download.py      # Filter by pub date
│   ├── eebo_download.py           # 1475-1700 texts
│   ├── oldbailey_download.py      # Trial proceedings
│   ├── caselaw_download.py        # US court cases
│   └── chronicling_download.py    # Newspaper OCR
├── process/
│   ├── date_filter.py             # Enforce temporal cutoffs
│   ├── ocr_clean.py               # Fix OCR errors
│   ├── dedup.py                   # Remove duplicates
│   └── contamination_check.py     # Detect anachronisms
└── stats/
    └── corpus_analysis.py         # Size, date distribution
```

### 2.4 Temporal Cutoff Strategy

For each target period, apply strict filtering:

| Cutoff | Include | Exclude |
|--------|---------|---------|
| **1850** | Pre-1850 texts only | Any reference to post-1850 events |
| **1900** | Pre-1900 texts only | Edwardian content, WWI, etc. |
| **1913** | Pre-WWI texts | Any WWI knowledge |
| **1950** | Pre-Cold War | Post-WWII events |

### 2.5 Contamination Prevention
- **Publication date filter**: Only texts published before cutoff
- **Content scan**: Search for known post-cutoff entities (WWI, Hitler, atomic bomb, etc.)
- **Modern edition detection**: Exclude reprints with modern introductions
- **OCR timestamp check**: Ensure OCR'd books are originals, not modern reprints

---

## Phase 3: Data Processing Pipeline

### 3.1 Download and Assemble
```bash
# Estimated data collection
# Gutenberg: ~15GB compressed
# EEBO: ~20GB XML
# Old Bailey: ~2GB
# Caselaw: ~50GB+ (filter to pre-cutoff)
# Chronicling America: Variable (select by date)
```

### 3.2 Processing Steps
1. **Normalize formats**: Convert XML/HTML/TEI to plain text
2. **Date extraction**: Parse publication dates from metadata
3. **Temporal filtering**: Strict cutoff enforcement
4. **OCR cleanup**: Pattern-based error correction
5. **Deduplication**: Fuzzy matching to remove overlaps
6. **Contamination audit**: Scan for anachronistic terms
7. **Quality filtering**: Remove garbage/corrupted texts

### 3.3 Sharding for Nanochat
- Target: ~250M characters per shard (nanochat format)
- Shuffle across sources to avoid domain clustering
- Save as parquet files matching nanochat's expected format

---

## Phase 4: Tokenizer Training

### 4.1 Historical Tokenizer
Train a new BPE tokenizer on the historical corpus:
- Captures period-appropriate vocabulary
- Handles archaic spellings (e.g., "honour", "colour", long-s)
- OCR error resilience

### 4.2 Vocab Considerations
- 65,536 vocab size (match nanochat default)
- May need special handling for:
  - 18th-19th century typography
  - Legal/scientific terminology
  - Newspaper column formatting

---

## Phase 5: Model Training

### 5.1 Hardware Options

| Option | Config | Time | Cost |
|--------|--------|------|------|
| **RTX 3090 (local)** | d20, reduced batch | ~2-3 weeks | Electricity only |
| **Lambda 8xH100** | d20-d26 | 4-12 hours | ~$100-300 |
| **Lambda 8xH100** | d34 | ~40 hours | ~$1000+ |

### 5.2 Training Configuration
```python
# Historical nanochat config
model_depth = 20  # Start conservative
tokens = 11_000_000_000  # ~11B tokens (Chinchilla optimal)
batch_size = 512  # Adjust for hardware
learning_rate = 3e-4  # Muon optimizer default
```

### 5.3 Training Stages (Modified for Historical)

1. **Pretraining**: Use historical corpus shards
2. **Midtraining**:
   - Generate synthetic historical conversations
   - Period-appropriate Q&A (no modern knowledge)
   - Historical reasoning tasks
3. **SFT**:
   - Curated period-accurate dialogues
   - Identity: "I am an assistant with knowledge up to [year]"
4. **RL (optional)**: May skip for research model

### 5.4 Multiple Cutoff Training
Train separate models for each cutoff:
- `nanochat-1850`: Victorian/pre-industrial
- `nanochat-1900`: Edwardian optimism
- `nanochat-1913`: Eve of WWI
- `nanochat-1950`: Post-WWII but pre-Cold War

---

## Phase 6: Evaluation & Analysis

### 6.1 Temporal Integrity Tests
- **Knowledge probes**: "Who won World War I?" (should fail for pre-1913)
- **Anachronism detection**: Ask about future events
- **Period beliefs**: Ask about eugenics, imperialism, progress (expect period views)

### 6.2 Qualitative Evaluation
- Generate essays on period topics
- Historical roleplay conversations
- Compare worldview across cutoffs

### 6.3 Research Questions to Explore
1. How does expressed optimism change across cutoffs?
2. What beliefs shift between 1900→1913→1950?
3. Can the model "predict" future events from its knowledge?
4. How do scientific explanations differ?

---

## Phase 7: Experiments to Run

### 7.1 Comparative Studies
- Same prompt, different cutoffs → How answers change
- Period-appropriate vs modern phrasing detection
- Historical forecasting accuracy

### 7.2 Creative Applications
- Generate "period fiction"
- Simulate historical debates
- Create educational dialogues from historical perspective

### 7.3 Research Applications
- Study linguistic change over time
- Explore evolution of ideas/beliefs
- Counterfactual history exploration

---

## Documentation Strategy

Since this is an exploratory/learning project with potential for publication/sharing, maintain comprehensive documentation:

### Technical Documentation
- `docs/pipeline/` - Step-by-step reproduction guides
- `docs/data/` - Corpus assembly methodology, statistics
- `docs/training/` - Hyperparameters, training logs, decisions

### Research Documentation
- `docs/findings/` - Observations about historical worldviews
- `docs/evaluation/` - Test results, qualitative analysis
- `notebooks/` - Jupyter notebooks for experiments and visualizations

### Code Documentation
- Clear docstrings in all scripts
- README files in each directory
- Inline comments for non-obvious decisions

---

## Implementation Roadmap

### Week 1-2: Foundation
- [ ] Clone nanochat, run on sample data to understand pipeline
- [ ] Set up data download scripts for all sources
- [ ] Begin Project Gutenberg download with date filtering

### Week 3-4: Data Assembly
- [ ] Complete downloads from all primary sources
- [ ] Build date filtering and contamination detection
- [ ] Process and clean all datasets
- [ ] Analyze corpus statistics (size, date distribution, domains)

### Week 5-6: Data Sharding
- [ ] Convert to nanochat shard format
- [ ] Create temporal subsets for each cutoff
- [ ] Verify contamination-free status

### Week 7-8: Tokenizer & Initial Training
- [ ] Train historical tokenizer
- [ ] Run small-scale training test (local GPU)
- [ ] Validate training is working

### Week 9-10: Full Training
- [ ] Rent cloud GPU (Lambda/etc.)
- [ ] Train d20 model on 1913 cutoff (first target)
- [ ] Evaluate and iterate

### Week 11+: Expansion
- [ ] Train additional cutoff models
- [ ] Develop evaluation suite
- [ ] Document findings
- [ ] Consider publishing/sharing

---

## Key References

### Nanochat
- Repository: https://github.com/karpathy/nanochat
- Discussion #1: "Introducing nanochat"
- Discussion #139: Customizing with identity/knowledge

### Historical LLMs
- Ranke-4B: https://github.com/DGoettlich/history-llms
- "Vintage LLMs": https://owainevans.github.io/talk-transcript.html
- PNAS paper: "Historical LLMs for behavioral science"

### Data Sources
- Project Gutenberg: https://huggingface.co/datasets/manu/project_gutenberg
- EEBO-TCP: https://textcreationpartnership.org/
- Old Bailey: https://www.oldbaileyonline.org/
- Chronicling America: https://chroniclingamerica.loc.gov/about/api/
- Caselaw Access: https://case.law/

---

## Questions for Future Consideration

1. **Tokenizer reuse**: Can we use nanochat's modern tokenizer on historical text, or is historical tokenizer critical?
2. **Model size tradeoffs**: Is d20 sufficient for historical knowledge, or need larger?
3. **Synthetic data generation**: How to generate period-appropriate SFT data without contamination?
4. **Multilingual**: Victorian/Edwardian texts often include French, Latin, Greek - handle or filter?
5. **Domain balance**: How to weight books vs newspapers vs legal documents?
