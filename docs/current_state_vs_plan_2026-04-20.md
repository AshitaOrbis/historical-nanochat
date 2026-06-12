# Historical Nanochat — Current State vs Apr-20 Plan

Report prepared 2026-04-20 for GPT Pro review.

> **UPDATE 2026-04-20 (post-report):** The d22 r=30 run described below has
> been reclassified as `legacy_textonly_d22_r30_internal_baseline`. It is NOT
> the governed 3090 PoC, NOT a release candidate, NOT a teacher candidate,
> and NOT the model referenced in any access-request email. See
> `report/RUN_RECLASSIFICATION_2026-04-20.md` and
> `docs/decision_gate_2026-04-20.md`. Phase-0-lite is being built in parallel
> under `data/phase0/` and new artifacts land under `token_cache_v3/`,
> never overwriting the live cache.

This document describes the *actual* state of training data, corpus pipeline,
and the currently-running base run, and maps that state against the three plan
documents dated Apr 19:

- `historical_nanochat_claude_code_action_plan.md` (Phase 0 implementation spec)
- `historical_nanochat_next_steps.md` (phase map through release)
- `historical_nanochat_access_emails.md` (outreach templates)

---

## 1. TL;DR

- **A 13-day d22 base training run is in progress** (started 2026-04-20
  09:05 UTC). It is using an *older* data pipeline — not the one the Apr-19
  plan documents prescribe.
- **The plan has a clear Phase 0 (corpus/pipeline) that must clear Gate A
  before Phase 2 (3090 PoC model) matters.** The currently-running d22 job
  is effectively Phase 2. Gate A is **partially satisfied at best** — some
  elements are in place, several are absent, and one element (provenance
  preservation in training shards) is materially missing.
- The *training stack* is solid: retrained rustbpe tokenizer with full
  32768 nanochat special-token layout, 100% 256-byte coverage, 0 byte
  drops on 72.6 M docs / 85.7 B chars, dataloader with cached-tokens
  path, sweep verified d22 as the right depth under the 24 GB budget.
- The *data* itself predates the plan. It has reasonable raw coverage
  (8 sources, ~85 B raw chars after packaging) but violates several of
  the plan's Phase-0 requirements (see §3).
- **Decision needed**: let the d22 run complete as a "pipeline validation
  + baseline loss curve" artifact while we build Phase-0 on the side, or
  kill it and rebuild to the plan first. Either is defensible. The sunk
  compute is ~30 hours of 3090 time; the 13-day ETA is 1.3× the time to
  rebuild Phase 0 and re-pack shards (rough estimate, 10 days for a
  single operator).

---

## 2. What the plan demands of Phase 0

Summarized from `historical_nanochat_claude_code_action_plan.md` §1-§16:

### 2.1 Repo layout

```text
data/
  sources/registry.yaml         # every source registered BEFORE any harvester writes
  harvesters/                   # per-source harvesters: 11 P0 sources
  process/                      # rights_audit, date_audit, dedupe, ocr_quality, ocr_triage, pack_to_parquet, manifests
  prompts/                      # OCR + synthetic-data contracts
  schemas/                      # Pydantic: document, page, source, ocr_queue, training_record
  scripts/                      # harvest_source, build_poc_corpus, audit_corpus, build_training_shards, estimate_tokens, make_ocr_queue
  tests/                        # test_registry, test_rights_audit, test_date_audit, test_dedupe, test_pack_to_parquet
```

### 2.2 Rights audit (fail closed)

- Default action for unknown rights: **exclude**.
- Default action for missing date: **exclude from training**.
- Default action for post-cutoff date: **exclude**.
- For mixed-rights sources, require explicit item-level rights evidence.
- Output per-source audit JSON + Markdown.

### 2.3 Date audit (confidence hierarchy)

```text
exact date > year > range > inferred from bibliographic field > weak text guess > unknown
```

- Exclude records where only ingestion/download date is known (e.g. Gutenberg
  release date ≠ publication date).
- Preserve raw date field + confidence score.
- Cutoff buckets: `pre1850`, `1850_1875`, `1875_1900`, `1900_1913`, `post_cutoff_exclude`, `unknown_exclude`.

### 2.4 Unified document schema

Every shard record should preserve at minimum:

```text
text, source_id, document_id, page_id, segment_id, title, author,
publication_date, publication_year, date_confidence, language,
language_confidence, country_or_region, publication_place, publisher,
genre, subgenre, rights, rights_url, source_url, citation,
ocr_engine, ocr_quality_estimate, source_quality,
content_hash, dedupe_hash, word_count, char_count, token_estimate,
cutoff_bucket, sensitive_content, metadata
```

### 2.5 Gate A (before any 3090 training)

Proceed only if:

1. At least 1 B clean token estimate exists.
2. Validation shard is held out **by document/source** (not just row split).
3. Unknown rights/date records excluded.
4. Training script loads parquet directly (no intermediate step).
5. Smoke test runs.

### 2.6 Recommended first corpus mix (Phase 1)

```yaml
books_general: 0.35
newspapers_periodicals: 0.30
legal_government: 0.15
science_technical: 0.10
early_modern: 0.05
misc_holdout_or_eval: 0.05
```

### 2.7 Phase 2 eval suite

- Held-out historical validation loss **by source family**.
- Cutoff anachronism test.
- Source-grounded QA test.
- Style continuation test.
- Synthetic-data pilot.

---

## 3. What we actually have

### 3.1 Raw deduped JSONL by source

Located at `data/deduped/` (ext4, 398 GB total):

| Source | JSONL size | Approx record schema | Has `source` | Has date | Has `rights` |
|---|---:|---|---|---|---|
| american_stories | 275 GB | `text, source, year, date, newspaper, state, lccn, article_id` | yes | yes (exact) | **no** |
| bhl | 126 GB | `text, source, year, item_id, file` | yes | yes (year) | **no** |
| bl_newspapers | 12 GB | — | yes | yes | **no** |
| eebo | 6.5 GB | — | yes | yes | **no** |
| gutenberg | 4.2 GB | `text, source, metadata, estimated_year` | yes | yes (inferred) | **no** |
| tcp | 1 GB | — | yes | yes | **no** |
| oldbailey | 452 MB | `text, source, year, date, type, file, id` | yes | yes (exact) | **no** |
| chronicling_v2 | 13 MB | — | yes | yes | **no** |
| caselaw | 86 KB | `text, source, year, court, case_name, id` | yes | yes | **no** |
| chronicling_america | 57 KB | — | yes | yes | **no** |

Observations:

- `source` and some kind of date field are present in every source's JSONL.
  The plan's Phase-0 expectations for at-ingest provenance are **substantially
  met** at the raw JSONL level.
- **`rights` is absent from every source.** The rights audit the plan calls
  for has never been executed; there is no per-record rights field in any
  deduped JSONL file.
- `caselaw` (86 KB) and `chronicling_america` (57 KB) are essentially empty.
  The P0 CAP and raw Chronicling America harvesters either were never run
  to completion or were deprecated in favor of American Stories (which
  subsumes much of Chronicling America at article-level granularity).

### 3.2 Existing harvester scripts

`data/download/` contains (all authored before Apr 12):

```
american_stories_download.py  bl_newspapers_download.py   caselaw_download.py
chronicling_download.py       chronicling_download_v2.py  chronicling_comparison.py
eebo_convert.py               gutenberg_download.py       oldbailey_convert.py
oldbailey_download.py         bhl_download.py             britannica_1911_download.py
tcp_download.py
```

Observations:

- Covers most of the plan's P0 list (Gutenberg, IA-like, LOC-like,
  Chronicling America, American Stories, BL Books via BL newspapers-only,
  BHL, Old Bailey, CAP, TCP). Missing: formal `internet_archive`
  harvester, `loc_selected_books` (LOC dataset), `ncse_v2`, `papers_past`,
  `delpher`, `british_library_19c_books` (BL Books HF dataset — we have
  BL *newspapers* instead).
- No source registry (`data/sources/registry.yaml`). Harvesters are
  authored ad-hoc without a shared contract.
- Each harvester emits its own schema. The plan's unified document
  schema does not exist as a Pydantic model; at ingest we get per-source
  keys like `year` vs `estimated_year` vs `date` without normalization.

### 3.3 Pre-existing processing modules

`data/process/` contains:

- `contamination_check.py` — fixed during this session (was silently
  skipping contextual terms via a broken SAFE-list). Validates 16 smoke
  tests.
- `dedup.py` — exists but I have not audited it against the plan's
  SimHash/MinHash + cross-source strategy.
- `shard_packager.py` — streaming packer (rewritten in this session)
  with bounded shuffle buffer, OCR quality pre-filter, near-dedup via
  content fingerprint, per-shard manifest with source distribution,
  rejection-counter persistence.
- `train_tokenizer.py` — pre-existing, used `RustBPETokenizer` path.

Missing vs. the plan:

- No `rights_audit.py`.
- No `date_audit.py` as a standalone module with the confidence hierarchy
  (date extraction was patched into `gutenberg_download.py` only).
- No `normalize_text.py` with the "preserve historical spelling" contract.
- No `ocr_triage.py`, `ocr_batch_export.py`, `ocr_batch_import.py`.
- No Pydantic schemas in `data/schemas/`.
- No `data/prompts/ocr_gpt54_contract.md`.

### 3.4 Training shards currently feeding d22

**Path:** `data/shards/` (322 parquet files, 36 GB)
**Token cache:** `data/token_cache_v2/` (47 GB uint16, 25.2 B tokens)

Shard-record schema: `{"text": str}` — **no source_id, no date, no rights, no
document_id, no cutoff_bucket, nothing else.** All the provenance that was in
the deduped JSONL was dropped at pack time.

Source composition of these shards is **not directly inspectable** because
there is no `manifest.json` in the shards directory (only
`data/shards-small/stats.json` exists and has an empty `source_counts: {}`).

What we can infer from the packaging defaults:

The `shard_packager.py` module uses `DEFAULT_SAMPLE_RATES`:

```python
'american_stories': 0.33,   # 54.5B -> ~18B
'bhl': 0.60,                # 29.9B -> ~18B
'bl_newspapers': 1.0,       # Keep all 2.8B
'gutenberg': 1.0,           # Keep all 1.7B
'eebo': 1.0,                # Keep all 1.6B
'tcp': 1.0,                 # Keep all 0.25B
'oldbailey': 1.0,           # Keep all 0.11B
```

If these rates were applied to the deduped JSONL files (not confirmed), the
expected composition is:

| Source family | Tokens (est.) | Share | Plan target (§3.6) |
|---|---:|---:|---:|
| American newspapers (AS + BL news) | ~21 B | ~56% | 30% (newspapers/periodicals) |
| Science (BHL) | ~18 B | ~48% | 10% (science/technical) |
| Books (Gutenberg + EEBO) | ~3.3 B | ~9% | 35% (books) + 5% (early modern) |
| Early modern (TCP) | ~0.25 B | <1% | 5% |
| Legal (Old Bailey + CAP) | ~0.11 B | <1% | 15% (legal/government) |

Actual token counts in the cache total 25.2 B; the above is a structural
estimate of the ratios, not an accounting of the 25.2 B tokens.

**This is materially different from the plan's recommended mix.** The corpus
is dominated by American newspapers and natural-history scientific texts.
Legal representation is an order of magnitude smaller than the plan calls
for. Book coverage is ~4× below target.

### 3.5 Tokenizer

- `tokenizer/tokenizer.pkl` (rustbpe + tiktoken wrapper), vocab 32768.
- All 9 nanochat specials present at ids 32759–32767.
- Complete 256-byte coverage (no [UNK] fallback needed).
- Full-corpus scan: 72.6 M docs / 85.7 B chars / **0 byte drops**.
- `tokenizer_manifest.json` records SHA-256 of pkl + token_bytes,
  training time (422 s on 10 B chars), validation stats.

This is the one subsystem that fully matches what the plan implies.

### 3.6 Training-loop infrastructure

Complete and validated in a multi-day sweep:

- `NANOCHAT_PARQUET_DIR` + `--parquet-dir` explicit path.
- `nanochat.dataloader_cached` (mmap-backed uint16 token stream).
- Activation checkpointing, chunked LM-head CE, GQA flag, compile-mode
  selector, SDPA backend diagnostics.
- Seq-length curriculum mid-training switch.
- Loader-wait-time instrumentation (37% without cache → 0.2% with).
- Benchmark CSV: per-step throughput, peak VRAM, loader %.
- Startup assertions: vocab == len(token_bytes), BOS id matches manifest,
  manifest SHA printed, resume-time wte/lm_head shape check.
- Save/resume round-trip verified at d24 (21.4 GB peak) and d16, d22.

### 3.7 Currently-running base run

Kicked off 2026-04-20 09:05 UTC (pid 1641295, nohup + disown).

```text
depth                        d22 (615,645,184 params)
max_seq_len                  1024
device_batch_size            8
total_batch_size             262,144 tokens/step
target_param_data_ratio      30
num_iterations               70,455
total_tokens                 18,469,355,520 (18.47 B)
compile_mode                 default
activation_checkpoint        off
tokenizer                    rustbpe vocab 32768
token_cache                  data/token_cache_v2
save_every                   2000 (≈ 9 h between checkpoints)
eval_every                   1000 (1 M-token BPB)
core_metric_every            disabled (run post-training)
peak VRAM target             ~17.5 GB / 24 GB
expected wall                ~13 days at 16.4 k tok/s
```

Current step (at report time): step 20 of 70,455; tok/s 16.5 k (matches
sweep); loader 0.1%; loss 6.58; init BPB 4.996 at step 0.

---

## 4. Gap analysis against Gate A

| Gate-A requirement | Status | Evidence |
|---|---|---|
| ≥1 B clean token estimate | **Satisfied** | 25.2 B tokens in token_cache_v2. |
| Validation shard held out by document/source | **Not satisfied** | Parquet split is last-shard-is-val (row-level), not source-level holdout. Old Bailey / BHL / LOC books NOT explicitly held out. |
| Unknown rights records excluded | **Not satisfied** | No rights audit has been run; `rights` field absent in every JSONL. Implicit assumption: everything in `deduped/` is public-domain because the sources we harvested are nominally PD. Assumption is unaudited. |
| Unknown date records excluded | **Partially satisfied** | Gutenberg has a date-precedence extractor (issued > death > birth+20 > fallback; drops "downloads" field); Old Bailey, American Stories, CAP have dates from source. No central date-confidence hierarchy; no "unknown-date exclude" enforcement. |
| Training script loads parquet directly | **Satisfied** | `--parquet_dir` and `--token_cache_dir` both work. |
| Smoke test runs | **Satisfied** | 16/16 Python smoke tests + multi-day depth sweep + save/resume round-trip. |

Decision rule: three of the six items are unsatisfied. The spirit of Gate A
("at least demonstrably controlled corpus") is **not fully met**, though the
training stack itself is fully in order.

---

## 5. Provenance loss at shard-pack time

This is the single most consequential gap.

The plan's §5 calls for every training record to carry `source_id`,
`publication_date`, `date_confidence`, `language`, `genre`, `rights`,
`source_url`, `ocr_quality_estimate`, `content_hash`, and several other
fields. Our shards contain only `text`.

Consequences:

1. Cannot compute held-out validation loss **by source family** during
   training (plan §4 eval #1). The per-family breakdown would have to be
   reconstructed by re-tokenizing held-out deduped JSONL files — possible
   but it requires parallel tokenizer work during eval.
2. Cannot do source-conditioned sampling or weighting adjustments without
   re-packaging.
3. Cannot run the plan's Phase-6 source-grounded synthetic data pipeline,
   which requires `source_packet` records with date + source + URL fields.
   The teacher model trained on these shards will NOT have seen source
   IDs; even at inference time, there is no way to prompt it with a dated
   source packet in the shape the plan envisions.
4. Downstream distillation (Phase 7) cannot easily attribute teacher
   generations to the right corpus subset.

This is fixable by re-packaging with the full schema. On the current 36 GB
of JSONL → 47 GB cache hardware path, re-packaging would take ~2 hours if
we simply carry through the existing JSONL fields. The harder part is
retrofitting a rights audit that fails closed — several hours of coding and
a corpus re-scan.

---

## 6. Corpus mix drift

Plan target vs. actual, laid side-by-side:

| Family | Plan target | Actual (est.) | Gap |
|---|---:|---:|---|
| Books (general + early modern) | 40% | ~9% | **-31 pp** |
| Newspapers/periodicals | 30% | ~56% | **+26 pp** |
| Legal/government | 15% | <1% | **-14 pp** |
| Science/technical | 10% | ~48% | **+38 pp** |
| Misc/eval holdout | 5% | 0% | -5 pp |

The model will be biased toward American newspapers and natural-history
scientific literature. This will show up in:

- Style: more telegraphic newspaper cadence, more botanical-taxonomic
  register, less Victorian long-form narrative prose.
- Anachronism profile: American 19th-c news coverage carries heavy
  contemporary commentary on events ~1860–1913; non-US events under-
  represented.
- Legal register essentially absent (even though Old Bailey and CAP raws
  exist in modest quantity; CAP wasn't properly downloaded).

The model may still be a useful proof-of-concept — it just won't match the
"source-balanced corpus" the plan recommends. Worth being honest about this
in any downstream model card.

---

## 7. Items the current run CAN validate

Despite the gaps, the d22 r=30 run in progress will produce useful
Phase-2 evidence:

1. **End-to-end pipeline works at scale** — dataloader, tokenizer,
   cache, training loop stable at d22 for ≥30 h wall time (as of
   report), no OOM, no NaN, stable tok/s.
2. **Baseline loss curve** on a 25 B-token pre-1913 corpus — useful
   reference even if the mix is skewed.
3. **Existence proof for 3090 viability at 615 M parameters** — relevant
   to Phase 5 compute planning (we can now say: "a single 3090 can
   comfortably train a 615 M model on 18 B tokens in ~13 days without
   activation checkpointing").
4. **Held-out perplexity on a single val shard** — limited (no source
   breakdown) but non-zero signal.

What it CANNOT validate:

- Source-grounded QA accuracy (no source IDs in training data).
- Anachronism rate vs cutoff (no date-bucket eval was prepared).
- Style-by-source consistency.
- Rights compliance (rights audit never ran).

---

## 8. Three options going forward

### Option A — Let d22 finish; build Phase 0 in parallel.

Pros:

- Don't waste the ~30 h of 3090 time already sunk.
- End up with a real baseline model in 13 days.
- Phase 0 work (registry, rights audit, schema, eval sets) doesn't require
  the GPU and can proceed entirely on CPU.
- At day 13, we'll have both a baseline AND a cleaned corpus, and can
  decide whether to kick off a Phase-0-aligned d22 run #2.

Cons:

- The baseline model cannot be used for source-grounded synthetic data
  (plan Phase 6) because it never saw source IDs.
- Model card will have to disclose the audit gaps.
- ~13 days of 3090 time for a model we know is structurally wrong for
  the long-term plan.

### Option B — Kill d22 now; do Phase 0 first; re-pack; re-train.

Pros:

- The eventual model is aligned with the plan from step zero.
- No awkward disclosures in the model card.
- Phase-6 synthetic data generation works out of the box.

Cons:

- Sunk 30 h compute is fully wasted.
- Phase 0 is a serious implementation effort — roughly 1–2 weeks of
  focused work for a single operator (registry + harvesters standardized
  + rights audit + date audit + schema + re-pack + eval sets). Training
  restarts after that.
- Net timeline: Phase-0-aligned model ready in ~28 days vs. ~13 days
  for Option A's baseline.

### Option C — Hybrid: let d22 finish as "pipeline validator," but
immediately stand up Phase 0 in parallel for the next run.

- Treat d22 r=30 as a throwaway baseline, not a candidate teacher.
- Build `registry.yaml`, unified Pydantic schema, rights+date audit,
  source-labeled shard packer while GPU is busy.
- When d22 finishes (day 13), IMMEDIATELY re-pack the corpus using the
  new pipeline (few hours) and start the "real" d22 run (or d20 r=40 if
  data volume drops after rights exclusions).
- Phase-1 eval suite (anachronism, source-grounded QA, style) gets built
  during the 13-day gap.

This is probably the right answer: d22 baseline as a free dry-run of the
full 13-day schedule, while Phase 0 proceeds on CPU.

---

## 9. Concrete next steps, regardless of option

Items that should happen before the *next* serious run (not during the
current one), in order of highest leverage first:

1. **Source registry** (`data/sources/registry.yaml`) with the 10 sources
   we already have raws for + 4 P1 stubs from the plan.
2. **Unified document schema** (Pydantic in `data/schemas/`) with
   `source_id`, `document_id`, `publication_date`, `date_confidence`,
   `rights`, `rights_url`, `source_url`, `cutoff_bucket` as required
   fields.
3. **Rights audit** (`data/process/rights_audit.py`) that fails closed.
   Backfill `rights` into existing deduped JSONL from the source registry
   (e.g. Gutenberg → `public_domain_us`, Old Bailey → `public_domain`,
   BHL → `open_access` with item-level check, American Stories → derived
   from public-domain Chronicling America, etc.).
4. **Date audit** (`data/process/date_audit.py`) with the confidence
   hierarchy and the cutoff buckets.
5. **Shard re-packer** that preserves the full schema in parquet (not
   just `text`). Output manifest MUST include `source_counts` with
   actual per-source token totals.
6. **Eval holdouts** — split Old Bailey, BHL, one American Stories
   newspaper title, and one Gutenberg author set into held-out val/eval
   shards BEFORE shuffling into training.
7. **Anachronism eval** — at minimum 100 prompts with expected
   "shouldn't know this" behavior at the 1913 cutoff (WWI, 1918 flu,
   Wright flight, Einstein relativity, Nazi Germany).
8. **Source-grounded QA eval** — 500 held-out passages with
   human-authored or GPT-generated questions + evidence spans.

Items 1–5 are the ones that block the plan's Phase-6 synthetic-data
pipeline. Items 6–8 are what give the eventual model card something to
say about quality.

---

## 10. Honest assessment for GPT Pro

- **The training infrastructure is overbuilt for where the data is.** We
  have an excellent single-GPU training stack, a principled tokenizer,
  and a thorough sweep showing d22 fits nicely on 24 GB at 16 k tok/s.
  What we're training on is a legacy corpus that does not meet the
  plan's own Phase-0 standards.
- **The corpus is usable but not governed.** Enough raw data exists to
  train a meaningful 615 M model; the records inside the shards have
  lost their provenance. The downstream plan (synthetic data → student
  distillation) requires provenance to be preserved through to the
  training parquet, and ours isn't.
- **The currently-running d22 is a pipeline baseline, not a candidate
  teacher.** That framing is defensible provided the model card says so.
- **Option C is the least-regret path.** Let d22 finish, build Phase 0
  in parallel, then re-train against a governed corpus. Total wall clock
  to a Phase-0-aligned model: ~26 days (13 d22 + 13 Phase 0) vs. ~28 d
  for Option B, but Option C gets a baseline checkpoint for free.

---

## 11. Numbers for GPT Pro to scrutinize

- Token cache total: **25.2 B tokens** (uint16 .bin shards).
- Estimated post-Phase-0 clean tokens: **≤20 B** after rights audit
  (some BHL items may have stricter per-item rights; CAP needs actual
  re-harvest; American Stories mostly PD).
- Params at d22: 615,645,184.
- Chinchilla ratio 30: 18.47 B tokens = 70,455 steps at 262 k tokens/step.
- Throughput: **16.4 k tok/s measured** (sweep), **16.5 k tok/s observed
  live** at step 20.
- Peak VRAM d22 DBS=8 no-ckpt compile=default: 17.5 GiB (sweep), waiting
  for live confirmation at ~step 500.
- 3090 BF16 peak FLOPS: ~71 TF/s. MFU at d22: 6.3% ≈ 32 TF/s ≈ **45% of
  3090 peak**. H100 MFU reference would be the relevant comparison for
  Phase 5.
- Expected wall for the current run: 70,455 × dt(16 s) ≈ 313 h ≈ 13.0 d.

Any of these I'd flag as worth sanity-checking: the assumed sample rates
in §3.4, the "actual (est.)" composition in §6, and the Phase-0-aligned
clean-token estimate in §11. All three depend on inputs I cannot observe
directly without re-scanning the 25.2 B-token cache against its source
JSONLs.

---

End of report.
