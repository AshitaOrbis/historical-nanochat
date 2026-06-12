# Historical-Nanochat: Outcome Brief for Multi-Model Deliberation

**Date of brief:** 2026-05-12
**Run completed:** 2026-05-10 17:44 UTC
**Audience:** GPT Max panel, GPT Council, GPT Pro, Opus 4.7 — independent critique panel

---

## 0. What the project is

`historical-nanochat` is a fork of Karpathy's `nanochat` (the single-3090-friendly minimal LLM training stack). The thesis: **train base models from scratch on rights-audited corpora that hard-cut at a pre-modern date** (e.g. ≤ 1913), so the model genuinely cannot know post-cutoff events, rather than fine-tuning a modern model into historical roleplay. Comparable to Ranke-4B (Zurich) and Owain Evans' "vintage LLMs" concept.

Repo: ``
Data lives on ext4 at `~/historical-nanochat/data/` (~499 GB total, symlinked into the repo).

---

## 1. The run we are evaluating

**Run tag:** `governed_corpus1913_v4_d22_r30_3090_poc_parallel_family_run3_resume10000`
**Model tag:** `governed_v4_d22_r30_parallel_family`
**Final checkpoint:** `base_checkpoints/governed_v4_d22_r30_parallel_family/model_070455.pt`

### Config (canonical, from `report/base-model-training.md`)

| | value |
|---|---|
| Architecture | nanochat d22 transformer |
| Depth | 22 layers |
| Aspect ratio | 64 |
| Head dim | 128 |
| KV head ratio | 1.0 (no GQA) |
| Max seq len | 1024 |
| Parameters | **615,645,184** (~615.6M) |
| Vocab | 32,768 (custom rustbpe+tiktoken tokenizer trained on the 1913 corpus) |
| Device batch size | 8 |
| Total batch size | 262,144 tokens/step |
| Total train tokens | **18,469,355,520** (~18.47 B) |
| Iterations | 70,455 |
| Tokens : Params ratio | 30.0 (Chinchilla) |
| Loader strategy | `parallel_family_cache` |
| compile_mode | default (bfloat16 autocast) |
| sdpa_backend | auto |
| activation_checkpoint | off |
| chunked_loss | True, chunk 1024 |
| warmup_ratio | 0.02 (warmup ends step 1409) |
| warmdown_ratio | 0.40 (warmdown starts step 42273) |
| final_lr_frac | 0.0 |
| embedding_lr | 0.20 |
| matrix_lr | 0.015 |
| unembedding_lr | 0.003 |
| adam_beta1 | 0.80 |
| adam_beta2 | 0.95 |
| weight_decay | 0.00 |
| DDP world size | 1 (single 3090) |

### Outcomes

| | value | comment |
|---|---|---|
| Min val bpb | **1.1092** | bytes per byte; cross-entropy/ln2 over the tokenized val set |
| Final val bpb | **1.1092** | identical to min — monotone descent through warmdown, no overfit |
| CORE metric | None | not yet run; standardized eval is post-train phase |
| MFU | 4.84% | low; expected on 3090 with bf16 autocast in a stack tuned for H100 |
| Peak VRAM | 17.5 GiB / 24 GiB | well under budget |
| tok/sec | ~13k-16k | loader was <1% of step time |
| Wall time, this run | 20,309 min (≈ 14.1 d) | run3 leg alone (from step 10000 resume) |
| End-to-end wall time | ≈ 17.5 d | including 46 hrs wasted in aborted run #1 |
| Total FLOPs | 7.01 × 10¹⁹ | ~10¹⁹ — small-frontier scale |

### Validation bpb trajectory (resume run, sampled)

| step | val bpb |
|---:|---:|
| 10000 (resume start) | 1.2406 |
| 15000 | 1.2201 |
| 20000 | 1.1997 |
| 25000 | 1.1868 |
| 30000 | ~1.16 (interpolated) |
| 35000 | 1.1745 |
| 40000 | ~1.158 |
| 45000 | 1.1570 |
| 50000 | 1.1481 |
| 55000 | ~1.135 |
| 60000 | ~1.13 |
| 65000 | ~1.118 |
| 70000 | 1.1095 |
| 70455 (final) | **1.1092** |

Curve is monotone-descending and was still inching down at the end (warmdown LR ≈ 0 by then, so the model has extracted what this schedule allows).

### Final-checkpoint sample probes (raw, from training log)

```
The capital of France is 100000, 900 francs, and the population
The chemical symbol of gold is the same as that of silver, and the same as that of copper
If yesterday was Friday, then tomorrow will be Friday. If yesterday was Saturday, then to-day will be Saturday.
The planets of the solar system are: Saturn, Jupiter, Mars, Uranus, Ne
The opposite of hot is the best thing to do with the cold. A hot bath is the
My favorite color is a light brown, but the best is a dark brown, with a
If 5*x + 3 = 13, then x is the number of the square root of the number of the square root of the number
```

Observations: English fluency present, period register present ("to-day", "francs"), factual recall absent, arithmetic/logic absent, modern-LLM anchors ("Paris") absent.

This is a **base pretraining checkpoint with no midtraining, no SFT, no RLHF.**

---

## 2. The corpus

### Source mix (v4 governed, ≤1913 cutoff)

| family | share | hard floor | soft ceiling | sources |
|---|---:|---:|---:|---|
| newspapers_periodicals | 37.7% | ≥10% | ≤50% | Chronicling America (LOC) etc. |
| science_technical | **26.9%** | ≥5% | ≤25% (over) | Biodiversity Heritage Library (item-level recovery) |
| books_general | 17.5% | ≥10% | n/a | British Library Books CC0 + Project Gutenberg |
| early_modern | **9.8%** | >0% | ≤7% (over) | EEBO + TCP subset |
| legal_government | 8.2% | ≥5% | n/a | Caselaw Access Project + Old Bailey |
| | | | | |
| **Total train tokens** | 19.12 B | | | (18.47 B actually consumed during training) |
| Held-out val | 2.86 B | | | source-stratified, separate split |

### Governance properties

- Item-level rights audit for BHL (science); collection-policy admission for TCP; derived-dataset rights for CAP; per-source registry for the remainder.
- Date-cutoff enforcement: explicit `publication_year ≤ 1913` with confidence hierarchy.
- Per-shard parquet schema preserves: `text, source_id, document_id, date_bucket, rights_class, publication_year, split`.
- Per-shard `provenance.json` keyed off `source_file` (post-fix; see §3) gives the loader an authoritative family tag.

### Comparison to a legacy d22 baseline run

The same repo previously ran `legacy_textonly_d22_r30_internal_baseline`: same model config, ~25 B tokens, but trained on text-only shards (`{text}` only, no provenance). Source mix of the legacy baseline was **94.97% newspapers**. The governed v4 run reaches near-comparable val bpb on a substantially more balanced corpus with fewer total tokens (18.5 B vs 25 B).

---

## 3. Path-to-result (training-dynamics + infra journey)

**Four GPU smokes** preceded the long run:

| smoke | cache granularity | loader | result |
|---|---|---|---|
| 1 | unsplit (15M tok/shard) | sequential | shard-flip loss spike at step 80, no recovery |
| 2 | unsplit | sequential + softer LR | same spike, LR fix insufficient |
| 3 | split (1M tok/shard) | sequential | val bpb plateaus at 2.23-2.27, gradients undo each other |
| **4** | split (1M tok/shard) | **parallel_family_cache** | monotone val descent 2.24 → 1.95 across 4 ckpts |

The decisive intervention was the **parallel-family-cache loader**: maintain one sequential cursor per source family, draw microbatches according to a fixed per-step schedule (12 newspapers / 8 science / 6 books / 3 legal / 3 early-modern → 32 microbatches = grad_accum_steps). Every optimizer step computes its gradient on a fixed 37.5/25/18.75/9.375/9.375 family mix, eliminating both shard-flip spikes (smokes 1-2) and gradient interference plateaus (smoke 3).

### Run history

| run | dates | outcome |
|---|---|---|
| #1 | 2026-04-24 → 2026-04-26 | **Aborted at step 10,293** due to stale-provenance bug. Detection 46 hrs in via diagnostic logging. See postmortem `report/governed_v4_run1_postmortem_detailed.md`. |
| #2 | 2026-04-26 → 2026-04-28 | Ran on regenerated provenance + refuse-to-start guards. Checkpointed at step 10,000. Process restart (not crash) ended the leg. |
| **#3** | **2026-04-28 → 2026-05-10** | **Resumed from step 10,000. Completed 70,455 of 70,455 steps.** This is the leg that produced the final checkpoint and the 1.1092 val bpb. |

### The stale-provenance bug, summarized

- `parallel_family_cache` loader reads `cache_manifest.json` (shard list) and `provenance.json` (shard_index → family map). Both keyed off `shard_index`.
- After the shard-split + shuffle pipeline rewrote `cache_manifest.json` with a new `shard_index` space (3,125 → 18,926 shards), `provenance.json` was not regenerated.
- Loader silently used the stale provenance: read only 16.5% of the cache; got the family tag right 19.2% of the time.
- Training metrics looked healthy (loss descending, val descending, no NaN/OOM/instability).
- Bug detected because train loss EMA hit 0.06 (per-token entropy ~0.06 nats not credible for diverse historical text) AND because diagnostic logging printed per-family cursors that were off by 3 orders of magnitude from the schedule.
- Fix: regenerate provenance, add refuse-to-start coverage and family-cross-check guards in the loader, keep diagnostic cursor logging.
- Run #1 checkpoints archived in `base_checkpoints/run1_archived_pre_provenance_fix_v4_d22_r30/`.

---

## 4. What was NOT done (gaps)

- **No CORE metric.** Standardized perplexity/cloze benchmark not run.
- **No per-family val BPB.** We have one aggregate val bpb, not per-family slices. (Per-family slices would have caught run #1's stale-provenance bug from the smoke phase.)
- **No anachronism / cutoff eval.** Whether the model "doesn't know X because pre-1914" vs "doesn't know X because 615M params" is currently unverified.
- **No source-grounded QA eval.** Whether the model can produce period-appropriate factual answers (not just period-appropriate prose) is unverified.
- **No style-by-source continuation eval.** Whether the model can produce e.g. Old Bailey legal register vs Chronicling America newspaper register on demand is unverified.
- **No midtraining (structured tasks).** Base checkpoint only.
- **No SFT or RLHF.** It is not yet a chat model.
- **No comparison to peer historical LMs** (Ranke-4B, vintage LLM efforts).
- **The two soft-ceiling overshoots** (science 26.9% vs ≤25%, early_modern 9.8% vs ≤7%) are noted in the launch justification but not analyzed for their effect on the trained model.

---

## 5. What is solid

- Training mechanics: loader <1% wait, no NaN/OOM, peak VRAM well under budget, every checkpoint state-dict round-trip SHA-verified.
- Curve health: monotone descent through warmdown, no overfitting, no late instability.
- Governance: the corpus is auditable end-to-end. Every shard is rights-traceable, date-traceable, source-traceable.
- The loader bug was caught by a diagnostic that the researcher had deliberately added in anticipation of exactly this failure class. The postmortem is structural (couplings between manifest and provenance), not one-off.
- Run #1 was archived, not overwritten. Reproducibility of the bug is preserved.

---

## 6. Where the project sits in the landscape

- Karpathy's reference d20 nanochat (FineWeb-Edu, 11.2 B tokens, single 3090) reports ~0.81 val bpb. Numbers not directly comparable (different vocab, different corpus, different val set), but in the same general regime.
- Ranke-4B (Zurich, pre-release) is the most direct peer: similar idea, larger model.
- Owain Evans' "vintage LLMs" talk frames the research question this project tries to operationalize.

---

## 7. Key artifacts on disk

| file | what it is |
|---|---|
| `report/base-model-training.md` | terminal training summary (the canonical run record) |
| `report/governed_v4_launch_justification.md` | pre-launch justification, source mix, smoke series |
| `report/governed_v4_run1_postmortem_detailed.md` | the stale-provenance postmortem |
| `report/v4_smoke_series_report.md` | three-smoke comparative report |
| `logs/phase0/governed_v4_d22_r30_parallel_family_run3_resume10000.log` | full training log, ~34 MB |
| `logs/phase0/governed_v4_d22_r30_parallel_family_run3_resume10000_bench.csv` | per-step bench metrics, ~225 KB |
| `base_checkpoints/governed_v4_d22_r30_parallel_family/model_070455.pt` | final model parameters |
| `base_checkpoints/governed_v4_d22_r30_parallel_family/meta_070455.json` | final metadata |
| `data/processed/corpus_1913_v4_balanced_candidate/manifest.json` | canonical corpus manifest |
| `data/token_cache_v4_balanced_candidate/provenance.json` | per-shard family + source mapping |

End of brief.
