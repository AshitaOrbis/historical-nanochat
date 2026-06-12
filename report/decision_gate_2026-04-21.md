# Decision Gate Report — `legacy_textonly_d22_r30_internal_baseline`

- **Generated:** 2026-04-21 05:05 UTC
- **Gate checkpoint:** T+14h from reclassification (2026-04-20 15:20 UTC)
- **This is the T+24 gate fired early** because Phase-0-lite reached pipeline-complete state sooner than estimated (<8 h wall for full-corpus pass vs. 1-3 d planned). Proceeding to decision.

---

## 1. Live baseline numbers (read-only from the running process)

| Metric | Value at T+14h | Baseline (at reclass) |
|---|---:|---:|
| Step | 2,907 / 70,455 (4.13%) | 1,406 (2.0%) |
| Tokens seen | 0.762 B / 18.47 B | 0.369 B |
| Train loss (latest) | 3.727 | ~3.80 |
| Val BPB @ step 1000 | 3.1844 | — |
| Val BPB @ step 2000 | **3.0788** | — |
| Val BPB trend | strictly descending | — |
| tok/sec (last 500 steps mean) | 16,260 | 16,374 |
| tok/sec delta vs baseline | **-0.7%** | — |
| Peak VRAM | 17.13 GiB | 17.13 GiB |
| Wall time | 773 min (~12.9 h) | — |
| ETA to 70,455 | 18,025 min (~12.5 d) | 18,430 min (~12.8 d) |

---

## 2. First-checkpoint verification (step 2000) — **PASSED**

Artifact: `report/verification_step002000_20260421T021526Z.md`

| Check | Result |
|---|---|
| checkpoint files present | PASS |
| optimizer shard (rank0) present | PASS |
| meta.json well-formed | PASS (n_layer=22, vocab=32768) |
| state_dict round-trip (save → load → hash) | PASS (sha256 match) |
| train loss descending | PASS (mean_first100=6.275 → mean_last100=3.700, drop=+2.575) |
| validation BPB trajectory | PASS (init 4.996 @ step 0 → 3.079 @ step 2000) |
| no OOM | PASS |
| no NaN | PASS |
| no compile recompile storm | PASS (0 hits) |
| bench CSV parseable | PASS |
| peak VRAM ≤ 22 GB | PASS (17.13 GB / 24 GB card) |
| tok/sec stability < 2% | PASS (stddev/mean = 0.18%) |

**Overall: PASS** on all 12 checks.

---

## 3. Sample probe snapshot

Not yet run at T+14h. Probe script (`tools/sample_probe.py`) is ready; prompts
fixed across victorian_prose / newspaper / bhl_science / legal_oldbailey /
cutoff_anachronism / cutoff_insideperiod. First actionable run point is the
step-5000 checkpoint (ETA ≈ T+27h). Recommend running against step 5000 and
step 10000 before the T+48h gate.

---

## 4. Phase-0-lite readiness — **COMPLETE at smoke scale, FULL cache in progress**

Priority-order status:

| # | Artifact | Status | Evidence |
|---|---|---|---|
| 1 | source registry | **done** | `data/phase0/sources/registry.yaml` (10 sources + 6 planned, cutoff_year=1914, fail-closed defaults) |
| 2 | unified doc + training schema | **done** | `data/phase0/schemas/*.py` (pydantic 2.13; Document / TrainingRecord / SourceEntry / audit enums) |
| 3 | rights audit (fail-closed) | **done, full corpus** | all 10 sources audited; per-source `manifests/rights_audit_<s>.jsonl.gz` |
| 4 | date audit + cutoff buckets | **done, full corpus** | all 10 sources audited; per-source `manifests/date_audit_<s>.jsonl.gz` |
| 5 | metadata-preserving repacker | **done, full corpus** | 4,451 governed parquet shards with full (source_id, document_id, date_bucket, rights_class, publication_year, split) schema |
| 6 | source_mix.json | **done** (char-estimate) | `manifests/source_mix.json` |
| 7 | held-out val by doc/source | **done** | `manifests/split_manifest.json`: 220M train / 2.22M val_by_document / 500 val_by_source |
| 8 | anachronism + source_qa evals | **done (code + prompts)** | `data/phase0/evals/` — runner will execute against checkpoint step 5000+ |
| 9 | token_cache_v3 builder | **smoke done; full in progress** | `token_cache_v3_smoke/` (12 shards, 3.59 B tokens, 17.7 min wall). Full build (4,451 shards) launched; est. completion T+20h. |
| 10 | dataloader + forward smoke | **done** | `report/token_cache_v3_smoke.md` — 7/7 PASS on CPU, BOS present, tiny GPT forward produces finite logits |

### Full-corpus combined audit + repack timing

- Total wall: 88 min (combined rights + date + repack in single pass per source)
- I/O: 417 GB read, 107 GB written
- nice -n 19, ionice -c 3, 1 worker
- Training impact: loader_pct_mean 0.150 → 0.168 during peak, tok/s 16,374 → 16,369 (-0.03%). **Well below the 2% threshold; training never warned.**

---

## 5. Governed corpus — actual composition

Total corpus: **59.67 B tokens (chars/4 estimate)** / **238.6 GB chars** / **222.25 M documents** / **4,451 parquet shards**.

Vs. legacy text-only corpus (25.2 B tokens): **+2.37× more tokens** because we kept pre-1914 American Stories in full rather than subsampling at 0.33.

| Family | Plan target | Governed actual | Gap vs plan |
|---|---:|---:|---:|
| books_general | 35% | **1.7%** | **-33.3 pp** |
| newspapers_periodicals | 30% | **95.4%** | **+65.4 pp** |
| legal_government | 15% | **0.2%** | **-14.8 pp** |
| science_technical | 10% | **0.0%** | **-10.0 pp** |
| early_modern | 5% | **2.7%** | -2.3 pp |
| other / holdout | 5% | 0.0% | -5.0 pp |

### Why the mix is this skewed

1. **american_stories ballooned**: 219.9 M docs kept (vs ~54 B tokens estimate in the prior report, actual is 217 GB chars ≈ **54.4 B tokens**). At 95.4% of corpus, it dominates everything.
2. **BHL (118 GB raw) was fully quarantined** by the fail-closed rights audit. Registry says `per_item_rights_required: true` because BHL mixes PD and open-access-non-PD items; the deduped JSONL has no per-record rights field, so every record was quarantined. This is governance **working as designed** — but it removes the entire science_technical family.
3. **TCP (1 GB raw) was fully quarantined** for the same reason. Small impact on mix (TCP was ~0.25 B tokens).
4. **CAP (caselaw) is empty** (10 rows total). The JSONL was never populated beyond a token harvest. This is a data-source blocker, not a governance blocker.

### Net per-source token counts (char-estimate)

| source | rows kept | chars | est. tokens | share |
|---|---:|---:|---:|---:|
| american_stories | 219,941,445 | 217.43 GB | 54.36 B | 91.1% |
| bl_newspapers | 2,171,100 | 10.66 GB | 2.67 B | 4.5% |
| eebo | 53,305 | 6.40 GB | 1.60 B | 2.7% |
| gutenberg | 10,352 | 4.08 GB | 1.02 B | 1.7% |
| oldbailey | 74,849 | 438 MB | 109 M | 0.18% |
| chronicling_v2 | 500 (held out) | 12.5 MB | 3.1 M | 0.005% |
| caselaw | 10 | 79 KB | 20 K | <0.001% |
| chronicling_america | 3 | 48 KB | 12 K | <0.001% |
| **bhl** | **0 (quarantined)** | 0 | 0 | 0 |
| **tcp** | **0 (quarantined)** | 0 | 0 | 0 |

---

## 6. What this implies for the next governed run

The governed corpus as-is is **not an improvement over the legacy baseline's mix** — it is still newspaper-dominated, and losing BHL makes the science hole worse, not better. A d22 run against this corpus would produce a model even more telegraphic-newspaper-biased than the legacy one, with worse science register.

**Before spending another 13 days of 3090 time on a governed run, the blockers need addressing:**

1. **BHL governance decision.** The fail-closed policy is technically correct but drops 118 GB of unique scientific content. Two defensible alternatives:
   - **(a)** Downgrade `per_item_rights_required` to `false` and accept the BHL corpus-wide rights statement as sufficient evidence. BHL publicly claims all ingested content is PD or open-access; documenting that claim in the registry preserves auditability without losing 20 B tokens.
   - **(b)** Re-harvest BHL via their API with rights metadata per-item. Larger lift (~1-3 d of harvester work + re-ingest), but yields strictly better governance.
   - **(c)** Drop BHL entirely and fill science_technical via an alternative source (Wellcome Collection, arXiv historical, Royal Society archive). Lift varies.
2. **Books shortfall.** Gutenberg alone gives 1.7% vs the 35% target. Either re-harvest a bigger books source (LOC Selected Books, BL 19c Books, IA) — plan-listed but not harvested — or accept a newspaper-heavy model.
3. **Legal shortfall.** Old Bailey 0.18% vs 15% target; CAP is 80 KB and must be re-harvested. CAP is the only clean path to closing the legal gap.
4. **TCP governance.** Small corpus impact (<0.25 B tokens) but the quarantine should either be resolved (phase-metadata re-harvest) or accepted as permanent loss.

---

## 7. Decision

**Continue to completion** the legacy_textonly_d22_r30 baseline, with the following conditions:

- [x] **CONTINUE** — baseline runs to step 70,455 (ETA ≈ T+13d). Primary value is the pipeline-validation artifact and the baseline loss curve.
- [ ] STOP at next checkpoint — *not chosen*
- [ ] re-scope at T+72h — *not needed*

### Rationale

Four facts drive "continue":

1. **Baseline is healthy.** First-checkpoint verification passed 12/12. Val BPB dropped 4.996 → 3.079 (step 2000), loss trending down, no instability, throughput stable. Nothing in the run is failing.
2. **Phase-0-lite is further along than planned but the next governed run is BLOCKED on upstream data work, not pipeline readiness.** Stopping the baseline now would free the GPU for a governed run that currently CANNOT be kicked off without first resolving BHL / CAP / books governance (items 1-4 above). Those fixes are data-harvest and rights-metadata work, measured in days, done on CPU. The baseline can run concurrently during that work.
3. **Zero training interference from Phase-0-lite.** The full-corpus combined pass (88 min, 417 GB read, 107 GB written, nice+ionice) moved training loader_pct_mean from 0.15% → 0.17% and tok/s from 16,374 → 16,369 (-0.03%). The CPU-only Phase-0-lite track can proceed through re-harvest + re-pack without touching the GPU.
4. **The sunk ~13h of 3090 time (to date) has produced measurable artifacts**: a validated checkpoint save/resume path, a loss curve reference, live benchmarks at d22. Stopping now discards all of that without yet having a governed corpus to train on.

### What has to happen BEFORE the next governed run starts

- Resolve BHL governance (option (a), (b), or (c) — flagged for human decision).
- Either re-harvest CAP + a books source, or accept a newspaper-heavy mix in the governed run.
- Finish the full `token_cache_v3` build (in progress, ETA T+20h) so exact per-source token counts are known.
- Write the governance memo amending `data/phase0/sources/registry.yaml` with the new per-item-rights policy.

Targeted date for governed run launch: **when legacy_textonly_d22_r30 completes**, assuming BHL resolution lands in the next ~5 days.

---

## 8. Non-negotiables recap (unchanged)

- Legacy checkpoints are **not** used as teachers.
- Legacy samples are labeled `legacy/internal/baseline`.
- Governed shards live under `data/phase0/governed_shards/`; `token_cache_v3` lives under `data/token_cache_v3`; neither touches `data/shards/` or `token_cache_v2`.
- Any outreach email that mentions a model must refer to a governed model, not this baseline.

---

## 9. Next actions

| # | Action | Owner | Target |
|---|---|---|---|
| 1 | Human decision on BHL policy (a/b/c) | user | T+48h |
| 2 | Full token_cache_v3 build complete | orchestrator (running) | T+20h |
| 3 | Sample probe @ step 5000 | orchestrator | T+27h |
| 4 | Sample probe @ step 10000 | orchestrator | T+48h |
| 5 | Anachronism + source_qa evals @ step 10000 | orchestrator | T+48h |
| 6 | T+48 review report | orchestrator | T+48h |
| 7 | If BHL resolved: re-run combined pass on BHL alone | orchestrator | after #1 |
| 8 | If CAP re-harvested: re-ingest into governed pipeline | orchestrator + user | indefinite |
