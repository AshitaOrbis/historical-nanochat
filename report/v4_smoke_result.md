# Governed v4 GPU Smoke Result — CONDITIONAL PASS

- **Run:** `governed_v4_smoke_d22_pre_poc`
- **Model tag:** `v4_smoke_d22`
- **Launched:** 2026-04-23 20:53:25 UTC
- **Completed:** 2026-04-23 21:47:10 UTC (~54 min wall, 200 iters)
- **Scope:** `v4_smoke_d22.scope` under `compute.slice` (isolated from CPU jobs)
- **Config snapshot:** `report/v4_smoke_launch_record.json`
- **Cache:** `token_cache_v4_balanced_candidate/` (train 3125 shards, val 90 shards, uniform-shuffled)

## Verdict

**MECHANICAL PASS** (pipeline works end-to-end) + **SOFT FAIL** on training dynamics.

Do NOT launch the 70,500-step long run with the current LR schedule. First run a second smoke with softened LR (warmup + reduced peak) to confirm the divergence is schedule-related, not data-related.

## Pass-criteria results

| Criterion | Threshold | Actual | Verdict |
|---|---:|---:|---|
| loader_pct_mean (last 150 steps) | ≤ 2% | **0.17%** | PASS |
| train loss first-100 mean > last-100 mean | required | 5.04 > 4.72 (barely) | **conditional** |
| validation BPB monotone descending | required | 1.60 → 1.84 → 1.69 → 1.63 | **non-monotone** |
| tok/sec within 10-20% of legacy 16,374 | -20 to +20% | **+0.5%** (16,378-16,450) | PASS |
| peak VRAM | ≤ 22 GiB | **17.13 GiB** (identical to legacy) | PASS |
| no OOM events | 0 | 0 | PASS |
| no NaN/inf | 0 | 0 | PASS |
| no compile recompile storm | ≤ 5 hits | 0 | PASS |
| checkpoint save @ step 100 | required | `model_000100.pt` + optim + meta | PASS |
| checkpoint save @ step 200 | required | same | PASS |
| state_dict round-trip (save→load→hash) | sha256 match | PASS (2911a48e4123...) | PASS |
| val loader works from separate val dir | required | 4 evals returned finite BPB | PASS |

## Training dynamics — the concern

Train loss trajectory (every 10 steps):

| step | loss | lrm | note |
|---:|---:|---:|---|
| 0 | 10.40 | 1.00 | random init, matches ln(32768) |
| 10 | 6.52 | 1.00 | |
| 20 | 5.69 | 1.00 | |
| 30 | 4.93 | 1.00 | |
| 40 | 4.59 | 1.00 | |
| 50 | 4.38 | 1.00 | val BPB 1.5957 |
| 60 | 4.11 | 1.00 | |
| 70 | **3.97** | 1.00 | **minimum** |
| 80 | 4.10 | 1.00 | ← divergence begins |
| 90 | 5.69 | 1.00 | loss spike |
| 100 | 6.01 | 1.00 | val BPB **1.8409** (worse) |
| 120 | — | ~0.85 | warmdown starts (`warmdown_ratio=0.4` × 200 → step 120) |
| 150 | 5.16 | 0.62 | recovering as LR drops |
| 200 | — | (end) | val BPB 1.6297 (recovered to ~step-50 level) |

**Interpretation:** peak LR was too high for v4's diverse batches. Loss diverged at step 80-100 while LR was still at peak. Warmdown (step 120+) brought LR down and the model recovered. For the 200-step smoke this was fine; the warmdown phase is 40% of the run.

**For the long run,** warmdown starts at step 42,300 of 70,500 — meaning the model would train at peak LR for **~7 days** on diverse batches. The divergence risk is enormous.

## Why v4 is harder than legacy at peak LR

Legacy corpus was 95% American newspapers — relatively homogeneous OCR English, stable perplexity across batches. v4 interleaves:

- BHL scientific monographs (Latin taxonomy, numeric/formulaic)
- BL Books page-level text (various OCR quality)
- CAP legal opinions (specialized jargon, long citations)
- American Stories (period newspapers)
- BL Newspapers (British newspapers, different register)
- EEBO early modern English (archaic spellings)

Per-batch loss distribution is much wider → gradient variance larger → small LR tolerance shrinks.

## Artifact integrity

- Train shards shuffled via `uniform_random seed=7` — preserves global family mix in every 100-shard window (~80% books, ~16% news, ~4% legal, <1% sci, rare early_modern per shard-count; token-weighted distribution matches the 19.12 B train target).
- Val shards shuffled same way.
- `cache_manifest.json` has `filename` field per entry binding logical index to the on-disk .bin file. No file renames needed.
- `state_dict` round-trip sha256 matches between save and reload.

## Recommendation

Before the 70,500-step long run, do **ONE** of:

### Option 1 — Add warmup + reduce peak LR (recommended)

Run a 2nd 400-step smoke with:

```
--warmup_ratio 0.02         # warm from 0 → peak over first 2% of steps
--embedding_lr 0.20          # down from 0.30
--matrix_lr 0.015            # down from 0.020
--unembedding_lr 0.003       # down from 0.004
```

At 400 steps this gives us LR peak for steps ~8 → ~240, so the "peak-LR window" is longer than the smoke and can be judged. If loss is monotone descending across the peak-LR window, we're safe for the long run.

Estimated cost: ~65 min wall.

### Option 2 — Reduce peak LR only

Same LR reduction but without warmup. Confirms LR reduction is sufficient.

### Option 3 — Investigate data-variance hypothesis

Re-enable the smoke on a bookified-only or newspapers-only v4 subset. If no divergence → confirms the issue is batch-mixing variance; addressable with LR.

I recommend **Option 1**. It's the least regret: if the divergence was LR, fixed. If it was data variance, warmup + lower LR also helps (less aggressive update steps).

## Triage against the plan's decision rules (user directive)

- v4 smoke passes mechanically **AND** train tokens ≥ 18 B: matches condition E.
- But: the mechanical pass hides a soft-fail on training dynamics.
- Honest reading: the long run would very likely diverge during the 42,300-step peak-LR phase. Launching it as-is would waste ~7 days of compute.
- Therefore: classify this as "v4 smoke fails on training dynamics" → run a second smoke with safer LR schedule before commit.

## Ready state

- v4 corpus: **ready** (manifest + 5 reports + 3125/90 parquet shards)
- v4 cache: **ready** (19.12 B train tokens + 2.86 B val tokens, shuffled)
- smoke infra: **ready** (isolated systemd scope, dataloader patched, smoke tool v4-aware)
- Second smoke: **requires decision on LR/warmup parameters**

## Next actions (pending user call)

1. Approve Option 1 LR-softened smoke, or pick alternate path.
2. If smoke #2 passes cleanly, capture a new launch record and kick off `governed_corpus1913_v4_d22_r30_3090_poc` with the matching LR schedule.
3. If smoke #2 also diverges, investigate data-level causes (per-shard loss profiling).
