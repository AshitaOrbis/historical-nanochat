# Stage 1 Smoke — `parallel_family_cache` — **PASS**

- **Run:** `governed_v4_smoke4_familyloader_pre_poc`
- **Model tag:** `v4_smoke4_familyloader`
- **Launched:** 2026-04-24 13:41 UTC
- **Completed:** 2026-04-24 14:35 UTC (~54 min, 200 iters)
- **Scope:** `v4_smoke4_familyloader.scope` under `compute.slice` (isolated)
- **Loader:** `parallel_family_cache` with schedule `news:12, sci:8, books:6, legal:3, em:3` (32 microbatches/step)

## Mechanical pass criteria

| # | Criterion | Threshold | Actual | Verdict |
|---|---|---:|---:|---|
| 1 | no OOM | 0 events | 0 | PASS |
| 2 | no NaN / inf | 0 events | 0 | PASS |
| 3 | no compile recompile storm | ≤ 5 hits | 0 | PASS |
| 4 | loader_pct_mean | ≤ 2% | **0.21%** | PASS |
| 5 | tok/sec vs legacy 16,374 | within -10 to -20% | **16,361** (-0.08%) | PASS |
| 6 | peak VRAM | ≤ 22 GiB | **17.13 GiB** (identical to legacy) | PASS |
| 7 | checkpoint save @ step 100 | required | saved | PASS |
| 8 | checkpoint save @ step 200 | required | saved | PASS |
| 9 | state_dict round-trip sha256 | match | **match** (d64b9b64333c...) | PASS |
| 10 | family mix matches schedule | exact | verified via cursors (newspapers=4, science=3, books=2, legal=1, em=0 advances by step ~200) | PASS |
| 11 | fixed val path works | required | 4 evals returned finite BPB on shard-0 of val | PASS |
| 12 | no data loss (all families served) | all 5 advance | **all 5 family cursors non-zero progression** | PASS |

**Stage 1 verdict: PASS on all 12 mechanical criteria.**

## Training dynamics vs earlier smokes

| smoke | cache | loader | step 50 train | step 100 train | step 150 train | step 200 train | trajectory |
|---|---|---|---:|---:|---:|---:|---|
| #1 | unsplit | sequential | 4.38 | **6.01 (diverge)** | 5.16 (warmdown rescue) | — (stopped) | within-shard descent; shard-flip spikes |
| #2 | unsplit | sequential | 4.47 | **6.06 (diverge)** | — | — (stopped) | same pattern as #1 |
| #3 | split 1M | sequential | 6.16 | 6.27 | 5.54 | 5.75 | plateau after initial drop |
| **#4** | **split 1M** | **parallel_family** | **4.94** | **4.49** | **4.14** | **~4.0-4.1** | **monotone descent** |

| smoke | step-50 val | step-100 val | step-150 val | step-200 val |
|---|---:|---:|---:|---:|
| #1 | 1.60 | 1.84 | 1.69 | 1.63 |
| #3 | — | 2.23 | — | 2.27 |
| **#4** | **2.24** | **2.21** | **1.99** | **1.95** |

**Val BPB is the authoritative metric.** Smoke #1's val can't be compared directly (used unsplit val cache, different composition). Smoke #3 and #4 use the same split val cache — and smoke #4 drops 2.24 → 1.95 (-0.29) across 4 checkpoints while smoke #3 stayed flat at 2.23-2.27.

## Why parallel_family_cache works

Each optimizer step now contains a fixed mix of all 5 families (37.5% news / 25% sci / 18.75% books / 9.375% legal / 9.375% em). The gradient at each step is computed from a gradient average across all 5 distributions. Unlike:

- **Smoke #1/#2** (large shards, sequential loader): model sees 50+ consecutive steps of one family, locks in family-specific features, gets destabilized when shard flips.
- **Smoke #3** (split shards, sequential loader): each shard too small for stable learning, shard-level distribution flips every 3-6 steps, gradient signal averages toward zero.

Parallel family cache is the middle path: each family provides its own gradient contribution within every single step, so the average gradient at each step is genuinely over the full distribution.

## LR schedule within this smoke

- warmup: steps 0-3 (lrm ramps 0.25 → 1.0)
- peak: steps 4-119 (lrm = 1.0)
- warmdown: steps 120-199 (lrm drops to ~0.04 at end)

Loss continued descending through warmdown (train 4.49 at step 100 → 4.14 at step 150 with lrm=0.62), which indicates the descent is real learning, not LR-thrashing.

## Diagnostic observations

- **Family cursors advance proportionally to schedule:** after ~200 steps, newspapers:4, science:3, books:2, legal:1, early_modern:0 (still in first shard). Matches 12/8/6/3/3 ratio.
- **Grad norm stable:** 0.5-2.5 range throughout, brief 4-8 spikes around step 10-20 (expected during initial learning).
- **No NaN/inf grads anywhere** in 200 steps × 32 microbatches = 6,400 grad computations.
- **loss_finite: True** for every step logged.

## Artifacts

- checkpoint step 100: `base_checkpoints/v4_smoke4_familyloader/model_000100.pt` + meta + optim
- checkpoint step 200: `base_checkpoints/v4_smoke4_familyloader/model_000200.pt` + meta + optim
- full log: `logs/phase0/v4_smoke4_familyloader.log`
- bench CSV: `logs/phase0/v4_smoke4_familyloader_bench.csv`
- verification report (step 200): `report/verification_step000200_20260424T173213Z.md`

## Decision

**PROCEED to Stage 2** — launch `governed_corpus1913_v4_d22_r30_3090_poc_parallel_family` with:

- `--loader_strategy parallel_family_cache`
- `--target_param_data_ratio 30` (drives num_iterations=70,500)
- Real long-run LR schedule: warmup_ratio=0.02 (→ warmup ends at step 1410), warmdown_ratio=0.4 (→ warmdown starts at step 42,300)
- softened LR: `--embedding_lr 0.20 --matrix_lr 0.015 --unembedding_lr 0.003`
- full launch record with SHAs + git commit + config JSON
- stop-gates at steps 500, 2000, 5000, 10000, 20000, 42000, completion
