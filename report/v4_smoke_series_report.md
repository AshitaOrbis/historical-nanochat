# v4 Smoke Series — Three-Run Comparative Report

Three GPU smokes were run against the `corpus_1913_v4_balanced_candidate` corpus / `token_cache_v4_balanced_candidate` (~19.12 B train + 2.86 B val). Each run used d22 / T=1024 / DBS=8 / total_batch=262144 / compile_mode=default / no activation checkpointing / isolated `compute.slice` scope. No OOM, NaN, compile instability, or cgroup cascade in any run — the **v4 pipeline is mechanically correct**.

| | smoke #1 | smoke #2 | smoke #3 |
|---|---|---|---|
| run name | `governed_v4_smoke_d22_pre_poc` | `governed_v4_smoke2_d22_softlr_pre_poc` | `governed_v4_smoke3_d22_splitshards_pre_poc` |
| steps | 200 | 200 (stopped) | 300 (stopped) of 400 |
| warmup_ratio | 0 | 0.02 | 0.02 |
| warmdown_ratio | 0.4 | 0.4 | 0.4 |
| embedding_lr | 0.30 | 0.20 | 0.20 |
| matrix_lr | 0.020 | 0.015 | 0.015 |
| unembedding_lr | 0.004 | 0.003 | 0.003 |
| train cache | 3,125 shards, 15M mean | same | **18,926 split sub-shards, 1M mean** |
| val cache | 90 shards, 32M mean | same | **2,881 split sub-shards, 1M mean** |
| compile stable | yes | yes | yes |
| OOM / NaN | none | none | none |
| peak VRAM | 17.13 GiB | 17.13 GiB | 17.13 GiB |
| tok/sec | 16,380 | 16,380 | 16,365 |
| loader_pct_mean | 0.17% | 0.17% | 0.17% |
| checkpoint save sha match | yes | yes | yes |
| step-0 val BPB | 4.996 | 4.996 | **3.894** |
| step-100 val BPB | 1.84 | 1.82 | 2.23 |
| step-200 val BPB | 1.63 | — (stopped) | 2.27 |
| step-300 val BPB | — | — | 2.23 |
| best train raw loss | 3.97 (step 70, then diverges) | — | 4.70 (step 190) |
| loss pattern | within-shard descent, between-shard spikes | same as #1 | noisy around 5.5-6.5, no sustained descent |
| **verdict** | **mechanical PASS, dynamics FAIL** (divergence at shard flip) | **mechanical PASS, dynamics FAIL** (same spike pattern) | **mechanical PASS, dynamics FAIL** (plateau) |

## What we learned

1. **Mechanical pipeline is solid.** The v4 corpus, token cache, dataloader, compile path, checkpoint save/resume, val loader, isolated scope, and crash-hardening all work across three runs with no faults.

2. **LR softening alone doesn't fix diverse-batch training.** Smoke #2 replicated smoke #1's spike-at-shard-flip pattern despite -33% peak LR.

3. **Shard-splitting fixed the "family-blocked" issue** (99.9% adjacent same-family → 27% after split). Diag logs show the dataloader now switches shards every 3-6 steps across all 5 families.

4. **But shard-splitting + softened LR produced a different failure:** loss plateau after ~100 steps. The model hits a "generalist floor" — good at averaging across families but not specializing enough to push past ~5.5 train loss / ~2.25 val BPB. Likely: 1M-token shards give only 4 gradient updates per shard before flipping, insufficient to "lock in" any family-specific pattern, and each flip partially undoes the last.

5. **Val BPB comparisons across smokes are unreliable** because val cache was split+shuffled differently per smoke. Only within-smoke trajectories are apples-to-apples.

## Fundamental question

Can d22 (616M params) productively train on v4's diverse 5-family corpus at these LRs and cache granularities, in a meaningful wall-clock? Three 200-400-step smokes don't definitively answer it. Observations:

- The "shard divergence" in smoke #1/#2 (loss 3.97 → 6.01 when shard flips legal → newspapers) **is real learning being overwritten**, not a mathematical pathology. Over a 70,500-step run, with hundreds of family rotations, the loss oscillations should average toward a real descent — but we can't prove that from a 200-400 step smoke.
- The plateau in smoke #3 may be a real failure OR may be a 300-step artifact. We'd need ~2,000-5,000 steps to know if the plateau persists or breaks.
- Legacy's 95%-homogeneous corpus trained cleanly because all batches came from one distribution. v4's 5-family mix is fundamentally harder.

## Three paths forward

### Path A — Launch long run with smoke #2 config, monitor aggressively

- Use **original (unsplit) cache** + **softened LR** (smoke #2 config).
- Run 70,500 steps; check val BPB every 2000 steps; stop if val diverges or plateaus by step 5000.
- **Pros:** real long-run signal. If loss dynamics are transient oscillations on short horizons but descend over 70k steps, we'll see that.
- **Cons:** ~7 days of 3090 time at risk if the divergence is sustained. Sunk cost if smoke wasn't predictive.
- **Fallback:** if val plateaus by step 5000, stop and try Path B.

### Path B — Write a parallel-shard dataloader

- Modify `dataloader_cached` to read from N shards simultaneously and round-robin tokens into each batch. This gives **batch-level** family mixing instead of shard-level.
- This is what the parquet dataloader already does via row-group striping — just need the cached-loader equivalent.
- **Pros:** addresses the root cause (per-batch family homogeneity) without retokenization.
- **Cons:** 4-8 hours of dataloader engineering + another smoke cycle.

### Path C — Accept that d22 + v4 diversity needs curriculum

- Train on one family at a time in sequence (newspapers for 20k, science for 15k, books for 15k, ...).
- Rebuild corpus as per-family caches rather than one mixed cache.
- **Pros:** within-family descent is proven.
- **Cons:** violates the "diverse training" principle of v4. Produces a model biased by the final family trained on. Not aligned with the governed PoC target.

## Recommendation

**Path A is the least-regret option.** The smoke runs prove the v4 pipeline is mechanically correct; they don't conclusively prove that long training will fail. Running the long run with stop-gates at steps 2000, 5000, 10000 lets us make an empirical call with real data. If it's clearly plateauing by step 5000, we pivot to Path B.

Explicit stop-gates for a Path A long run:

| gate | step | criterion | on-fail action |
|---|---:|---|---|
| first checkpoint | 2000 | val BPB < 2.50 AND train loss < 5.0 | stop, review |
| early progress | 5000 | val BPB < 2.20 AND train loss < 4.5 | stop, pivot to Path B |
| mid-run | 20000 | val BPB < 1.80 | stop, eval results |
| mid-run | 42000 | val BPB < 1.50 (entering warmdown) | review |
| completion | 70500 | val BPB < 1.30 (preferred) | publish as PoC |

## Artifacts

- smoke #1: `report/v4_smoke_launch_record.json`, `report/v4_smoke_result.md`, `base_checkpoints/v4_smoke_d22/*`
- smoke #2: `report/v4_smoke2_launch_record.json`, `logs/phase0/v4_smoke2_d22.log`
- smoke #3: `logs/phase0/v4_smoke3_d22.log`, `base_checkpoints/v4_smoke3_d22/model_000200.pt`
- post-split cache: `token_cache_v4_balanced_candidate/` (shuffled w/ seed 11)
- crash postmortem: `report/crash_2026-04-21_oom_cgroup_postmortem.md`
- recovery plan: `report/governed_corpus_recovery_plan_2026-04-21.md`

## Ready state summary

- v4 corpus: ready
- v4 cache: ready (split+shuffled)
- Training stack: mechanically validated across 3 smokes
- **Long-run launch: BLOCKED on user decision between Path A / B / C**

GPU is idle (1.6 GB / 24 GB). No CPU jobs running. Legacy step-4000 checkpoint preserved as fallback (verified 12/12 PASS).
