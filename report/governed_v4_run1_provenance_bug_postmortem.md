# Postmortem — Governed v4 Run #1 Stale-Provenance Bug

- **Run:** `governed_corpus1913_v4_d22_r30_3090_poc_parallel_family` (run #1)
- **Launched:** 2026-04-24 17:40 UTC
- **Stopped:** 2026-04-26 ~17:00 UTC (step 10,293 of 70,455 = 14.6%)
- **Wall time wasted:** ~46 hours of GPU
- **Severity:** training-data integrity (the model was learning, but not from the corpus the run claimed)

## What happened

The `parallel_family_cache` loader builds its per-family shard lists from `provenance.json`. The bug: after I split the cache shards (3,125 → 18,926 sub-shards) and re-shuffled the manifest, **`provenance.json` was never regenerated**.

State at run-1 launch:

| artifact | shard count | shard_index range |
|---|---:|---:|
| `cache_manifest.json` (post-split, post-shuffle) | 18,926 | 0..18,925 |
| `provenance.json` (pre-split) | 3,125 | 0..3,124 |

The loader iterates `provenance.json`'s `per_shard` entries and does `manifest_by_idx.get(sidx)` to look up the manifest entry. Provenance referenced indices 0..3,124 — but those indices in the **shuffled post-split manifest** were random shards (not the originals).

A diagnostic showed only **601 of 3,125 (19.2%)** of provenance "this shard is family X" claims actually matched the shuffled shard's true family. The other 80.8% were wrong — and silently so.

The loader's `manifest_by_idx.get(sidx)` returned `None` for indices 3,125..18,925 (those weren't in provenance), and `continue`'d past them. Net effect:

- Loader served microbatches from the first 3,125 manifest shards only (16.5% of the cache).
- Even within that subset, family-tag was wrong 80.8% of the time, so the "newspapers" microbatches were drawn from a random mix of all families, "science" likewise, etc.
- The schedule's intended 12/8/6/3/3 family ratio became meaningless.
- The model was training on a mystery subset with random family assignment.

## How it was detected

Routine status check showed train loss EMA = 0.06 at step 10,000. That's anomalously low — language models on diverse text should plateau around 1.5-2.5 nats per token, not 0.06. Cross-checked against val BPB 1.33 (sane) and family cursor advancement: newspapers cursor was at 1 after consuming a schedule-implied ~1B tokens. Expected ~1,049 shard advances; observed 1. That gap is what unmasked the partition staleness.

## Why pre-launch checks didn't catch this

- The `parallel_family_cache` unit tests use the cache pointed-to by the test (the same broken provenance). All 5 tests passed because the loader behaves consistently with whatever provenance says — even if provenance is wrong. The tests verify mechanism, not data integrity.
- The Stage-1 smoke (smoke #4) ran the same broken loader. It passed because the model still learned *something* — even with random family assignment, the model trains on actual cache tokens. Val BPB still descended (2.24 → 1.95). The bug is invisible from training metrics alone.
- The launch-record SHAs captured the broken provenance hash. Reproducible, just reproducibly broken.

## Fix

### Immediate — regenerate provenance from the post-split manifest

```bash
python -m data.phase0.process.build_token_cache_v4 --skip-train --skip-val
```

This rebuilds `provenance.json` by parsing each manifest shard's `source_file` for the family prefix. After regeneration:

| family | shards |
|---|---:|
| books_general | 2,804 |
| early_modern | 1,871 |
| legal_government | 1,627 |
| newspapers_periodicals | 7,468 |
| science_technical | 5,156 |
| **total** | **18,926** ✓ |

### Defensive — loader sanity check (`dataloader_cached.py`)

Added two refuse-to-start guards to `_load_family_shard_lists`:

1. **Coverage:** if provenance covers < 95% of manifest shards, raise `RuntimeError` with regenerate command.
2. **Family-cross-check:** for each provenance entry, parse the manifest shard's `source_file` and confirm the family matches. If > 5% disagree, raise `RuntimeError`.

Either guard would have caught this bug before run-1 launch. Tests now pass; loader correctly partitions all 18,926 shards.

### Process — pipeline-step ordering

The bug was structurally caused by `split_cache_shards.py` and `shuffle_cache_manifest.py` reassigning `shard_index` values without invalidating the dependent `provenance.json`. Adding to the prevention checklist:

- **Any script that mutates a manifest's shard_index space MUST regenerate or invalidate dependent provenance/index files.**
- `split_cache_shards.py` should rewrite `provenance.json` at end-of-script (or delete it and force regeneration before training).
- `shuffle_cache_manifest.py` should preserve `(source_file, family)` regardless of shard_index reassignment — done implicitly because shuffle preserves the entries — but the dependent provenance still needs regeneration.

## Was the run-1 model salvageable?

Looking at run-1's val BPB trajectory:

| step | val BPB |
|---:|---:|
| 0 | 3.8935 |
| 1000 | 1.6248 |
| 2000 | 1.4599 |
| 5000 | 1.3328 |
| 10000 | 1.3260 |

The model was learning — val BPB descended cleanly, plateauing around 1.33. But it was learning from a **mystery subset**: 16.5% of the cache, with random family tags. We can't honestly call this output a "governed corpus_1913_v4 model" because it wasn't trained on what we said it was.

Decision: discard the run-1 checkpoints (keep them in `base_checkpoints/governed_v4_d22_r30_parallel_family/` for archive/diagnostic only). Restart fresh from step 0 with corrected provenance.

## Cost

- 46 hours of 3090 time burned (run-1 wall clock from launch to detection)
- 5 checkpoints saved that won't be used as the governed PoC artifact
- ~10 GB disk for those checkpoints (preserved for potential diagnostic re-eval)

## What's preserved

- step-2000, 4000, 6000, 8000, 10000 checkpoints from run-1 stay in `base_checkpoints/governed_v4_d22_r30_parallel_family/` (renamed dir TBD, see below)
- run-1 log, bench CSV, diagnostic log
- run-1 launch record (`governed_v4_long_launch_record.json`) with the broken-provenance SHA captured for reproducibility

## Restart plan

- Move/rename run-1 checkpoint dir to `base_checkpoints/run1_archived_pre_provenance_fix_v4_d22_r30/`
- Capture a new launch record (`governed_v4_long_launch_record_run2.json`) with the corrected provenance SHA
- Launch run #2 fresh from step 0 with same config (parallel_family_cache, softened LR, real long-run schedule)
- Same stop-gates at 500/2000/5000/10000/20000/42000/completion
- Update model card to disclose: run-1 was discarded due to provenance bug; run-2 is the genuine governed PoC

## Lessons

1. **A passing smoke does not validate data integrity** — only metric integrity. Both can pass simultaneously while the corpus being trained on is wrong.
2. **Manifests and provenance are two coupled artifacts; touching one without rebuilding the other is the same kind of bug as a dangling pointer.**
3. **The loader should refuse to start under suspicious provenance, not silently degrade.** Now it does.
4. **Family cursor advancement is the canonical health signal for parallel_family_cache.** If a family with N shards has cursor << expected after K steps, the partition is probably broken.
5. **At a 46-hour wall-clock cost, this kind of bug is catastrophic if undetected; at a 1-hour smoke cost, it's invisible.** A real cure is the loader-level guard, not "more careful smoking."

## Status

- run-1 stopped, GPU idle
- provenance regenerated, manifest covers all 18,926 shards
- loader sanity-check guard added; 5/5 unit tests pass
- ready to launch run #2
