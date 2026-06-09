# Governed v4 d22 r30 3090 PoC — Launch Justification

## Run

**`governed_corpus1913_v4_d22_r30_3090_poc_parallel_family`**

- Launched: 2026-04-24 17:40 UTC
- Host: requiem (NVIDIA RTX 3090, 24 GB)
- Scope: isolated systemd user scope `governed_v4_d22_r30_parallel_family.scope` under `compute.slice`
- Expected wall time: ~13 days (70,455 steps × ~16 s/step)
- Full launch record (git commit, SHAs, config): `report/governed_v4_long_launch_record.json`

## Why this run — in one paragraph

We have a governed, provenance-preserving pre-1914 training corpus that meets every hard source-family floor (newspapers ≤50%, books ≥10%, legal ≥5%, science ≥5%, early_modern >0%). Four GPU smokes established that the **mechanical pipeline is sound** and identified the exact failure mode of a naive cached dataloader on diverse data. The fourth smoke, with a family-balanced dataloader, produced the first v4 run with **monotone multi-checkpoint validation descent**. This launch is the first training run that can honestly be called a governed 3090 proof-of-concept candidate.

---

## What changed from the legacy baseline

The legacy run (`legacy_textonly_d22_r30_internal_baseline`) was a useful training-stack validator but not a governed model. It trained on 25 B tokens of text-only shards that dropped source/date/rights metadata at pack time. Its source mix was ~95% American newspapers.

The v4 corpus is a different artifact:

| | legacy | v4 governed |
|---|---|---|
| source-level rights audit | none | item-level for BHL; collection-policy for TCP; derived-dataset for CAP; registry for remainder |
| date cutoff enforcement | implicit | explicit ≤ 1913 with confidence hierarchy |
| parquet schema | `{text}` only | `{text, source_id, document_id, date_bucket, rights_class, publication_year, split, ...}` |
| science_technical share | effectively 0% (BHL dropped) | 26.9% (BHL item-level recovery) |
| books_general share | 1.7% (Gutenberg only) | 17.5% (BL Books CC0 + Gutenberg) |
| legal_government share | 0.18% (Old Bailey only) | 8.2% (CAP reharvest + Old Bailey) |
| early_modern share | 3.1% (EEBO only) | 9.8% (EEBO + TCP subset policy) |
| newspapers_periodicals share | 94.97% | 37.7% |
| total train tokens | 25 B | 19.12 B |
| held-out val | last shard of train | separate 2.86 B val split, source-stratified |

The v4 corpus is substantially more balanced; every source admission is auditable back to an item, subset, or collection policy.

---

## The training-dynamics problem and how we solved it

### Smoke 1 — sequential cache, legacy LR

Ran d22 / 200 steps / original v4 cache / legacy LR. Train loss descended cleanly while the dataloader stayed on one shard (steps 0-70: loss 10.4 → 3.97). **When the shard flipped at step 80 from legal to newspapers, loss spiked to 6.01 and did not recover before warmdown.** The cached dataloader was reading each 15 M-token shard end-to-end, meaning ~50-60 consecutive steps of one family, then a sudden distribution flip.

Mechanical pass, dynamics fail.

### Smoke 2 — sequential cache, softened LR

Same setup with LR reduced ~33% (embedding 0.20, matrix 0.015, unembedding 0.003, warmup 0.02). The same shard-flip spike appeared at the same step. LR softening alone did not fix the root cause.

Mechanical pass, dynamics fail.

### Smoke 3 — split shards, softened LR

Split every large `.bin` shard to ≤2 M tokens (3,125 shards → 18,926 sub-shards; 90 val → 2,881 val sub-shards). Re-shuffled manifests. The dataloader now switches shards every 3-6 steps instead of every 50-60.

Initial descent happened (step 0-50), but val BPB **plateaued at 2.23-2.27** across steps 100/200/300 with no further improvement. Training loss was noisy around 5.5-6.5 with no net progress. Root cause: each sub-shard gave only 4 gradient updates before flipping to a new family, and per-shard gradients partially undid each other. The model converged to a "generalist floor" without specializing.

Mechanical pass, dynamics fail (different failure mode than smokes 1-2).

### Smoke 4 — parallel_family_cache — the decisive experiment

Kept the split shards. Wrote a new cached dataloader that **maintains one sequential cursor per source family** and serves microbatches according to a fixed per-step schedule:

```
newspapers_periodicals: 12 microbatches per optimizer step
science_technical:       8
books_general:           6
legal_government:        3
early_modern:            3
                        ---
                        32 microbatches (grad_accum_steps)
```

Every optimizer step now computes its gradient from a fixed 37.5%/25%/18.75%/9.375%/9.375% family mix. No more within-step homogeneity. No more shard-flip spikes (there are no shard flips per-step — families are drawn in parallel).

Result:

| step | train loss EMA | val BPB |
|---:|---:|---:|
| 50 | 4.94 | 2.2387 |
| 100 | 4.49 | 2.2096 |
| 150 | 4.14 | 1.9912 |
| 200 | ~4.0 | **1.9475** |

**Monotone descent across all four checkpoints** — the first v4 smoke where every eval point is lower than the previous. Contrast with smoke 3 which plateaued at ~2.23 across the same span.

Mechanical checks: all 12 pass (no OOM/NaN/compile issue, loader 0.21%, tok/s 16,361 = -0.08% of legacy baseline, peak VRAM 17.13 GiB identical to legacy, state-dict sha256 round-trips on both saved checkpoints).

**Smoke 4 is the evidence that justifies launching the long run.**

---

## Head-to-head smoke comparison

| | cache granularity | loader strategy | step-100 train | step-200 val trajectory | verdict |
|---|---|---|---:|---:|---|
| smoke 1 | unsplit (15 M/shard) | sequential | 6.01 (post-spike) | 1.60 → 1.84 → 1.69 → 1.63 (oscillating) | dynamics FAIL |
| smoke 2 | unsplit | sequential | 6.06 (post-spike) | — (stopped early) | dynamics FAIL |
| smoke 3 | split (1 M/shard) | sequential | 6.27 | 2.23 → 2.27 → 2.23 (plateau) | dynamics FAIL |
| **smoke 4** | **split (1 M/shard)** | **parallel_family** | **4.49** | **2.24 → 2.21 → 1.99 → 1.95 (monotone)** | **PASS** |

---

## The launched long-run configuration

Everything below matches smoke 4 except the iteration count (now 70,455 for the real Chinchilla-ratio=30 budget) and the cadence flags (save every 2k, val every 1k, sample every 5k).

```
depth:                        22
params:                       615,645,184
max_seq_len:                  1024
device_batch_size:            8
total_batch_size:             262,144 tokens/step
target_param_data_ratio:      30 (Chinchilla)
num_iterations:               70,455 (derived)
total_tokens:                 18,469,355,520
compile_mode:                 default (bfloat16 autocast)
sdpa_backend:                 auto
activation_checkpoint:        off
chunked_loss:                 True, chunk 1024

loader_strategy:              parallel_family_cache
token_cache_train_dir:        token_cache_v4_balanced_candidate/train  (3125 -> 18926 split shards, shuffled seed=11)
token_cache_val_dir:          token_cache_v4_balanced_candidate/val    (90   -> 2881  split shards, shuffled seed=11)
provenance_index:             provenance.json (per-shard source_id + family)

warmup_ratio:                 0.02   (warmup ends at step 1409)
warmdown_ratio:               0.4    (warmdown starts at step 42273)
peak LR duration:             40,864 steps
embedding_lr:                 0.20   (legacy was 0.30)
matrix_lr:                    0.015  (legacy was 0.020)
unembedding_lr:               0.003  (legacy was 0.004)

save_every:                   2000
eval_every:                   1000
eval_tokens:                  262,144
core_metric_every:            -1 (disabled during training; post-train only)
sample_every:                 5000

diagnostic_logging:           on (raw/EMA loss, grad norm, param norm, per-group LR,
                                 NaN/inf detection, current-microbatch family, family cursors)
```

---

## Stop-gates

The run will be halted if any of these gates fail. Each gate reviews the usual metrics plus the family-balance signal from the loader.

| step | criterion |
|---:|---|
| 500 | no OOM/NaN/inf; family mix matches schedule; train EMA not exploding; val BPB not sharply worse than step 0 |
| 2000 | first checkpoint save/reload sha-verified; train EMA clearly below early-run baseline; val BPB meaningfully below step 0 |
| 5000 | train EMA continues downward; val BPB continues downward; ≥ 4/5 source-family val slices improve |
| 10000 | progress remains visible; no family persistently harmed by sampling bug; checkpoint/resume reliable |
| 20000 | val BPB still learning; samples not dominated by one family |
| 42000 | warmdown-entry review; run healthy; val BPB improving or stable |
| completion | run full eval battery |

If a gate fails, the fallback order is: (1) adjust LR one bracket, (2) adjust warmup_ratio, (3) adjust family schedule, (4) consider a smaller model.

---

## Crash-hardening in effect

Learned from the 2026-04-21 OOM cascade postmortem (`report/crash_2026-04-21_oom_cgroup_postmortem.md`):

- Training runs in its own systemd scope under `compute.slice`. CPU corpus jobs run under `background.slice`. A cgroup-wide OOM in one scope cannot cascade to another.
- `OOMPolicy=continue` per scope — OOM kill targets the offending process only, not the whole scope.
- ShardWriter now enforces `max_buffer_bytes=500 MB` so no long-text repacker can grow to 57 GB RAM again.
- All CPU corpus jobs (audit, repack, reharvest, tokenize, split, shuffle) finished before this launch.
- Disk: 1.8 TB NVMe, ~350 GB free; sufficient for 13 d of checkpoints + bench CSVs.

---

## What this run will produce

- One checkpoint every 2,000 steps (`base_checkpoints/governed_v4_d22_r30_parallel_family/`).
- Val BPB data point every 1,000 steps — enough to interpolate the learning curve.
- One sample probe every 5,000 steps (labeled `governed_v4`, not `legacy/internal/baseline`).
- Full benchmark CSV (tok/sec, dt, peak VRAM, loader wait, MFU).
- Per-step diagnostic log: grad norm, param norm, update-to-param ratio, NaN/inf counter, current microbatch family.

At completion, post-training eval battery runs:

1. held-out BPB by source family (separate val slices)
2. cutoff / anachronism eval
3. source-grounded QA eval
4. style-by-source continuation eval
5. synthetic-data pilot (the whole point of preserving provenance in parquet)
6. model card + run card

---

## Artifacts index

| file | what it is |
|---|---|
| `report/governed_v4_long_launch_record.json` | SHAs, git commit, full config, LR schedule, stop-gate definitions |
| `report/v4_smoke4_stage1_result.md` | Stage 1 (smoke 4) mechanical pass report |
| `report/v4_smoke_series_report.md` | Three-smoke comparative report (smokes 1-3) |
| `report/v4_smoke_result.md` | Smoke 1 standalone report |
| `report/crash_2026-04-21_oom_cgroup_postmortem.md` | OOM/cgroup crash postmortem; crash-hardening rules |
| `report/governed_corpus_recovery_plan_2026-04-21.md` | Five-scenario source-mix projection; rationale for v4 recovery work |
| `data/processed/corpus_1913_v4_balanced_candidate/manifest.json` | Canonical corpus manifest |
| `data/processed/corpus_1913_v4_balanced_candidate/reports/source_mix.md` | Final actual source mix |
| `data/token_cache_v4_balanced_candidate/provenance.json` | Per-shard family + source mapping used by the loader |
| `nanochat/nanochat/dataloader_cached.py` | `parallel_family_cache` loader implementation |
| `nanochat/tests/test_family_loader.py` | Unit tests (5/5 pass): schedule correctness, cursor advance, deterministic resume, provenance guard, schedule-mismatch guard |
| `logs/phase0/governed_v4_d22_r30_parallel_family.log` | live training log |

---

## Honest caveats

- The corpus is soft-cap-over by a small margin: science_technical actual 26.9% vs preferred ≤25% ceiling, early_modern actual 9.78% vs preferred ≤7% ceiling. All hard floors are satisfied. The model card will disclose both.
- The family schedule (12/8/6/3/3) is close to but not identical to the global token mix (37.7 / 26.9 / 17.5 / 8.2 / 9.8). Close enough that the effective training distribution is within rounding of the corpus; exact cross-family weight is not a guaranteed invariant of this loader.
- 200 training steps are not a conclusive predictor of 70,500-step behavior. The run is gated precisely because smokes can only verify pipeline correctness and early-phase dynamics, not full-training generalization.
- Legacy `d22_r30` step-4000 checkpoint remains preserved as the fallback path if this run fails a gate catastrophically.
