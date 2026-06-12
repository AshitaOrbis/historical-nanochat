# Governed v4 GPU Smoke Test Plan

This plan runs a short, measured GPU smoke against `token_cache_v4_balanced_candidate` before committing to the long governed run. The purpose is to confirm the v4 cache + dataloader + training stack work end-to-end at d22 scale with no OOM / NaN / throughput regression vs legacy.

## Pre-flight gates (must all pass before GPU smoke)

1. `corpus_1913_v4_balanced_candidate/manifest.json` exists with `train_tokens_est >= 18 B`.
2. `token_cache_v4_balanced_candidate/train/cache_manifest.json` exists; `total_tokens` matches the sum of `.bin` file sizes / 2.
3. `token_cache_v4_balanced_candidate/val/cache_manifest.json` exists.
4. `token_cache_v4_balanced_candidate/provenance.json` shows per-family share matches the corpus `source_mix.md`.
5. `smoke_token_cache_v3.py --cache-dir <v4 cache>` passes all 7 CPU checks (manifest well-formed, tokens in vocab range, BOS present, tiny GPT forward on CPU produces finite logits).
6. GPU is idle (`nvidia-smi --query-gpu=memory.used --format=csv` returns < 2 GB).
7. `reports/source_mix.md` + `reports/rights_audit.md` + `reports/date_distribution.md` + `reports/recovery_sources.md` + `reports/known_limitations.md` all present.
8. SHAs captured (see "Pre-launch identity record" below).

## Smoke config (exact)

```
run:                      governed_v4_smoke_d22_pre_poc
depth:                    22
max_seq_len:              1024
device_batch_size:        8
total_batch_size:         262_144
target_param_data_ratio:  -1         # not budget-driven; --num_iterations hard-coded
num_iterations:           200        # 200 measured iterations after compile warmup
compile_mode:             default
sdpa_backend:             auto
chunked_loss:             True
loss_chunk_size:          1024
activation_checkpoint:    False
tokenizer:                /tokenizer/tokenizer.pkl (vocab 32768, SHA from manifest)
token_cache_dir:          data/token_cache_v4_balanced_candidate/train
val_cache_dir:            data/token_cache_v4_balanced_candidate/val
eval_every:               50         # 4 val checks during the 200-iter smoke
eval_tokens:              262_144
core_metric_every:        -1         # no CORE during smoke
sample_every:             -1         # no sampling during smoke
save_every:               100        # save at step 100, then at 200 end-of-run
benchmark_csv:            logs/phase0/v4_smoke_bench.csv
```

## Isolation requirements

The training process **must** launch in a separate systemd user scope, not in the tmux cgroup hosting the CPU jobs:

```
systemd-run --user --scope \
  --slice=compute.slice \
  --unit=v4_smoke_d22 \
  --property=OOMPolicy=continue \
  --collect -- <training invocation>
```

No CPU-heavy corpus job may run in the same cgroup as training. (Postmortem: `report/crash_2026-04-21_oom_cgroup_postmortem.md`.)

## Pass criteria

| Metric | Pass threshold |
|---|---|
| loader_pct_mean (over last 150 steps) | ≤ 2% |
| train loss trajectory | descending; first-100 mean > last-100 mean |
| validation BPB | decreasing monotonically across the 4 evals |
| tok/sec (last 150 steps mean) | within 10-20% of legacy d22 baseline of 16,374 (acceptable: ≥ 13,000) |
| peak VRAM | ≤ 22 GiB (24 GB card budget) |
| OOM events | 0 |
| NaN / inf events | 0 |
| compile recompile storms | ≤ 5 |
| checkpoint save @ step 100 | succeeds; state_dict round-trip sha256 match |
| checkpoint resume @ step 100 | succeeds; one subsequent step produces finite loss |
| source-family val availability | val BPB reportable per source_id family (fails closed if val shards miss a family) |

## On-fail triage (decision rule)

Map the fail mode to the user's decision rule:

- **Cache / dataloader bug** (e.g. manifest mismatch, shape error, bad dtype) → attempt quick fix (<2 h); if fix eludes, fallback path D: resume legacy from step 4000.
- **OOM on d22** → fallback path E: try d20 r40 on v4 (approx 10% smaller working set); if that also OOMs, resume legacy.
- **Throughput regression > 20%** → investigate (likely loader_wait issue from v4 being on a different disk layout). If fix obvious, apply; else fall back to legacy.
- **NaN / inf** → halt immediately; do not resume a NaN'ed checkpoint. Resume legacy from step 4000 and debug v4 on CPU.
- **Checkpoint save/resume fails** → blocker; do not launch long run. Root-cause fix required.

## Pre-launch identity record

Before any training step runs, capture:

```
report/v4_smoke_launch_record.json:
  git_commit:                  <output of `git -C ~/claudeworkspace/research/historical-nanochat rev-parse HEAD`>
  tokenizer_sha:               <SHA-256 of tokenizer/tokenizer.pkl>
  tokenizer_manifest_sha:      <SHA-256 of tokenizer/tokenizer_manifest.json>
  corpus_manifest_sha:         <SHA-256 of data/processed/corpus_1913_v4_balanced_candidate/manifest.json>
  token_cache_train_sha:       <SHA-256 of token_cache_v4_balanced_candidate/train/cache_manifest.json>
  token_cache_val_sha:         <SHA-256 of token_cache_v4_balanced_candidate/val/cache_manifest.json>
  source_mix_sha:              <SHA-256 of manifests/source_mix.json or reports/source_mix.md>
  rights_audit_sha:            <SHA-256 of manifests/rights_audit_*.jsonl.gz concatenated>
  date_audit_sha:              <SHA-256 of manifests/date_audit_*.jsonl.gz concatenated>
  run_config_json:             <full `--flag value` snapshot of base_train.py invocation>
  systemd_scope_command:       <the `systemd-run --user --scope ...` invocation>
  launch_utc:                  <ISO-8601 timestamp>
  hostname:                    requiem
  gpu_nvidia_smi:              <nvidia-smi -q output snapshot>
```

## Long-run launch decision (post-smoke)

If smoke passes AND `train_tokens >= 18 B`:
→ **Launch `governed_corpus1913_v4_d22_r30_3090_poc`** using the same config but `num_iterations = 70,500` and `save_every = 2000`.

If smoke passes AND `train_tokens` is 12-18 B:
→ Choose d20 r40 or d22 r20-r25 based on the actual token count. Re-emit the config as `governed_corpus1913_v4_d20_r40_3090_poc` or equivalent.

If smoke passes but source mix still badly imbalanced (e.g. newspapers > 50% or books < 10%):
→ Rebuild balanced cache with tighter caps; do not launch with the current cache.

If smoke fails for any hard pass criterion:
→ Apply the matching triage branch above.

## Non-negotiables

- The long run uses the name `governed_corpus1913_v4_d22_r30_3090_poc`. Do not reuse the legacy name.
- The long run must launch in its own systemd scope (`--slice=compute.slice --unit=governed_v4_d22_r30`).
- CPU corpus jobs must remain in `background.slice`, never sharing a scope with training.
- The first governed checkpoint (step 2000) must be verified with `tools/verify_first_checkpoint.py` before the run is trusted.
- Sample probes (`tools/sample_probe.py`) run at step 5000 and step 10000 against the governed corpus with the label `governed_v4` (not `legacy/internal/baseline`).

## What this IS NOT

- This smoke is not a CORE eval. CORE runs post-training on a stable checkpoint.
- This smoke is not a final model-card artifact. The long run produces the model card.
- Passing the smoke does not mean the v4 corpus is the "right" corpus — it means the pipeline is mechanically correct and the mix is within thresholds. Quality of the trained model is judged at post-train eval.
