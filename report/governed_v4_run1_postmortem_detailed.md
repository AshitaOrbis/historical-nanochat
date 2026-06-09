# Postmortem — Governed v4 Run #1 Stale-Provenance Bug

A maximally detailed account of the bug, the diagnostic chain, the remediation, and the lessons.

- **Run ID:** `governed_corpus1913_v4_d22_r30_3090_poc_parallel_family` (run #1)
- **Launched:** 2026-04-24 17:40 UTC
- **Detected:** 2026-04-26 ~17:00 UTC during routine status check
- **Stopped:** 2026-04-26 ~17:05 UTC at step 10,293 of 70,455 (14.61%)
- **Wall clock wasted:** 45.8 hours of GPU
- **Detection latency from bug to detection:** ~46 hours
- **Detection latency from launch to detection:** ~46 hours (bug present from t=0)
- **Severity classification:** training-data integrity (not a crash, not a metric corruption — the model was learning, but on a different corpus than the run claimed)

---

## TL;DR

The `parallel_family_cache` dataloader builds its per-family shard lists by joining `cache_manifest.json` (the on-disk shard index) against `provenance.json` (the family-tag for each shard). After the cache shards were split from 3,125 → 18,926 sub-shards and re-shuffled, **provenance.json was never regenerated**, so it still referenced the 3,125 pre-split shard indices. After re-shuffling, those indices in the new manifest pointed to random shards. The loader silently used the stale provenance, served microbatches from only 16.5% of the cache, and got the family tag right on only 19.2% of those shards. The result: 46 hours of training on a mystery 1/6 subset of the corpus with effectively random family weighting.

The bug was undetectable from training metrics alone (loss descended, val BPB descended, no NaN/OOM/instability) and was first noticed because train loss EMA was anomalously low (0.06). Family-cursor inspection then revealed the partition staleness directly: newspapers cursor at 1 after consuming a schedule-implied 1 billion tokens, where 1,049 was expected.

The fix is small (regenerate provenance + add two refuse-to-start guards in the loader), the lesson is structural (any pipeline step that mutates a manifest's `shard_index` space invalidates dependent provenance/index files, and the loader must enforce this).

---

## Architecture preamble — what artifacts were involved

Three coupled artifacts under `data/token_cache_v4_balanced_candidate/`:

```
token_cache_v4_balanced_candidate/
├── train/
│   ├── cache_manifest.json     ← shard list with shard_index, source_file, tokens, bytes, filename
│   ├── shard_00000.bin         ← uint16 token stream for shard with index 0
│   ├── shard_00000.meta.json
│   ├── ... (18,926 .bin files after split)
│   └── shard_18925.meta.json
├── val/
│   └── ... (2,881 .bin files after split)
└── provenance.json             ← per-split: shard_index → family + source_id
```

Of those, **`cache_manifest.json` and `provenance.json` are coupled by `shard_index`.** The dataloader uses `manifest_by_idx[shard_index]` to look up actual shard files, and uses `provenance.per_shard[i].family` to know which family that shard belongs to. If either side's `shard_index` space changes, the link is broken.

The build process is:

1. `data/phase0/process/build_v4_balanced.py` selects governed parquet shards by family budget, hard-links them into `corpus_1913_v4_balanced_candidate/{train,val}/`. Each parquet shard has an `source_file` name like `shard_<family>_<source_id>_NNNNNN.parquet`.
2. `data/phase0/process/build_token_cache_v4.py` calls `scripts.build_token_cache` which reads each parquet, tokenizes, writes `shard_NNNNN.bin` + `shard_NNNNN.meta.json` per shard, and emits `cache_manifest.json` enumerating them. **It then writes `provenance.json`, deriving `family` from each `source_file` name.** Both files start coherent.
3. `data/phase0/process/split_cache_shards.py` splits any `.bin` shard whose token count exceeds `--max-tokens-per-shard`. It rewrites `cache_manifest.json` with new shard_index values 0..N (N grew from 3,125 to 18,926). It does NOT touch `provenance.json`.
4. `data/phase0/process/shuffle_cache_manifest.py` shuffles the manifest entries and reassigns `shard_index` in shuffle order. Adds a `filename` field to each entry so the on-disk file binding is preserved across renumbering. It does NOT touch `provenance.json`.

After steps 3 + 4, `cache_manifest.json` had 18,926 shards with newly-shuffled shard_index values 0..18925. `provenance.json` still had its 3,125 pre-split entries with shard_index values 0..3124. Both files individually were internally consistent; **the inter-file linkage was silently broken.**

---

## What the loader did

The relevant code (post-fix, before fix this had no guards):

```python
def _load_family_shard_lists(cache_dir: str) -> dict[str, list[dict]]:
    prov_path = ".../provenance.json"
    prov = json.load(open(prov_path))
    per_shard = prov["splits"]["train"]["per_shard"]   # 3,125 entries
    manifest = _load_manifest(cache_dir)                # 18,926 entries
    manifest_by_idx = {s["shard_index"]: s for s in manifest["shards"]}

    by_family: dict[str, list[dict]] = {}
    for rec in per_shard:                # iterate provenance entries
        sidx = rec["shard_index"]        # 0..3124
        fam = rec["family"]
        mentry = manifest_by_idx.get(sidx)
        if mentry is None:
            continue                     # silently skip
        by_family.setdefault(fam, []).append(mentry)
    return by_family
```

What this did in practice:

- For each provenance entry's `shard_index ∈ [0, 3125)`, look up the manifest shard with that index. After shuffle, that manifest entry was a **random** shard (not the original).
- Provenance claims "shard_index 42 is family newspapers_periodicals" — but in the post-shuffle manifest, shard_index 42 was, say, a `shard_books_general_blbooks_001234.parquet`-derived sub-shard.
- The shard got added to `by_family["newspapers_periodicals"]` even though it was actually a books shard.
- Provenance entries with `shard_index ≥ 3125` simply didn't exist (provenance only had 3,125 entries), so 15,801 manifest shards were never added to any family list — the loader had no awareness they existed.

The schedule (`12 newspapers / 8 sci / 6 books / 3 legal / 3 em` per optimizer step) drew microbatches from these mis-labeled lists. So a "newspapers microbatch" was statistically a 1-in-5 chance of being actual newspapers, a 1-in-5 chance of being books, etc.

---

## Why no metrics caught it

The training run looked **outwardly healthy** for 46 hours. Every standard signal said "fine":

| signal | observed | plausible? |
|---|---:|---|
| train loss EMA descent | 10.40 → 0.06 over 10,293 steps | only at the end did this look anomalous |
| val BPB descent | 3.89 → 1.33 across 11 evals | yes — model is learning |
| no NaN / inf | confirmed | yes |
| no OOM | confirmed | yes |
| no compile recompile storm | 0 hits | yes |
| loader_pct | 0.1-0.3% | yes |
| tok/sec | 16,300-16,400 (matches legacy baseline) | yes |
| peak VRAM | 17.13 GiB (identical to legacy) | yes |
| state-dict round-trip sha256 | matched on every checkpoint | yes |
| family cursors advancing | newspapers=1, books=503, sci=13, legal=14, em=0 | **NO** — but only visible if you do the math |
| family microbatch distribution | claimed correct by schedule | unverifiable from any standard log line |

The cursor pattern was the smoking gun: `newspapers_periodicals: 1` after 10,291 optimizer steps was numerically impossible if the schedule was being honored. At 12 newspaper microbatches/step × 8,193 tokens × 10,291 steps = 1.012 billion newspaper tokens consumed. Newspaper sub-shards average 964K tokens each, so we should have advanced through ~1,049 shards. We had advanced through 1.

The reason this was visible at all: my earlier diagnostic logging change (`--diagnostic_logging`) had added family-cursor printout to the per-step log line. Without that, the bug would have been invisible until the model was deployed and someone evaluated it on per-family slices and found that one family had memorized a tiny number of shards while another had no representation at all.

---

## How I detected it

A routine "what's happening" status check at 46 hours wall clock. Three signals in sequence:

### Signal 1 — anomalously low train loss EMA

```
step 10287/70455 (14.60%) | loss: 0.061442 | lrm: 1.00 | tok/sec: 16,408
```

Cross-entropy of 0.061 nats per token would mean ~94% per-token confidence. For a 615 M-param model on diverse historical text, that's not credible. The val BPB at the same step was 1.326 — which translates back to roughly 3.2 nats per token (val BPB × bytes-per-token × ln(2)). The 50× gap between train and val cross-entropy is impossible to explain by ordinary train/val mismatch.

### Signal 2 — diagnostic log shows raw_loss is also tiny

```
[diag] step 10283 | raw_loss: 0.0525 | ema_loss: 0.0594 | grad_norm: 0.1603
```

Confirmed it wasn't an EMA bug — raw per-microbatch loss was also in the 0.05-0.10 range. So the loss readings were honest. The training itself was producing this very-low loss.

### Signal 3 — family cursor mismatch

```
fam_cursors={'newspapers_periodicals': 1, 'science_technical': 13,
             'books_general': 503, 'legal_government': 14,
             'early_modern': 0}
```

Books cursor 503 looked sensible (6 microbatches/step × 10,291 steps × 8,193 tokens / ~1.19M tokens/shard ≈ 425 shards expected, observed 503 — close, allowing for step granularity and shard-size variance).

Newspapers cursor 1 versus expected 1,049 was off by three orders of magnitude.

### Verification — diagnostic Python

A small one-shot diagnostic confirmed the partition staleness:

```python
# Read what the loader sees
prov = json.load(open(".../provenance.json"))
ps = prov["splits"]["train"]["per_shard"]
manifest = json.load(open(".../cache_manifest.json"))
manifest_by_idx = {s["shard_index"]: s for s in manifest["shards"]}

# For each shard the loader assigns to a family, parse the actual
# family from source_file and compare.
mismatches = Counter()
for r in ps:
    me = manifest_by_idx.get(r["shard_index"])
    if me is None: continue
    actual = parse_family(me["source_file"])
    mismatches[(r["family"], actual)] += 1
```

Result: 601 of 3,125 (19.2%) loader classifications were correct. 80.8% of the loader's family tags were wrong — and the loader didn't know about 15,801 of the 18,926 manifest shards at all.

---

## Why pre-launch validation didn't catch it

Five things had to go wrong simultaneously, and they did:

### 1. Unit tests for the family loader passed

`nanochat/tests/test_family_loader.py` has five tests:

| test | what it verifies |
|---|---|
| `test_schedule_produces_expected_family_mix_per_step` | over 96 microbatches, the count of microbatches *labeled* family X equals the schedule's count |
| `test_cursors_advance` | over 640 microbatches, at least one cursor moves |
| `test_resume_produces_same_next_microbatch` | the deterministic-resume property |
| `test_refuse_if_provenance_missing` | loader raises when `provenance.json` is absent |
| `test_schedule_mismatch_raises` | loader raises when schedule doesn't sum to grad_accum_steps |

All five pass with broken provenance. They verify mechanism (the loader correctly serves microbatches according to the schedule and persisted state) — they do not verify *whether the schedule's family labels match the actual content of the served shards*. The tests had no way to fail in this scenario.

### 2. The Stage-1 mechanical smoke (smoke #4) passed

A 200-step GPU smoke confirmed the loader could feed the training stack without crashes, NaN, OOM, or compile issues, and that val BPB descended monotonically (2.24 → 1.95 across 4 checkpoints). What I didn't notice at the time: the smoke was running on the same broken provenance. The model was learning *something* — even with random family assignment, the model trains on real cache tokens, just from the wrong subset. Val BPB descent was a real positive signal about training mechanics, but it told us nothing about which shards were being read.

In retrospect, smoke #4 should have included a "per-family val BPB" check — measure validation loss separately on books-only, science-only, etc. slices. If the family-balanced loader is working, all per-family val slices should improve. If it's broken, the per-family slices that the loader is actually reading from will improve while the others stay flat. We didn't have per-family val plumbed.

### 3. The launch-record SHA capture was reproducible-but-broken

`governed_v4_long_launch_record.json` captured `token_cache_provenance_sha = 2046f30210f7492c40daf3162bf85284488c45b41b56a80e5ada59d2c86d99c6`. That hash is deterministic — re-running the launch would consume the same broken provenance file. Reproducibility was preserved, but reproducibility of a bug isn't validation. The SHA capture was a control, not a check.

### 4. Provenance regeneration was a manual step, not a script-pipeline guarantee

`build_token_cache_v4.py --skip-train --skip-val` regenerates provenance.json from the current manifest. I ran it once early on (after the original tokenization) and never again. The downstream scripts (`split_cache_shards.py`, `shuffle_cache_manifest.py`) didn't trigger or even check provenance regeneration. Each of those scripts independently re-wrote `cache_manifest.json` with different `shard_index` values, with no awareness that an external file's references depended on the index space they were rewriting.

### 5. Coupling between manifest and provenance was implicit, not enforced

There's no schema-level link between the two files. A `shard_index` in provenance is a foreign key into manifest, but nothing in the data model declares that. The loader merrily accepted whatever the lookup returned, including `None` (handled by `continue` — the most dangerous keyword in the loader, in retrospect).

---

## The fix

### Immediate — regenerate provenance from the post-split manifest

```bash
cd /home/user/historical-nanochat
PYTHONPATH=/home/user/historical-nanochat \
  /home/user/claudeworkspace/research/historical-nanochat/.venv/bin/python \
  -m data.phase0.process.build_token_cache_v4 --skip-train --skip-val
```

The `--skip-train --skip-val` flags tell the script to skip retokenization and only regenerate `provenance.json` by parsing each manifest shard's `source_file` field. After regeneration:

| family | shards in regenerated provenance |
|---|---:|
| books_general | 2,804 |
| early_modern | 1,871 |
| legal_government | 1,627 |
| newspapers_periodicals | 7,468 |
| science_technical | 5,156 |
| **total** | **18,926** ✓ matches manifest exactly |

`provenance.json` SHA changed: `2046f3021...` (broken) → `55ffa71c30b54f0f...` (correct).

### Defensive — loader sanity-check guards

`_load_family_shard_lists()` in `nanochat/nanochat/dataloader_cached.py` now refuses to start under the conditions that would have caught run #1:

```python
# After populating by_family from provenance:
coverage = matched / max(1, len(manifest["shards"]))
if coverage < 0.95:
    raise RuntimeError(
        f"parallel_family_cache: provenance.json covers only "
        f"{coverage*100:.1f}% of manifest shards ({matched}/{len(manifest['shards'])}). "
        "Regenerate provenance via "
        "`python -m data.phase0.process.build_token_cache_v4 --skip-train --skip-val`."
    )

# Family-cross-check: parse family from source_file, compare to provenance:
if family_mismatches > 0:
    mismatch_pct = family_mismatches / max(1, matched) * 100
    if mismatch_pct > 5.0:
        raise RuntimeError(
            f"parallel_family_cache: {family_mismatches} of {matched} shards have a "
            f"provenance family that does not match the source_file family "
            f"({mismatch_pct:.1f}%). Provenance is stale — regenerate via "
            "`python -m data.phase0.process.build_token_cache_v4 --skip-train --skip-val`."
        )
```

Either guard alone would have caught the bug at training start:

- **Coverage guard:** would have triggered immediately. 3,125 of 18,926 = 16.5% coverage, far below the 95% threshold.
- **Family-cross-check:** would have triggered at 80.8% mismatch, far above the 5% threshold.

The 95% coverage threshold (rather than 100%) tolerates minor manifest churn. The 5% mismatch threshold (rather than 0%) tolerates rare edge cases like manifest entries with non-standard `source_file` names. Both thresholds are configurable in code if needed.

The guards run inside `_load_family_shard_lists`, which the loader calls once at construction. There is no runtime cost during training, and no path through the loader that bypasses the guards.

### Operational — pipeline-step ordering

Three rules added to the prevention checklist (now in the postmortem repository):

> **Rule 1.** Any script that mutates a manifest's `shard_index` space MUST regenerate or invalidate dependent provenance/index files in the same run, or refuse to complete.

> **Rule 2.** `cache_manifest.json` and `provenance.json` are a coupled pair. They share `shard_index` as a foreign key. If you can't guarantee they were rebuilt together, treat the dataset as corrupt until proven otherwise.

> **Rule 3.** A loader that selects from a manifest based on an external index file MUST validate the join before serving any data. Silent skip-on-miss is forbidden.

Applied to the relevant scripts:

| script | mutates `shard_index` space? | required action |
|---|---|---|
| `build_token_cache_v4.py` | yes (writes manifest fresh) | also writes provenance — already correct |
| `split_cache_shards.py` | yes | should regenerate provenance at end-of-script. Currently relies on user to run `build_token_cache_v4 --skip-train --skip-val` afterward. **TODO: bake into the splitter script.** |
| `shuffle_cache_manifest.py` | yes (reassigns shard_index in shuffle order) | preserves `(source_file, family)` per entry implicitly (shuffles entries, not field contents), so provenance keyed off `shard_index` becomes wrong but provenance keyed off `source_file` would be fine. Idea for future: provenance.json should key off a stable identifier (`source_file` or a content-hash), not the volatile `shard_index`. |
| `repacker_v3.py` | no | safe |
| `combined_audit_and_pack.py` | no | safe |
| `build_v4_balanced.py` | no (does not write manifest) | safe |

### Diagnostic improvement — log family microbatch distribution per N steps

For run #2 onward, the diagnostic log captures `current_microbatch_family` for the prefetched-next microbatch. Future improvement: log the histogram of family microbatches actually consumed across the last N optimizer steps. This is the canonical health signal for `parallel_family_cache` and should be visible in the per-step log without needing post-hoc analysis. **TODO for run #3 (if needed)** — not blocking run #2.

---

## Was the run #1 model salvageable?

**No.** The model was trained on a 16.5% subset of the cache with random family weighting. We could not honestly call it a "balanced governed v4 PoC."

Quantitatively, the situation was:

- Effective corpus: ~3.16 B tokens (16.5% of 19.12 B)
- Effective family mix: random sampling within the first 3,125 shards of a randomly-shuffled manifest. Approximately reflects the global token-share distribution for those 3,125 shards but with weights distorted by the bogus schedule. Reconstructing the exact effective mix would require offline analysis of which shard_index values 0..3124 corresponded to which actual families post-shuffle.
- Val BPB at run-1 step 10,000: 1.326. This is a real measurement on a real model, but it doesn't measure what we said it measured.
- Train loss EMA at step 10,000: 0.06 — anomalously low, consistent with the model essentially memorizing a small subset.

Continuing run #1 to completion would have produced a model card we could not write honestly. Stopping was correct.

The run-1 checkpoints (step 2000, 4000, 6000, 8000, 10000) were preserved under `base_checkpoints/run1_archived_pre_provenance_fix_v4_d22_r30/` for diagnostic purposes only. They are NOT the governed PoC artifact.

---

## Cost accounting

| item | cost |
|---|---:|
| GPU wall time (run-1 launch to detection) | 45.8 hours |
| GPU wall time (run-1 stop + provenance regen + run-2 launch) | ~30 min |
| Disk space, run-1 checkpoints retained | ~10 GB |
| Token-cache disk space (unaffected) | ~38 GB train + 5.5 GB val |
| Engineering: provenance regeneration | already supported by existing script (no new code) |
| Engineering: loader guards | ~30 lines of Python in `_load_family_shard_lists` |
| Engineering: postmortem documentation | ~2 hours |

The run-2 launch on 2026-04-26 ~17:30 UTC starts the 13-day clock again. ETA to first checkpoint (step 2000) is ~9 hours from launch.

---

## Lessons

1. **A passing smoke does not validate data integrity.** It validates training mechanics. Both can pass simultaneously while the corpus being trained on is silently wrong. The smoke harness needs per-family val checks (or a "what fraction of cache is actually being read" telemetry) to catch this class of bug.

2. **Manifests and provenance are a coupled pair.** Treating them as independent files is the same kind of mistake as a dangling pointer in C. Better long-term fix: provenance should key off `source_file` (a content-stable identifier) rather than `shard_index` (a position-volatile one).

3. **Silent skip-on-miss is dangerous.** `manifest_by_idx.get(sidx)` returning `None` and the loader doing `continue` is graceful in some senses, but in this case it allowed the loader to function with 5/6 of the cache invisible to it. The fix (refuse to start if coverage drops below threshold) is two lines and would have caught run #1 immediately.

4. **The canonical health signal for parallel_family_cache is family-cursor advancement, not loss/throughput/VRAM.** Loss/throughput/VRAM tell you the GPU is happy. Cursor advancement tells you the data is what you think it is. Future training infrastructure should put cursor health alongside loss in the per-step log.

5. **At a 46-hour wall-clock cost, this kind of bug is catastrophic; at a 30-second smoke cost (with proper guards), it's invisible.** The economic case for moving validation up the pipeline is overwhelming.

6. **"Reproducible" is not "correct".** The run-1 launch record SHA-pinned the broken provenance. Reproducibility ensured the bug was reliably present, not that it was absent. SHA capture is a control, not a check.

7. **Diagnostic logging that nobody reads at the right time is no different from no diagnostic logging.** The cursor mismatch was visible in the per-step diag log from step 0 of run-1, but nobody looked until 46 hours in. The fix is automated alerting (if cursor advancement is anomalous over the first N steps, raise), not a process change ("look at the log more often").

---

## Status of run #2

- Launched: 2026-04-26 ~17:30 UTC
- Provenance: regenerated, coverage 100%, family-cross-check 100% match
- Loader: refuse-to-start guards active (would have caught the bug)
- Config: identical to run #1 (parallel_family_cache, softened LR, real long-run schedule)
- Stop-gates: 500, 2000, 5000, 10000, 20000, 42000, completion
- Step 0 confirmed healthy: loss 10.40 (random init), lrm 0.00 (warmup begin), provenance loaded for 18,926 shards
- ETA to first checkpoint (step 2000): ~9 h
- ETA to completion: ~13 d

Run #1 archive at `base_checkpoints/run1_archived_pre_provenance_fix_v4_d22_r30/`.

---

## Artifact index — postmortem and related

| file | purpose |
|---|---|
| `report/governed_v4_run1_postmortem_detailed.md` | this document |
| `report/governed_v4_run1_provenance_bug_postmortem.md` | shorter postmortem, written immediately on detection |
| `report/governed_v4_long_launch_record.json` | run-1 launch record, broken-provenance SHA captured |
| `report/governed_v4_long_launch_record_run2.json` | run-2 launch record, corrected-provenance SHA |
| `report/governed_v4_launch_justification.md` | original justification for the long run (still valid for run #2) |
| `report/v4_smoke4_stage1_result.md` | Stage-1 smoke that "passed" with broken provenance |
| `nanochat/nanochat/dataloader_cached.py` | now contains the refuse-to-start guards |
| `nanochat/tests/test_family_loader.py` | unit tests (5/5 still pass after guard addition) |
| `data/phase0/process/build_token_cache_v4.py` | provenance regenerator |
| `data/phase0/process/split_cache_shards.py` | TODO: bake provenance regeneration into end-of-script |
| `data/phase0/process/shuffle_cache_manifest.py` | TODO: same |
| `base_checkpoints/run1_archived_pre_provenance_fix_v4_d22_r30/` | run-1 checkpoints, archived, not for governed PoC use |
| `logs/phase0/governed_v4_d22_r30_parallel_family.log` | run-1 training log (ended at step 10,293) |
| `logs/phase0/governed_v4_d22_r30_parallel_family_run2.log` | run-2 training log (live) |

End of postmortem.
