# Run Reclassification — 2026-04-20

## Decision

The depth-22, ratio-30 base training run started 2026-04-20 09:05 UTC is hereby
reclassified as:

> **`legacy_textonly_d22_r30_internal_baseline`**

It is explicitly **NOT**:

- a governed 3090 PoC
- a release candidate
- a teacher candidate for any distillation pipeline
- the model referenced in any access-request outreach email

## What the run IS for

- A 13-day end-to-end pipeline validator under real GPU load: dataloader,
  tokenizer, cache path, compile mode, SDPA backend, save/resume, peak
  VRAM, throughput stability.
- A baseline loss curve on a 25.2 B-token pre-1913 text-only corpus with
  a known, documented governance gap.
- A forcing function for building Phase-0-lite governance (source registry,
  rights audit, date audit, metadata-preserving repacker) on CPU in the
  13-day window.
- A free reference point to compare any governed-shard run against.

## What the run IS NOT allowed to be

- Published on HuggingFace, model zoo, or any public registry under a name
  that implies it meets the project's own Phase-0 requirements.
- Referenced in any outreach email, access request, or collaboration
  pitch as representing the project's governed corpus capability.
- Used as a teacher for distillation into a released student.
- Cited as evidence that Phase 0 was satisfied.

Any external artifact that mentions this run must carry the tag
`legacy/internal/baseline` and include a one-paragraph disclosure of:

1. which Phase-0 requirements the run did NOT satisfy,
2. that shards carry `{text}` only, with no `source_id`, `date`, `rights`, or
   `document_id` provenance columns,
3. that the source mix was not governed against the plan's target mix.

## Why this framing is honest

- Shards contain only `text`. Provenance was dropped at pack time.
- No `rights_audit.py` has run. The "everything in `deduped/` is PD" belief
  is unaudited.
- No `date_audit.py` with a confidence hierarchy has run. Some sources have
  exact dates; several do not.
- Validation shard is a row-level last-shard split, not a document- and
  source-disjoint holdout.
- Source mix is estimated at ~56% newspapers / ~48% BHL science vs the
  plan target of 30/10.
- The governed mix (`source_mix.json`), the per-source held-out val set,
  and the anachronism/source-grounded QA evals all do not exist yet.

## 24-48 hour decision gate

See `decision_gate_2026-04-20.md`. At T+24h and T+48h from reclassification,
the gate will ask:

- Is Phase-0-lite producing governed shards on a 1-3 day timeline?
- Would a governed d22 run within ~30 days outperform `legacy_textonly_d22_r30`
  enough to justify stopping the legacy run at its next checkpoint?
- Or is the legacy run cheap enough to finish and keep as a pure pipeline
  validator?

Until the gate fires, **do not disturb the running process**. It is pid
`1641295`. Its outputs live under:

- log: `logs/d22_r30_20260420_090531.log`
- bench CSV: `logs/d22_r30_20260420_090531_bench.csv`
- checkpoints: `base_checkpoints/` (save_every=2000; first save at step 2000)
- token cache (READ-ONLY during this run):
  `/home/user/historical-nanochat/data/token_cache_v2/`

Any Phase-0-lite write must target new paths
(`token_cache_v3/`, `data/processed/governed/`, etc.) and must not touch the
live cache or live shards directory.

## Labels that ride with every downstream artifact

```
run_id:          legacy_textonly_d22_r30_internal_baseline
start:           2026-04-20T09:05:31Z
depth:           22
params:          615,645,184
target_ratio:    30 (Chinchilla)
total_tokens:    18,469,355,520
tokenizer:       rustbpe_32768 (/tokenizer/tokenizer.pkl)
token_cache:     token_cache_v2 (text-only, no provenance)
shard_schema:    {text}
governance:      NONE (no rights/date audit, no source_id retention)
use_as_teacher:  FORBIDDEN
use_in_release:  FORBIDDEN
use_in_eval:     AS BASELINE ONLY, labelled "legacy/internal/baseline"
```
