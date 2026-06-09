# 24-48 Hour Decision Gate — legacy_textonly_d22_r30_internal_baseline

Created 2026-04-20 alongside the reclassification. Fires at:

- **T+24h:** 2026-04-21 ~15:00 UTC (informational; no action forced)
- **T+48h:** 2026-04-22 ~15:00 UTC (continue-vs-stop decision required)

## Scope

This gate governs the legacy baseline run only. It does NOT decide whether
to start a governed run — that decision depends on Phase-0-lite readiness
and belongs to a separate gate (Gate A from the action plan).

## Inputs the gate consumes

From the running baseline (read-only):

1. **Live step / token count** from `logs/d22_r30_20260420_090531.log`
2. **Train loss trajectory** — last 500 steps smoothed, slope test
3. **Validation BPB history** from the `Step NNNNN | Validation bpb: X` lines
   (eval_every=1000, expect ~48 val points in 48h)
4. **Benchmark CSV** `logs/d22_r30_20260420_090531_bench.csv`: tok/sec,
   loader %, peak VRAM
5. **First checkpoint verification** — run `tools/verify_first_checkpoint.py`
   against `base_checkpoints/d22_r30/step_02000/` as soon as the first save
   completes (expected ~step 2000, ~T+2h from reclassification due to the
   saves_every cadence)
6. **Sample probe output** — `report/baseline_samples/step_NNNNN/` every ~5000
   steps (first probe eligible at step 5000)

From the Phase-0-lite rebuild:

7. **Source registry status** — is `data/phase0/sources/registry.yaml`
   complete and validated?
8. **Rights audit status** — has `rights_audit.py` run to completion across
   deduped JSONL? Does it emit a keep/drop manifest with per-source token
   counts?
9. **Date audit status** — has `date_audit.py` assigned date buckets and
   flagged unknown-date records for exclusion?
10. **Repacker readiness** — has `repacker_v3.py` passed a smoke test on a
    small subset (e.g., oldbailey + caselaw) producing parquet with the full
    metadata schema?
11. **token_cache_v3 estimate** — given rights+date exclusions, what's the
    projected token count? How does it compare to v2's 25.2 B?
12. **Projected source mix after repack** — compared to the plan's target
    (40/30/15/10/5) how close can we get with the corpus we already have?

## Gate rules at T+48h

Decide **continue to completion** when ALL of:

- Train loss is descending (slope < 0 over last 5k steps, ignoring short-term
  oscillation).
- Val BPB is descending and is well below the init 4.996 (expected to be
  below 3.0 by step ~3-5k given the 3.18 reading at step 1000).
- No OOM, no NaN, no persistent compile recompile storms.
- Checkpoint save/reload round-trip verified at step 2000.
- EITHER: Phase-0-lite is NOT on track to produce governed shards within
  the 13-day baseline wall clock (i.e., parallel track will not beat this
  run's completion), so finishing the baseline costs nothing over
  restarting.
- OR: Phase-0-lite IS on track, but the governed run's token count after
  exclusions is projected at <10 B, which is too small to justify kicking
  off a parallel d22 training while the baseline still offers a loss
  reference.

Decide **stop at next checkpoint** when ANY of:

- Persistent instability (NaN, OOM after the run has been stable, tok/s
  drift > 10% negative trend, recompile storms, checkpoint corruption).
- Val BPB stalled or rising for > 3 consecutive evals (could indicate data
  loop, loss-chunked CE bug, or lr schedule mis-tuning).
- Phase-0-lite is producing governed shards within ~1-3 days AND projected
  token count is >=15 B AND projected source mix is materially closer to
  plan targets. In that case, the legacy run's 13-day remaining wall clock
  is burning GPU that could instead run the governed d22.
- User-level pivot (e.g., pivot away from 3090 budget, pivot to d16 or d20,
  pivot to a teacher-student schedule that needs provenance-carrying shards
  from the start).

Decide **continue but re-scope at next gate** when:

- Phase-0-lite state is mixed: registry + schemas done, rights audit done,
  but repacker untested; projected token count unclear. Re-check in 48h.

## Output of this gate

Write a one-page `report/decision_gate_report_<date>.md` containing:

- Section 1: live numbers (step, tokens, loss, val BPB, tok/s, peak VRAM).
- Section 2: Phase-0-lite readiness table (one row per item 1-9 in the
  plan priority order with status ∈ {todo, in_progress, done, blocked}).
- Section 3: projected post-audit token count and source mix.
- Section 4: decision — continue / stop / re-scope — with rationale.
- Section 5: if stop, the exact step number to stop at and the checkpoint
  path to preserve.

## Non-negotiables at the gate

- No decision made without the first-checkpoint verification having passed.
  If step 2000 save has failed or not completed by T+48h, the gate
  automatically becomes "debug first, decide later."
- No decision made without a fresh sample probe if the run has crossed
  step 5000.
- No decision that discards the legacy checkpoints without first archiving
  at least the last good checkpoint and its verification report, in case
  it proves useful as a pipeline baseline comparator.
