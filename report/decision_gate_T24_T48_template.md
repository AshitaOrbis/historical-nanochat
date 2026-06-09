# Decision Gate Report — `legacy_textonly_d22_r30_internal_baseline`

Generated: <UTC timestamp>
Gate checkpoint: T+<hh>h from reclassification (2026-04-20 15:20 UTC)

---

## 1. Live baseline numbers (read-only from the running process)

| Metric | Value | Source |
|---|---:|---|
| Step | `____` / 70,455 | `logs/d22_r30_20260420_090531.log` |
| Tokens seen | `____` B / 18.47 B | step × 262,144 |
| Train loss (last 500 steps mean) | `____` | log parse |
| Val BPB (last) | `____` | `Step NNNNN \| Validation bpb:` lines |
| Val BPB (trend last 3 evals) | ↓ / flat / ↑ | |
| tok/sec (last 500 steps mean) | `____` | bench CSV |
| Peak VRAM | `____` GiB | bench CSV |
| Wall time | `____` h | log |
| ETA | `____` h | log |

At reclassification (T+0): step 1406 / 70,455 (2.0%), loss ~3.8,
val BPB 3.18 at step 1000, peak VRAM 17.13 GiB, tok/sec 16.4k,
ETA ~13 days.

---

## 2. First-checkpoint verification (step 2000)

Run: `python tools/verify_first_checkpoint.py --step 2000 --device cpu`

| Check | PASS / FAIL / WARN | Detail |
|---|---|---|
| checkpoint files present | | |
| meta.json well-formed | | |
| state_dict round-trip | | |
| train loss descending | | |
| validation BPB trajectory | | |
| no OOM | | |
| no NaN | | |
| no compile recompile storm | | |
| bench CSV parseable | | |
| peak VRAM <= 22 GB | | |
| tok/sec stability < 2% | | |

Overall: `PASS / FAIL`

Artifact: `report/verification_step002000_<ts>.md`

---

## 3. Sample probe snapshot

Latest probe step: `step_<NNNNN>`
Location: `report/baseline_samples/step_<NNNNN>/`

Category takeaways (qualitative, labeled "legacy/internal/baseline"):

| Category | Sample behavior | Notes |
|---|---|---|
| victorian_prose | | |
| newspaper | | |
| bhl_science | | |
| legal_oldbailey | | |
| cutoff_anachronism | | |
| cutoff_insideperiod | | |

---

## 4. Phase-0-lite readiness

Priority order (as specified at reclassification). Status ∈
{todo, in_progress, done, blocked}.

| # | Artifact | Status | Path | Notes |
|---|---|---|---|---|
| 1 | source registry | done | `data/phase0/sources/registry.yaml` | 10 sources, 6 planned |
| 2 | unified doc + training schema | done | `data/phase0/schemas/` | pydantic, smoke-tested |
| 3 | rights audit (fail-closed) | done (smoke) | `data/phase0/process/rights_audit.py` | full-corpus run TBD |
| 4 | date audit + cutoff buckets | done (smoke) | `data/phase0/process/date_audit.py` | full-corpus run TBD |
| 5 | metadata-preserving repacker | done (smoke) | `data/phase0/process/repacker_v3.py` | full repack TBD |
| 6 | source_mix.json | done (smoke) | `data/phase0/manifests/source_mix.json` | full totals TBD |
| 7 | held-out val by doc/source | done | `data/phase0/process/split_holdouts.py` | chronicling_v2 full-source |
| 8 | anachronism + source_qa evals | done | `data/phase0/evals/` | prompts + runner ready |
| 9 | token_cache_v3 builder | done (stub) | `data/phase0/process/build_token_cache_v3.py` | provenance index tested |

What's blocking a governed run:

- **Full rights+date audit across 398 GB deduped/** — each source needs a
  full pass (not smoke). Estimated wall time: ~4-8 h single-threaded on
  CPU given the 398 GB size; most of that is I/O on the bhl and
  american_stories files. Can parallelize by source.
- **Full repack** — same scale. Estimated ~6-12 h on CPU.
- **token_cache_v3 build** — once repack is done, ~1-2 h on CPU (same
  order as the v2 cache build time).

---

## 5. Governed-corpus projections

Inputs (fill from `data/phase0/manifests/source_mix.json` after full repack):

| Family | Plan target | Projected share | Gap |
|---|---:|---:|---:|
| books_general | 35% | `____%` | `____` pp |
| newspapers_periodicals | 30% | `____%` | `____` pp |
| legal_government | 15% | `____%` | `____` pp |
| science_technical | 10% | `____%` | `____` pp |
| early_modern | 5% | `____%` | `____` pp |
| misc / holdout | 5% | `____%` | `____` pp |

Total projected governed token count: `____` B
(legacy v2 comparison: 25.2 B).

Materially closer to plan target? `yes / partially / no`

---

## 6. Decision

Choose ONE:

- [ ] **continue to completion** — legacy baseline finishes at step 70,455.
      Governed run is decoupled and waits for Phase-0-lite full repack.
- [ ] **stop at next checkpoint** — stop the legacy run at its next
      `save_every=2000` boundary. Preserve the last good checkpoint dir.
      Reallocate GPU to a governed d22 (or d20) run once token_cache_v3
      is ready.
- [ ] **re-scope at T+72h** — not enough evidence; gather another 24 h
      of Phase-0-lite progress and re-check.

Rationale:

> `<write 3-6 sentence justification citing the numbers above>`

If **stop at next checkpoint**, record:

- stop step: `____`
- preserved checkpoint: `base_checkpoints/d22_r30/step_<NNNNNN>/`
- archival tar: `<path>.tar.zst`
- who archives: `<person/cron>`

If **continue to completion**, record:

- ETA at completion: `____` UTC
- parallel governed track deadline: `____` UTC
  (must produce token_cache_v3 before legacy run ends so the next run
  can start immediately)

---

## 7. Non-negotiables recap

- Legacy checkpoints are NOT used as teachers.
- Legacy samples are labeled `legacy/internal/baseline`.
- Governed shards live under `data/phase0/governed_shards/` and never
  overwrite `data/shards/` or `token_cache_v2`.
- Any outreach email that references a model must refer to a governed
  model, not this baseline.
