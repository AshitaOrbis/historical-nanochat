# Crash Postmortem — 2026-04-21 OOM cgroup cascade

- **Date / time of failure:** 2026-04-21 15:29:48 UTC (local 08:29:48)
- **Duration before noticed:** ~73 minutes (session resumed 09:42 local)
- **Scope of damage:** all Python processes running in the `tmux-spawn-468a2bf3` cgroup scope, including:
  - `legacy_textonly_d22_r30_internal_baseline` training (PID 1641295) at step 5249 of 70,455 (lost ~3,249 steps ≈ 13 h compute)
  - BHL item-level gate rerun (PID 4175203 — the OOM offender)
  - CAP full reharvest (PID 4176777)
  - BL Books harvester (PID 4176779)
  - Loader-wait monitor (PID 2268233)

## Root cause

`ShardWriter` in `data/phase0/process/repacker_v3.py` buffered output rows in a pure-Python `dict[str, list]` until either `target_rows_per_shard` (default 50,000) was reached OR the user explicitly called `flush()`. There was **no bytes-budget guard**.

The BHL deduped JSONL contains scientific monographs whose `text` fields average ~1 MB per record (and spike to many MB for multi-volume works). When `combined_audit_and_pack` was run with `--bhl-gate` on the full 214,921-record BHL JSONL, the ShardWriter accumulated ~45,900 admitted rows before its 50,000-row flush threshold was hit. That buffer held:

- anon-rss = **57,515,712 kB ≈ 57.5 GB**
- total-vm = **113,757,764 kB ≈ 113 GB**
- pgtables = 204 MB (page table overhead of holding that much text in Python lists)

The machine has 64 GB RAM. The kernel OOM killer selected PID 4175203 (the BHL gate, correctly identifying it as the offender):

```
Apr 21 08:29:48 requiem kernel: Out of memory: Killed process 4175203 (python)
  total-vm:113757764kB, anon-rss:57515712kB, file-rss:2032kB, shmem-rss:0kB,
  UID:1000 pgtables:199560kB oom_score_adj:0
```

## Blast radius — why training also died

systemd's default `OOMPolicy` for user scope units is `kill` (terminate the **entire** cgroup scope on any OOM kill within it). The tmux-spawn-468a2bf3 cgroup scope contained **all** processes launched from the same tmux pane, including training. At 08:31:18 — 90 seconds after the initial BHL kill — systemd brought down the rest of the scope:

```
Apr 21 08:31:18 requiem systemd[1492]: tmux-spawn-468a2bf3-6e06-4381-b8d6-93f6862c8024.scope:
  Failed with result 'oom-kill'.
```

**`nohup` and `disown` do not protect against cgroup-wide kills.** They prevent SIGHUP on terminal close; they don't exempt a process from systemd-scope lifecycle events.

## Fixes implemented

### Code fixes

1. **`ShardWriter.max_buffer_bytes = 500 MB`** (`data/phase0/process/repacker_v3.py`):
   the writer now flushes when EITHER `row_count >= target_rows_per_shard` OR `buffered_bytes >= max_buffer_bytes` — whichever trips first. A runaway text-heavy stream now flushes early.
2. **`--rows-per-shard 5000`** (down from the 50,000 default) for BHL item-level gate runs and all other long-text corpora (BHL, BL Books, CAP). Belt-and-suspenders against the bytes budget.
3. **Subset-policy (TCP) + item-level gate (BHL)** are now first-class in `combined_audit_and_pack.py` so the audit doesn't have to buffer extra decisions in memory.

### Operational fixes

4. **All CPU jobs launch via `systemd-run --user --scope --slice=background.slice`** with their own transient scope. Each job got its own unit name (`phase0-bhl-gate`, `phase0-cap-full`, `phase0-blbooks`, `phase0-cap-repack`, `phase0-cap-refine`, `phase0-blbooks-repack`). An OOM kill in one scope no longer cascades to any other scope.
5. **`OOMPolicy=continue`** is set per scope where supported, so even same-scope sibling processes survive a kill.
6. **`MemoryMax=8G-30G`** per CPU scope attempted (note: systemd --user transient scopes sometimes report `MemoryMax=infinity` due to delegation quirks; the bytes-budget fix at the ShardWriter level is the real safety net).

## What was salvaged

| Artifact | State |
|---|---|
| Step-2000 legacy checkpoint (`base_checkpoints/d22_r30/*_002000.*`) | intact + verified (12/12 pass on `tools/verify_first_checkpoint.py`) |
| v3 governed shards (4,451 parquet files in `data/phase0/governed_shards/`) | intact |
| `token_cache_v3` (66.69 B tokens, 4,451 shards) | intact |
| All v3 manifests + reports | intact |
| BHL item-level recovery per_record JSONL (214,921 entries) | intact |
| TCP v4 subset-policy shard (7,485 admitted) | intact |
| BL Books 1890s decade JSONL (2.48M pages) | intact (from earlier test) |
| BHL recovery reports + source_mix + decision_gate | intact |

## What was lost

| Artifact | Loss |
|---|---|
| Legacy training wall-clock | ~13 h of compute between step 2000 and crash at step 5249 |
| Partial CAP shards 0-9 output | rebuilt cleanly after restart |
| Partial BHL gate output (had written 0 final parquet; all counts were in-memory) | rebuilt cleanly |
| Partial BL Books 1510-1889 output (had only fetched 1510-1799 URLs; no JSONL written successfully due to 403 issue) | rebuilt cleanly with User-Agent fix |
| Loader-wait monitor jsonl post-crash | N/A; restarted fresh |

Nothing load-bearing was lost.

## What must be revalidated

Done as part of restart:

- [x] v3 shard count + manifest integrity (4,451 shards, `token_cache_v3_provenance.json` matches)
- [x] Step-2000 checkpoint state_dict round-trip (`verification_step002000_*.md` passes)
- [x] All restarted CPU jobs write cleanly and are still in an isolated scope
- [x] `ShardWriter` flush is triggered by both thresholds (verified via TCP smoke that wrote correctly)
- [x] BHL gate rerun completes without OOM (post-fix: 66,021 rows kept in 10.9 min, zero OOM events in journal)

## Prevention checklist (permanent)

Copy into every new corpus-building / long-running job's README or launch script:

1. **ShardWriter (or any row buffer) must flush on row limit OR byte limit.** Hard floor: 500 MB for text corpora. Will assert this at instantiation.
2. **ShardWriter must log buffer bytes periodically** (every N rows or every K MB) so we can diagnose memory growth in progress.
3. **Any full-corpus CPU job must have a MemoryMax or equivalent cgroup limit.** `--property=MemoryMax=8G` (or higher if justified). If the scope reports MemoryMax=infinity despite the flag (systemd user-scope quirk), make the bytes-budget fix at the application layer the authoritative defense.
4. **Any full-corpus CPU job must run outside the training cgroup/scope.** Preferred: `systemd-run --user --scope --slice=background.slice --unit=<name> --collect`.
5. **Training launch docs must warn that `nohup` and `disown` do not protect against cgroup-wide OOM kills.** They only protect against SIGHUP on terminal close. Use a dedicated systemd scope for training too (`--slice=compute.slice` or equivalent).
6. **Repacker and similar long-buffer code must refuse unbounded in-memory buffering.** Default: `max_buffer_bytes = 500 MB`. Trying to set it to `None` or `0` should raise.
7. **All corpus-building jobs must write to a `tmp/<job_id>/` staging path and atomically rename only after manifest validation passes.** Do not mix partial crash outputs with restarted outputs. (*New rule added 2026-04-21 post-crash.*)
8. **Any harvester that fetches public HTTP URLs must set a descriptive User-Agent header.** bl.iro.bl.uk returns 403 without one.
9. **Monitor the training log's `loader_pct` during any concurrent CPU work.** Pause / throttle if mean > 2% for >10 min.

## Lessons

- Python list-of-strings is a terrible memory container for large-text corpora. 45k × 1 MB text buffers = 45 GB anon-rss before any flush.
- Default systemd `OOMPolicy=kill` is a **session-level footgun** when you run multiple heavy jobs in the same tmux pane. Always spawn long-running jobs in their own scope.
- `nohup` is not a shield against cgroup lifecycle events — document this for future engineers.
- Fail closed in governance code (BHL per-item-rights) is correct for correctness but can produce severe corpus imbalance; always model the corpus impact before committing to the policy. Our recovery pass (preston → BHL item.txt join) turned the 0% science family share into 20%+ without loosening governance.
