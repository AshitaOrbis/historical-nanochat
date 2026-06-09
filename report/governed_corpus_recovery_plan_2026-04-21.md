# Governed Corpus Recovery Plan — 2026-04-21

Status: `governed_v3_newspaper_dominated_smoke_corpus` is built and smoke-tested end-to-end. It is **not** usable as-is for a release-candidate 3090 PoC run. This document inventories what recovery work lands us inside the user's restart thresholds and what the minimum reharvest is to satisfy the preferred target.

Hard restart thresholds (user directive):
- newspapers_periodicals ≤ 50%
- books_general ≥ 10%
- legal_government ≥ 5%
- science_technical ≥ 5%
- early_modern > 0%

Preferred target band:
- books 20-35%
- newspapers 30-45%
- legal 5-15%
- science 5-15%
- early_modern 2-5%

---

## 1. Current `governed_v3_newspaper_dominated_smoke_corpus` (exact tokenizer)

Source: `/home/user/historical-nanochat/data/token_cache_v3/cache_manifest.json`

- Total tokens: **66,692,365,636 (66.69 B)**
- Docs: 222,251,564 across 4,451 governed parquet shards
- Built in 3.65 h nice+ionice; no GPU interference

| family | tokens | share | vs floor |
|---|---:|---:|---|
| newspapers_periodicals | 63,334,441,117 | **94.97%** | +44.97 pp over 50% ceiling |
| early_modern | 2,043,451,654 | 3.06% | above 0% floor, in 2-5% preferred |
| books_general | 1,131,504,219 | 1.70% | -8.30 pp below 10% floor |
| legal_government | 182,968,646 | 0.27% | -4.73 pp below 5% floor |
| science_technical | **0** | **0.00%** | -5.00 pp below 5% floor |

Per-source detail:

| source | docs | tokens | share |
|---|---:|---:|---:|
| american_stories | 219,941,445 | 60,130,314,785 | 90.16% |
| bl_newspapers | 2,171,100 | 3,199,860,981 | 4.80% |
| eebo | 53,305 | 2,043,451,654 | 3.06% |
| gutenberg | 10,352 | 1,131,504,219 | 1.70% |
| oldbailey | 74,849 | 182,949,437 | 0.27% |
| chronicling_v2 | 500 | 4,249,907 | 0.01% |
| caselaw | 10 | 19,209 | ~0% |
| chronicling_america | 3 | 15,444 | ~0% |

---

## 2. BHL item-level rights recovery — **DONE**

Script: `data/phase0/process/bhl_rights_recovery.py`
Outputs: `manifests/bhl_rights_recovery.json`, `reports/bhl_rights_recovery_report.md`, per-record `bhl_rights_recovery.per_record.jsonl.gz`

Join chain implemented:
- preston provenance (archive.org preston-bhl, ~248 MB × 2 files, n-triples RDF)
  → IA identifier (via `hash://sha256/... <hasVersion> https://archive.org/download/<ia_id>/...` pattern)
- BHL item.txt (157 MB, BarCode / MARCItemID as join key)
  → CopyrightStatus + RightsStatement + LicenseType + Year

Classification (after v2 classifier patch — fixed "NOT_IN_COPYRIGHT" false positives):

| decision | rows | est. tokens |
|---|---:|---:|
| include_train_strict_pd | **87,094** | **18,188,976,946 (18.19 B)** |
| include_train_noncommercial_cc | 509 | 50,486,297 |
| quarantine_unknown_rights | 33,518 | 4,316,435,765 |
| quarantine_in_copyright | 711 | 91,747,714 |
| exclude_post_cutoff | 93,089 | 7,295,818,625 |

Rights basis: **item_metadata** (per-item CopyrightStatus + RightsStatement from BHL's canonical item.txt, joined via Preston's content-addressed dump manifest). This is exactly the basis the user required — not a blanket source-registry fallback.

The strict-PD pool is the "Public domain." / "Not in copyright." / "No known copyright restrictions." / "NOT_IN_COPYRIGHT" / "NIC" equivalence class. The NC-CC pool is tiny (509 items, 50 M tokens) — not worth a separate non-commercial variant unless we explicitly commit to a research-only corpus.

**Recoverable for release-candidate training: ~18.19 B tokens, strict PD.**

---

## 3. TCP recovery — **DONE (report only; patch + rerun pending)**

Outputs: `reports/tcp_recovery_report.md`

- ECCO-TCP: 2,474 rows → ~107.3 M tokens (chars/4) — source-level PD via Text Creation Partnership
- Evans-TCP: 5,011 rows → ~145.9 M tokens — source-level PD via TCP
- No other TCP subsets present in our harvest.

**Recoverable: ~253 M tokens (chars/4); expected ~220-260 M at actual tokenizer.**

Rights basis: **collection_policy** (both subsets are explicit PD releases from TCP / Oxford / Michigan).

Implementation requires a small patch to `rights_audit.py` to read subset-level policy from the registry when `per_item_rights_required=false`. About 1 hour of work + trivial re-run.

---

## 4. CAP reharvest — **PLAN ONLY; HARVEST DEFERRED**

Outputs: `reports/cap_reharvest_plan.md`

- Source chosen: `common-pile/caselaw_access_project` (HF, ungated, 175 shards of `cap_*.jsonl.gz`, CC0).
- Decision-date extraction: regex on first ~2000 chars of each opinion (month-name / numeric patterns).
- Expected pre-1914 cases: ~1 M based on CAP's public year distribution.
- **Token estimate: 500 M – 1.5 B tokens** (average opinion 500-1000 tokens).

Not executed this session because:
- Full download 60-80 GB while token_cache_v3 build was still writing 137 GB to disk.
- Needs a dedicated `cap_reharvest.py` module (~4-8 h of coding + testing).
- Better run overnight.

Rights basis: **government_work** (US court opinions not copyrightable) + **derived_dataset_policy** (common-pile / CAP CC0 labels).

---

## 5. Books recovery survey — **SURVEY ONLY**

Outputs: `reports/books_recovery_survey.md`

Recommended priority order:

| source | ungated? | license | token est. | effort |
|---|---|---|---:|---|
| **BL Books** (`TheBritishLibrary/blbooks` HF) | yes | CC0-1.0 | **~2.5 B** | 1-2 d |
| **LOC Selected Books** | yes (LOC data.gov) | PD / gov | ~1-3 B | 2-4 d |
| **Internet Archive PD books** | yes | per-item | 5-20 B | 3-5 d + dedupe |
| HathiTrust | partial | mixed | 10-30 B | BLOCKED on access |
| Gutenberg (expansion) | already | PD | +0.5-1 B | minimal |

Minimum realistic books recovery in 1-2 weeks: **BL Books (~2.5 B) + 1 IA PD books scrape (~3 B) = ~5.5 B book tokens** on top of existing 1.13 B from Gutenberg → **~6.6 B books total**.

---

## 6. Projected source mix under 4 scenarios

All projections use the exact tokenizer for the existing v3 data plus chars/4 estimates for recoveries.

### Scenario A — **Do nothing** (current v3 corpus)

| family | tokens | share | meets floor? |
|---|---:|---:|---|
| newspapers | 63.33 B | **94.97%** | NO (>50%) |
| science | 0 | 0.00% | NO (<5%) |
| books | 1.13 B | 1.70% | NO (<10%) |
| legal | 0.18 B | 0.27% | NO (<5%) |
| early_modern | 2.04 B | 3.06% | yes |

**Verdict: does NOT meet thresholds. No restart authorized.**

### Scenario B — **BHL + TCP only** (the cheapest recoveries)

Assumes BHL strict-PD 18.19 B + TCP 0.25 B added.

| family | tokens | share | meets floor? |
|---|---:|---:|---|
| newspapers | 63.33 B | 74.6% | NO (>50%) |
| science | 18.19 B | 21.4% | yes (too high for 5-15% preferred) |
| early_modern | 2.29 B | 2.7% | yes |
| books | 1.13 B | 1.3% | NO (<10%) |
| legal | 0.18 B | 0.2% | NO (<5%) |

**Verdict: meets science + early_modern floors, fails newspapers/books/legal. No restart.**

### Scenario C — **BHL + TCP + CAP + BL Books** (moderate effort, 1-2 wks)

Assumes BHL 18.19 B, TCP 0.25 B, CAP 1.0 B (midpoint), BL Books 2.5 B.

| family | tokens | share | meets floor? |
|---|---:|---:|---|
| newspapers | 63.33 B | 71.0% | NO (>50%) |
| science | 18.19 B | 20.4% | yes |
| books | 3.63 B | 4.1% | NO (<10%) |
| early_modern | 2.29 B | 2.6% | yes |
| legal | 1.18 B | 1.3% | NO (<5%) |

**Verdict: still fails newspapers + books + legal. No restart.**

### Scenario D — **Scenario C + IA PD books 5 B + CAP 3 B + american_stories downsample 0.3** (aggressive, 2-3 wks)

Assumes IA PD books 5 B, CAP reharvested to 3 B (upper realistic), AS subsampled to 30% of 60.13 = 18.04 B.

| family | tokens | share | meets floor? | preferred? |
|---|---:|---:|---|---|
| newspapers | 21.24 B | **40.0%** | yes | yes (30-45%) |
| science (BHL) | 18.19 B | **34.3%** | yes | NO (too high; 5-15% preferred) |
| books | 8.63 B | **16.3%** | yes | NO (on edge of 20-35%) |
| early_modern | 2.29 B | 4.3% | yes | yes (2-5%) |
| legal | 3.18 B | 6.0% | yes | yes (5-15%) |

Total: **53.53 B tokens**.

**Verdict: meets ALL floors. Science too high for preferred band — would downsample BHL to 12 B to land science at ~22%. A 3090 d22 run at Chinchilla ratio 30 needs 18.47 B tokens; 53 B is more than 2× that, giving room for holdouts and one-epoch training.**

### Scenario E — **Preferred-zone target** (full reharvest + dedupe, 3-4 wks)

Assumes BHL downsampled to 12 B, AS 0.3× (18.04 B), CAP 3 B, BL Books + LOC + IA dedupe 15 B, TCP 0.25 B, eebo+gutenberg unchanged (3.17 B).

| family | tokens | share | in preferred band? |
|---|---:|---:|---|
| newspapers | 21.24 B | 41.0% | yes (30-45%) |
| books | 16.13 B | 31.1% | yes (20-35%) |
| science (BHL) | 12.00 B | 23.1% | NO (>15%) |
| legal | 3.18 B | 6.1% | yes (5-15%) |
| early_modern | 2.29 B | 4.4% | yes (2-5%) |

Total: **51.84 B tokens.** Could downsample BHL further to 8 B to land science at ~16%; or keep 12 B and accept a slightly science-heavy but preferred-zone corpus. Acceptable either way.

---

## 7. Recommendation

1. **Do not restart training on v3.** `governed_v3_newspaper_dominated_smoke_corpus` stays labeled as a smoke/validation artifact only.
2. **Legacy baseline (`legacy_textonly_d22_r30_internal_baseline`) continues to completion** (ETA ~12.5 d remaining) unless it becomes unstable. It remains the training-stack + loss-curve reference.
3. **Immediate CPU priorities (parallel to legacy baseline):**
   - a. Patch `rights_audit.py` + `combined_audit_and_pack.py` to read the subset-level TCP policy; re-run TCP (trivially fast, <1 min).
   - b. Re-run combined pass for **BHL at item-level** using `bhl_rights_recovery.per_record.jsonl.gz` as the keep/drop gate. This adds `science_technical ≈ 21.4%` to the governed corpus. Expected wall: ~1-2 h nice+ionice.
   - c. Implement `cap_reharvest.py` and run the CAP reharvest overnight.
   - d. Implement the BL Books harvester (via HF snapshot_download + raw file URLs; NOT the legacy loader script).
4. **Target Scenario D or E before a governed long run.** The minimum-acceptable target is **Scenario D** (all floors met, science a bit high). Scenario E is only worth pursuing if a 4-week harvest window exists.
5. **When all four recoveries land:**
   - a. Re-run `combined_audit_and_pack` across the new sources.
   - b. Re-build `token_cache_v3 → token_cache_v4` with the corrected mix. Label the new cache `governed_v4_balanced_candidate`.
   - c. Re-run the CPU dataloader smoke.
   - d. Run a short GPU smoke (e.g. 1000 steps at d22) to verify the new cache feeds correctly.
   - e. At that point, a governed long run is authorized.

---

## 8. What was NOT attempted this session

- Actual CAP download. Plan documented, harvester module not yet written.
- BL Books / LOC / IA downloads. Survey only.
- Rewriting `rights_audit.py` for per-subset policy. Design documented in tcp_recovery_report.md.
- Building `governed_v4_balanced_candidate` token cache. Requires recoveries above to land first.
- GPU smoke test against v3. Legacy baseline is still on the GPU; will queue this for after recoveries land or after legacy baseline ends.

---

## 9. Companion artifacts

| Artifact | Path |
|---|---|
| Corpus label (NOT a release candidate) | `data/phase0/LABEL.md` |
| BHL item-level classifications | `data/phase0/manifests/bhl_rights_recovery.json` |
| BHL per-record decisions | `data/phase0/manifests/bhl_rights_recovery.per_record.jsonl.gz` |
| BHL recovery report | `data/phase0/reports/bhl_rights_recovery_report.md` |
| TCP recovery report | `data/phase0/reports/tcp_recovery_report.md` |
| CAP reharvest plan | `data/phase0/reports/cap_reharvest_plan.md` |
| Books recovery survey | `data/phase0/reports/books_recovery_survey.md` |
| Current v3 source_mix | `data/phase0/manifests/source_mix.json` |
| Current v3 split_manifest | `data/phase0/manifests/split_manifest.json` |
| Current v3 cache manifest | `data/token_cache_v3/cache_manifest.json` |
| Combined pass summary | `data/phase0/reports/combined_audit_and_pack_summary.json` |

---

## 10. Training and disk state at report time

- Training: step 5010 / 70,455 (7.11%), loader_pct_mean 0.146%, tok/s 16,353, loss 3.48 (mean last 98), ETA ~12.1 d remaining. No warnings.
- Disk: 159 GB free after v3 cache build (66.69 B uint16 tokens = 133 GB).
- No OOM, NaN, compile storms, or checkpoint anomalies since reclassification.
