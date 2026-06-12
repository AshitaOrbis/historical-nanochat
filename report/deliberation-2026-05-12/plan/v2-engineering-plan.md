# V2 Engineering Plan for Historical-Nanochat

Prepared: 2026-05-14

Scope: concrete pre-cloud engineering plan for a single Lambda 8xH100 SXM nanochat d24 run on the 1913 corpus. The locked design is accepted: 1913 cutoff, pre-WWI d24 cloud run, Talkie-1930 plus modern reference anchors, one cloud-train run, and possible but not mandatory Token Superposition Training (TST).

Evidence checked locally:

- `report/governed_v4_run1_postmortem_detailed.md`: stale provenance served only 16.5% of cache with 19.2% correct family tags while training metrics looked healthy.
- `report/deliberation-2026-05-12/round1-validity/empirical-findings.md`: current headline validation reads 262,144 tokens from the first Gutenberg val shard, not the full held-out split.
- `nanochat/nanochat/dataloader_cached.py`: current `parallel_family_cache` still raises for `world_size > 1`, so 8xH100 launch needs loader work.
- `tokenizer/tokenizer_manifest.json`: tokenizer input is `data/shards`, not proven final train-only v4 data.
- `data/processed/corpus_1913_v4_balanced_candidate/reports/source_mix.md`: actual train mix is about 40.27 / 20.13 / 20.40 / 10.07 / 9.13, with several important val-only sources.
- `data/phase0/README.md`: cross-source SimHash/MinHash dedup is explicitly not implemented.

External checks:

- Lambda current pricing page lists 8x H100 SXM at `$3.99/GPU/hr`, before tax: https://lambda.ai/pricing
- Lambda docs say persistent storage must be attached before instance start; data outside the mount is erased on termination: https://docs.lambda.ai/public-cloud/on-demand/getting-started/
- TST paper is arXiv 2605.06546, submitted 2026-05-07, with two-phase superposition plus recovery and up to 2.5x equal-loss time reduction at 10B A1B scale: https://arxiv.org/abs/2605.06546
- Talkie base anchor exists as `talkie-lm/talkie-1930-13b-base`, 13B, 260B pre-1931 English tokens, Apache-2.0: https://huggingface.co/talkie-lm/talkie-1930-13b-base
- nanochat d24 public reference reports 1.384B params, 8.8B tokens, 3.04h on 8xH100: https://github.com/karpathy/nanochat/discussions/481

## 1. Consensus

1. **Do not launch the d24 cloud run from the current v1/v4 path.** The current cache/provenance failure class has been partially guarded, but the data identity model is still rooted in mutable `shard_index`, and the 8-GPU path is not implemented for `parallel_family_cache`. [unanimous]

2. **The single most important pre-run task is cache/provenance redesign.** Stable IDs must be rooted in `source_file` plus document/segment/content identity, not position. `split_cache_shards.py` and `shuffle_cache_manifest.py` must regenerate or invalidate provenance in the same operation. [unanimous]

3. **Document-level split semantics and source/family metadata must survive tokenization and cache transforms.** Without `document_id`, `source_id`, `family`, `split`, tokenizer hash, and token-span metadata in or beside the cache, dedup, eval, and probes cannot be trusted. [unanimous]

4. **Per-family validation is load-bearing.** The present `val_bpb` is a real learning signal on a fixed Gutenberg prefix, but it is not a broad validation metric. The launch gate must be per-family bpb across all five families and multiple shards, first run on the existing v1 615M checkpoint. [skeptic, architect, risk analyst, empiricist]

5. **Targeted cross-source near-duplicate dedup must precede launch.** The minimum target set is `american_stories` train vs `chronicling_v2`/Chronicling America val, BL Books/TCP/EEBO classics and editions, and long works split across shards. Full global dedup is not required before this run. [unanimous]

6. **Accept the actual corpus mix unless dedup materially damages it.** Do not rebuild to match the older brief percentages. Make the actual mix the canonical v2 mix and train schedule, unless high-confidence dedup removes enough material to shift family shares by more than about 2-3 percentage points or leaves a family under-supported. [unanimous]

7. **Tokenizer audit is mandatory; tokenizer regeneration is conditional.** The existing tokenizer corpus is not proven final train-only v4. If it saw post-1913/modern material, regenerate. If it only saw pre-1914 train+val-style material, the cloud run can proceed with disclosure, but the project must not claim "governance-clean synthetic data generator" until train-only tokenizer provenance is clean. [skeptic, architect, risk analyst, empiricist]

8. **Cutoff policy should be publication-year cutoff plus disclosure and probes, not a broad pre-run semantic filter.** Content-level filtering for all anachronism is too open-ended and may delete legitimate pre-1914 anticipation. Do only high-precision modern-paratext quarantine before training. Move semantic anachronism to eval. [unanimous]

9. **TST is a gated optimization branch, not a launch dependency.** The paper is promising, but this run's main risk is validity and data path integrity, not a `$96` GPU bill. TST must pass a fixed-budget smoke from step 0, with phase metadata and resume checks, or be dropped. [unanimous]

10. **Anchor logistics need adapters, not pre-run full results.** Talkie and the modern reference must be identified and the probe/logprob interface must exist before launch, but full anchor evaluation can finish after the cloud checkpoint exists. Do not headline raw bpb comparisons across different tokenizers. [skeptic, architect, risk analyst, empiricist]

11. **Cloud logistics are a hard launch gate.** Persistent filesystem, checkpoint sync, tmux strategy, upload hashes, and restore proof must be verified before the full run. [unanimous]

## 2. Disagreements

### Disagreement 1: narrow launch gate vs governance-first rebuild

**The disagreement:** Should the pre-cloud plan be a minimal set of blockers, or a fuller governance-first rebuild before any Lambda run?

**Skeptic's strongest argument:** The cloud run is cheap and short. The scarce resource is single-author attention. The right gate is "do not produce unusable weights," not "solve historical-LM measurement." Under this view, block only on cache truthfulness, bounded leakage, a working eval harness, and checkpoint safety. Full semantic filtering, corpus-mix correction, full anchor runs, and TST productionization are scope creep.

**Architect's strongest argument:** The 8xH100 run is not just a 3-hour command. The current loader path is single-GPU only, cache identity still depends on a known-bad foreign key, and synthetic-data ambitions require governance-clean instrumentation. A fast launch that cannot prove document splits, source/family metadata, DDP family mix, or per-family eval would repeat the v1 pattern at larger scale.

**Adjudication:** The Architect is stronger on actual launch blockers because `parallel_family_cache` currently refuses `world_size > 1` and because provenance remains structurally coupled to mutable `shard_index`. The Skeptic is stronger on rejecting open-ended scientific cleanup. The winning plan is a bounded governance rebuild: stable cache identity, document metadata, targeted dedup, DDP loader, per-family eval, and cloud preflight are critical path; semantic filtering, old-mix rebuild, full anchors, and synthetic generation are deferred.

### Disagreement 2: tokenizer acceptance vs train-only tokenizer

**The disagreement:** Is a tokenizer audit plus disclosure enough, or should a train-only tokenizer be mandatory before cloud launch?

**Acceptance side, strongest argument (Skeptic/Risk Analyst):** Tokenizer training on validation-like pre-1914 text is a measurement confound, not the same as weight training on validation text. If the tokenizer did not see post-cutoff or modern text, retraining it can consume time and force cache regeneration without changing the main cloud-run validity decision. Accept and disclose if the audit is clean enough.

**Regeneration side, strongest argument (Architect/Empiricist):** The tokenizer is part of the measurement system. If v2 is supposed to be the instrument for future trustworthy synthetic data, tokenizer provenance should be train-only and tied to the frozen corpus. Otherwise val bpb and probes inherit an avoidable alignment/leakage caveat.

**Adjudication:** Conditional. For the cloud base-model run, audit is the blocker and regeneration is required only if the tokenizer saw post-1913/modern material or cannot be reconstructed. For "governance-clean synthetic data generator" claims, train-only tokenizer provenance is mandatory before making that claim. This preserves the Skeptic's scope control while honoring the Architect's governance standard.

### Disagreement 3: near-duplicate threshold

**The disagreement:** What level of train-val near-duplicate overlap should block launch?

**Strict side, strongest argument (Risk Analyst/Empiricist):** Historical newspapers and editions have high base rates for syndication and repeated public-domain text. If even about 1% of val tokens are high-confidence near-duplicates of train, headline evaluation can be distorted, especially on small family slices. A low threshold forces early triage.

**Looser side, strongest argument (Skeptic/Architect):** Some repeated text is historically meaningful. Over-aggressive dedup can delete syndication and classic-edition circulation, which are part of the historical record. A hard low threshold may turn dedup into corpus redesign.

**Adjudication:** Use two thresholds. If targeted high-confidence train-val overlap exceeds 1% of val tokens in any affected slice, pause launch and triage with sampled false-positive review. If after quarantine more than 5% high-confidence train-val overlap remains, or if cleanup shifts a family by more than 2-3 percentage points, block launch and rebuild the split. This is stricter for eval leakage while avoiding deletion of historically meaningful reuse without review.

### Disagreement 4: TST integration

**The disagreement:** Should TST be integrated before this one cloud run because it must start at step 0?

**Pro-TST side, strongest argument (Architect):** If TST works, this is the only chance to use it on the d24 run. The paper reports meaningful time reduction without changing architecture, tokenizer, optimizer, data, or parallelism. A clean feature-flagged implementation could buy speed or quality.

**Anti-TST side, strongest argument (Skeptic/Risk Analyst/Empiricist):** TST changes the training objective and creates a new failure surface. This run's bottleneck is interpretability, not compute cost. A failed TST run would be harder to diagnose because corpus, loader, objective, and eval would all be changing together.

**Adjudication:** TST may run only as a parallel spike with a hard deadline and explicit kill criteria. It must not delay cache/provenance, DDP loader, dedup, eval, or cloud preflight. If not green by the end of week 2, standard d24 wins.

### Disagreement 5: modern reference choice

**The disagreement:** Should the modern reference be a nanochat d24/FineWeb checkpoint or a Talkie-modern sibling such as `talkie-web-13b-base`?

**nanochat-reference side, strongest argument (Skeptic/Architect):** The cloud run is nanochat d24. A modern nanochat/FineWeb-style reference best isolates corpus/cutoff effects from architecture/training-stack effects, if weights and tokenizer provenance are available.

**Talkie-modern side, strongest argument (Empiricist):** Talkie-1930 and a Talkie-web sibling would give a cleaner historical-vs-modern pair inside the same external model family, avoiding a three-way architecture mismatch.

**Adjudication:** Pick the modern anchor by adapter quality, not aesthetics. Preferred: a public nanochat d24/FineWeb checkpoint with clear weights/tokenizer and logprob access. Fallback: Talkie-web base if it gives a cleaner paired historical/modern comparison. In either case, probes must use model-native tokenizers and byte-normalized logprob reporting; raw bpb across tokenizers is not a headline metric.

## 3. Open Questions

1. **What exact documents trained the 32,768-vocab BPE?** Surfaced by Empiricist and reinforced by Architect. The manifest path is not enough; the script and input snapshot must be reconstructed.

2. **How much targeted near-duplicate leakage actually exists?** Surfaced by all agents. The project has not measured `american_stories` train vs Chronicling-style val overlap, multi-edition classics overlap, or long-work shard leakage.

3. **What is the best DDP family-balanced loader design?** Surfaced most concretely by Architect. Current `parallel_family_cache` is single-GPU. The design must prove per-rank striping, resume determinism, and expected family mix.

4. **Can document-level split semantics be recovered cheaply from current artifacts, or does v2 require retokenization from parquet?** Synthesis-emergent. If current cache lacks enough doc-span metadata, cache rebuild becomes unavoidable.

5. **Which modern reference is actually available with usable likelihood/logprob evaluation?** Surfaced by Architect and Empiricist. This should be resolved before probe implementation, but full anchor eval can follow cloud training.

6. **What probes will Pro recommend, and do they require generation, likelihood, calibration, or source-grounded scoring?** Surfaced by the original query and Skeptic/Empiricist. Reserve adapter time now; do not block on final probe content.

7. **Will Lambda have 8xH100 SXM availability on launch day?** Surfaced by Risk Analyst and cloud docs. Verify day-of and keep RunPod Secure Cloud fallback ready.

8. **Can TST be implemented without contaminating the standard d24 result?** Surfaced by all agents. This is irreducible until a local smoke proves phase switching, resume, throughput, and no obvious eval regression.

9. **Is the immediate goal a credible pre-WWI base checkpoint or a governance-clean synthetic-data generator?** Synthesis-emergent. The first can launch after bounded audit/disclosure; the second requires stricter train-only tokenizer and provenance standards before claims or generation.

## 4. Final Recommendation

Build a conservative V2 cloud-run package and launch one standard nanochat d24 run only after the data identity, DDP loader, targeted dedup, per-family eval, and cloud-persistence gates pass. Treat TST as a separate deadline-bound optimization spike. Do not rebuild the corpus to match old mix numbers. Do not do full semantic content filtering before launch. Do not headline raw cross-model bpb.

Confidence: **high** that this direction is right, because every agent converged on the same core blockers and local evidence confirms the two most serious ones: stale-provenance risk and missing multi-GPU support. Confidence is **medium** on the exact timeline, because tokenizer regeneration, dedup severity, and DDP loader complexity are open variables.

Conditions that would change the recommendation:

- If targeted dedup finds >5% unremediated high-confidence train-val overlap, rebuild the split before launch.
- If tokenizer audit shows post-1913 or modern text in tokenizer training, regenerate tokenizer and cache before launch.
- If DDP family-balanced loader cannot pass 8-rank smoke quickly, either use a different governed DDP-safe loader with measured family mix or delay the cloud run.
- If TST passes all gates by the fixed deadline with >=1.25x measured wall-clock gain and no unclear bpb/probe regression, it can become the cloud-run path. Otherwise standard d24 remains the path.
- If the researcher insists the next artifact must generate trustworthy synthetic data immediately, raise the gate: train-only tokenizer, stricter provenance, and synthetic-data eval move from deferrable to critical.

Cheapest decisive next action: **write and implement the V2 cache/provenance contract, starting with stable source-file-rooted IDs and automatic provenance regeneration/invalidation in `split_cache_shards.py` and `shuffle_cache_manifest.py`.** This unblocks dedup keys, tokenizer/cache rebuild decisions, DDP loader audits, per-family eval, and launch records.

### 4.1 Critical Path

These tasks are the shortest sequence that must complete before the full Lambda d24 run can launch.

| # | Task | Output artifact | Dependencies | Estimate | Kill criterion |
|---:|---|---|---|---:|---|
| 0 | Freeze V2 run contract | `report/deliberation-2026-05-12/plan/v2-run-contract.md`: claims, non-claims, corpus version, cutoff policy, launch gates | locked docs | 2-3 human h | If it still says "validated historical LLM" or "cannot know post-1914", rewrite before coding |
| 1 | V2 cache/provenance contract | `docs/v2_cache_provenance_contract.md` and JSON schema for stable IDs, doc spans, source/family/split metadata, tokenizer hash | Task 0 | 5-8 human h | If IDs still depend on mutable `shard_index`, no launch |
| 2 | Patch provenance-mutating scripts | Updated `build_token_cache_v4.py`, `split_cache_shards.py`, `shuffle_cache_manifest.py`, plus refusal tests | Task 1 | 14-22 human h, <2 CPU h tests | If split/shuffle can still rewrite manifest without regenerating or invalidating provenance, no launch |
| 3 | Preserve document split and metadata through cache | Cache sidecars or rebuilt cache metadata with `document_id`, `source_id`, `family`, `split`, token spans, content/doc digest | Tasks 1-2 | 10-18 human h, 4-16 CPU h | If document-level split semantics disappear after tokenization, no launch |
| 4 | Tokenizer audit decision | `report/deliberation-2026-05-12/plan/tokenizer_v2_audit.md`; optionally `tokenizer_v2_trainonly/` | Task 0; before final cache build | 2-4 human h audit; conditional 6-12 human h and 4-12 CPU h rebuild | If tokenizer saw post-1913/modern material or cannot be reconstructed, regenerate or block governance-clean claims |
| 5 | Corpus policy lock | `report/deliberation-2026-05-12/plan/v2_corpus_card.md`: actual mix, train schedule, val-only sources, cutoff disclosure | Tasks 0-4 | 3-5 human h | If schedule contradicts the frozen corpus card or wraps a family heavily without disclosure, revise |
| 6 | Targeted cross-source near-duplicate dedup | `report/deliberation-2026-05-12/plan/dedup_v2_policy.md`, `dedup_audit.jsonl`, quarantine/remap manifest | Tasks 1-5 | 12-22 human h, 6-24 CPU h | Pause at >1% high-confidence overlap in any val slice; block at >5% remaining overlap or >2-3 pp family-share shift |
| 7 | Rebuild final V2 cache and audit | `data/token_cache_v2_1913_cloud/{train,val}`, manifests, provenance, hashes, audit JSON | Tasks 2-6 | 4-8 human h, 8-24 CPU h | Any missing family/source/split metadata, tokenizer SHA mismatch, unstable hash, or guard mismatch blocks launch |
| 8 | DDP-safe family-balanced loader | Patched loader, unit tests, resume-state tests, measured 8-rank family-mix smoke plan | Task 7 | 12-20 human h, local GPU optional | If 8-rank striping/resume cannot preserve expected family mix, no 8xH100 launch |
| 9 | Per-family eval harness | `tools/eval_per_family_bpb.py`; v1 615M baseline JSON across all five families and multiple shards | Tasks 7-8 | 8-14 human h, 1-3 GPU h | If v1 cannot be evaluated reproducibly per family, do not launch d24 |
| 10 | Probe and anchor interface contract | `eval/anchors.yaml`, `eval/probes/schema.json`, raw-text logprob/generation adapter stubs | Task 9 interface | 4-8 human h | If probes cannot run through a model-native adapter on v1, defer probe claims and narrow launch record |
| 11 | Cloud runbook and preflight scripts | `cloud/lambda_bootstrap.sh`, `cloud/run_d24.sh`, `cloud/sync_checkpoints.sh`, `ops/lambda_d24_runbook.md` | Tasks 7-10 | 6-10 human h, 1-2 CPU/upload h | If checkpoint path is not persistent and off-node synced, no launch |
| 12 | Lambda 8-GPU smoke | 50-100 step smoke report with startup guards, DDP mix, checkpoint save/sync/restore proof | Tasks 8-11 | 1-2 human h, 0.25-0.5 H100 h | Any failed guard, wrong family mix, no restorable checkpoint, or wrong storage path blocks full run |
| 13 | Full cloud launch | Lambda 8xH100 d24 run plus post-run per-family eval and launch record | Tasks 0-12 | 1-2 human h active, 3-6 H100 h | Abort if preflight hashes differ, storage path is ephemeral, or run diverges/NaNs |

### 4.2 Parallel Tasks

These can run alongside the critical path but must not delay it.

| Task | Output artifact | Dependencies | Estimate | Kill criterion |
|---|---|---|---:|---|
| TST gate | `report/deliberation-2026-05-12/plan/tst_gate.md`; optional feature branch/flag | final cache/eval interface preferred | Audit 4h; implementation 12-24h; 2-6 GPU h | Kill if >24 human h, no finite 1k-step smoke, no clean phase resume, <1.25x wall-clock gain, or unclear bpb/probe regression |
| Talkie anchor logistics | Adapter smoke for `talkie-lm/talkie-1930-13b-base`; precision note for logprobs | probe schema | 4-8 human h | If reliable logprobs are unavailable within 4h, defer to generation-only probes for Talkie |
| Modern reference logistics | Chosen modern anchor and adapter smoke | probe schema | 4-8 human h | If no clear checkpoint/tokenizer/logprob path, use a declared unmatched reference and avoid direct bpb claims |
| High-precision cutoff/paratext scan | `report/deliberation-2026-05-12/plan/v2_cutoff_scan.md` | corpus card | 4-8 human/CPU h | If precision is low or review burden exceeds one weekend, move to probes/disclosure |
| Pro probe reserve | JSONL schema, scoring harness stub, 6-10h capacity reserved after Pro output | eval interface | 6-10 human h | If Pro probes require unsupported APIs, reduce to the common model-native subset |
| RunPod fallback | `ops/runpod_fallback.md` with network-volume path and launch checklist | cloud runbook | 3-5 human h | If Lambda is available and smoke passes, keep as contingency only |

### 4.3 Deferrable Tasks

- Full semantic content-anachronism filtering.
- Full all-pairs/global near-duplicate dedup beyond targeted leakage checks.
- Rebuilding the corpus to match the old 17.5% / 8.2% brief mix.
- Moving val-only sources into train to improve source diversity.
- SFT, RLHF, chat tuning, synthetic data generation, or public assistant demo.
- Hugging Face release polish, model card publication, and broad public writeup.
- 2B/3B expansion, isoFLOP scaling grids, or Chinchilla-applicability pivot.
- Full Talkie and modern anchor evaluation before cloud training, as opposed to adapter readiness.

### 4.4 Explicit Rejects

- Do not launch the current v1/v4 cache plus current single-GPU `parallel_family_cache` on 8 GPUs.
- Do not make TST mandatory.
- Do not spend cloud time debugging tokenizer/provenance/cache contracts.
- Do not compare raw bpb across custom tokenizer, Talkie, and modern reference as a headline result.
- Do not run broad semantic filtering that deletes legitimate pre-1914 anticipation of later events.
- Do not rebuild the corpus only to match stale brief percentages.
- Do not trust aggregate `val_bpb` as a launch gate.
- Do not start synthetic data generation before v2 provenance and eval are green.

### 4.5 Consolidated Kill Criteria

Cloud no-go if any remain true:

- Manifest/provenance linkage still depends on mutable `shard_index`.
- `split_cache_shards.py` or `shuffle_cache_manifest.py` can stale provenance.
- Document-level split and source/family metadata cannot be audited through tokenization.
- Tokenizer provenance is unknown and cannot be disclosed precisely.
- Targeted high-confidence train-val duplicates remain above the agreed threshold.
- DDP loader cannot measure and resume the intended family mix across 8 ranks.
- Per-family eval cannot run reproducibly on all five families.
- Lambda smoke cannot save, sync, and restore a checkpoint from persistent storage.
- Public or internal launch contract still implies "validated historical LLM" or "post-1914 ignorance proved."

### 4.6 Timeline

Recommended standard path, without TST production integration and without tokenizer regeneration: **75-105 human hours**, **1-3 CPU days**, **2-5 local GPU hours**, **0.25-0.5 H100 smoke hours**, then **3-6 H100 hours** for the final run. At single-author evening/weekend pace: **4-6 calendar weeks**.

If tokenizer regeneration or a clean TST integration is added: **90-130 human hours** and **6-8 calendar weeks**. Set the TST deadline at the end of week 2; after that, standard d24 wins.

If targeted dedup finds major leakage or the DDP loader is harder than expected: stop and replan rather than renting the node anyway.

### 4.7 Single Most Important Next Task

**Implement the V2 cache/provenance contract.** Specifically: define stable IDs rooted in `source_file` plus document/segment/content hash; carry document split, family, source, tokenizer hash, and token spans through cache; patch `split_cache_shards.py` and `shuffle_cache_manifest.py` so they regenerate or invalidate provenance automatically; add startup refusal tests that reproduce the stale-provenance bug class.

Until this is done, dedup reports, tokenizer/cache rebuilds, per-family eval, DDP loader tests, and cloud launch records all rest on the same class of failure that already wasted 45.8 GPU hours.

## 5. Attribution Map

| Claim | Contributing agents |
|---|---|
| Cloud run should not launch from the current data path | Skeptic, Architect, Risk Analyst, Empiricist |
| Cache/provenance redesign is the top blocker | Skeptic, Architect, Risk Analyst, Empiricist |
| Stable IDs need `source_file` plus doc/segment/content identity, not only `source_file` or `shard_index` | Skeptic, Architect, Risk Analyst, Empiricist |
| `split_cache_shards.py` and `shuffle_cache_manifest.py` can re-induce the stale-provenance class | Skeptic, Architect, Risk Analyst, Empiricist |
| DDP loader support is a real 8xH100 blocker | Architect; synthesis verified in local code |
| Per-family eval on all five families is load-bearing | Skeptic, Architect, Risk Analyst, Empiricist |
| Current validation bpb is narrow and should not be a headline | Skeptic, Risk Analyst, Empiricist; inherited from locked validity synthesis |
| Targeted cross-source near-duplicate dedup is required; global dedup is deferrable | Skeptic, Architect, Risk Analyst, Empiricist |
| Tokenizer audit is mandatory; regeneration is conditional for cloud launch but mandatory for strong governance-clean synthetic-data claims | Architect, Empiricist, Skeptic, Risk Analyst; synthesis adjudication |
| Accept actual corpus mix; do not rebuild to old brief percentages | Skeptic, Architect, Risk Analyst, Empiricist |
| Publication-year cutoff should be disclosed and probed, not broadly semantically filtered before launch | Skeptic, Architect, Risk Analyst, Empiricist |
| TST is promising but should be gated and optional | Skeptic, Architect, Risk Analyst, Empiricist |
| Anchor adapters are needed before launch; full anchor results can follow | Skeptic, Architect, Empiricist |
| Cloud persistent filesystem and checkpoint restore proof are hard gates | Skeptic, Architect, Risk Analyst, Empiricist |
| Timeline should be closer to 4-6 weeks than a one-week patch if DDP and cache redesign are included | Architect, Risk Analyst, Empiricist; synthesis weights Architect due to local DDP blocker |
