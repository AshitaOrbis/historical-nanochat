# Historical Nanochat — Remaining Claude Code Instructions

Prepared for copy/paste into Claude Code after the running `legacy_textonly_d22_r30_internal_baseline` passed step-2000 checkpoint verification and the Phase-0-lite audit/repack job began.

## Current context to preserve

```text
Current GPU run:
- Name: legacy_textonly_d22_r30_internal_baseline
- Status: internal training-stack / baseline run only
- Not a governed PoC, release candidate, teacher model, or access-request model
- Step-2000 checkpoint verification: PASS on all checks
- Loss: 6.28 -> 3.70 by step 2000
- Val BPB: 4.996 -> 3.079 by step 2000
- Checkpoint sha256 round-trip: PASS
- Peak VRAM: ~17.13 GiB
- tok/sec stddev: ~0.18%
- Loader wait during concurrent audit/repack: ~0.15%, below threshold

Current CPU work:
- Combined audit+repack running nice/ionice low priority
- Six small sources completed
- EEBO mid-flight at last report
- caselaw and chronicling_america harvests are structurally incomplete
- TCP currently quarantined by fail-closed per-item rights policy
- Gutenberg kept ~10,352 docs / ~4 GB chars

Operating principle:
Keep the legacy run alive as a safety net unless it becomes unstable or the governed corpus/token_cache_v3 is ready, sane, smoke-tested, and worth restarting on.
```

---

## Prompt 1 — Continue audit/repack and report at the BL/BHL transition

```text
Continue the combined audit+repack job exactly as launched:
- nice -n 19
- ionice -c 3
- no GPU disruption
- keep the loader monitor running

At the scheduled BL/BHL transition check-in, report:

1. Training impact
   - current training step
   - current tok/s
   - loader_mean %
   - loader p95 if available
   - peak VRAM
   - any OOM/NaN/recompile/checkpoint anomaly

2. Audit/repack progress
   - source currently processing
   - sources completed
   - rows/docs kept
   - rows/docs quarantined
   - chars/tokens estimated kept
   - chars/tokens estimated quarantined
   - elapsed time
   - ETA to complete all sources

3. Early source-specific findings
   - Gutenberg kept/excluded counts and date-confidence breakdown
   - TCP quarantine reason and whether this is policy-correct or too strict
   - caselaw/chronicling_america structural-incomplete status
   - EEBO status
   - BL newspapers status when it begins
   - BHL status when it begins

4. Safety thresholds
   - If loader_mean rises above 2% for more than 10 minutes, throttle or pause audit/repack and report.
   - If training tok/s drops more than 10% sustained, throttle or pause audit/repack and report.
   - If disk IO causes dataloader stalls, pause audit/repack and keep the GPU run healthy.

Do not change the running training process unless needed to preserve it.
```

---

## Prompt 2 — Full Phase-0-lite audit/repack acceptance criteria

```text
For the full Phase-0-lite corpus pass, produce concrete artifacts before recommending a governed restart.

Required outputs:

data/phase0/manifests/full_rights_audit.json
data/phase0/manifests/full_date_audit.json
data/phase0/manifests/source_mix.json
data/phase0/manifests/split_report.json
data/phase0/manifests/repack_manifest.json

data/phase0/reports/rights_audit.md
data/phase0/reports/date_distribution.md
data/phase0/reports/source_mix.md
data/phase0/reports/split_report.md
data/phase0/reports/known_limitations.md

data/processed/corpus_1913_v3/
  manifest.json
  train/*.parquet
  val/*.parquet
  eval/*.jsonl
  reports/*.md

data/token_cache_v3/
  manifest.json
  token shards
  provenance index

The governed parquet records must preserve, at minimum:

text
source_id
document_id
segment_id
title
author
publication_date
publication_year
date_confidence
language
genre
rights
rights_url
rights_decision_basis
source_url
source_quality
ocr_quality_estimate
content_hash
dedupe_cluster_id
cutoff_bucket
split
weight_hint
metadata

Acceptance gates:

1. Unknown rights records are excluded or quarantined.
2. Unknown date records are excluded or quarantined unless explicitly marked eval-only.
3. Post-cutoff records are excluded.
4. Validation split is held out by document/source, not merely by row.
5. Source-family token counts are computed from actual packed records.
6. Repack manifest ties every shard to source/date/rights summaries.
7. token_cache_v3 refuses to build from legacy text-only shards.
8. pytest/smoke tests pass.
9. A tiny dataloader smoke test succeeds on token_cache_v3.
```

---

## Prompt 3 — Rights-audit nuance for old JSONL without per-row rights

```text
During the full rights audit, do not treat absence of per-row rights fields as automatically fatal for every source.

Instead, classify the basis for each inclusion decision.

Add or preserve this field:

rights_decision_basis:
  item_metadata
  source_registry
  collection_policy
  derived_dataset_policy
  local_access_config
  unknown

Use fail-closed behavior:

- If source has explicit item-level rights, prefer item_metadata.
- If source is known open/public-domain at collection level, allow source_registry or collection_policy, but record that basis honestly.
- If source is derived from a public-domain source such as Chronicling America, use derived_dataset_policy and preserve the upstream source relationship.
- If rights are mixed and item-level evidence is missing, quarantine/exclude.
- If rights are unknown, quarantine/exclude.
- If source is request_access/restricted/paid_skip without explicit access_granted in a local uncommitted config, exclude.

Reports should include counts by:

include_train
include_eval_only
include_metadata_only
quarantine_unknown_rights
quarantine_unknown_date
exclude_post_cutoff
exclude_restricted
exclude_paid_or_no_access

Also include counts by rights_decision_basis so the eventual model card can say exactly how records were admitted.

Specific current-source notes:

- caselaw: currently structurally incomplete; keep metadata/reporting, but do not treat as meaningful training coverage until reharvested.
- chronicling_america: currently structurally incomplete; American Stories may cover the article-level newspaper role for now.
- TCP: if all records are quarantined by per-item policy, report whether source-level public-domain TCP subsets can be admitted under source_registry/collection_policy. Do not silently include. If policy remains too uncertain, keep quarantined and state the cost in tokens/source mix.
- Gutenberg: include only records with acceptable original-publication-date confidence; do not use Project Gutenberg release date as original publication date.
```

---

## Prompt 4 — Date audit and cutoff enforcement

```text
Run the full date audit with the project cutoff:

cutoff = 1913-12-31

Use this confidence hierarchy:

exact date > year > range > inferred from bibliographic field > weak text guess > unknown

Cutoff buckets:

pre1850
1850_1875
1875_1900
1900_1913
post_cutoff_exclude
unknown_exclude

Rules:

1. Exact date <= cutoff: eligible if rights pass.
2. Year <= cutoff year: eligible if rights pass.
3. Date range ending <= cutoff: eligible if rights pass.
4. Date range crossing cutoff: exclude unless segment-level date is known pre-cutoff.
5. Unknown date: exclude from training by default.
6. Source ingestion/download/release date is not publication date.
7. Project Gutenberg release date must not be used as original publication date.
8. Preserve raw date field and normalized date decision.

Report:

- counts by date_confidence
- counts by cutoff_bucket
- counts by source_id
- examples of excluded unknown-date records
- examples of excluded post-cutoff records
- any source-specific date parser caveats
- estimated token loss from date exclusions
```

---

## Prompt 5 — Source mix and weighting decision

```text
After full repack, compute actual source mix from packed/tokenized records, not from legacy sample-rate estimates.

Report:

1. Total governed train tokens.
2. Total governed val tokens.
3. Total governed eval records.
4. Token counts by source_id.
5. Token counts by source family:
   - books_general
   - newspapers_periodicals
   - legal_government
   - science_technical
   - early_modern
   - misc_holdout_or_eval
6. Deviation from the target mix:
   books_general: 0.35
   newspapers_periodicals: 0.30
   legal_government: 0.15
   science_technical: 0.10
   early_modern: 0.05
   misc_holdout_or_eval: 0.05

Then recommend one of:

A. Use unweighted governed corpus.
B. Downsample overrepresented families.
C. Use sampling weights in training/cache build.
D. Rebuild corpus before training because imbalance is too severe.

Do not recommend a governed long run until source_mix.json and source_mix.md exist.

Special note:
If legal_government remains near zero because caselaw is structurally incomplete and Old Bailey is small, explicitly flag the gap. Do not hide it by relabeling other content as legal/government.
```

---

## Prompt 6 — Governed token-cache v3 build

```text
Build token_cache_v3 only after full governed parquet exists.

Requirements:

- token_cache_v3 must be built from data/processed/corpus_1913_v3, not from data/shards or token_cache_v2.
- Preserve a provenance index mapping token-cache spans back to:
  source_id
  document_id
  segment_id
  split
  shard path
  token count
- Write manifest.json with:
  tokenizer manifest SHA
  corpus manifest SHA
  source_mix SHA
  rights audit SHA
  date audit SHA
  train token count
  val token count
  build timestamp
  build command
- Refuse to run if input parquet has only {"text": str} and no metadata fields.
- Refuse to overwrite token_cache_v2.
- Refuse to build if val split is not document/source held out.

After build, run:

1. dataloader smoke test;
2. validation split smoke test;
3. provenance index spot-check;
4. one small batch decode/inspect check;
5. source-family val availability check.
```

---

## Prompt 7 — Governed-cache GPU smoke test before stopping legacy run

```text
Before stopping the running legacy baseline, test the governed training path.

Run a short GPU smoke test using token_cache_v3:

depth: d22
max_seq_len: 1024
device_batch_size: 8
total_batch_size: 262144
compile_mode: default
activation_checkpointing: off
tokenizer: native nanochat vocab 32768
token_cache: data/token_cache_v3
duration: 100-200 iterations
checkpoint save/resume: yes
tiny validation: yes
CORE eval: no

Report:

- tokens/sec
- peak VRAM
- loader wait %
- train loss movement
- validation BPB
- source-family validation availability
- checkpoint save success
- checkpoint reload success
- whether compile/default remains stable
- whether any metadata/provenance code path slows the dataloader

Decision rule:

- If governed v3 throughput is within about 10-15% of legacy throughput and validation works, it is safe to restart on governed corpus.
- If governed v3 is much slower, fix cache/dataloader before restarting.
- If governed v3 fails save/resume or metadata assertions, do not stop the legacy baseline yet.
```

---

## Prompt 8 — T+24 / T+48 decision gate

```text
At T+24 and T+48, fill the decision gate report.

Inputs:

1. Legacy baseline status
   - step
   - tokens trained
   - train loss
   - validation BPB
   - tokens/sec
   - loader wait
   - peak VRAM
   - checkpoint status
   - qualitative sample path
   - any instability

2. Phase-0-lite status
   - rights audit complete: yes/no
   - date audit complete: yes/no
   - full repack complete: yes/no
   - source_mix report complete: yes/no
   - validation split by document/source complete: yes/no
   - token_cache_v3 complete: yes/no
   - governed GPU smoke complete: yes/no

3. Governed corpus status
   - train tokens
   - val tokens
   - source-family mix
   - excluded tokens by rights/date
   - major source gaps
   - known limitations

Decision rules:

A. If token_cache_v3 is ready within ~24-48h, has >=15B governed train tokens, source/date/rights reports are sane, and the governed GPU smoke passes:
   stop legacy baseline at a clean checkpoint and restart governed run.

B. If token_cache_v3 is ready but governed train tokens are 10-15B:
   recommend adjusted run, likely d20 ratio=30-40 or d22 ratio=20-25.

C. If governed train tokens are <10B or source mix is badly broken:
   keep legacy baseline running while fixing corpus.

D. If Phase 0 will take several more days:
   keep legacy baseline running; do not idle GPU.

E. If legacy baseline becomes unstable:
   stop it and focus on governed corpus.

The decision report must explicitly state one of:
- continue legacy;
- stop legacy and restart governed;
- continue both until next gate.
```

---

## Prompt 9 — Choosing model size for the governed restart

```text
When token_cache_v3 is ready, choose the governed run based on actual train token count, source mix, and throughput.

Use this rule:

If governed train tokens >=18B and d22 smoke is stable:
  run d22 ratio=30 again.

If governed train tokens are 12-18B:
  choose either:
  - d20 ratio=40 if the goal is a mature smaller model;
  - d22 ratio=20-25 if the goal is capacity and the corpus is high quality.

If governed train tokens are 8-12B:
  choose d18 or d20, not d22 ratio=30.

If governed train tokens <8B:
  treat as a smaller governed PoC corpus; do not oversize model.

If source mix is still badly imbalanced:
  prefer source weighting/downsampling over blindly training on all tokens.

Do not choose d24 for the governed restart unless:
- d24 smoke passes on token_cache_v3;
- VRAM peak remains below ~21GB;
- expected wall time is acceptable;
- source/eval reports are ready.

When making the recommendation, include:
- params;
- target train tokens;
- expected wall time;
- peak VRAM from smoke/sweep;
- ratio tokens:params;
- why this choice fits the governed corpus.
```

---

## Prompt 10 — Add source-grounded prompt shape to baseline probes

```text
Extend sample_probe_prompts.yaml with a category:

source_grounded_prompt_shape

Add prompts that look like the future source-grounded synthetic-data format, even though the running model is only a legacy baseline.

Example prompt shapes:

1.
Source: Old Bailey
Date: 1888-04-02
Passage: [short held-out passage]
Question: What was the charge in this proceeding?
Answer only from the passage.

2.
Source: Biodiversity Heritage Library
Date: 1896
Passage: [short held-out scientific passage]
Question: What organism or specimen is being described?
Answer only from the passage.

3.
Source: American Stories
Date: 1905-06-12
Passage: [short article passage]
Question: Summarize the event in one sentence using only evidence from the passage.

4.
Source: Project Gutenberg
Date: 1890
Passage: [short literary passage]
Task: Continue in the same style for one paragraph without adding modern references.

Every output must be labeled:
legacy/internal/baseline

Purpose:
This is not to claim the legacy model is source-grounded. It creates a baseline for later governed/SFT comparisons.
```

---

## Prompt 11 — Minimal eval readiness for the governed run

```text
Before the governed run becomes the public-facing 3090 PoC, ensure these eval files exist:

data/processed/corpus_1913_v3/eval/anachronism_eval.jsonl
data/processed/corpus_1913_v3/eval/source_grounded_qa.jsonl
data/processed/corpus_1913_v3/eval/style_source_eval.jsonl
data/processed/corpus_1913_v3/eval/heldout_manifest.json

Minimum contents:

Anachronism eval:
- at least 100 prompts
- includes post-1913 leakage probes:
  WWI as known fact
  1918 influenza
  League of Nations
  Nazi Germany
  antibiotics/penicillin as standard treatment
  post-1913 political leaders/events
- includes in-period controls so the model is not rewarded for refusing everything

Source-grounded QA:
- at least 500 held-out passages if feasible
- from at least Old Bailey, BHL, American Stories, books/general
- each item includes evidence_spans
- each item includes source_id, document_id, date, rights

Style/source eval:
- newspaper brief
- Victorian book continuation
- legal summary
- scientific description
- period letter or formal prose

The eval runner should produce:
- overall score
- score by source family
- examples of failures
- anachronism/leak flags

Do not block corpus building on perfect evals, but do not call the governed run public-facing until minimal evals exist.
```

---

## Prompt 12 — Model card and run-card discipline

```text
Create or update run cards with strict naming.

For the current running model:

Name:
legacy_textonly_d22_r30_internal_baseline

Allowed claims:
- validates 3090 d22 training stack
- validates tokenizer/cache/checkpoint path
- provides internal baseline loss/BPB/generation samples
- useful for comparison against governed corpus runs

Disallowed claims:
- governed PoC
- release candidate
- teacher model
- source-grounded model
- rights-audited corpus model
- access-request evidence model, except as a training-stack demo

Required disclosure:
This checkpoint was trained on legacy text-only shards produced before the Phase-0 provenance-preserving corpus pipeline was completed. Source-level provenance, rights/date audit fields, and source-family validation splits were not preserved in the training shards.

For the governed restart model:

Name format:
governed_corpus1913_v3_[depth]_[ratio]_3090_poc

Required model card sections:
- cutoff date
- tokenizer
- model size
- context length
- train tokens
- source mix
- rights policy
- date policy
- validation split policy
- OCR limitations
- known source imbalance
- anachronism eval
- source-grounded QA eval
- sample generations
- intended use
- not intended use
- synthetic-data caveats
```

---

## Prompt 13 — Outreach remains blocked until governed evidence exists

```text
Do not send major access requests yet, except for low-friction clarification questions that do not claim a completed PoC.

Block major outreach until these exist:

1. governed corpus report;
2. metadata schema;
3. rights/date audit report;
4. source mix report;
5. 3090 PoC run report, preferably governed;
6. anachronism eval report;
7. source-grounded QA eval report;
8. no-redistribution statement for restricted full text.

When outreach starts, use the wording from the email pack:
- non-commercial research;
- public-domain or research-accessible materials;
- provenance-preserving;
- controlled bulk access, not scraping;
- no redistribution of restricted full text;
- source-grounded synthetic data, not hallucinated roleplay.

The legacy baseline can be mentioned only as:
a training-stack baseline on unaudited legacy text shards,
not as the project’s governed proof-of-concept model.
```

---

## Prompt 14 — Synthetic-data pilot should wait for governed source packets

```text
Do not run a serious synthetic-data pilot from legacy text-only shards.

The synthetic-data pilot requires source packets.

Source packet schema:

source_id
document_id
segment_id
date
title
publication_place
genre
rights
source_url
passage

Generated example schema:

task_type
cutoff
source_packet_id
question
answer
evidence_spans
uncertainty
anachronism_check
teacher_model
judge_model
quality_score

Task types to prepare:

source_grounded_qa
period_summary
newspaper_brief
legal_case_summary
scientific_description
historical_glossary
letter_rewrite_in_period_style
evidence_extraction
what_can_be_inferred
what_cannot_be_inferred
anachronism_detection
timeline_safe_refusal
primary_source_comparison

The first pilot should be small:
10k-100k examples

Do not claim synthetic-data utility until a smaller student improves on held-out source-grounded evals.
```

---

## Prompt 15 — Phase-0-lite completion report

```text
When Phase-0-lite finishes, produce one consolidated report:

report/phase0_lite_completion_2026-04-XX.md

Include:

1. Executive summary
   - whether Phase-0-lite passed
   - whether governed restart is recommended
   - recommended model/run config

2. Corpus totals
   - documents
   - segments
   - characters
   - estimated tokens
   - cached tokens

3. Rights audit
   - included/excluded/quarantined counts
   - counts by rights_decision_basis
   - source-level caveats

4. Date audit
   - counts by date_confidence
   - counts by cutoff_bucket
   - examples of excluded records

5. Source mix
   - by source_id
   - by source family
   - comparison to target mix
   - proposed weighting/downsampling

6. Splits
   - train/val/eval
   - document/source holdout logic
   - heldout sources

7. Repack/cache
   - parquet paths
   - token_cache_v3 path
   - manifest SHAs
   - tokenizer SHA

8. Eval readiness
   - anachronism prompts count
   - source-grounded QA count
   - style eval count

9. Governed smoke test
   - tok/s
   - VRAM
   - loader wait
   - save/resume
   - BPB/loss movement

10. Decision
   - continue legacy baseline
   - stop legacy and restart governed
   - wait for more corpus work
   - selected model depth/ratio

Also include a short appendix listing sources that were structurally incomplete, excluded, or quarantined:
- caselaw
- chronicling_america
- TCP
- any others discovered during the full audit
```

---

## Prompt 16 — If governed cache is ready soon, restart procedure

```text
If token_cache_v3 is ready, sane, and smoke-tested within the T+24/T+48 window, use this restart procedure.

1. Let the legacy run reach the next clean checkpoint.
2. Run verify_first_checkpoint.py or equivalent checkpoint integrity check on that latest checkpoint.
3. Copy/record:
   - run config
   - checkpoint path
   - logs
   - sample probes
   - loss/BPB curves so far
4. Stop legacy run gracefully.
5. Mark legacy run status:
   stopped_cleanly_for_governed_restart
6. Start governed run using the selected config.
7. After 200-500 governed steps:
   - force checkpoint save;
   - reload once;
   - confirm loss finite and descending;
   - confirm source-family validation path works.
8. Only then let governed run continue unattended.

Do not delete the legacy checkpoint. It is useful as a training-stack baseline and fallback.
```

---

## Prompt 17 — If Phase-0-lite reveals too few governed tokens

```text
If the governed corpus is much smaller than expected after rights/date exclusions, do not force the original d22 ratio=30 plan.

Use these fallback choices:

- >=18B governed train tokens:
  d22 ratio=30 is acceptable.

- 12B-18B governed train tokens:
  prefer d20 ratio=30-40 or d22 ratio=20-25.

- 8B-12B governed train tokens:
  prefer d18/d20.

- <8B governed train tokens:
  treat this as a smaller governed corpus PoC; use d16/d18 and focus on eval/corpus quality, not scale.

If source imbalance is severe:
- propose downsampling or source weights;
- report expected effective tokens after balancing;
- do not hide the imbalance by training on all available tokens and calling it governed-balanced.
```

---

## Prompt 18 — If TCP is quarantined but seems salvageable

```text
Investigate TCP quarantine separately after the main audit/repack completes.

Goal:
Determine whether TCP public-domain subsets can be included under a defensible source_registry or collection_policy basis without violating fail-closed rights logic.

Report:

1. Which TCP subset(s) are present locally:
   - EEBO-TCP Phase I
   - ECCO-TCP
   - Evans-TCP
   - EarlyPrint or other derived set

2. Source/collection rights evidence available locally or in registry.

3. Whether item-level rights metadata exists.

4. Whether source-level public-domain admission is defensible.

5. If defensible, propose a policy patch:
   rights_decision_basis = collection_policy or source_registry
   rights = public_domain_or_open_research
   include_train = true

6. If not defensible, keep TCP quarantined and document the token/source-mix cost.

Do not include TCP silently. Any admission must be explained in rights_audit.md.
```

---

## Prompt 19 — If caselaw / Chronicling America are structurally incomplete

```text
For caselaw and chronicling_america, preserve the finding that the existing harvests are structurally incomplete.

Do not use tiny/incomplete harvests to claim legal or raw-Chronicling coverage.

Add to known_limitations.md:

- caselaw_access_project: local harvest currently contains only a tiny structural sample; not representative; excluded or metadata-only until reharvested.
- chronicling_america: local direct harvest currently contains only a tiny structural sample; American Stories serves as the main newspaper source for now; raw page OCR reharvest remains future work.

Add follow-up issues:

[corpus] Reharvest Caselaw Access Project or HF mirror with cutoff <=1913
[corpus] Reharvest Chronicling America raw OCR pages for OCR benchmark/source-grounding gaps

If governed run proceeds before these are fixed, the model card must disclose:
- legal/government undercoverage;
- direct raw Chronicling America undercoverage;
- reliance on American Stories for newspaper article text.
```

---

## Prompt 20 — Keep final claims disciplined

```text
For all generated reports, maintain this distinction:

legacy_textonly_d22_r30_internal_baseline:
- internal baseline
- training-stack validation
- unaudited legacy text-only corpus
- no provenance-preserving training shards
- not a teacher model

corpus_1913_v3 governed run:
- candidate 3090 proof-of-concept only if rights/date/source/eval artifacts exist
- still not automatically a final teacher model
- teacher utility requires a later student improvement experiment

Do not write language implying:
- “best historical chatbot”
- “release candidate”
- “teacher model”
- “source-grounded model”
- “rights-cleared corpus”

unless the corresponding evidence exists.

Allowed phrasing after a governed successful run:
“A governed 3090-trained historical nanochat proof of concept with documented source mix, rights/date filtering, and preliminary anachronism/source-grounded evaluation.”
```
