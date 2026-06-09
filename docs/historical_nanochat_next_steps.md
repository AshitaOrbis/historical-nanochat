# Historical Nanochat: Further Potential Steps After the First Proof of Concept

Created: 2026-04-19

This file is the roadmap after the immediate Claude Code implementation phase. It assumes the repo has a working open-source ingestion pipeline, rights/date audit, OCR triage queue, parquet shard builder, and a first small 3090-trained proof-of-concept model.

Companion files:

- `historical_nanochat_claude_code_action_plan.md` — immediate coding-agent implementation spec.
- `historical_nanochat_access_emails.md` — outreach and access request templates.

---

## 1. Phase map

```text
Phase 0: repo/data plumbing
Phase 1: open-source proof-of-concept corpus
Phase 2: 3090 proof-of-concept model
Phase 3: OCR correction benchmark and selective GPT-5.4 OCR pipeline
Phase 4: access requests and larger corpus expansion
Phase 5: serious rented-GPU teacher model
Phase 6: source-grounded synthetic data generation
Phase 7: student-model distillation/evaluation
Phase 8: public release package
```

---

## 2. Phase 0 — repo/data plumbing

Complete before training matters.

Deliverables:

- source registry;
- source harvesters for P0 open sources;
- unified metadata schema;
- rights audit;
- date/cutoff audit;
- OCR quality scoring;
- dedupe;
- streaming parquet packer;
- historical nanochat dataloader wiring;
- single-3090 scripts;
- source/eval reports.

Success criteria:

- Can build a small corpus from at least 5 source families.
- Can generate parquet train/val shards.
- Can reproduce the corpus from manifests.
- Can explain every exclusion due to rights/date/OCR/dedupe.
- Can train a tiny smoke-test model end-to-end.

---

## 3. Phase 1 — open-source proof-of-concept corpus

Recommended first corpus:

```text
Cutoff: 1913-12-31
Language: English first
Target size: 2B-10B usable tokens depending on storage/time
Sources:
- Project Gutenberg, date-known subset only
- Internet Archive public-domain subset, tightly filtered
- LOC Selected Digitized Books
- British Library 19th Century Digitised Books
- American Stories
- Chronicling America raw OCR for gaps/OCR benchmark
- Old Bailey
- Caselaw Access Project
- Biodiversity Heritage Library
- EEBO/ECCO/Evans TCP
- NCSE v2.0
```

Important design choice:

Do not let easy sources dominate. A model trained mostly on newspapers will sound different from one trained mostly on books. Start with simple source weights, then tune.

Suggested initial source weights:

```yaml
books_general: 0.35
newspapers_periodicals: 0.30
legal_government: 0.15
science_technical: 0.10
early_modern: 0.05
misc_holdout_or_eval: 0.05
```

Reports to produce:

```text
source_mix.md
date_distribution.md
language_distribution.md
rights_audit.md
ocr_quality.md
dedupe_report.md
known_limitations.md
```

---

## 4. Phase 2 — 3090 proof-of-concept model

Purpose:

- Validate that the data pipeline produces trainable shards.
- Beat or at least compare credibly with small historical baselines such as Mr. Chatterbox-like models.
- Generate enough evidence to justify access requests and rented-GPU spend.

Recommended run:

```text
GPU: RTX 3090 24GB
Model: nanochat d14/d16-style, not d20 as default
Context: 1024
Precision: BF16/TF32 where supported
Memory: activation checkpointing + chunked loss
Training duration: 3-14 days depending on ambition
Data: corpus_1913_v0
```

Evaluation:

1. Held-out historical validation loss by source family.
2. Cutoff anachronism test.
3. Source-grounded QA test.
4. Style continuation test.
5. Synthetic-data pilot: generate 10k-100k source-grounded examples and train a tiny student.

Compare against:

- base nanochat of similar size if available;
- Mr. Chatterbox if weights/eval are accessible;
- Violet 160M/1.4B if accessible;
- TimeCapsuleLLM v2 if accessible;
- a modern general model prompted to answer as pre-1913, for qualitative anachronism comparison.

Expected result:

- A 3090 model can be a strong proof of concept.
- It should not be treated as the final teacher unless evals show it produces reliable source-grounded data.

---

## 5. Phase 3 — OCR correction benchmark

This may become the broadly useful artifact even if the first model is small.

Goal:

Build a benchmark and pipeline comparing existing OCR, local OCR, and GPT-5.4-style multimodal OCR/correction on historical pages.

Benchmark sample:

```text
1,000 pages total
- 100 Chronicling America page scans
- 100 Internet Archive book pages
- 100 British Library book pages if images available
- 100 BHL scientific pages
- 100 legal/government pages
- 100 pages with tables
- 100 pre-1850 pages
- 100 dense newspaper classified/ad pages
- 100 non-English pages
- 100 handwritten/mixed difficult pages if rights permit
```

For each page, store:

```json
{
  "source_id": "...",
  "document_id": "...",
  "page_id": "...",
  "date": "...",
  "rights": "...",
  "image_url_or_path": "...",
  "source_ocr": "...",
  "local_ocr": "...",
  "gpt54_ocr": "...",
  "manual_gold_sample": "...",
  "quality_metrics": {},
  "human_notes": "..."
}
```

Metrics:

- character error rate on manually checked samples;
- word error rate;
- layout preservation score;
- table preservation score;
- hallucination/modernization rate;
- cost per page;
- throughput;
- downstream validation-loss effect.

Recommended OCR cascade:

```text
1. Use existing source OCR.
2. Score quality.
3. Keep good OCR unchanged.
4. For bad/high-value pages, try local OCR if cheap.
5. Send only selected high-value failures to GPT-5.4.
6. Store corrected text and original OCR side by side.
7. Use corrected OCR for training only after validation.
```

Deliverable:

```text
historical_ocr_benchmark_v0/
  pages.jsonl
  source_ocr/
  local_ocr/
  gpt54_ocr/
  gold_samples/
  metrics.md
  cost_report.md
  examples.md
```

---

## 6. Phase 4 — access requests and corpus expansion

Send emails after PoC. Prioritize:

1. Harvard Library Public Domain Corpus.
2. HathiTrust public-domain full-view text.
3. Canadiana/CRKN.
4. JSTOR Text Analysis / early journal content.
5. British Library newspaper guidance/access.
6. Trove API/bulk workflows.
7. BnF/Gallica for French expansion.
8. Delpher post-1879 only if Dutch/multilingual model is planned.

Make the request credible by attaching:

- PoC corpus/model report;
- metadata schema;
- rights/date audit policy;
- no-redistribution pledge;
- eval results;
- OCR benchmark sample.

Possible expansion corpora:

```text
corpus_1850_v0
corpus_1875_v0
corpus_1900_v0
corpus_1913_v0
corpus_1913_multilingual_v0
corpus_1913_source_grounded_v0
```

---

## 7. Phase 5 — serious rented-GPU teacher model

A strong synthetic-data teacher likely needs more than the 3090 proof of concept.

Minimum serious target:

```text
Parameters: 1.2B-2B
Context: 2048
Tokens: 20B-40B high-quality historical tokens
Compute: rented H100/H200/B200 or multiple RTX Pro 6000-class GPUs
Cutoff: choose one primary cutoff first, likely 1900 or 1913
Posttraining: source-grounded SFT, not generic modern chat SFT
```

Ambitious target:

```text
Parameters: 3B-4B
Context: 2048-4096 if feasible
Tokens: 60B-80B curated historical tokens
Multiple cutoffs: 1850, 1875, 1900, 1913
Teacher output: 1M-10M filtered source-grounded examples
```

Training runs to compare:

```text
A. books only
B. newspapers only
C. books + newspapers
D. books + newspapers + law/government/science
E. source-balanced corpus
F. source-balanced corpus + OCR-corrected high-value pages
```

Do not spend on the big run until:

- corpus stats are stable;
- evaluation suite is ready;
- resume/checkpointing works;
- at least one small model trains successfully;
- synthetic-data pilot shows promise.

---

## 8. Phase 6 — source-grounded synthetic data generation

The synthetic-data teacher should not freely invent historical Q&A. It should generate from dated source packets.

Source packet schema:

```json
{
  "source_id": "old_bailey",
  "document_id": "...",
  "segment_id": "...",
  "date": "1894-03-12",
  "title": "...",
  "publication_place": "...",
  "genre": "legal",
  "rights": "public_domain",
  "source_url": "...",
  "passage": "..."
}
```

Generated example schema:

```json
{
  "task_type": "source_grounded_qa",
  "cutoff": "1913-12-31",
  "source_packet_id": "...",
  "question": "...",
  "answer": "...",
  "evidence_spans": ["..."],
  "uncertainty": "...",
  "anachronism_check": {
    "post_cutoff_terms": [],
    "modern_framing": false,
    "unsupported_claims": []
  },
  "teacher_model": "...",
  "judge_model": "...",
  "quality_score": 0.0
}
```

Task types to generate:

```text
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
```

Quality filters:

- answer must cite evidence span;
- no post-cutoff facts;
- no unsupported named entities;
- no modern moralizing unless task asks for modern analysis;
- preserve period terms but allow explanatory notes;
- reject hallucinated citations;
- use a separate judge model or rule-based detector for anachronisms.

---

## 9. Phase 7 — student-model training and proof of synthetic-data utility

This is the key scientific claim: the historical teacher is worthwhile if it improves smaller models.

Student sizes:

```text
100M-160M
300M-400M
700M
```

Training sets:

```text
Raw historical text only
Raw historical text + generic SFT
Raw historical text + teacher source-grounded synthetic data
Raw historical text + modern frontier synthetic data
Raw historical text + mixed teacher/frontier judged data
```

Metrics:

1. Historical validation loss.
2. Source-grounded QA accuracy.
3. Evidence citation accuracy.
4. Cutoff/anachronism rate.
5. Style authenticity judged blind.
6. Refusal/uncertainty calibration for post-cutoff questions.
7. Student performance per generated token.

Winning condition:

A smaller student trained on historical-teacher data beats the same student trained on raw text and/or modern-prompted synthetic data on source-grounded, time-locked evals.

---

## 10. Phase 8 — release package

Potential public artifacts:

```text
1. Corpus registry and manifests, excluding restricted full text.
2. Open-source harvesters for public-domain sources.
3. OCR quality benchmark.
4. GPT-5.4 OCR correction prompts and pipeline.
5. Corrected OCR subset where rights allow redistribution.
6. Historical nanochat model weights, if license permits.
7. Source-grounded synthetic dataset, only from sources allowing derived release.
8. Evaluation suite for anachronism/source-grounded historical QA.
9. Model card and data card.
```

Model card should include:

- cutoff date;
- training data source mix;
- excluded sources;
- rights policy;
- OCR limitations;
- known anachronism failures;
- recommended use;
- prohibited/unsafe use;
- synthetic-data generation caveats;
- evaluation results;
- contamination analysis.

---

## 11. Data sources to keep on the watchlist

Books/general:

- Harvard Library Public Domain Corpus.
- HathiTrust public-domain full-view text.
- Internet Archive public-domain scans/OCR.
- Project Gutenberg.
- British Library 19th Century Digitised Books.
- Library of Congress Selected Digitized Books.
- Google Books public-domain routes, only if terms/access are clear.

Newspapers/periodicals:

- Chronicling America.
- American Stories.
- NCSE v2.0.
- Papers Past.
- Trove.
- Delpher.
- Gallica/BnF press.
- Europeana Newspapers.
- British Library / Living with Machines related datasets.
- National Library of Finland historical newspapers/journals.
- BNE Hemeroteca Digital.
- Welsh Newspapers Online.

Legal/government/social history:

- Old Bailey.
- Caselaw Access Project.
- Century of Lawmaking / Congressional Globe.
- govinfo historical bulk datasets.
- Founders Online.
- FRASER.
- NARA catalog/open data.
- WPA Slave Narratives / Born in Slavery, with careful ethical tagging.

Science/technical:

- Biodiversity Heritage Library.
- Making of America.
- JSTOR early journal content via request.
- Royal Society public-domain material, if accessible.
- historical patent corpora, if rights/access are clean.

Early modern/pre-1800:

- EEBO-TCP public-domain subsets.
- ECCO-TCP.
- Evans-TCP.
- EarlyPrint.
- Deutsches Textarchiv.
- TextGrid Digital Library.

Multilingual later:

- Gallica/BnF.
- BNE Biblioteca/Hemeroteca Digital.
- Delpher.
- National Library of Finland.
- Aozora Bunko.
- OpenITI.
- Perseus/Scaife.
- Sefaria, with license-by-text filtering.
- Chinese Text Project / CBETA only after dedicated licensing review.

---

## 12. Risks and mitigations

### Risk: Rights contamination

Mitigation:

- fail closed;
- preserve rights fields;
- no active downloader for request/restricted sources;
- rights audit report for every shard.

### Risk: Date contamination

Mitigation:

- date confidence;
- source-specific date parsing;
- cutoff buckets;
- exclude unknown date by default;
- test 1850/1875/1900/1913 cutoffs.

### Risk: OCR hallucination from GPT-5.4 correction

Mitigation:

- never replace source OCR silently;
- preserve original OCR;
- require JSON with uncertainty;
- sample human audit;
- measure hallucinated/modernized text;
- use GPT-5.4 on high-value bad pages only.

### Risk: Over-American or over-Victorian corpus

Mitigation:

- source weights;
- source-balanced validation;
- add Canadian/Australian/New Zealand/French/Dutch sources after PoC.

### Risk: Synthetic data amplifies teacher errors

Mitigation:

- source-ground every example;
- evidence spans mandatory;
- judge/filter generated data;
- train small student experiments before scaling.

### Risk: The model is charming but technically not better

Mitigation:

- predefine evals;
- compare baselines;
- publish failure cases;
- focus on source-grounded utility, not roleplay demos.

---

## 13. Decision gates

### Gate A: before 3090 training

Proceed only if:

- at least 1B clean token estimate exists;
- validation shard is held out by document/source;
- unknown rights/date records excluded;
- training script loads parquet directly;
- smoke test runs.

### Gate B: before access requests

Proceed only if:

- PoC corpus report exists;
- PoC model or at least corpus benchmark exists;
- project page explains rights/date policy;
- request templates are customized.

### Gate C: before rented cluster

Proceed only if:

- 3090 PoC trained end-to-end;
- eval suite detects anachronisms;
- data shards are deduped and stable;
- OCR correction benchmark has results;
- training recipe is rehearsed;
- rollback/resume/checkpointing works.

### Gate D: before claiming synthetic-data utility

Proceed only if:

- at least one student model improves on held-out source-grounded evals;
- synthetic examples include evidence spans;
- generated data is filtered for post-cutoff contamination;
- comparisons include modern synthetic-data baselines.

---

## 14. Suggested issue backlog

```text
[corpus] Add registry schema and validation
[corpus] Implement Gutenberg harvester
[corpus] Implement LOC Selected Digitized Books harvester
[corpus] Implement Old Bailey harvester
[corpus] Implement American Stories harvester
[corpus] Implement BL Books harvester
[corpus] Implement BHL harvester
[corpus] Implement CAP harvester
[corpus] Implement TCP harvester
[corpus] Implement rights audit
[corpus] Implement date audit and cutoff buckets
[corpus] Implement streaming parquet packer
[ocr] Implement OCR quality scoring
[ocr] Implement GPT-5.4 OCR queue export/import
[ocr] Build 1,000-page OCR benchmark sample
[train] Add parquet-dir support to nanochat
[train] Add 3090 training scripts
[train] Add activation checkpointing
[train] Add chunked loss
[eval] Add cutoff/anachronism eval
[eval] Add source-grounded QA eval
[eval] Add baseline comparison harness
[outreach] Send HathiTrust request after PoC
[outreach] Send Harvard PDC request after PoC
[outreach] Send Canadiana request after PoC
[synth] Implement source-packet synthetic-data generation
[synth] Implement synthetic-data filters/judges
[student] Train 160M student on raw vs synthetic comparison
```

---

## 15. Recommended first three milestones

### Milestone 1: “Corpus builds”

Time target: first implementation sprint.

Deliverable:

```text
data/processed/corpus_1913_poc_small/
```

With at least:

- Gutenberg subset;
- LOC books subset;
- Old Bailey;
- American Stories sample;
- BL books sample;
- rights/date reports;
- parquet train/val shards.

### Milestone 2: “3090 model trains”

Deliverable:

- d12/d14 smoke run;
- d14/d16 longer run;
- basic eval report;
- model card draft.

### Milestone 3: “Synthetic data test works”

Deliverable:

- 10k-100k source-grounded examples;
- 100M-300M student comparison;
- evidence-span validation;
- anachronism rate comparison.

Only after these should the big rented run or access-request push become the main focus.
