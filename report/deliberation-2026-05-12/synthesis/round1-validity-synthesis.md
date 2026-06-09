# Round 1 — Validity Synthesis

Four independent panels reviewed the historical-nanochat run. This synthesis surfaces points of agreement, disagreements, and contributions unique to each panel. Citations: [Max] = GPT Max HCOM 4-agent, [Council] = codex-council 4-agent, [Opus] = Opus 4.7 subagent, [Pro] = GPT-5.5 Pro single-session.

---

## 1. Unanimous findings (4-of-4)

| # | Finding | Confidence |
|---|---|---|
| 1.1 | The 1.1092 bpb is measured on a fixed deterministic 262,144-token prefix of one Gutenberg books val shard, NOT on the 2.86 B held-out source-stratified split. The brief's framing is misleading. | Highest. Code-verified during R1. |
| 1.2 | The trained checkpoint exists, the run completed mechanically, no NaN/OOM/instability, checkpoint round-trip clean, peak VRAM under budget, loader <1% wait. | Highest. |
| 1.3 | Run #1's stale-provenance bug establishes that ordinary training-health metrics (loss, val bpb, throughput, VRAM) are *insufficient* for data-integrity claims. The corrected run's loader guards address the specific failure class but do not generalize. | High. |
| 1.4 | Effective training distribution ≠ corpus inventory. The 12/8/6/3/3 schedule governs the model's exposure, not the cache mix. | High. |
| 1.5 | Sample completions show surface English fluency and period register at the token level but **no** factual recall, arithmetic, or basic logic. They cannot validate "period-appropriate competence" beyond style. | High. |
| 1.6 | Comparing 1.1092 to peer LMs is not protocol-comparable (different vocab, corpus, val protocol, tokenizer). | High. |
| 1.7 | "Monotone descent" is overstated. Real trajectory has multiple local upticks (12000→13000, 22000→25000, 29000→34000). | Verified. |
| 1.8 | Major gaps: no CORE metric, no per-family val, no anachronism eval, no source-grounded QA, no midtraining, no document-level audit. | All four panels list these. |

**Convergent verdict (all four):** Mechanically credible corrected training run with a real but narrow learning signal, NOT a validated historical governed-base-model PoC. Promising artifact with serious unresolved construct-validity gaps.

---

## 2. Substantive new contributions per panel

### [Opus] — highest-value unique findings (panel-of-record for these)

| # | Finding | Evidence cited |
|---|---|---|
| 2.1 | **Gutenberg has 1,132 val shards but ZERO train shards.** The brief lists Gutenberg as a books_general train source; it is val-only. The 1.1092 bpb measures cross-source held-out generalization to a source the model never saw, not within-distribution loss. | provenance.json + source_mix.md verified |
| 2.2 | **The brief's corpus mix is materially wrong.** Actual built-corpus mix is closer to 40.27 / 20.40 / 20.13 / 9.13 / 10.07, not 37.7 / 26.9 / 17.5 / 9.8 / 8.2. Three of five families (books, legal, early_modern) have **single-source train coverage** because Gutenberg, TCP, Old Bailey, Chronicling-America are all train-empty. | source_mix.md |
| 2.3 | **No cross-source near-duplicate dedup.** Explicitly acknowledged in `data/phase0/README.md:120` under "What's NOT here." | README quoted |
| 2.4 | **Two split mechanisms exist and don't coordinate.** `split_holdouts.py` does document-disjoint hash-based; `build_v4_balanced.py` (the actually-used script) does whole-shard splitting and ignores the `split` column. | Code-verified |
| 2.5 | **The "guards" are documentation, not safety nets.** Provenance is still keyed by volatile `shard_index` (postmortem itself names `source_file` as right fix); two of three upstream scripts (`split_cache_shards.py`, `shuffle_cache_manifest.py`) still don't auto-regenerate provenance. | TODOs in postmortem |
| 2.6 | **Unique training tokens are ~16.18 B, not 18.47 B; true tokens:params is ~26.3:1, not Chinchilla 30:1.** Books and legal wrapped (1.04× and 1.11× cycles) during warmdown, when LR was already decaying. | meta_070455.json + cycle arithmetic |
| 2.7 | **Date-cutoff enforcement is publication-year-only.** A 1913 newspaper forecasting WWI is in training; the project thesis is defeasible at the corpus level, not only at the eval level. | Conceptual; supported by data_audit.py docstring warning |

### [Pro] — highest-value unique findings

| # | Finding | Significance |
|---|---|---|
| 3.1 | **The bpb has at least 7 measurement-theory problems beyond the 262 k issue.** Effective statistical sample is *less* than 262 k tokens because adjacent tokens are highly dependent — it's one contiguous prefix of one shard with 9 documents, not 262 k IID samples. No uncertainty estimate. Repeated eval on same prefix → weak validation-set overuse. "Final = minimum" doesn't establish monotone generalization. Bpb is compression-proxy, not historical-validity measure. The probe is **books_general** during late warmdown books re-exposure, creating a within-family confound. Lack of anchor for external comparability. |
| 3.2 | **Tokenizer is part of the measurement system and possibly part of the leakage surface.** A 32,768-vocab tokenizer trained on the 1913 corpus is highly aligned with the orthography, names, and boilerplate of that corpus; bpb on a Gutenberg prefix using that tokenizer partially measures tokenizer-corpus alignment, not generalization. Possible tokenizer-training leakage if val text was in tokenizer training data. |
| 3.3 | **Books/legal late re-exposure interacts with the books-only validation probe.** The final reported loss may be unusually favorable to books-style language because warmdown included repeated books tokens. Not just "the eval is books" but "the eval is books **after** late optimization revisited books." |
| 3.4 | **Cutoff-notion taxonomy.** Distinguishes: source publication year cutoff, edition cutoff, content cutoff, model-knowledge cutoff. Each implies a different verification protocol. The current pipeline enforces only the first two; the latter two are not established. |
| 3.5 | **A 12-line failure-class table** enumerating silent data-integrity failure modes that current loader guards would NOT catch: wrong content with correct tag, document-level cross-split leakage, wrong publication_year inherited from original work, wrong tokenizer, normalization mismatch, resume cursor mismatch, eval/train cache mix-up, within-family imbalance, truncated/padded shards, document-boundary corruption, metadata regenerated from corrupted state, semantic anachronism. |

### [Max] — highest-value unique findings

| # | Finding | Significance |
|---|---|---|
| 4.1 | First panel to flag that **the family schedule (12/8/6/3/3 = 37.5/25/18.75/9.375/9.375) is close to but not identical to the corpus mix**, with the implication that books/legal are slightly over-sampled relative to inventory. |
| 4.2 | Frames the right validity label as **"promising completed run artifact with serious unresolved construct-validity gaps,"** which became the convergent framing across all four panels. |
| 4.3 | Notes that the model is "good at predicting clean English literary prose" because the tokenizer is also good at compressing it — but this was elaborated more fully by Pro. |

### [Council] — highest-value unique findings

| # | Finding | Significance |
|---|---|---|
| 5.1 | **Architect's three-tier validity decomposition**: (a) completed governed pretraining run, (b) broad five-family held-out language model, (c) usable historical base model. The success-or-failure verdict depends on not collapsing these layers. |
| 5.2 | **Skeptic's positive framing of the prefix-bpb signal**: even if narrow, descent on a held-out prefix is *real evidence of held-out-data generalization*, not noise. This is the most charitable reading of the headline number and survives even after Opus shows it's cross-source. |
| 5.3 | First panel to explicitly call out **the validation-cache narrowness as a "0.0092% of held-out tokens" measurement** — making the gap quantitatively concrete. |

---

## 3. Disagreements / modulations

There are no hard disagreements among the panels — they converge on the headline. Modulations:

- **[Opus] vs [Max+Council+Pro] on the 1.1092 construct.** Max/Council/Pro frame 1.1092 as a "narrow held-out probe." Opus argues it's specifically a **cross-source held-out generalization** measure (Gutenberg is train-empty), which is a *stronger* construct than implied by "narrow probe." This cuts both ways: more demanding construct + the descent on the more demanding construct is more impressive than the panels gave credit for, but the brief's framing is now even more wrong. Opus is correct on the facts (verified). The synthesis treats this as an **Opus refinement** of the consensus, not a disagreement.

- **[Pro] vs [Opus] on whether tokenizer alignment is a measurement-theory issue or a possible leakage.** Pro raises tokenizer-training-leakage as a possible weak form of leakage (if val text was in tokenizer training); Opus mentions it once but doesn't develop. Pro's framing is more careful: tokenizer training is "not the same as weight training on validation text, but it can still improve segmentation of validation-specific strings."

- **[Pro] vs [Council/Max] on "promising."** Pro pushes back on Max's "promising completed run artifact" label as too warm. Pro: "promising" should not be attached to the validation result. The validation result is too compromised to carry that adjective. This is a real semantic distinction.

---

## 4. Blindspots NONE of the four panels caught

What I cannot find in any of the four reports:

(a) **Repro of run #2 → run #3 continuity.** The training was interrupted at step 10,000 of run #2 and resumed in run #3. The optimizer state, RNG state, dataloader cursor, and tokenizer were all reloaded — but **was the val cache identical between runs?** If the val cache was rebuilt between run #2 and run #3 (it shouldn't have been, but the v4 build pipeline is multi-step), the pre- and post-resume val bpb numbers would not be commensurate. The trajectory presented as continuous may have a hidden discontinuity at step 10,000.

(b) **What was the tokenizer training corpus, exactly?** Pro and Opus both flag tokenizer-content alignment as a confound. Neither verifies what corpus the 32,768-vocab tokenizer was actually trained on. If it was trained on the v3 governed corpus (pre-Gutenberg-train-exclusion), then Gutenberg WAS in the tokenizer training. If it was trained on the v4 cache (which excludes Gutenberg from train), the tokenizer is naive to Gutenberg orthography and the bpb measures both modeling AND tokenizer-out-of-distribution.

(c) **The "Chinchilla 30:1" framing in the brief comes from `target_param_data_ratio: 30` in the config, which is the SCHEDULED ratio.** Opus correctly notes that unique-tokens ratio is ~26.3:1. But none of the panels asked whether Chinchilla optimality is itself the right target for a domain-restricted, single-source-per-family corpus. Chinchilla scaling laws were derived on FineWeb-Edu-style mixed-web text. Historical corpora may have different scaling-optimal points.

(d) **No panel asked about the optimizer state size or learning-rate-schedule continuity across resume.** The optim_010000_rank0.pt file is 2.6 GB. Resume reads it, restores AdamW first/second moments and step counters. If anything in the schedule (warmup ratio, warmdown ratio, peak LR) was changed between run #2 and run #3, the model trained for ~14 days on a learning-rate trajectory that diverged from its 70,455-step plan from the start. No verification of this.

(e) **No panel computed the actual epoch count per family in tokens-trained terms with LR weighting.** Books and legal wrapped during warmdown when peak LR was ~0.005 (33% of peak). So the **effective LR-weighted** second pass on books/legal contributes maybe 10-20% as much to weight updates as the first pass. This refines Pro's §F point but with a number nobody computed.

(f) **What's the val/train alignment in tokenization?** Was the val cache tokenized with the same tokenizer as train, in the same encoding configuration? If the val cache was tokenized at a different point in the pipeline (different normalization, different BPE merges), bpb is meaningless. The brief says val cache lives at `/data/token_cache_v4_balanced_candidate/val/` parallel to `/train/`, which strongly implies same tokenizer — but it's not verified.

---

## 5. The headline validity verdict (synthesized)

**Mechanically credible corrected training run. Headline metric measures the wrong construct (cross-source held-out prefix on a train-empty source, not aggregate val). The trained-on corpus is materially less diverse than the brief reports (single-source coverage in books, legal, science families). At least one major class of contamination (cross-source near-duplicates) is acknowledged unmitigated. Date-cutoff enforcement is publication-year-only and cannot in principle catch content-semantic anachronism. The "governance" story repairs the known stale-provenance bug class but leaves the structural manifest/provenance coupling intact, and the post-incident safety net is two assertions + a researcher manually reading diagnostic cursor logs.**

**Correct label: "Mechanically valid corrected run with a real but mis-described learning signal on a less-diverse corpus than reported, and several construct-validity gaps the project does not yet close."**

This is not a failure. It is the legitimate output of an honest single-3090 training experiment that worked, plus a partially mis-framed report about it. The underlying artifacts — checkpoint, governance pipeline, smoke series, postmortem culture — are real and unusually mature for a single-author project. The validity gaps are addressable but unaddressed.

---

## 6. What R2 (thesis) and R3 (next-actions) must take as fixed

Going into R2 and R3:

1. **Stop using 1.1092 as the headline result.** It is a Gutenberg-books prefix tracking number, not an aggregate val measure. Any public framing of the run that treats 1.1092 as the model's "validation loss" is misleading.

2. **The brief's corpus mix needs correcting.** Three families have single-source train coverage. Gutenberg, TCP, Old Bailey, Chronicling-America are val-only.

3. **The thesis is "post-1914 publication-year cutoff," not "post-cutoff knowledge."** The corpus contains 1913 forecasts about post-1914 events.

4. **The model is a base pretraining checkpoint with no midtraining, SFT, or RLHF.** Anything calling it a "model" should distinguish "base model" from "chat model."

5. **The interesting facts** that survived this review intact: a single-3090 14-day training produced a 615M-param transformer on a multi-source pre-1914 corpus; the per-family loader strategy was the decisive innovation; the postmortem culture caught a 46-hour data-integrity bug that would have been invisible to standard metrics. These are real contributions.
