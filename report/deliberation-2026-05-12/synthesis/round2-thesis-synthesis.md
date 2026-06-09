# Round 2 — Thesis Synthesis

Citations: [Max], [Council], [Pro], [Opus].

---

## 1. Unanimous: what the artifacts CANNOT support as a thesis

| Drop this | Why |
|---|---|
| "Validated historical / vintage LLM" | Capability not measured; no per-family eval, no anachronism eval. |
| "The model genuinely cannot know post-1914 events" | Cutoff is publication-year-only; pre-1914 docs contain forecasts of post-1914 events. Defeasible at the corpus layer, not just unverified. |
| "Aggregate validation bpb 1.1092" | It's a 262 k Gutenberg-books prefix, not the held-out split. |
| "Balanced five-family corpus" (in the sense of source diversity) | 3 of 5 families have single-source train coverage. Gutenberg, TCP, Old Bailey, Chronicling-America are val-only. |
| "Period-appropriate factual competence" | Sample log shows surface fluency, no factual recall, no arithmetic, no logic. |
| "Ranke-class on a 3090" | Ranke-4B is 4B params × 80 B tokens × multi-cutoff. This is 615 M × ~16 B unique tokens × one cutoff. |
| "Loader solved heterogeneous pretraining" | One smoke-series rescue, no ablations against alternatives, only locally validated. |
| "Novel methodology / new method" | The loader is a workaround, not a method. The postmortem culture is normalized in academic ML / SRE — venue-novel, not method-novel. |
| "Small-model vintage LLM novelty" | Talkie-1930 (13B, pre-1931), Ranke-4B, Evans-vintage conceptually pre-existing. |

## 2. Unanimous: what the artifacts CAN support

- A **mechanically valid corrected commodity-GPU training run** completed on a rights-audited pre-1914 corpus.
- A **narrow cross-source held-out compression signal** on a Gutenberg prefix.
- An **auditable governance trail** — per-shard provenance, an archived failed run, a diagnostic-cursor logging discipline that caught a 46-hour data-integrity bug invisible to standard metrics.
- The **family-scheduled loader** as a documented local fix for the upstream cache-shape choice.
- A **transparent ledger** of where apparent historical-LM validity dissolves under audit.

The first-pass thesis all four panels recommend: **"mechanically valid commodity-GPU historical pretraining run + an unusually visible account of where its historical-LLM claims exceeded its actual corpus mechanics."** That sentence survives every panel's vote.

## 3. The real disagreement: how to FRAME what's been done

This is where the panels diverge.

| Panel | Recommended frame | Risk profile |
|---|---|---|
| [Max] | Postmortem / field report | Safest. Risk: under-claims. |
| [Council] | Methods/postmortem note + corpus card | Safe. Adds methods-paper-after-weekend-evals option (workshop note). |
| [Pro] | "Worked anatomy of claim-collapse in a historical LLM project" — provenance-governance case study under commodity constraints | Sharpest. Names the central content (claim-collapse) honestly. Best calibrated thesis sentence: *"historical-nanochat is a mechanically valid but construct-limited attempt to adapt a minimal LLM training harness to nominally time-bounded historical corpora; its scientific value lies in the exposed governance mechanics — source coverage, split semantics, cutoff granularity, validation interpretation, and loader invariants — not in demonstrated historical competence."* |
| [Opus] | **First prototype of a measurement instrument for time-isolated language modeling, plus its calibration log** — Thomist/Aristotelian instrument-making register | Highest payoff if the researcher actually wants to keep working at the next scale. Frames the artifact as v1 of an instrument-class rather than v1 of a product. |

The four frames are not contradictory. They're nested:

```
postmortem (Max)
    ⊂ methods/postmortem (Council)
        ⊂ provenance-governance case study (Pro)
            ⊂ instrument prototype + calibration log (Opus)
```

Pro adds: *the case study's distinctive content is exposed claim-collapse*. Opus adds: *the case study is best understood as instrument-making in the techne sense*. Both extensions land on the same underlying object, framed for different audiences.

## 4. Findings unique to one panel

### [Opus] — 6 contributions not in any other panel

| # | Finding | Confidence |
|---|---|---|
| 4.1 | **The family loader is a workaround, not a method.** Modern corpora (SlimPajama, DCLM, FineWeb) interleave families at the cache layer at tokenization time; the family-scheduled loader is the cheapest fix for a per-family-contiguous-shard cache that shouldn't have been built that way. The right unrun ablation is "loader-fix vs cache-fix," not "compare loader schedules." | ~80% |
| 4.2 | **The postmortem culture is venue-novel, not method-novel.** Model cards (Mitchell 2018), datasheets for datasets (Gebru 2018), reproducibility appendices, the SRE postmortem tradition — all normalized in academic ML / engineering culture. What's unusual is a single-author hobby project applying it. | high |
| 4.3 | **615M is structurally insufficient for the original capability thesis.** Factual recall in modern LMs emerges roughly at 1-7B+ with substantial post-training. Below that, "doesn't know post-1914 events" is indistinguishable from "doesn't know much of anything." The cutoff hypothesis is not testable at 615M, no matter how thorough the evals. The current artifact is a **calibration model for an experiment that requires a bigger model.** | ~85% |
| 4.4 | **The 1913 cutoff is theoretically arbitrary.** Why 1913 specifically? 1800 / 1850 / 1900 / 1913 / 1939 / 1968 are all defensible. The implicit "1914 is the modernity breakpoint" is itself a claim that should be argued or relativized, not presented as a corpus fact. The Aristotelian register lands: a pre-1913 corpus is not "a vintage LLM" — it is specifically a **pre-Sarajevo LLM**. | ~70% |
| 4.5 | **"Governed" should be specified to a concrete distinctive tuple.** vs Ranke (year filter only?), vs Common Pile (rights but no historical cutoff), vs Gutenberg-only LMs (no rights provenance per item), vs Pile Books2/3 (contested rights, no audit). This project's actual distinctive governance is: per-item rights audit + publication-year cutoff + preserved through the cache layer + archived failure as evidence. That tuple is plausibly genuinely new at single-author hobby scale; without specification, "governed" is just a word. | ~75% |
| 4.6 | **A thesis-adjacent open scaling question lives in the data.** The 26.3:1 unique-tokens:params number plus the absence of any Chinchilla-applicability measurement for low-entropy historical corpora puts the project adjacent to a real research question: *do pre-modern corpora sit at different points on the Chinchilla curve than modern web text?* The 615M run is too small to answer it. The project could open it. | ~55% on the answer, ~80% on "this is the most interesting unasked question adjacent to the artifact." |

### [Pro] — 4 contributions not in any other panel

| # | Finding | Significance |
|---|---|---|
| 5.1 | **"Worked anatomy of claim-collapse"** as the operative thesis content. Useful because historical/vintage LLMs are especially vulnerable to exactly the semantic-overclaim failures this project documents. | Names the central content precisely. |
| 5.2 | **Talkie-1930 (13 B, pre-1931 text) is now part of the comparison class**, with explicit OCR/leakage/benchmark discussion. Capability novelty is even less available; the legitimate niche is *small-scale provenance audit*, not *small-scale vintage LLM*. | Updates the competitive landscape. |
| 5.3 | **"Historical style ≠ historical competence."** A model can sound archaic and bookish while failing source coverage, chronology, factuality, and anachronism resistance. The thesis must not let style substitute for competence. | Sharp boundary nobody else drew. |
| 5.4 | **Software-engineering lesson framing**: "historical-LLM validity requires structural invariants, not documentation promises." The loader "guards" are improved awareness, not proof of safety. | Generalizable lesson. |

### [Max] — distinctive contributions

| # | Finding | |
|---|---|---|
| 6.1 | Final-recommendation sentence is the cleanest one-line version: *"a single-author, single-3090 project mechanically completed a 615M base-pretraining run on a rights/date/source-governed pre-1914 corpus; the corrected run shows a narrow held-out Gutenberg learning signal, but the main contribution is the audit trail showing that temporal-governed LMs fail first at provenance, mixture accounting, tokenizer/split semantics, and evaluation design."* | Best single-sentence "as-is" thesis from the panels. |
| 6.2 | Adjudication that **"the positive artifact is the premise, the postmortem/evaluation boundary is the thesis."** | Useful framing primitive. |

### [Council] — distinctive contributions

| # | Finding | |
|---|---|---|
| 7.1 | Frames the disagreement axis as **"as positive as Architect/Empiricist (artifact exists, real learning signal) vs as guarded as Skeptic/Risk-Analyst (don't let 'prototype' smuggle in unsupported validation)."** Adjudication is the precise rule "use positive language for the artifact's reality, guarded language for the claim boundary." | Best meta-rule for writing tone. |

## 5. The synthesized thesis recommendation

**For the researcher's actual psychological profile** (Investigative-100, Intellect-90, CRT-7/7, Thomist-Aristotelian receptive, allergic to careerist/networking framings, energized by architectural/instrument-making/civilizational scope), the Opus reframe lands hardest. It is also the most honest about what the artifact actually is.

But Pro is right that the **operative content** of the thesis is the worked anatomy of claim-collapse. And Max is right that **the postmortem/evaluation boundary is the load-bearing thesis content**, not the model.

These compose:

### THE STRONGEST DEFENSIBLE THESIS (as of today, no additional work)

> **"This is the first prototype of a measurement instrument for time-isolated language modeling. Its calibration log — the rights-audited pre-1914 corpus, the family-scheduled loader, the archived stale-provenance postmortem, the narrow Gutenberg-prefix learning signal at 1.1092 bpb, and the inventory of which historical-LLM claims survive audit and which collapse under it — is the contribution. The 615M checkpoint itself is the smallest measurement the instrument can take; using it to test 'does training-time temporal isolation produce post-cutoff ignorance' would require the next scale up and an honest characterization of the publication-year cutoff's semantic limits."**

### THE STRONGEST DEFENSIBLE THESIS (after one weekend of evals)

Per-family bpb (verifies whether the model's compression generalizes beyond Gutenberg books), tokenizer-train-set audit (rules out tokenizer-content leakage), CORE-or-equivalent external anchor (places the checkpoint somewhere comparable), matched pre/post-cutoff anachronism logprob set (measures whether the cutoff is even visible at this scale), source-register continuation (style probe), and cross-source document near-duplicate scan (closes or sharpens the contamination question).

> **"Same instrument framing; calibrated case study; the calibration log now includes the readings, not just the rig."**

Note: the weekend evals can *measure* the publication-year-cutoff problem. They cannot *close* it. The thesis must explicitly say this.

### THESES WORTH DROPPING (committee list)

- "validated historical LLM"
- "model genuinely cannot know post-1914 events"
- "aggregate validation bpb of 1.1092"
- "balanced five-family corpus" (with implication of source diversity)
- "Ranke-class on a 3090"
- "period-appropriate factual competence"
- "loader solved heterogeneous pretraining"
- "novel methodology"
- "unusual postmortem culture" (substitute: "applying industrial-ML documentation discipline at hobby scale")

### PUBLICATION FORM

A **technical field report / long-form blog post**, framed as instrument-making + calibration log, with:

- Model card (the 615M base checkpoint, honestly described).
- Corpus card (the v4 governed pipeline, with the train-empty source disclosures).
- Postmortem appendix (the stale-provenance bug, archived).
- Eval appendix (after weekend evals, or labeled "future-work" if not run).
- Either a defended choice of cutoff (1913 specifically, with reason), or a relativization (1913 was the pragmatic choice given corpus availability; thesis applies to any pre-modern cutoff).

**NOT** a methods paper, **NOT** a corpus paper, **NOT** a model release, **NOT** a capability paper. The "venue" question is less important than the "claim genre" — audit / case study / calibration log, not capability/method/model claim.

## 6. The thesis-relevant question NONE of the panels asked

- **Does the researcher actually want to scale to the next instrument size, or is this run already the destination?** If the answer is "I built this and I'm done," then the instrument framing is wasted — the artifact stands as a single calibration log of a one-off run, and the publication form should match (blog post, no model card hosting, no successor experiment). If the answer is "this was v1 of a research program," the instrument framing is load-bearing and the next-scale plan should be drafted now. Round 3 will need to ask this directly.

## 7. What R3 (next actions) must take as fixed

1. The thesis is *instrument prototype + calibration log + claim-collapse anatomy*, not *small vintage LLM*.
2. The 1.1092 bpb is a calibration reading on the narrowest validated probe, not a headline.
3. The 615M scale is too small to test the original cutoff hypothesis. Any next-action plan that includes "use this checkpoint to demonstrate the cutoff thesis" is a non-starter; it needs to be replaced with either "scale up to test the thesis" or "claim something other than the cutoff thesis."
4. Weekend evals are *measurement of the existing instrument*, not *upgrades to the model*.
5. The "publication-year ≠ content-knowledge cutoff" problem is a corpus-level structural finding that must be named in any public framing.
6. The "governed" tuple needs concrete specification before being used as a contribution claim.
