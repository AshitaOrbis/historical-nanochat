## 1. Consensus

- The as-is thesis is **not** “validated historical/vintage LLM.” It is a mechanically valid single-3090 base-pretraining artifact plus a governance/evaluation postmortem. `[unanimous]`
- The positive artifact is real: a 615M checkpoint exists, training completed, the family loader plausibly stabilized the run, and the Gutenberg-prefix loss descent is a narrow but real held-out learning signal. `[unanimous]`
- The 1.1092 bpb must be labeled as a fixed Gutenberg-books-prefix probe, not aggregate validation, historical competence, or external competitiveness. `[unanimous]`
- Minimal follow-up should **measure and bound**, not launder the model into a stronger claim: per-family/source bpb with uncertainty, tokenizer/provenance checks, matched anachronism logprobs, CORE/perplexity anchors, and source-register probes. `[skeptic, architect, empiricist; risk-analyst endorsed core subset]`
- The right publication form is a technical blog/field report/postmortem with model card, corpus/provenance card, and eval appendix. Not a model announcement; not yet a methods or corpus paper. `[unanimous]`
- The genuinely new contribution is the conjunction: governed historical corpus attempt, family-balanced loader, consumer-GPU execution, and unusually candid postmortem culture. Not scale, capability, or parity with Ranke-4B. `[unanimous]`
- Drop inflated theses: “validated vintage LLM,” “post-1913 ignorance,” “Ranke-4B competitor,” “aggregate 1.1092 validation,” “period factual competence,” “chat model,” “clean 1913 epistemic boundary,” and “general new loader method.” `[unanimous]`

## 2. Disagreements

**The disagreement:** Is the central story a failure/postmortem or a positive prototype?

**Failure/postmortem side:** Skeptic and Risk Analyst argue that the main danger is thesis inflation. The most durable thing here is not model capability but the discovery that smooth training curves and clean checkpoints can coexist with wrong provenance, wrong metric semantics, and weak construct validity.

**Positive-prototype side:** Empiricist and Architect argue that calling it merely a failure loses signal. The corrected checkpoint exists, the run completed, the loader appears locally decisive, and the Gutenberg result is real within its narrow construct.

**Adjudication:** The stronger synthesis is dual but asymmetric: the **positive artifact is the premise**, while the **postmortem/evaluation boundary is the thesis**. Phrase it as “a successful small run whose success exposed what historical-governed pretraining still fails to measure.”

---

**The disagreement:** How much can one weekend of evals upgrade the thesis?

**Upgrade side:** Architect, Empiricist, and Risk Analyst think weekend evals can turn the artifact into a calibrated case study: family/source generalization, cutoff selectivity, tokenizer leakage checks, and external anchors.

**Constraint side:** Skeptic insists this cannot become historical competence. A 615M base checkpoint with weak factual/arithmetic samples cannot be upgraded into a reliable vintage model by diagnostics alone.

**Adjudication:** Both are right, but Skeptic sets the ceiling. Weekend work can support “measured governed historical base-pretraining case study,” not “competent historical model.”

---

**The disagreement:** Is the corpus pipeline or family loader itself a publishable method?

**Method-candidate side:** Architect and Empiricist see the family-balanced loader as the strongest technical candidate contribution, supported by the smoke-series contrast.

**Too-early side:** Skeptic and Risk Analyst point out missing ablations, source-coverage caveats, no dedup story, and the fact that this proves only local rescue, not general method.

**Adjudication:** Claim it as a **local engineering recipe and case-study contribution**, not a methods paper result. A methods claim needs ablations or replication across mixtures.

---

**The disagreement:** Should “single 3090 feasibility” be a headline?

**Headline side:** Architect and Risk Analyst treat the single-author/single-3090 constraint as part of the contribution.

**Caution side:** Skeptic and Empiricist warn that one completed configuration does not establish broad consumer-GPU feasibility.

**Adjudication:** Use “realized feasibility under this exact configuration.” Do not generalize to historical LMs, larger models, stricter cutoffs, or useful assistants.

## 3. Open Questions

- What are per-family and per-source bpb numbers, with uncertainty? `[unanimous]`
- Was the tokenizer trained on any validation text, and does tokenizer-corpus alignment explain part of the Gutenberg result? `[skeptic, architect, empiricist]`
- Does the model distinguish pre-cutoff facts from post-cutoff foils better than a general-ignorance baseline? `[unanimous]`
- Does performance collapse outside Gutenberg/books, especially given the narrow validation probe? `[skeptic, empiricist]`
- Can CORE or another external anchor place this checkpoint in any meaningful comparison frame? `[architect, empiricist, skeptic]`
- Is the family loader generally useful, or only locally useful for this corpus/run? `[skeptic, architect, risk-analyst, empiricist]`
- How should Ranke-4B be compared if no protocol-comparable public evals exist? `[unanimous]`
- What exact corpus artifacts can be released or documented enough to make the governance story auditable? `[skeptic, risk-analyst, empiricist]`

## 4. Final Recommendation

Publish this as a **technical postmortem / field report**: “a governed historical pretraining postmortem from one RTX 3090.” The as-is thesis should be:

> A single-author, single-3090 project mechanically completed a 615M base-pretraining run on a rights/date/source-governed pre-1914 corpus; the corrected run shows a narrow held-out Gutenberg learning signal, but the main contribution is the audit trail showing that temporal-governed LMs fail first at provenance, mixture accounting, tokenizer/split semantics, and evaluation design.

With one weekend of follow-up, the strongest thesis becomes:

> This is a calibrated case study of low-budget governed historical base pretraining, with measured family/source generalization and bounded evidence about cutoff selectivity, while remaining explicitly not a chat model, not a validated historical reasoner, and not a Ranke-4B competitor.

Confidence: **high** that the publication boundary is right; **medium** that weekend evals will strengthen rather than narrow the thesis, because the probes may show Gutenberg/books was unusually favorable or reveal tokenizer/split issues.

Conditions that would change the recommendation: widen the claim only if per-family/source bpb is stable, tokenizer/provenance separation is clean, external anchors are sane, and matched anachronism probes show pre-cutoff advantage over post-cutoff foils. Narrow it if evals collapse outside Gutenberg, leakage appears, or cutoff probes show only general 615M ignorance.

Cheapest decisive next action: run the weekend eval bundle in this order: tokenizer/provenance audit, per-family/source bpb with uncertainty, CORE/perplexity anchor, matched anachronism logprob set, then source-register continuation/diagnostic QA. That directly tests whether the current thesis can become “measured case study” or must remain “postmortem plus narrow artifact.”

## 5. Attribution Map

| Claim | Contributing agent(s) |
|---|---|
| Not a validated historical/vintage LLM | Skeptic, Architect, Risk Analyst, Empiricist |
| As-is thesis should center governance/evaluation postmortem | Skeptic, Risk Analyst, Architect, Empiricist |
| Positive checkpoint and narrow Gutenberg learning signal are real | Empiricist, Skeptic, Architect |
| 1.1092 must not be treated as aggregate validation | Skeptic, Risk Analyst, Empiricist, Architect |
| Weekend evals can support a calibrated case study, not competence | Architect, Empiricist, Risk Analyst, Skeptic |
| Family loader is a candidate local engineering contribution, not proven general method | Architect, Empiricist, Skeptic, Risk Analyst |
| Publication form: technical postmortem plus model/corpus cards | Architect, Risk Analyst, Empiricist, Skeptic |
| Ranke-4B comparison should be constraint/transparency, not capability parity | Risk Analyst, Skeptic, Architect, Empiricist |
| Interesting non-flattering thesis: historical isolation is instrument-building, not a date cutoff | Skeptic, Empiricist, Architect, Risk Analyst |

