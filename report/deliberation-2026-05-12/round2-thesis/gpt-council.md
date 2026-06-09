# Round 2 Thesis - GPT Council Synthesis

Synthesizer note: this integrates the Skeptic, Architect, Risk Analyst, and Empiricist Round 2 outputs. I did not read `gpt-max.md`. The locked Round 1 validity result is treated as a constraint, not reopened.

## 1. Consensus

1. **The as-is thesis must be a methods/postmortem thesis, not a model-capability thesis.** [unanimous]
   The run supports a mechanically valid single-3090 615M base pretraining artifact on an auditable publication-year-bounded corpus. It does not support "validated historical LLM", "historical chatbot", "time-locked mind", or broad historical competence. Skeptic and Risk Analyst state this most sharply; Architect and Empiricist converge on the same boundary.

2. **The 1.1092 bpb is real but narrow.** [unanimous]
   It is a fixed 262,144-token Gutenberg books-prefix signal, not aggregate validation over the 2.86B-token held-out cache. The charitable form, emphasized by Skeptic and Empiricist, is that it measures cross-source held-out learning on a train-empty Gutenberg source. The limiting form, emphasized by Risk Analyst, is that it must not become the headline result.

3. **The strongest positive result is the validity machinery around the run.** [unanimous]
   The family-scheduled loader, provenance checks, stale-provenance postmortem, archived failed run, and source-aware audit trail are more publishable than the checkpoint's behavioral ability. Architect emphasizes the design path; Risk Analyst emphasizes the failure detectors; Skeptic and Empiricist emphasize that ordinary loss/VRAM/throughput metrics would have hidden the original data-integrity failure.

4. **One weekend of evals can improve the thesis, but cannot turn this into a mature historical model paper.** [unanimous]
   All four agents recommend minimal follow-up centered on per-family/source-stratified bpb, CORE for nanochat comparability, and controlled pre/post cutoff or anachronism probes. The weekend can make the artifact an evidence report; it cannot create SFT/RLHF, factual competence, a clean corpus paper, or Ranke-class capability.

5. **The right public form is a technical field report/blog plus model-card and postmortem appendices.** [unanimous]
   None of the agents recommend a model-release announcement, benchmark paper, or strong methods paper as-is. Empiricist allows that a workshop-style methods/corpus note could become defensible after the weekend evals; Risk Analyst and Skeptic are more conservative.

6. **Comparisons must be deflationary and precise.** [unanimous]
   Compared with Ranke-4B, this is not a scale or capability competitor. Compared with Owain Evans' vintage LLM framing, this is not conceptual novelty. Compared with standard nanochat, this is not "small LLM from scratch" novelty. The contribution is the conjunction: commodity-GPU historical corpus governance, family-scheduled loading, and visible failure analysis.

7. **The tempting inflated claims should be dropped.** [unanimous]
   Drop "validated historical LLM", "cannot know post-1914", "aggregate validation bpb", "balanced five-family corpus" if that implies source diversity, "Ranke on a 3090", "period-appropriate factual competence", "loader solved heterogeneous pretraining", and "postmortem proves robust governance."

## 2. Disagreements

### Disagreement 1: How positive should the as-is thesis sound?

**Each side's strongest argument.** Skeptic and Risk Analyst argue for a guarded thesis: the project is a mechanically completed base run and a failure-taxonomy/postmortem artifact. They warn that words like "prototype", "governed base model", or "vintage LLM" can smuggle in validation the evidence does not provide. Their strongest point is that Round 1 locked multiple construct-validity gaps: no per-family eval, no CORE, no anachronism eval, publication-year-only cutoff, and single-source train coverage in three families.

Architect and Empiricist allow a somewhat more positive phrasing: a "serious single-GPU methods artifact" or "historical base-model pipeline" is legitimate because the checkpoint exists, the corrected loader ran to completion, and the fixed Gutenberg-prefix bpb is a real learning signal rather than noise. Their strongest point is that underclaiming to "nothing happened" would waste the actual artifact: a 14-day consumer-GPU run with auditable governance and a concrete provenance failure repaired.

**Adjudication.** The stronger synthesis is positive but not capability-positive. Use Architect/Empiricist language for the artifact's reality, but Skeptic/Risk Analyst language for the claim boundary. The phrase "historical base-model run" is defensible; "validated historical model" is not. "Vintage-LLM methods field report" is safer than "vintage LLM prototype" unless the weekend evals pass.

### Disagreement 2: What is genuinely new?

**Each side's strongest argument.** Empiricist and Architect put the family-scheduled loader high: the smoke series suggests sequential loading produced shard-flip spikes or plateaus, while per-step family mixing stabilized training. This is the most concrete engineering contribution.

Skeptic and Risk Analyst caution that the loader is not yet a proven general method. There is no clean ablation beyond the smoke sequence, no source-aware downstream eval, and no proof that the schedule is optimal. They put more weight on the postmortem culture and validity machinery than on the loader as a method.

**Adjudication.** The novelty is not one ingredient. Single-3090 budget alone is not new because nanochat already makes that legible. A date cutoff alone is not new because Evans and Ranke occupy that conceptual territory. The loader alone is promising but under-validated. The strongest defensible novelty is the package: a small-budget historical-pretraining run where the corpus governance, family-scheduled loader, validation failure, and postmortem are all inspectable.

### Disagreement 3: How much should the Gutenberg-prefix result be valued?

**Each side's strongest argument.** Skeptic and Empiricist give the result its strongest legitimate interpretation: because Gutenberg is train-empty, descent on the fixed prefix is cross-source held-out learning, not mere training loss. It is narrow, but it is real evidence that the model learned some transferable English/book modeling.

Risk Analyst argues the result is dangerous as a public center of gravity. It is one deterministic slice, books-only, not representative of five families, and could easily be misread as "validation loss." Architect's draft also keeps it as supporting evidence, not thesis.

**Adjudication.** Include the Gutenberg result in the evidence table and model card; do not put it in the title, abstract headline, or central thesis. It is "real but mis-described", exactly as Round 1 concluded.

### Disagreement 4: What publication form becomes possible after a weekend?

**Each side's strongest argument.** Empiricist says that if weekend evals produce per-family bpb, CORE, source-conditioned continuations, cutoff probes, and tokenizer/duplicate checks, the work could become a short methods/corpus workshop note. This is plausible because the value would be the measurement package, not the model's strength.

Risk Analyst and Skeptic resist moving too fast: a weekend cannot repair corpus characterization, near-duplicate policy, publication-year/content-cutoff ambiguity, or provide clean loader ablations. They would keep the form as a technical blog or postmortem-methods note.

**Adjudication.** As-is: technical field report/blog plus model card and postmortem appendix. After weekend evals: workshop-style methods/postmortem note may be defensible if results are reported with failures intact. Still not a strong standalone corpus paper, model paper, or benchmark paper.

### Disagreement 5: Should the project use heroic/forge framing?

**Each side's strongest argument.** Risk Analyst notes that "forge record, not trophy" can motivate the researcher while preserving humility. Architect similarly suggests the literary frame belongs in essay voice, not in claims.

Skeptic's implied objection is that heroic framing can become a path for inflated claims if it substitutes for measurement. Empiricist's evidence-bound posture supports that concern.

**Adjudication.** Use the forge frame only as a narrative wrapper for disciplined workmanship under constraint: build, break, audit, and disclose. Do not let it become evidence. A good title can be plain; a subtitle or closing paragraph can carry the "forge log" tone.

## 3. Open Questions

1. **Per-family and per-source generalization.** No agent can answer whether the model compresses newspapers, legal, science, early-modern, and books coherently because the logged bpb only touches the Gutenberg prefix. Surfaced by Skeptic, Risk Analyst, Empiricist, Architect.

2. **Cutoff behavior versus small-model ignorance.** No agent can distinguish "does not know post-1914 because governed corpus" from "does not know much because 615M base-only model." Surfaced by Skeptic, Risk Analyst, Empiricist.

3. **Tokenizer and duplicate leakage.** Empiricist explicitly asks whether Gutenberg validation text entered tokenizer training or near-duplicate sources, and Round 1 also left tokenizer/corpus leakage under-verified. This matters because bpb can partially reflect tokenizer-corpus alignment.

4. **Corrected corpus table and source coverage.** The public writeup needs one canonical table distinguishing inventory, effective loader schedule, train sources, val-only sources, and single-source family coverage. Surfaced by Skeptic and Risk Analyst; reinforced by Round 1 synthesis.

5. **Loader generality.** The smoke series supports the family-scheduled loader locally, but no agent can answer whether it generalizes across corpora, schedules, model sizes, or random seeds. Surfaced by Skeptic, Architect, Risk Analyst, Empiricist.

6. **Comparison baseline availability.** A legacy/internal nanochat baseline may be useful only if the same eval harness can run and caveats are explicit. Surfaced by Architect and Empiricist.

7. **Ranke-4B status is a moving comparison target.** The current public prerelease describes a much larger 4B, 80B-token-per-cutoff, multi-cutoff setup with posttraining/evals; if Ranke's final release changes its transparency or small-budget posture, this project's comparative novelty may narrow. Synthesis-emergent from external comparison.

## 4. Final Recommendation

### Required Thesis Answers

1. **Strongest legitimate thesis as-is**

   Historical-nanochat is a mechanically credible single-author, single-RTX-3090 historical pretraining field report: a 615M nanochat-style base checkpoint trained from scratch on an auditable publication-year-bounded pre-1914 corpus, whose main contribution is the validity machinery around the run: family-scheduled loading, provenance checks, an archived stale-provenance failure, and the demonstration that ordinary training-health signals can look fine while the data-governance story is wrong. The 1.1092 bpb is supporting evidence of real but narrow Gutenberg-prefix cross-source learning, not aggregate validation.

2. **Strongest legitimate thesis with minimal follow-up**

   With one weekend of evals, the strongest defensible thesis is conditional:

   > A commodity-GPU historical-pretraining pipeline can produce a mechanically sound 615M pre-1914 base checkpoint whose source-family language modeling and cutoff behavior are measured explicitly, while also showing which governance checks are required before claiming historical competence.

   Minimal follow-up:
   - Per-family and per-source held-out bpb with fixed token budgets and uncertainty intervals.
   - CORE through the nanochat path for standard comparability, explicitly not as a historical-validity proof.
   - Pre/post cutoff logprob probes using base-model continuations or multiple-choice likelihood, with positive pre-1914 controls and post-1914 negative controls.
   - Source-conditioned continuation probes for books, legal, science, newspapers, and early-modern register.
   - Tokenizer/duplicate audit focused on whether validation text or near-duplicates influenced tokenizer or train data.

   Cheapest decisive experiment: per-family bpb plus a pre/post cutoff logprob probe. If those fail, the thesis stays a postmortem/methods artifact. If they pass, it can cautiously become a measured historical-base-model prototype field report.

3. **Secondary theses**

   - The stale-provenance postmortem is a first-class result: run #1 shows loss, val bpb, throughput, and VRAM can all pass while corpus identity is wrong.
   - The family-scheduled loader is a real local engineering contribution, supported by the smoke series, but should be claimed as an effective intervention in this run series rather than a solved general method.
   - The Gutenberg-prefix bpb is not worthless: it is a real train-empty-source held-out signal, just narrow and mis-described.
   - The single-3090 story matters only when joined to provenance and validation discipline; budget alone is not novel.
   - The negative behavioral result matters: a 615M base-only checkpoint can learn surface register without factual recall, arithmetic, or robust logic.

4. **Theses the researcher is tempted by that should be dropped**

   - "Validated historical LLM" or "validated governed-base-model PoC."
   - "The model cannot know post-1914 events."
   - "1.1092 bpb is aggregate validation."
   - "Competitive with Ranke-4B."
   - "Family-balanced corpus" if it implies multi-source train coverage across all families.
   - "Family-balanced loader solved heterogeneous historical pretraining."
   - "Period-appropriate factual competence."
   - "Chinchilla-optimal 30:1" without stating scheduled-token versus unique-token exposure.
   - "Postmortem guards prove robust governance."

5. **Appropriate publication form**

   As-is: technical field report or blog post plus model card, run card, corrected corpus/source table, and postmortem appendix. Use the spirit of model cards and dataset datasheets: intended use, non-use, corpus composition, metrics that are invalid, and failure modes.

   After weekend evals: possibly a short workshop-style methods/postmortem note if it includes the evaluation failures as well as successes. Not yet a strong model paper, benchmark paper, or standalone corpus paper.

   Title shape: "A Single-GPU Pre-1914 Nanochat Base Run: Corpus Governance, Loader Design, and What the Validation Metric Did Not Measure."

6. **What's genuinely new versus existing work**

   - **Versus Ranke-4B:** not scale, capability, posttraining, or concept. Ranke's public prerelease describes 4B Qwen3-style models, 80B training tokens per historical cutoff, a much larger historical corpus, SFT/RL, and explicit cutoff evaluation ambitions. Historical-nanochat's defensible edge is transparent hobby-scale execution and a visible failure/postmortem trail.
   - **Versus Owain Evans' vintage LLM concept:** not conceptual novelty. Evans already frames date-bounded training and future-leakage control as the core idea. Historical-nanochat contributes a concrete stress test showing that shard provenance, split semantics, tokenizer/corpus leakage, validation representativeness, and publication-year cutoff semantics are operationally central.
   - **Versus standard nanochat replications:** not "training a small model cheaply" and not superior bpb/CORE. The novelty is adapting nanochat into a governed historical-corpus lab with family-scheduled loading and source-aware validity demands.
   - **The real contribution:** the conjunction of small-budget execution, historical corpus governance, family-scheduled loader, and disciplined postmortem culture. If forced to name one contribution, choose "validity machinery for vintage LLM attempts", not "the model."

7. **Unified thesis recommendation**

   Publish this as a technical field report and model-card package about a single-3090, 615M-parameter, publication-year-bounded pre-1914 nanochat base run that succeeded mechanically and exposed the real research problem: vintage LLMs do not become valid because the loss curve falls under a date cutoff. They become credible only when source manifests, loader schedules, validation slices, tokenizer provenance, and cutoff semantics are made inspectable. The checkpoint learned a real but narrow Gutenberg-prefix signal and a surface historical register; it has not demonstrated broad source-family generalization, historical factual competence, or post-1914 ignorance. The interesting result is the forge log: build the run, break the governance story, catch the break, fix one class of failure, and state exactly what remains unproven.

### Confidence

High confidence in the direction: publish as a methods/postmortem field report, not as a model-capability claim. All agents converge, and Round 1 evidence strongly constrains the thesis.

Medium confidence in the strongest weekend-upgraded thesis because the needed evals are clear and cheap, but their outcomes are unknown.

Low confidence in any specific historical-capability claim until per-family bpb and cutoff probes exist.

### Conditions That Would Change the Recommendation

- If per-family bpb only works on books/Gutenberg, downgrade to a negative validation/postmortem report.
- If pre/post cutoff probes show post-1914 preference, drop "time-locked behavior" entirely.
- If pre-1914 controls are also random, do not claim historical knowledge; claim only language modeling and governance.
- If tokenizer or near-duplicate audit finds validation contamination, reframe the Gutenberg bpb as compromised measurement rather than held-out signal.
- If source-aware evals pass across families and cutoff probes behave cleanly, cautiously upgrade to "measured historical base-model prototype", still not chat model.

### Cheapest Next Action

Run a weekend eval bundle with a hard stop after two core scripts:

1. Per-family/source bpb over fixed equal token budgets from the held-out cache, including Gutenberg separately.
2. Pre/post cutoff logprob probes with matched pre-1914 positives, post-1914 negatives, and general-ignorance controls.

Add CORE if time remains because it anchors nanochat comparison, but do not let CORE substitute for the historical validity tests.

## 5. Attribution Map

| Claim | Contributing agent(s) |
|---|---|
| As-is thesis should be methods/postmortem, not validated historical LLM | Skeptic, Architect, Risk Analyst, Empiricist |
| 1.1092 bpb is fixed Gutenberg-prefix signal, not aggregate validation | Skeptic, Risk Analyst, Empiricist, Round 1 locked artifacts |
| Gutenberg signal is still real cross-source held-out learning | Skeptic, Empiricist |
| Do not headline 1.1092 | Risk Analyst, Skeptic, Architect |
| Weekend evals should prioritize per-family bpb and cutoff/anachronism probes | Skeptic, Risk Analyst, Empiricist, Architect |
| CORE is useful for nanochat comparability but not historical validity | Skeptic, Empiricist, Architect |
| Loader is a real local engineering contribution | Architect, Empiricist, Risk Analyst |
| Loader should not be claimed as a solved general method | Skeptic, Risk Analyst |
| Publication should be field report/blog plus model card/postmortem | Skeptic, Risk Analyst, Empiricist, Architect |
| Ranke comparison must be non-competitive and deflationary | Skeptic, Risk Analyst, Empiricist, Architect |
| Evans comparison means concept is not new | Skeptic, Empiricist, Architect |
| Standard nanochat comparison means single-GPU small-model training is not enough novelty | Skeptic, Empiricist, Architect |
| Main novelty is the combined validity machinery: governance, loader, eval failure, postmortem | Synthesis from all four agents |
| Heroic/forge framing is acceptable only as narrative wrapper, not evidence | Risk Analyst, Architect, synthesis |
