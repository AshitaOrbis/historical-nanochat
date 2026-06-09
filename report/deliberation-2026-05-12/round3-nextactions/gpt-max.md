# Round 3 - Next Actions Synthesis

Source note: the Empiricist section in the prompt was empty, so this synthesis uses the on-disk file `round3-nextactions/persona-empiricist.md` as the fourth persona input.

## 1. Consensus

- The branch-choice question is not "is the model good enough?" It is: closure on a v1 calibration artifact, or commitment to a v2 instrument program whose first work is measurement repair, not another training run. [unanimous]

- Publish-and-move-on is the default recommendation after one disciplined weekend eval bundle. All four agents treat the current artifact as already able to support the locked thesis: instrument prototype, calibration log, and claim-collapse anatomy. [unanimous]

- Build-next-version is rational only behind gates: named hypotheses, stable data/provenance invariants, per-family validation, tokenizer/split audit, cross-source dedup, cutoff policy, and a capped compute plan. [unanimous]

- The 615M checkpoint is structurally insufficient for the original cutoff-capability thesis. At this scale, "does not know post-1914 facts" is confounded with "does not know much factual material at all." [unanimous]

- The 1.1092 bpb result must not be a headline. It is a fixed 262k-token Gutenberg-prefix calibration reading, not aggregate validation, not broad held-out generalization, and not evidence of historical competence. [unanimous]

- Publication-year cutoff is not semantic time isolation. It cannot in principle exclude all content-semantic anachronism; the v1 artifact must disclose this as a structural limitation, not a pending polish issue. [unanimous]

- Immediate 1-3B scaling is rejected. A 1B run is the first plausible next-scale pilot only after gates; 3B is external-compute or months-scale territory, not a default single-3090 next action. [unanimous]

- Cache/provenance redesign beats another loader patch for v2. A loader patch can be acceptable for v1 evaluation guards, but a v2 instrument needs stable IDs, source/document metadata through tokenization, deterministic split semantics, and refusal tests. [architect, risk-analyst, empiricist, skeptic]

- The minimum weekend eval bundle should be falsification-oriented. Each item must be decisive against a named claim rather than decorative support for a writeup. [unanimous]

- Release should prioritize GitHub materials: report, eval scripts, configs, manifests/provenance summaries, cards, postmortems, and exact commands. HuggingFace weights are optional and conditional on blunt model-card framing. [unanimous, with risk-analyst and empiricist most restrictive]

- Highest teaching per GPU-hour is evaluating the existing checkpoint, especially per-family bpb and matched cutoff logprob probes. Highest teaching per human-hour is claim hygiene: tokenizer/split audit, duplicate pilot, claim ledger, field report, and model/corpus cards. [unanimous]

## 2. Disagreements

### 2.1 Checkpoint release threshold

**The disagreement:** Should the 615M checkpoint be released on HuggingFace as part of publication, or should public release focus on code, cards, logs, evals, and reports?

**Release-if-carded side:** The Architect treats weights/tokenizer as optional secondary artifacts: release them only with a blunt base-model card that makes no SFT/RLHF, narrow validation, source limitations, and cutoff semantics unavoidable. The Skeptic also allows a research base-model release if the card is hard-edged and the checkpoint is not marketed as a demo or assistant.

**More restrictive side:** The Risk Analyst and Empiricist argue that checkpoint publication is not neutral packaging. It can invite "vintage assistant" framing, misuse, or ridicule, while adding less value than the report and reproducible eval bundle. Empiricist is especially clear: because the checkpoint has not demonstrated historical competence, factual competence, arithmetic, or broad held-out generalization, release weights only if the eval bundle and model card travel beside it.

**Adjudication:** Risk Analyst and Empiricist have the stronger case. The publication value is in the calibration log and claim-collapse anatomy, not in turning the model into the public object. Recommended policy: GitHub release is default; HuggingFace tokenizer/config/card are acceptable; HuggingFace checkpoint release waits until the weekend eval bundle is attached and the repository name/card prevent assistant framing.

### 2.2 Desire for an instrument program vs evidential gates

**The disagreement:** Is the researcher's desire to continue sufficient to choose build-next-version, or must the weekend gates justify it independently?

**Motivation-sensitive side:** Architect, Risk Analyst, and Empiricist all allow branch 2 if the researcher explicitly wants a months-scale instrument program after understanding the opportunity cost. They treat the branch choice partly as a values decision: closure artifact versus new program.

**Gate-first side:** Skeptic argues that desire is evidence about motivation, not experimental validity. If per-family validation, dedup, tokenizer/split audit, cutoff probes, or cache/provenance invariants fail, scale-up must be canceled rather than reclassified as "still useful."

**Adjudication:** Skeptic is right about validity; the others are right about project choice. Desire can choose between valid options, but it cannot override failed gates. The clean rule is: branch 2 requires both an active yes from the researcher and passing instrument gates. Either missing condition defaults to publish-and-move-on.

### 2.3 Sequence: write first or eval first

**The disagreement:** Should the next action start with writing the field report or with the weekend eval bundle?

**Write-first side:** Skeptic and Risk Analyst rank the field report or claim ledger very high because writing forces claim hygiene and creates the artifact even if evals disappoint. Risk Analyst notes that inability to state limitations plainly is itself a publication blocker.

**Eval-first side:** Architect and Empiricist put the weekend eval bundle at the hinge because it determines whether the artifact can be calibrated beyond the Gutenberg prefix or must be narrowed to a postmortem-only frame.

**Adjudication:** This is mostly a scheduling disagreement, not a conceptual one. The highest-EV sequence is: spend 2-4 hours freezing the claim ledger and report outline, run the weekend eval bundle, then write the final report/cards with the measured results. That preserves claim hygiene without letting prose outrun evidence.

### 2.4 Loader patch vs cache redesign

**The disagreement:** Is a hardened loader patch enough for v2, or does v2 require cache/provenance redesign?

**Patch-tolerant side:** Empiricist allows a 4-8 hour loader patch keyed to stable source/document identity as a temporary guard, especially for v1 evaluation or short-term refusal checks.

**Redesign side:** Architect and Risk Analyst argue that loader patching is not a v2 foundation. V2 should make cache/provenance stable by construction: content-stable IDs, document-level split semantics, source/family metadata carried through tokenization, interleaved or explicitly sampled validation, and startup refusal if cache and provenance diverge. Skeptic supports this by warning that scale amplifies unresolved invariants.

**Adjudication:** Redesign wins for v2. A patch is acceptable only as a temporary guard for evaluating v1 or preventing a known repeat failure. It should not be used to justify a 1B run.

### 2.5 Chinchilla-for-historical-corpora priority

**The disagreement:** Should the historical-corpus scaling-law question be part of the next version or treated as an adjacent project?

**Include-as-side-study side:** Architect and Empiricist see a real adjacent question: historical governed corpora may have different compute-optimal token/parameter ratios than modern web mixtures. A small grid using 125M/300M/existing 615M and shared per-family evals could teach something before 1B.

**Do-not-distract side:** Skeptic and Risk Analyst warn that one big run cannot answer the question and that it can easily distract from the core v2 requirement: stable measurement. Risk Analyst says design now, answer later.

**Adjudication:** Treat it as adjacent and optional. Include it only after invariant repair and only as a small scaling grid with explicit curves and stop rules. Do not let "Chinchilla for historical corpora" become a reason to skip dedup, cutoff policy, or eval validity.

## 3. Open questions

- Does the researcher actively want a months-scale v2 instrument program after seeing that the first milestones are cache redesign, dedup, eval harnesses, and kill gates rather than a satisfying larger run? Surfaced by Architect, Risk Analyst, Empiricist, and Skeptic.

- What do per-family held-out readings look like once evaluation samples multiple shards and families rather than the fixed Gutenberg prefix? Surfaced by all agents.

- Was the tokenizer trained on text that includes validation content, and are train/val tokenizer/cache semantics identical across the run? Surfaced by Empiricist and grounded in R1 synthesis.

- What is the cross-source and train/val near-duplicate rate after document/passage-level dedup checks? Surfaced by all agents; Risk Analyst treats this as a high-impact scale risk.

- Is there any nonzero cutoff-visible signal at 615M on matched pre/post-cutoff probes, or do all such probes collapse into generic small-model incompetence? Surfaced by all agents, with Empiricist requiring a named hypothesis.

- Can source-register continuation show reliable family/style control beyond anecdotal samples? Surfaced by Architect, Risk Analyst, and Empiricist.

- What is the actual throughput, memory, and wall-time cost for a 1B Chinchilla-like run on the current 3090 stack after v2 infrastructure changes? Surfaced by Skeptic, Risk Analyst, and Empiricist.

- Are rights constraints compatible with releasing tokenizer, checkpoint, token caches, or only metadata/report/code? Surfaced by Risk Analyst and Empiricist.

- Should 1913 be defended as a pre-Sarajevo publication-year probe, or relativized as one arbitrary cutoff among several? Surfaced by Architect, Risk Analyst, Empiricist, and Skeptic.

- Can content-semantic anachronism filtering achieve tolerable false positive/negative rates, or should it be explicitly abandoned as a corpus guarantee and moved to audit/eval? Surfaced by all agents.

- Are historical corpora compute-optimal at a different token/parameter ratio than modern web mixtures? Surfaced by Architect and Empiricist; treated as adjacent by Risk Analyst and Skeptic.

## 4. Final recommendation

**Recommendation:** Publish-and-move-on by default, after one disciplined weekend eval bundle. Treat the artifact as the destination unless three things are true: the weekend evals are clean enough, the researcher actively wants a months-scale instrument program, and v2 has pre-registered kill gates plus repaired data/eval invariants before any scale-up.

**Confidence:** High that this direction is right, because all four agents independently converge on it and R1/R2 lock the core evidence limits. Medium confidence on exact cost estimates because eval plumbing, dedup runtime, rights constraints, and 3090 throughput can surprise.

### 4.1 The branch-choice question

Ask it this way:

> Do I want to publish v1 as an instrument calibration/postmortem artifact, or do I want to spend the next project cycle building a v2 instrument capable of testing the cutoff hypothesis at 1B+ scale?

If the answer is not an active yes to the second, choose publish-and-move-on.

### 4.2 Branch 1 - Publish-and-move-on

Ranked by expected value per hour:

| Rank | Action | Cost | Blockers | Falsifier / stop rule |
|---:|---|---:|---|---|
| 1 | Freeze claim ledger and report outline | 2-4 human hours, 0 GPU | Emotional pull to rescue the original capability thesis | If the outline still headlines 1.1092 or "historical LLM," narrow to postmortem frame before continuing |
| 2 | Run the minimum weekend eval bundle | 12-25 human hours, likely under 2 GPU-hours plus CPU time | Eval harness friction; unclear slice definitions | Severe split leakage, duplicate contamination, tokenizer ambiguity, or total cross-family collapse means publish as postmortem/calibration failure, not model artifact |
| 3 | Write the technical field report / calibration log | 8-16 human hours | Temptation to make the model the hero | If limitations cannot be stated plainly, do not publish yet |
| 4 | Write model card and corpus card | 4-8 human hours | Rights/source ambiguity; release-framing drift | Any sentence that lets a reader infer "validated historical LLM" blocks release |
| 5 | Release GitHub materials | 4-8 human hours | Separating reproducibility metadata from restricted data | If rights constraints cannot be separated from raw data/token caches, release metadata and scripts only |
| 6 | Optional HuggingFace release | 3-6 human hours | Misuse/assistant framing; incomplete eval bundle | Do not release checkpoint unless the eval bundle, card, and repo naming make base-model incapability impossible to miss |
| 7 | Freeze v1 | 2-4 human hours | New task accretion | Only an explicit branch-2 commitment reopens training work |

What to write:

- A long-form technical field report framed as "first prototype of a measurement instrument for time-isolated language modeling, plus calibration log and worked anatomy of claim-collapse."
- Sections: run record, corpus/governance tuple, loader and stale-provenance postmortem, what 1.1092 actually measures, weekend eval results, what claims survived, what claims collapsed, and what v2 would require.
- Appendices: model card, corpus card, eval appendix, stale-provenance postmortem, exact commands/configs, release notes.

Minimum viable release:

- GitHub: report, configs, eval scripts, provenance/manifest summaries, cards, postmortems, exact commands, and reproducibility instructions.
- Do not release raw corpora unless rights allow it.
- Do not release run #1 as a usable model artifact.
- HuggingFace: tokenizer/config/card are acceptable if clearly marked research-only; checkpoint release is conditional and secondary.

Model card must disclose:

- 615M base pretraining checkpoint.
- No midtraining, SFT, RLHF, chat competence, or assistant behavior.
- 1.1092 is a fixed 262k-token Gutenberg-prefix calibration reading, not aggregate validation.
- No demonstrated factual competence, arithmetic, logic, source-grounded QA, or cutoff competence.
- Publication-year cutoff only; no semantic time-isolation guarantee.
- Known source/diversity caveats, train-empty sources, family exposure asymmetries, and no cross-source dedup unless the weekend scan changes that.
- Intended use is research audit/calibration only, not factual historical QA or roleplay deployment.

### 4.3 Branch 2 - Build-next-version

Ranked by expected value per hour:

| Rank | Action | Cost | Blockers | Falsifier / stop rule |
|---:|---|---:|---|---|
| 1 | Write hypotheses and kill gates | 4-8 human hours | Inability to name a capability distinction scale could decide | If no named hypothesis would change behavior, do not build v2 |
| 2 | Gate on the weekend evals | 12-25 human hours, under 2 GPU-hours plus CPU | Eval plumbing; ambiguous labels | Stop if per-family bpb is incoherent, duplicate risk is high, tokenizer/split semantics are dirty, or cutoff probes show only generic incompetence |
| 3 | Redesign cache/provenance, not just loader | 3-7 days human/CPU; a 4-8 hour loader patch is only temporary | Current shard/provenance coupling; metadata drift | Stop if the pipeline cannot preserve source/document/family/split invariants by construction and refuse divergence at startup |
| 4 | Run cross-source near-dedup | 2-5 days CPU/disk/human for pilot/full pass depending depth | OCR noise; heterogeneous metadata; disk pressure | High train/val or cross-source duplicate rate invalidates simple scale-up until fixed |
| 5 | Decide cutoff policy | 4-8 hours to write policy; 3-10 days if testing a semantic filter | Treating 1913 as self-evident; high filter error rates | If 1913 cannot be defended, relativize it; if semantic filtering lacks measured error rates, abandon semantic-purity claims |
| 6 | Build v2 eval harness before training | 20-40 human hours, low GPU | Evals that cannot separate small-model incompetence from cutoff isolation | Stop if evals cannot distinguish broad modeling, source-register control, source-grounded QA, and cutoff-visible behavior |
| 7 | Optional small scaling grid for historical Chinchilla question | 20-60 GPU-hours plus analysis, after invariant repair | Variance and corpus defects overwhelming curves | Stop if curves are too noisy or confounded to inform 1B |
| 8 | 1B pilot only after gates | Plausibly multi-week 3090 run plus engineering | Throughput/memory limits; opportunity cost | Cancel if v2 gates fail, evals remain non-decisive, or compute plan exceeds cap |
| 9 | 3B design target, not near-term action | External compute or months-scale commitment | Single-3090 economics | Do not attempt as default local run |

Branch-2 thesis:

- The v2 question is not "can we train a bigger old-text model?"
- The v2 question is: "At a capability-relevant scale, does a publication-year-isolated corpus produce measurable cutoff behavior under an eval suite that can separate temporal isolation from generic incapability?"

Cache redesign requirements:

- Stable content/document/source IDs.
- Document-level split semantics.
- Source/family metadata carried through tokenization and cache build.
- Per-family validation loaders or interleaved validation cache.
- Provenance regenerated from build artifacts, not manually synchronized.
- Startup refusal if cache, manifest, provenance, tokenizer, or split metadata diverge.

Cutoff policy:

- Defend 1913 as a pragmatic pre-Sarajevo publication-year probe, or relativize it as one cutoff among many.
- Do not claim semantic isolation unless a measured filter exists.
- Prefer treating content-semantic anachronism as an audit/eval dimension, not an admission guarantee.

Scaling:

- 1B is the first plausible next-scale pilot after gates.
- 3B is not a single-3090 next action.
- The Chinchilla-for-historical-corpora question is adjacent. It deserves a small controlled grid only after the instrument is stable.

### 4.4 Explicit rejects

- Immediate 1-3B scale-up.
- SFT, RLHF, or chat demo as a rescue.
- "Vintage LLM" or "validated historical LLM" branding.
- Headlining 1.1092 bpb.
- Calling 1.1092 aggregate validation.
- Claiming post-1914 ignorance or semantic time isolation.
- Comparing bpb to Ranke/FineWeb/nanochat as if protocols are comparable.
- Treating loader patching as equivalent to cache/provenance redesign.
- Trying to perfectly filter content-semantic anachronism without measured false positive/negative rates.
- Broad new corpus collection before dedup, split, cache, and eval invariants exist.
- Publishing weights without a blunt card and adjacent eval bundle.
- Letting "instrument program" become an identity claim that overrides kill gates.

### 4.5 Minimum-viable weekend eval bundle

Ordered by expected value:

| Rank | Eval | Cost | Decisive against this claim | Falsifying result |
|---:|---|---:|---|---|
| 1 | Reproduce the 262k Gutenberg-prefix bpb, then compute per-family bpb on multiple held-out shards | 3-6 hours, under 1 GPU-hour | "The learning signal is broad across the governed corpus" | Only Gutenberg/books looks coherent, or family readings are unstable/chaotic |
| 2 | Tokenizer training-corpus, split-semantics, and val-cache continuity audit | 1-3 hours, 0 GPU | "The bpb is cleanly interpretable" | Tokenizer trained on val-specific text without disclosure, train/val cache mismatch, or run2/run3 continuity ambiguity |
| 3 | Cross-source near-duplicate pilot | 4-8 hours CPU/human | "Held-out/source independence is credible" | Material train/val or cross-source overlap |
| 4 | Matched pre/post-cutoff logprob probes | 4-6 hours, under 1 GPU-hour | "615M already shows cutoff-visible knowledge behavior" | No separation beyond generic incompetence |
| 5 | Source-register continuation/cloze probes by family | 3-5 hours, low GPU | "Period/source style generalizes beyond anecdotal samples" | Failure to condition on legal/newspaper/science/books/early-modern registers |
| 6 | Tiny source-grounded QA sanity set | 2-4 hours, low GPU | "The checkpoint has period-appropriate factual competence" | Factual failures like existing samples; this should block competence claims |
| 7 | External anchor such as CORE-equivalent if plumbing is close | 2-4 hours | "We can place this model relative to anything else" | Harness too costly or incomparable; leave out rather than force it |

Total expected cost: 19-36 human hours if all items are run; likely under 2 GPU-hours plus CPU/disk time. If time is tight, do ranks 1-4 first.

### 4.6 Cost-benefit framing

- Highest "what would this teach me" per GPU-hour: per-family bpb and matched cutoff logprob probes, because they use the existing checkpoint to test whether any broad or cutoff-visible signal exists.
- Highest "what would this teach me" per human-hour: tokenizer/split audit and claim ledger, because either can prevent weeks of wasted training or public overclaim.
- Highest overall EV/hour: weekend eval bundle plus field report. It either produces a clean calibrated v1 or decisively narrows the artifact to an honest postmortem.
- Lowest EV/hour: immediate scale-up. It consumes weeks of 3090 time while preserving the same interpretability failures.

### 4.7 What would change this recommendation

Shift toward build-next-version if all or most of these are true:

- Weekend evals show stable cross-family held-out compression.
- Tokenizer/split/val-cache continuity audit is clean.
- Cross-source duplicate pilot finds low contamination.
- Matched cutoff probes show at least weak cutoff-sensitive behavior rather than generic failure.
- Source-register and source-grounded probes show enough signal to motivate scale.
- The researcher explicitly accepts a months-scale instrument-building cycle with kill gates and a compute cap.
- External compute becomes available for 3B, or 1B throughput is empirically much better than expected.

Shift harder toward publish-only or private postmortem if any of these occur:

- Duplicate leakage or broken split semantics compromises the calibration reading.
- Per-family bpb collapses outside Gutenberg/books.
- Tokenizer or cache continuity is ambiguous enough to make bpb uninterpretable.
- Rights constraints block meaningful public model/corpus release.
- The draft keeps drifting into capability claims.
- V2 planning starts with parameter count before invariants.

### 4.8 Final unified recommendation

Publish-and-move-on after one weekend eval gate. The existing run has already earned a useful artifact: a single-author, single-3090 calibration log showing how historical-LLM claims collapse at provenance, split semantics, source coverage, tokenizer/eval interpretation, and cutoff granularity. Do not spend another multi-week 3090 cycle to avoid finishing. Reopen build-next-version only if the weekend measurements are clean, the researcher actively wants the opportunity cost, and v2 begins with hypotheses, cache/provenance redesign, dedup, cutoff policy, and eval gates before training.

## 5. Attribution map

| Claim | Contributing agents |
|---|---|
| Branch choice is closure artifact vs instrument program | Architect, Empiricist, Risk Analyst, Skeptic |
| Publish-and-move-on is default | Skeptic, Architect, Risk Analyst, Empiricist |
| Weekend evals should be gates, not decoration | Empiricist, Skeptic, Risk Analyst, Architect |
| Desire to continue cannot override failed gates | Skeptic, supported by Risk Analyst and Empiricist |
| The artifact's public thesis is instrument prototype + calibration log + claim-collapse anatomy | Architect, Empiricist, Skeptic, Risk Analyst, grounded in R2 synthesis |
| 615M cannot test the original cutoff-capability thesis | Architect, Skeptic, Risk Analyst, Empiricist |
| 1.1092 is a narrow Gutenberg-prefix calibration reading | All agents, grounded in R1 empirical findings |
| Publication-year cutoff is not semantic isolation | All agents, grounded in R1/R2 synthesis |
| GitHub artifacts should be primary release | Architect, Risk Analyst, Empiricist, Skeptic |
| HF checkpoint release should be conditional and secondary | Risk Analyst, Empiricist, with conditional support from Architect and Skeptic |
| Cache redesign is required for v2 | Architect, Risk Analyst, Skeptic, Empiricist |
| Loader patch is only temporary | Empiricist, Architect, Risk Analyst |
| Cross-source dedup is mandatory before scale | Architect, Risk Analyst, Empiricist, Skeptic |
| Defend or relativize 1913 | Architect, Risk Analyst, Empiricist, Skeptic |
| Abandon semantic anachronism as a guarantee unless a measured filter exists | Empiricist, Risk Analyst, Architect, Skeptic |
| 1B is a conditional pilot; 3B is not a local default | Skeptic, Risk Analyst, Empiricist, Architect |
| Chinchilla-for-historical-corpora is adjacent, not answered by v1 | Architect, Empiricist, Risk Analyst, Skeptic |
| Highest teaching per GPU-hour is existing-checkpoint eval | Empiricist, Skeptic, Risk Analyst, Architect |
| Highest teaching per human-hour is claim hygiene/audits/writeup | Empiricist, Risk Analyst, Skeptic, Architect |
