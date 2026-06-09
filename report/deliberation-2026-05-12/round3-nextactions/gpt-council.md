# Round 3 Next Actions - GPT Council Synthesis

Source note: this integrates the Skeptic, Architect, Risk Analyst, and Empiricist Round 3 outputs supplied for this synthesis, with R1 validity and R2 thesis treated as locked constraints. The thesis remains: first prototype of a measurement instrument for time-isolated language modeling, plus calibration log and worked anatomy of claim-collapse. The 615M scale cannot test the original cutoff-capability thesis. The 1.1092 bpb is a calibration reading on the narrowest validated probe.

## 1. Consensus

- **Default branch is publish-and-move-on after one tight weekend eval bundle.** All four agents converge here. The current artifact is valuable enough to write up, but not clean enough to justify immediate scale-up. [unanimous]

- **Branch 2 is not "make the model bigger"; it is a new instrument program.** The first work would be eval harnesses, cache/provenance redesign, split semantics, dedup, and cutoff/anachronism definitions. Architect frames this most constructively; Skeptic and Risk Analyst make it a hard gate; Empiricist ties it to decisive hypotheses. [unanimous]

- **Do not use `1.1092 bpb` as aggregate validation or as evidence of historical competence.** It is a fixed Gutenberg-prefix calibration signal. Empiricist and Risk Analyst emphasize the empirical narrowness; Skeptic stresses it is still real within its construct; Architect accepts it only as a calibration reading. [unanimous]

- **The existing 615M checkpoint cannot test "post-1914 ignorance."** Failure to know modern facts at this scale is confounded with generic incapability. Publication-year cutoff also cannot guarantee content-semantic cutoff. [unanimous]

- **Weekend evals should be gates, not decoration.** They should decide whether the writeup is an instrument-calibration report, a stricter postmortem, or the start of a v2 program. [unanimous]

- **Cache redesign beats another loader patch for v2.** A loader patch can be acceptable for v1 evaluation and short-term guards, but any new training program needs stable identities, provenance invariants, source/document/split metadata carried through tokenization, and refusal tests. [unanimous, with Empiricist allowing a temporary patch only]

- **Cross-source dedup and train-val contamination checks are mandatory before scale or strong held-out claims.** Dedup may preserve historically meaningful circulation inside train, but it cannot be skipped for evaluated splits. [unanimous]

- **1B is conditional; 3B is not a local next action on the 3090.** Agents differ on exact cost, but all reject immediate 3B local training and immediate 1-3B scaling. [unanimous]

- **SFT/RLHF/chat demos are the wrong next move.** They would turn a measurement project into a demo project and make claims harder to interpret. [unanimous]

- **Highest teaching value per GPU-hour is existing-checkpoint evaluation.** Per-family bpb, matched cutoff logprobs, source-register probes, and CORE/external anchors teach more per GPU-hour than a new run. [unanimous]

- **Highest teaching value per human-hour is claim hygiene and audit work.** Claim ledger, tokenizer/split audit, duplicate pilot, corpus/model cards, and field-report writing dominate immediate human-hour ROI. [unanimous]

## 2. Disagreements

### 2.1 Is this v1 of a continuing instrument program, or already the destination?

**The disagreement:** The agents agree on the default, but differ on how live Branch 2 should be after the weekend.

**Publish-and-move-on side:** Skeptic, Risk Analyst, and Empiricist argue that "instrument program" can become an identity trap. The missing piece is not merely scale. The missing pieces are construct-valid measurements, stable cache/provenance invariants, split/dedup integrity, and explicit cutoff semantics. A bigger run before those fixes mostly buys a more expensive ambiguity. Empiricist adds the base-rate argument: the researcher has already spent weeks of GPU time and many days of human time, while the marginal evidence per hour is now much higher for calibration and writeup.

**Build-next-version side:** Architect keeps Branch 2 alive as a legitimate research program if the researcher wants a multi-month commitment. The 615M checkpoint is below the scale needed for the original capability thesis, so a better instrument plus a 1B-class pilot could eventually test something the current model cannot. Risk Analyst and Empiricist also allow this if the weekend bundle is clean and a falsifiable v2 hypothesis exists.

**Adjudication:** Publish-and-move-on is stronger as the default. Architect's branch survives only if modified by Skeptic/Risk gates: no scale-up until the weekend evals are clean enough, the user explicitly wants a multi-month program, and the next run has a named hypothesis with kill criteria.

### 2.2 How positive should the public framing be?

**The disagreement:** Is "instrument prototype" appropriately positive, or does it smuggle in unsupported validation?

**Positive-artifact side:** Architect and Empiricist stress that underclaiming to "nothing happened" would waste the real achievement: a completed single-3090 615M base-pretraining run, a corrected family-scheduled loader, an archived provenance failure, and a real learning trajectory on a held-out slice. The contribution is not model capability, but the instrument, calibration trail, and failure anatomy.

**Caution side:** Skeptic and Risk Analyst warn that "prototype" can drift into "validated historical LLM." The 1.1092 number is narrow, the cutoff is publication-year-only, the model samples do not show factual competence, and public readers will overread any checkpoint release unless the claims are bluntly fenced.

**Adjudication:** Use positive language for the artifact's reality and guarded language for capability. The correct genre is a technical field report / calibration log / claim-collapse case study, not a model paper, not a capability paper, and not "vintage LLM" branding.

### 2.3 Should the checkpoint be released?

**The disagreement:** Is a public checkpoint a useful artifact or a source of overclaim?

**Release-if-carded side:** Architect and Skeptic allow a secondary research-artifact release if the model card makes the limitations hard to miss: 615M base checkpoint, no SFT/RLHF, no demonstrated historical competence, narrow validation reading, publication-year-only cutoff, no semantic anachronism guarantee, and caveats beside the weights.

**Restrictive side:** Empiricist and Risk Analyst see release as non-neutral. If weights become the public object, the project will be judged as a weak historical model instead of a strong audit/calibration report. Empiricist recommends releasing code, configs, eval scripts, reports, cards, tokenizer metadata, and provenance summaries first.

**Adjudication:** GitHub/report artifacts should be primary. A HuggingFace checkpoint is optional and secondary, only after the weekend bundle and only with a blunt model card. No chat demo.

### 2.4 Loader patch versus cache redesign

**The disagreement:** Can v2 proceed with a hardened loader, or does it need a redesigned cache/provenance layer?

**Patch-tolerant side:** Empiricist allows a 4-8 hour loader patch for short-term checks, especially to key checks to stable source/document identity and make v1 eval safer.

**Redesign side:** Architect, Risk Analyst, and Skeptic argue that v2 needs provenance correctness by construction. The v1 failure was a schema-coupling failure. A startup guard is not the same as stable IDs, atomic manifests, split-aware dedup, train-only tokenizer inputs, source/file/document metadata through tokenization, and golden small-corpus tests.

**Adjudication:** Redesign wins for Branch 2. Loader patching is acceptable only for v1 evaluation or temporary refusal checks.

### 2.5 Is Chinchilla-for-historical-corpora a core next question?

**The disagreement:** The historical-corpus scaling question is real, but may distract from the writeup.

**Include-as-side-study side:** Architect and Empiricist see a valuable adjacent question: historical governed corpora may not share modern web-text compute-optimal token/parameter ratios. A small scaling grid could teach something before 1B.

**Defer side:** Skeptic and Risk Analyst warn that Chinchilla language can become another scale-up justification before the measurement rig is fixed. The current run cannot answer the question, and a noisy scaling grid would be expensive theater.

**Adjudication:** Include it only in Branch 2 after the eval/data rig is stable. It is not a reason to delay Branch 1 publication and not a reason to launch 1B now.

## 3. Open Questions

- Does the researcher actively want a multi-month v2 instrument program, or is closure the real objective? Surfaced by Architect, Risk Analyst, Empiricist, and Skeptic; sharpened by synthesis.

- What exactly was the tokenizer trained on, and did tokenizer training include validation text or Gutenberg/source-specific text? Surfaced by Empiricist, Risk Analyst, and Skeptic; inherited from R1/R2.

- Does per-family and per-source bpb show coherent compression beyond the Gutenberg prefix? Surfaced by all agents.

- Are train/val/source boundaries clean under exact and near-duplicate checks? Surfaced by all agents.

- Does the 615M checkpoint show any matched pre/post-cutoff logprob differential, or only generic ignorance? Surfaced by all agents.

- Can source-register continuation or cloze probes show style/source control beyond anecdotal samples? Surfaced by Architect, Risk Analyst, and Empiricist.

- Should 1913 be defended as a meaningful pre-Sarajevo publication-year cutoff, or relativized as one pragmatic cutoff among several? Surfaced by all agents.

- Can a 1B-class model fit and train on the local 3090 with acceptable throughput after v2 changes? Surfaced by Skeptic, Risk Analyst, and Empiricist.

- Do historical/OCR-heavy governed corpora have a different compute-optimal token/parameter ratio than modern web mixtures? Surfaced by Architect and Empiricist; treated as adjacent by Skeptic and Risk Analyst.

- Are rights, provenance, and artifact packaging clean enough for public checkpoint release, or only for report/code release? Surfaced by Risk Analyst and Empiricist.

## 4. Final Recommendation

### Recommendation

Run one bounded weekend calibration bundle, then publish-and-move-on by default. The project has already produced the publishable object: a mechanically completed single-3090 historical pretraining artifact, an auditable governance trail, and a worked anatomy of how stronger historical-LM claims collapse under provenance, split, validation, and cutoff scrutiny. Build v2 only if the weekend bundle is clean enough and the researcher explicitly chooses a multi-month instrument program whose first milestones are measurement and data-invariant work, not a larger training launch.

Confidence: high that the direction is right. Medium on exact cost estimates because the eval harness friction, duplicate-scan runtime, tokenizer audit state, and 1B throughput are open variables. Low confidence that a 615M or immediate 1B run can answer the original semantic cutoff thesis without new instrumentation.

### Branch 1: Publish-and-Move-On

| Rank | Action | EV/hr | Cost | Blocker | Falsifier / downgrade |
|---:|---|---|---|---|---|
| 1 | Run the weekend eval bundle and freeze a claim ledger | Very high | 12-28 human h, 3-8 GPU h, CPU/I/O for dedup | small eval scripts, noisy item writing | severe leakage, duplicate contamination, or total non-books collapse turns the writeup into stricter postmortem |
| 2 | Write the field report as instrument prototype + calibration log + claim-collapse anatomy | Very high | 10-20 human h | capability-language temptation | any abstract/headline treats `1.1092` as aggregate validation or implies historical competence |
| 3 | Add corpus card, model card, and run manifest | High | 6-10 h | source-mix ambiguity, train-empty source disclosures, rights limits | cannot state train/val sources, tokenizer inputs, split method, duplicate policy, cutoff semantics |
| 4 | Archive reproducibility bundle: logs, hashes, eval scripts, checkpoint manifest, postmortem | High | 2-8 h | disk pressure, large artifacts | missing final checkpoint/tokenizer/cache provenance hashes make claims unauditable |
| 5 | Release selectively | Medium | 2-6 h | public misread of weights | checkpoint release lacks hard caveats; any demo becomes evidence of competence |

Release rule: GitHub/report artifacts first. Weights/tokenizer only as a research artifact with the eval bundle and a blunt model card. No chat demo.

### Branch 2: Build-Next-Version

| Rank | Action | EV/hr | Cost | Blocker | Falsifier / stop rule |
|---:|---|---|---|---|---|
| 1 | State the v2 hypothesis and kill criteria before coding | Very high | 3-6 h | "better model" motivation | cannot name what scale would teach beyond "larger is better" |
| 2 | Build eval harness first: per-family/source bpb, source QA, matched cutoff/anachronism logprobs, external anchor | Very high | 1-2 days, <8 GPU h | prompt/data design, harness friction | metrics are unstable, vibes-driven, or not tied to named claims |
| 3 | Redesign cache/provenance rather than relying on loader patch | Very high | 3-10 human days plus CPU/I/O | existing scripts assume volatile shard index | manifest mutation can still desynchronize provenance without tests failing |
| 4 | Run cross-source dedup and train-val contamination checks | High | 2-7 days CPU/human | OCR normalization, weak document boundaries | material train-val overlap or no defensible duplicate policy |
| 5 | Define cutoff construct and anachronism policy | High | 4-8 h policy; 1-3 weeks if building semantic audit workflow | ambiguous texts, forecasts, editions | cannot defend anything beyond publication-year cutoff, or filter error rates are high |
| 6 | Run a small historical-corpus scaling grid before 1B | Medium-high | roughly 150-300 GPU h plus 20-40 human h | noisy curves, automated eval not ready | eval noise/source imbalance exceeds scaling differences |
| 7 | Attempt 1B only after gates pass; treat 3B as cloud/multi-GPU only | Medium / low now | 1B: roughly 4-8 calendar weeks local after engineering; 3B local: reject | throughput, VRAM, opportunity cost | weekend/v2 gates fail, 1B smoke throughput collapses, or no cutoff-visible signal |

Cache verdict: for v1 evaluation, a loader patch is acceptable. For v2 training, redesign the cache/provenance layer with stable source/document/content IDs, split-aware metadata, train-only tokenizer inputs, atomic manifests, per-family eval manifests, and golden small-corpus invariant tests.

### Explicit Rejects

- Launching a 1-3B run now.
- Local 3B training on the 3090 as the next action.
- Using `1.1092 bpb` as headline aggregate validation.
- Claiming the 615M model demonstrates post-1914 ignorance.
- Treating publication-year cutoff as content-semantic cutoff.
- SFT, RLHF, chat tuning, or chat demos before base/corpus evals.
- Handpicked generations as evidence.
- Apples-to-apples bpb comparisons to Ranke, Talkie, FineWeb-Edu nanochat, GPT-2, or peer models.
- Loader-only patching as v2 infrastructure.
- Skipping cross-source dedup because duplicates may represent historical circulation.
- Expanding sources before split semantics, source coverage, and dedup are under control.
- Another long run whose stopping rule is "loss goes down."

### Minimum-Viable Weekend Eval Bundle

48-hour cap. Stop early if a prerequisite fails badly enough to narrow the writeup.

| Order | Eval | Cost | Decisive against |
|---:|---|---|---|
| 1 | Claim ledger plus reproduction of current Gutenberg-prefix bpb | 2-4 h, <1 GPU h | "the current headline is cleanly worded and reproducible" |
| 2 | Per-family/per-source bpb over multiple held-out slices | 3-6 h, 1-3 GPU h | "the learning signal generalizes beyond Gutenberg" |
| 3 | Tokenizer/split/cache/resume-continuity audit | 2-4 h, 0 GPU | "the calibration reading is uncontaminated and comparable" |
| 4 | Train-val/source exact and near-duplicate pilot, prioritizing evaluated slices | 4-8 h CPU/human | "held-out means held-out enough" |
| 5 | Matched pre/post-cutoff logprob set, including period forecasts and controls | 4-8 h, <1 GPU h | "cutoff signal is visible at 615M" |
| 6 | Source-register continuation or cloze probe by family | 3-5 h, <1 GPU h | "period/source style is more than anecdotal" |
| 7 | CORE or external anchor only if already close to runnable | 2-6 h, 1-3 GPU h | "the checkpoint has a comparable base-LM anchor" |
| 8 | One-page calibration table: claim, metric, result, allowed wording | 2-4 h | "public prose will not inflate the readings" |

Highest teaching per GPU-hour: per-family bpb, matched cutoff logprobs, source-register probes, CORE subset if runnable. Highest teaching per human-hour: claim ledger, tokenizer/split audit, duplicate pilot on evaluated slices, field-report writeup.

### Conditions That Would Change the Recommendation

Move toward Branch 2 if the weekend bundle shows clean tokenizer/split provenance, low duplicate overlap in evaluated slices, coherent non-Gutenberg per-family bpb, at least some source-grounded/source-register competence, a measurable cutoff/logprob differential or a clean null that motivates scale, and the researcher actively wants the time cost.

Move harder toward Branch 1, or even "postmortem only," if duplicate leakage is material, tokenizer inputs are validation-heavy and hard to quantify, non-books family evals collapse, cutoff probes are pure noise, rights/provenance packaging is unresolved, or the v2 motivation is mostly sunk cost.

Cheapest next action: make a one-page claim ledger, then run per-family bpb plus tokenizer/split audit. Those two checks are the fastest way to determine whether the weekend can produce a calibrated field report or must become a stricter postmortem.

## 5. Attribution Map

| Claim | Contributing agent(s) |
|---|---|
| Publish-and-move-on after a weekend eval is the default | Skeptic, Architect, Risk Analyst, Empiricist |
| Branch 2 is a multi-month instrument program, not a natural continuation | Architect, Empiricist, Skeptic, Risk Analyst |
| The thesis is instrument prototype + calibration log + claim-collapse anatomy | Architect, Empiricist, Skeptic, Risk Analyst, grounded in R2 |
| `1.1092 bpb` is a narrow Gutenberg-prefix calibration reading | Empiricist, Risk Analyst, Skeptic, Architect, grounded in R1 |
| 615M cannot test the original cutoff-capability thesis | Architect, Skeptic, Risk Analyst, Empiricist |
| Weekend evals should be decisive against named claims | Empiricist, Skeptic, Risk Analyst, Architect |
| Cache redesign is required for v2 | Architect, Risk Analyst, Skeptic, Empiricist |
| Loader patch is acceptable only as temporary/v1 guard | Empiricist, Architect, Risk Analyst |
| Cross-source dedup is mandatory before scale | Risk Analyst, Empiricist, Skeptic, Architect |
| Content-semantic anachronism cannot be solved by publication-year cutoff | Risk Analyst, Empiricist, Skeptic, Architect |
| 1B is conditional and 3B local should be rejected | Skeptic, Risk Analyst, Empiricist, Architect |
| Chinchilla-for-historical-corpora is an adjacent Branch 2 question | Architect, Empiricist, with caution from Skeptic and Risk Analyst |
| Checkpoint release should be secondary and hard-carded | Empiricist, Risk Analyst, with conditional support from Architect and Skeptic |
| Reject SFT/RLHF/chat demos as next action | Skeptic, Risk Analyst, Empiricist, Architect |
| Highest GPU-hour ROI is existing-checkpoint eval | Empiricist, Skeptic, Risk Analyst, Architect |
| Highest human-hour ROI is claim ledger, audits, and writeup | Empiricist, Risk Analyst, Skeptic, Architect |
