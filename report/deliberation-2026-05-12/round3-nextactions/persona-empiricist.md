# empiricist — final position

## My initial position (Phase 1 draft, ~3 sentences)

The branch-choice question is whether to publish a calibrated v1 artifact, or to commit to a new measurement program whose first real test requires fixing data/eval invariants and scaling beyond 615M. My blind recommendation was publish-and-move-on by default after a tight weekend eval gate: the current run supports an instrument prototype, calibration log, and claim-collapse anatomy, not the original cutoff-capability thesis. Building v2 is only evidence-rational if the researcher explicitly wants the opportunity cost and first proves the instrument can take stable readings across families, splits, dedup, and cutoff probes.

## What I learned from peers

- From **skeptic**: The most important warning is that "instrument program" can become an identity trap that smuggles in scale-up before the falsifiable hypothesis is stated.
- From **architect**: The cleanest branch split is artifact closure versus program commitment, with the weekend eval bundle as the hinge between them.
- From **risk-analyst**: Release and scale decisions should be framed as risk thresholds: capability overclaim, eval theater, and data-invariant debt dominate the downside.

## Disagreements I sharpened

I mostly agree with the peers, but I sharpen one point: "possibly release the checkpoint" should not be treated as a neutral packaging choice. Empirically, the checkpoint has not demonstrated historical competence, factual competence, arithmetic, or broad held-out generalization. I would release weights only if the model card makes the base-model incapability impossible to miss and the eval bundle is released beside it; otherwise release code, cards, logs, scripts, tokenizer metadata, and reports without turning the model into the public object.

I also push back on any branch-2 language that makes a 1-3B build sound like a near follow-on. At current run economics, 1B at a Chinchilla-like token count is plausibly a multi-week 3090 run after engineering work; 3B is effectively out of single-3090 scope unless the researcher accepts months of wall time or rents larger compute. The empirically decisive part is not "bigger model" in isolation, but bigger model after the measurement rig stops lying in known ways.

## Where I integrated peer reasoning

I integrated the skeptic's warning by treating branch 2 as a falsifiable experimental program, not as the natural next identity of the project. I integrated the architect's framing by making "publish v1" and "build v2" genuinely separate plans with different blockers and stop conditions. I integrated the risk-analyst's release caution by putting model-card disclosures and non-release conditions inside branch 1, rather than treating publication as simple cleanup.

## My final position

### What's established

The completed 615M run is mechanically real: it finished, checkpoints exist, and the corrected loader produced a real learning trajectory. But the strongest validated metric, 1.1092 bpb, is a fixed 262k-token Gutenberg-prefix calibration reading, not aggregate validation and not historical competence. The model samples show surface English and period tokens, but no reliable factual recall, arithmetic, or logic. Publication-year cutoff is also not semantic cutoff; pre-1914 documents can contain future forecasts and post-cutoff concepts.

### What's contested or unknown

The current evidence does not tell us whether compression generalizes across families, whether train/val/source boundaries are clean after cross-source duplication, whether tokenizer training leaks useful val-specific information, whether the model has any cutoff-visible preferences, or whether the family loader is a robust method versus a local workaround. Those are the decisive unknowns.

### Base rates

For a single-author 3090 project, the base rate favors finishing the calibrated report over starting another long training run. Small LMs often look fluent before they become factually useful, and weak eval harnesses routinely produce overclaims. The project has already paid weeks of GPU time and many days of human time; the marginal evidence per hour is now much higher for measurement and write-up than for immediate scale-up.

### Branch-choice question

Ask this crisply: "Do I want to publish v1 as an instrument calibration/postmortem artifact, or do I want to spend the next project cycle building a v2 instrument capable of testing the cutoff hypothesis at 1B+ scale?" If the answer is not an active yes to the second, choose publish-and-move-on.

### Branch 1: publish-and-move-on

1. Run the minimum weekend eval bundle. Cost: 12-25 human hours, likely under 2 GPU-hours plus CPU time. Hypothesis tested: "the existing artifact can be honestly calibrated beyond the Gutenberg prefix." Falsifier: severe split leakage, duplicate contamination, or total cross-family collapse; then publish as postmortem, not model artifact.

2. Write the field report plus corpus/model cards. Cost: 8-16 hours. Hypothesis tested: "the contribution is intelligible without capability overclaim." Falsifier: if 1.1092 or "historical LLM" keeps becoming the headline, narrow the genre to postmortem only.

3. Release selectively. Cost: 4-8 hours. Put on GitHub: code, configs, eval scripts, reports, provenance summaries, cards, and reproduction instructions. Put on HuggingFace only if checkpoint/tokenizer have a blunt model card. Do not release raw corpora or token caches unless rights allow it; do not release a chat demo. The card must disclose: 615M base model, no SFT/RLHF, no demonstrated factual competence, 1.1092 as narrow Gutenberg-prefix bpb, publication-year-only cutoff, no semantic anachronism guarantee, source/diversity caveats, no cross-source dedup unless completed, and research-artifact use.

4. Freeze v1. Cost: 2-4 hours. Tag the repo, archive logs/checkpoints, and stop training work. Falsifier: only the researcher's explicit decision to build v2.

### Branch 2: build-next-version

1. Gate on the weekend evals first. Cost as above. Stop if per-family bpb is incoherent, duplicates are high, tokenizer/split audit is dirty, or cutoff probes show only generic incompetence.

2. Fix cache/provenance before scale. A loader patch is 4-8 hours and should key checks to stable source/document identity, but it is not enough for v2. A cache redesign is 3-7 days: interleaved shards, stable IDs, source-family metadata preserved through tokenization, deterministic split semantics, per-family val loaders, and refusal tests. Hypothesis tested: "the instrument can preserve the construct it claims to measure."

3. Run cross-source dedup. Cost: 2-5 days on CPU/disk for MinHash/SimHash or n-gram overlap at document/passage level. Hypothesis tested: "held-out and source-family readings are not artifacts of duplicated text." Falsifier: high duplicate rates across train/val or across supposedly independent sources.

4. Decide the cutoff construct. Cost: 4-8 hours to write the policy; 3-10 days if attempting heuristic semantic filtering. My recommendation: relativize or defend 1913 as publication-year isolation, not semantic isolation. Semantic anachronism filtering should be treated as an audit and eval problem, not a promise of purity, unless a measured filter shows tolerable false positive/negative rates.

5. Scale only after the instrument is stable. A 1B, ~30B-token run is plausibly weeks on the current 3090; 3B with ~90B tokens is out of scope without larger compute. Hypothesis tested: "at a capability-relevant scale, a publication-year-isolated corpus produces measurable cutoff behavior." Falsifier: no improvement on factual/source-grounded probes in a smaller pilot, or no stable eval separation.

6. Treat Chinchilla-for-historical-corpora as a separate research question. Cheapest version: train a small scaling grid, e.g. 125M/300M plus the existing 615M on shared per-family evals, before committing to 1B+. Cost: several GPU-days, not weeks. Hypothesis tested: "historical governed corpora have a different compute-optimal token/parameter ratio than modern web mixtures."

### Minimum-viable weekend eval bundle

1. Reproduce the current 262k Gutenberg-prefix bpb and add per-family bpb over multiple held-out shards. Cost: 3-6 hours, under 1 GPU-hour. Decisive against "the learning signal is broad."
2. Audit tokenizer training corpus, split semantics, and val-cache continuity. Cost: 1-3 hours, no GPU. Decisive against "the bpb is cleanly interpretable."
3. Run a cross-source near-duplicate pilot. Cost: 4-8 hours CPU/human. Decisive against "held-out/source independence is credible."
4. Run matched pre/post-cutoff logprob probes. Cost: 4-6 hours, under 1 GPU-hour. Decisive against "615M already shows cutoff-visible knowledge behavior."
5. Run source-register continuation/cloze probes for each family. Cost: 3-5 hours. Decisive against "period/source style generalizes beyond anecdotal samples."
6. Add one external anchor such as CORE-equivalent perplexity if the harness is already close. Cost: 2-4 hours. Decisive against "we can place this model relative to anything else."

### Explicit rejects

Reject SFT/chat demos on the 615M checkpoint, headline use of 1.1092, "vintage LLM" branding, claims of post-1914 ignorance, peer-model bpb comparisons, scaling before cache/dedup/eval fixes, treating loader patching as equivalent to cache redesign, and large semantic-filtering work unless it is tied to measured error rates.

### Empirical verdict

Highest teaching per GPU-hour: per-family bpb and matched cutoff logprob probes. Highest teaching per human-hour: tokenizer/split audit, duplicate pilot, and the field-report write-up that prevents future claim inflation. The unified recommendation is publish-and-move-on after the weekend eval bundle unless the results are surprisingly clean and the researcher actively wants a months-scale v2 program.

## Confidence and what would change my mind

Confidence: high on the default recommendation, medium on exact cost estimates. I would shift toward build-next-version if weekend evals show stable cross-family compression, low duplicate contamination, clean tokenizer/split provenance, at least some source-grounded competence, and a nonzero cutoff signal, and if the researcher explicitly accepts the compute/time cost. I would shift further toward "private postmortem only" if the weekend audit finds severe leakage, broken split semantics, or rights/release blockers.
