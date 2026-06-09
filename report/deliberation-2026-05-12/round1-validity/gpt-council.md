# CLAIMS THAT ARE WELL-SUPPORTED (with evidence)

The corrected run completed as a real mechanical training artifact. All four agents converge on the narrow claim that a 615M-class d22 nanochat model trained through 70,455 optimizer steps under the repaired `parallel_family_cache` setup, consumed about 18.47B scheduled training tokens, and produced final checkpoint artifacts. The Skeptic emphasized that the final report records the expected config, parameter count, steps, optimizer tokens, final checkpoint, wall time, MFU, and memory, and that an error scan did not show Traceback/OOM/CUDA/NaN-finite failures. The Architect framed this as the strongest validity layer: completed run, final checkpoint, metadata, optimizer state, and sane logs. The Risk Analyst agreed this should not be lost when downscoping the claim. The Empiricist summarized the same distinction as "operational run validity is real."

The stale-provenance bug from run #1 is not evidenced in the current corrected cache/run artifacts. The Skeptic and Architect both report a clean current manifest/provenance join: 18,926/18,926 train shards matched and 2,881/2,881 val shards matched, with zero family mismatches. The Architect notes that `parallel_family_cache` now has coverage and family-cross-check guards. The Risk Analyst agrees the known stale-provenance class appears fixed at this layer, while warning that this does not prove source-date, rights, or split-leakage correctness end to end.

The logged `1.1092` bpb is a real measurement, but only for a narrow deterministic validation prefix. This is the central empirical finding and is endorsed by all four agents. The configured `eval_tokens` is 262,144. `base_train.py` rebuilds a fresh val loader at each eval and computes 32 eval batches from `eval_tokens / (device_batch_size * seq_len * world_size)`. The cached loader starts from sorted shard index 0 when no resume state is supplied. The first val shard is a Gutenberg/books shard of 999,563 tokens and 9 documents. Therefore the repeated validation metric is over roughly the first 262k tokens of that first Gutenberg/books val shard, about 0.0092% of the 2.86B-token held-out split. The Empiricist, Skeptic, Architect, and Risk Analyst all treat this as verified rather than speculative.

The fixed-prefix bpb trajectory is still evidence of learning on that slice. The Skeptic explicitly says the prefix bpb is not meaningless: it is a real held-out prefix, not known train leakage, and improvement there is evidence of language-model learning on at least one held-out source slice. The Architect calls this a plausible learning-signal PoC only in a narrow sense. The Empiricist says the decreasing trajectory is construct-valid for what it measures. The Risk Analyst's bottom-line verdict matches this: the corrected run improved a fixed 262k-token Gutenberg/books validation prefix to 1.1092 without obvious optimization collapse.

The effective training distribution is the loader schedule, not simply the corpus inventory. Skeptic, Risk Analyst, and Empiricist all flag this. The family schedule is fixed at 12/8/6/3/3 microbatches, so exposure follows that schedule rather than a naive "one pass over the governed corpus" interpretation. Final metadata/wrap counts indicate smaller families such as books/legal wrapped while larger families did not. This does not invalidate the run, but it does constrain what "trained on 18.5B tokens from a balanced corpus" can mean.

Run #1 is valid evidence that ordinary training health metrics are insufficient for corpus-integrity claims. Skeptic and Risk Analyst both put weight on the postmortem: loss, val bpb, and throughput looked healthy for about 46 hours while the loader used stale provenance, touched only a fraction of the intended cache, and assigned wrong family tags. This directly undermines any argument that smooth loss curves alone establish data correctness.

# CLAIMS THAT ARE PLAUSIBLE BUT UNVERIFIED (with what would verify them)

The corrected run probably trained on the intended repaired five-family cache, but this is not the same as a raw-to-model governance proof. The current manifest/provenance agreement, corrected launch record, loader guards, and coherent metadata support this claim; Skeptic and Architect give it meaningful weight, and Risk Analyst accepts the stale-provenance class as addressed. What would verify the stronger version is an end-to-end audit tying raw documents, dates, rights/source labels, split assignment, cache shards, provenance entries, loader cursors, and final family exposure together.

The model likely learned more than just the first Gutenberg prefix, but broad held-out generalization across the five source families is unmeasured. Architect's "A plus cautious B" classification is defensible only if "B" means a narrow learning signal rather than validated historical-model competence. Skeptic and Risk Analyst are stronger on the broader claim: without full, randomized, or per-family validation, no agent can infer corpus-wide performance from the logged metric. Verification would require validation metrics that actually cover the held-out split or source/family-stratified subsets.

"No overfit" is plausible only in the weak sense that the fixed prefix ended at its minimum logged bpb. It is not verified as a corpus-level training claim. Skeptic and Risk Analyst both note that the series had many local upward moves and that the validation target is the same tiny deterministic prefix each time. Verification would require train-vs-val behavior on a broader held-out sample and evidence that the prefix is not unusually easy or over-represented.

The governed pre-1914 corpus may induce period-appropriate behavior, but the run does not establish that behaviorally. The samples show surface fluency and some period-ish register, which Architect and Empiricist treat as compatible with learning. Skeptic and Risk Analyst correctly downgrade the stronger claim because the same samples show factual, arithmetic, and logic failures. Verification would require pre-1914 control prompts, post-cutoff/anachronism probes, source-grounded QA, and evidence that failures on modern content are selective rather than just general incompetence.

The comparison to a legacy d22 baseline may be useful narrative context, but it is not verified as a fair performance comparison. Risk Analyst is strongest here: bpb is bytes-normalized but not protocol-normalized when tokenizer, corpus, validation slice, and evaluation procedure differ. Verification would require a common eval protocol or at least a clearly matched validation target.

# CLAIMS THAT ARE WEAK OR LIKELY FALSE

The claim that `1.1092` bpb summarizes the 2.86B-token held-out validation set is weak and likely false. This is the highest-confidence adverse finding. All agents independently converge on the same mechanism: `eval_tokens=262,144`, fresh sequential val loader each eval, 32 batches, deterministic start at shard 0, first shard Gutenberg/books. The reported bpb should be labeled a fixed 262k-token Gutenberg-prefix bpb, not an aggregate held-out-corpus bpb.

The brief's "monotone descent through warmdown, no overfit" framing is overstated. Skeptic identifies many local regressions in the logged bpb series; Risk Analyst counts the risk as high and serious; Empiricist's verified findings say the trend is strongly downward but not monotone. "Final equals min on the fixed prefix" is supportable. "Monotone descent" and corpus-level "no overfit" are not.

The claim that the run demonstrates a successful historical governed-base-model PoC is not supported on validity grounds. Architect's decomposition is useful: the run is a valid mechanical artifact, maybe a narrow learning-signal artifact, but not a validated historical-model success claim. Skeptic, Risk Analyst, and Empiricist have the stronger case here because the load-bearing constructs - broad held-out generalization, source/family balance, cutoff behavior, factual competence, and source-grounded historical QA - are not measured.

The sample completions should not be read as evidence of historical competence. Skeptic and Risk Analyst both stress that "The capital of France is 100000..." and similar outputs are compatible with a small base model lacking reliable factual recall, arithmetic, chronology, and logic. Empiricist agrees the samples show fluent English and some surface register, not period-appropriate competence beyond token-level style.

The claim that the corpus is "balanced" in the sense of equal or complete unique exposure is weak. The model's effective distribution follows the fixed family microbatch schedule and final cursor/wrap behavior, not just the inventory percentages. This point comes from Skeptic, Risk Analyst, and Empiricist. It is not fatal to the run, but it makes simple corpus-pass language misleading.

The claim that the stale-provenance issue is fully closed as a governance concern is too strong. The specific stale-provenance failure appears fixed in the corrected artifacts, but Risk Analyst is right that raw publication dates, rights classification, source identity, duplicate document leakage, and split integrity are separate risks. Skeptic's point that run #1 passed ordinary health checks while wrong should make this distinction stricter, not softer.

# THINGS THE BRIEF AVOIDS OR HIDES

The brief's most important hidden move is treating the existence of a 2.86B source-stratified held-out split as if the logged bpb measured that split. Empiricist, Skeptic, Architect, and Risk Analyst all reject that implication. The held-out cache may exist and be source-stratified; the training-time eval appears to touch only the first fixed Gutenberg/books prefix.

The brief discloses many gaps, but it underweights the consequences of the validation-semantics gap. "Final/min val bpb 1.1092" carries most of the success narrative, yet the metric is not the construct the prose implies. This is not a small caveat; it changes the validity class of the result.

The brief presents the stale-provenance bug as a resolved history item but does not fully absorb its epistemic lesson. Skeptic and Risk Analyst both emphasize that run #1 shows ordinary health metrics and passing smokes can coexist with severe data-integrity failure. The corrected run deserves credit, but only artifact-level checks should carry governance claims.

The brief blurs corpus inventory, loader schedule, and realized exposure. The governed corpus mix and train-token total are real facts, but the model saw data according to the 12/8/6/3/3 family schedule and final cursor/wrap dynamics. This matters for any claim about balanced training or family coverage.

The brief lets qualitative samples do more rhetorical work than they can bear. It includes raw failures that should sharply limit any behavioral claim, but the surrounding narrative still risks reading them as historical-model evidence. The agents agree samples are smoke evidence at most.

The brief does not establish source governance at the raw-document level. It discusses governed pre-1914 data and provenance repair, but the panel did not see evidence sufficient to close date correctness, rights/source classification, document identity, duplicate leakage, or train/val contamination risks.

# CRITICAL OPEN QUESTIONS BEFORE THIS PoC CAN BE CALLED A SUCCESS

What is the model's bpb on the full held-out split, or on a defensible randomized/source-stratified/family-stratified held-out evaluation? This is the central open question from all four agents.

How different is the fixed Gutenberg-prefix bpb from random books bpb, other Gutenberg shards, non-book families, and source-weighted aggregate validation? Risk Analyst surfaces the failure mode; Empiricist's shard-order finding makes it load-bearing.

Do the final family cursors, wrap counts, and shard histories exactly match the intended 12/8/6/3/3 family schedule over the completed run? Skeptic and Risk Analyst both flag effective exposure as a validity condition.

Is there any train/val leakage at the document or near-duplicate level? Risk Analyst raises this as separate from manifest/provenance agreement.

Are raw source labels actually governed: publication years <=1913, rights status correct, source family correct, and collection metadata trustworthy? Risk Analyst raises this; Skeptic's run #1 critique explains why high-level metadata agreement is not enough.

Does the model show selective historical cutoff behavior, or does it simply fail both pre-1914 and post-1914 factual probes? Skeptic, Risk Analyst, and Architect all treat this as unresolved; Empiricist's sample review reinforces it.

Can the model answer source-grounded or period-grounded factual questions better than chance or better than a non-governed baseline under the same protocol? This is synthesis-emergent from the sample failures and the missing CORE/source-grounded QA gap.

What exactly is the intended success construct: "completed governed pretraining run," "broad five-family held-out language model," or "usable historical base model"? Architect separates these layers cleanly, and the final verdict depends on not collapsing them.

# OVERALL VALIDITY VERDICT (one paragraph)

This is a mechanically credible corrected training run, not a validated historical governed-base-model PoC. The strongest supported claim is narrow: a 615M d22 model completed 70,455 steps on the repaired `parallel_family_cache` setup, produced coherent checkpoint artifacts, and improved a repeated fixed 262k-token Gutenberg/books validation prefix to `1.1092` bpb without obvious optimization collapse. The central adverse finding is also high-confidence: the headline bpb is not an evaluation over the 2.86B-token held-out set and should not be used as evidence of broad source-stratified generalization. Architect's more constructive classification survives only as "valid mechanical artifact plus narrow learning signal"; Skeptic, Risk Analyst, and Empiricist have the stronger case against calling this a successful historical-model PoC. The validity label should be: completed corrected run with a real but narrow learning metric and serious unresolved construct-validity gaps.
