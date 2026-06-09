# CLAIMS THAT ARE WELL-SUPPORTED (with evidence)

The corrected run completed mechanically. All four agents endorse the narrow claim that a 615,645,184-parameter d22 base model trained through 70,455 steps under the `parallel_family_cache` loader, consuming about 18.469B scheduled training tokens and producing the final checkpoint `model_070455.pt`. The canonical run record, final metadata, and run logs support the final/min logged validation bpb of 1.1092, a coherent run identity, and absence of obvious NaN/OOM collapse.

Run #1 was invalid for the governed-PoC claim. The postmortem evidence is strong: stale provenance after split/shuffle covered only a minority of cache shards, mislabeled most family tags, and produced about 46 hours of outwardly healthy training under a wrong data path. This is not just a historical blemish; it proves that ordinary loss, throughput, and smoke health checks were insufficient to detect a material governance failure.

The run #2/#3 repair addressed the specific stale-provenance failure mode. The logs show a regenerated shard-family lookup for 18,926 train shards, and the loader guards for provenance coverage and family/source-file mismatch would have caught the original broken lookup. This supports "the known provenance bug was repaired for the completed run," not "end-to-end governance was proven."

The logged validation trajectory is a real training-health signal, but only a narrow one. The bpb trend improved substantially and the final point was the best recorded value. That supports optimization progress and lack of obvious collapse on the measured slice. It does not support broad held-out corpus validity.

The effective training distribution was the loader schedule, not simply the corpus inventory. The fixed 12/8/6/3/3 microbatch schedule implies roughly 37.5% newspapers, 25% science, 18.75% books, 9.375% legal, and 9.375% early-modern exposure. Final metadata showing books and legal wrap counts of 1 supports the agents' shared conclusion that some smaller families repeated while larger families had unvisited tail.

# CLAIMS THAT ARE PLAUSIBLE BUT UNVERIFIED (with what would verify them)

End-to-end corpus governance is plausible but not established. It would require an audit tying rights/date/source metadata through raw records, processed parquet, token cache, manifests, provenance files, train/val split construction, and loader consumption for both training and validation.

Broad five-family modeling quality is unverified. It would require full or randomized held-out evaluation, reported by family and preferably by source family/source collection, with explicit weighting. The current headline bpb cannot carry this claim because all four agents independently identified the validation-slice problem.

Period-appropriate prose is plausible at the surface level, but the samples do not validate it. Verification would require a held-out behavioral battery or blind human/style evaluation that distinguishes genuine period register from token-level period flavor such as "to-day" or older currency terms.

Cutoff/anachronism behavior is unverified. Small samples that omit modern anchors, or fail to answer modern factual prompts, do not prove cutoff fidelity because a 615M base model may fail for capacity, weak factuality, or prompt instability. Verification would require retained raw outputs on pre/post-cutoff probes and an evaluation design that distinguishes historical cutoff behavior from generic incapacity.

The corpus-mix story may be correct, but the artifact trail needs reconciliation. Agents noted a discrepancy between token-cache family shares and at least one processed manifest/source-mix report. That may reflect estimate-vs-tokenized-cache drift or a stale report, but it cannot be left implicit while corpus composition is a central validity claim.

External comparability is plausible only in a weak narrative sense. BPB is bytes-normalized, but the custom tokenizer, corpus, validation slice, and evaluation distribution make comparisons to Karpathy nanochat, legacy baselines, or Ranke-style efforts non-decisive unless the same checkpoint and baselines are evaluated under a common protocol.

# CLAIMS THAT ARE WEAK OR LIKELY FALSE

The claim that 1.1092 bpb summarizes the 2.86B-token held-out corpus is weak and probably false. The agents agree that `base_train.py` appears to rebuild a fresh sequential validation loader at each evaluation, with `eval_tokens=262144`, `device_batch_size=8`, and `seq_len=1024`, yielding only 32 eval batches. The first validation shard is a Gutenberg/books shard of about 1M tokens, so the logged bpb is best interpreted as a repeated fixed prefix probe unless a separate aggregate validation artifact exists.

Literal "monotone descent" is false. The trend improved and the final value is the minimum, but the raw series contains multiple upward moves. The supported statement is "downward trend with late improvement," not monotone descent.

The final sample completions are weak evidence for historical competence. They show some fluency and period-ish surface forms, but they also contain factual, arithmetic, and reasoning failures. They cannot validate source-grounded QA, cutoff behavior, broad period register, or factual reliability.

"Trained on 18.47B unique governed tokens" would be misleading. The better description is "trained on 18.47B scheduled tokens from the governed token cache under a family-balanced loader." Books and legal wrapped; larger families did not fully exhaust.

"Governance fixed" is too strong. The repaired provenance lookup and guards address the known loader-family bug, but they do not prove publication-year accuracy, rights accuracy, source identity, deduplication, train/val isolation, validation-family balance, or future robustness to manifest mutation.

# THINGS THE BRIEF AVOIDS OR HIDES

The brief does not foreground that the headline validation metric appears to be a tiny deterministic Gutenberg/books-prefix eval rather than an aggregate source-stratified validation result. This is the largest validity gap because the bpb number is the main quantitative success signal.

The brief underplays the implication of smoke #4 passing while provenance was broken. The important lesson is not only that run #1 was fixed; it is that the existing smoke and health metrics failed to detect a class of silent data-integrity failure.

The brief blurs corpus composition and effective training exposure. The governed cache mix, the loader schedule, the wrapped families, and the unvisited tail are related but not identical. This matters because the model's learned distribution follows the actual schedule, not the abstract corpus inventory.

The brief lists missing evals, but treats them too much like future polish rather than unresolved validity gates. No CORE metric, no per-family validation, no source-grounded QA, no anachronism/cutoff eval, and no midtraining comparison mean the historical-model claim has not been measured at the level it is being discussed.

The brief does not fully separate "stable optimization" from "valid historical model." A clean run curve, final checkpoint, and low loader wait are necessary operational evidence. They are not evidence of cutoff fidelity, balanced family competence, or historical factual behavior.

The brief risks over-reading a custom-tokenizer bpb. Even as bytes per byte, the score is not standardized across corpora, tokenizers, validation distributions, or loader protocols. Without common evaluation, it is a within-run signal, not a peer-comparable headline.

# CRITICAL OPEN QUESTIONS BEFORE THIS PoC CAN BE CALLED A SUCCESS

Does the final checkpoint achieve comparable bpb on full or randomized validation slices across all five source families, not only the apparent Gutenberg/books prefix?

How large is the per-family and per-source spread in validation loss, especially for newspapers, BHL/science, CAP/legal, and EEBO/TCP-style early-modern material?

Are train and validation splits document-clean, deduplicated at the right level, and free from source leakage?

Can the project audit date, rights, source identity, and family labels end-to-end from raw artifacts to token-cache shards to loader consumption?

Does the model show period/cutoff behavior in a way that is distinguishable from generic small-model incapacity or factual unreliability?

Is the fixed family schedule, including books/legal wrapping and larger-family unvisited tail, the intended estimand for the PoC, or did it change the claim being tested?

Are any external comparisons being made under a shared tokenizer/eval protocol or a clearly justified bridge metric?

Did any planned stop-gate evaluation actually run after completion, or is the success claim still resting on the training log, metadata, and samples?

# OVERALL VALIDITY VERDICT (one paragraph)

This is a mechanically successful corrected training run, not yet a validated historical governed-base-model PoC. The strongest supported claim is narrow: a 615M d22 model completed the corrected family-balanced training run, produced a coherent final checkpoint, and improved a repeated logged validation slice to 1.1092 bpb without obvious optimization collapse. The broader claims do not survive scrutiny: the headline bpb appears not to be an aggregate held-out-corpus metric, no per-family/full validation or cutoff/anachronism battery has been run, the effective training exposure differs from a simple governed-corpus pass, and the samples are far too weak to establish period-appropriate competence. The right validity label is therefore "promising completed run artifact with serious unresolved construct-validity gaps," not "successful validated historical PoC."
