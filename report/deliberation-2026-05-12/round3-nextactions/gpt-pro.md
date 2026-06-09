1. POINTS WHERE I AGREE WITH BOTH PRIOR PANELS

The core branch decision is correctly framed: finish v1 as a calibrated artifact versus enter a v2 measurement-instrument program. It is not “is the model impressive?”

I agree that publish-and-move-on is the default. The v1 artifact already has value as a worked calibration/postmortem object; further training would mostly convert sunk cost into more sunk cost unless the measurement substrate is repaired first.

I agree that 615M cannot carry the cutoff-capability thesis. It can support claims about the instrument, the corpus, the training run, the failure modes, and the anatomy of claim collapse. It cannot support strong claims about whether time-isolated language modeling “works” in the factual-recall sense.

I agree that the 1.1092 bpb result belongs in an evidence table, not the headline. It is important, but it is too easy for readers to misread it as “the model is historically good” or “the aggregate validation is strong.”

I agree that v2 begins with data/invariant repair, not more GPU time: provenance, cache design, split invariants, dedup, per-family accounting, cutoff policy, and reproducibility surfaces.

I also agree that HF checkpoint release should be conditional, while GitHub/code/config/report release should be default. The checkpoint is easier to misuse rhetorically than the repo and postmortem.

2. POINTS WHERE I DISAGREE OR MODULATE

The prior panels are probably still too generous about the phrase “weekend eval bundle.” Some items belong in a weekend bundle; others are really v2 infrastructure tasks wearing weekend clothes.

The weekend bundle should not try to make the instrument “good.” It should decide three things only:

Can the v1 report be honestly published?

Can the checkpoint be released without misleading users?

Which exact invariant failures must block v2 training?

Anything that does not answer one of those should be cut.

I would push back especially on dedup. A weekend can support an overlap/leakage smoke audit. It cannot support a serious corpus-wide dedup solution over ~16B unique tokens unless the infrastructure already exists. Full dedup, source-level canonicalization, near-duplicate policy, and cache redesign are v2 work.

I would also push back on qualitative prompt testing. For a 615M model, anachronism/factual-recall prompts are mostly illustrative. They are not decisive against the time-cutoff thesis because the model is structurally too small. A small prompt suite can be useful for the model card, but it should not be treated as an eval pillar.

The sequence should be stricter than “run evals, then write.” The first act must be a frozen claim ledger with pass/fail implications. Otherwise the weekend bundle becomes a wandering salvage operation.

The better sequence is:

freeze claims → freeze artifacts → assign each eval to a claim it can kill → run only those evals → write the release decision table → then write prose.

I would also modulate the HF release recommendation. The checkpoint should not be released merely because the eval bundle “passes.” It should be released only if the model card makes the limitations impossible to miss. The danger is not that the checkpoint is bad; the danger is that third parties will cite it as if it proves the original stronger thesis.

On costing: any estimate that treats v2 provenance/cache redesign plus dedup plus cutoff-policy repair as a small patch is likely off by more than 2x. That is not a weekend patch. It is probably the main project. Conversely, per-family bpb evaluation and basic token/accounting audits are probably cheaper than feared if the data and checkpoint are already organized.

The largest hidden cost is not GPU; it is human attention after fatigue. The researcher has already spent weeks of GPU and many days of effort. The next unit of work must reduce ambiguity, not create a new heroic maze.

3. ADDITIONAL NEXT-ACTION POINTS NOT RAISED (highest value)

First: create a release decision matrix before running anything else. Rows should be claims; columns should be “evidence required,” “weekend test,” “pass condition,” “fail consequence,” and “where this appears in report/card.” This prevents every result from becoming narratable.

Second: make an explicit artifact freeze bundle: checkpoint hash, tokenizer hash, git commit, config, train logs, data manifest, eval manifest, and exact command lines. This is more valuable than one more eval because it converts the project from “I trained something” into “this object can be inspected.”

Third: write the model card red box first, not last. The red box should say, in plain language: 615M is insufficient to test the cutoff capability thesis; publication-year cutoff cannot rule out semantic anachronism; three families have single-source training coverage; books/legal were warmdown-wrapped; 1.1092 bpb is cross-source Gutenberg held-out, not aggregate validation.

Fourth: do a reader-misuse audit. Ask: “What false headline could someone write after skimming this?” Then edit the report and model card to make that headline difficult. This matters more than another decorative chart.

Fifth: add an external cold-reader check, even if informal. One technically literate person should read only the abstract, evidence table, and model card, then answer: “What do you think this proves?” If they overclaim, the artifact is not ready.

Sixth: consider a no-v2 cooling-off gate. After publication, do not start v2 training for 30 days. During that time, only collect bug reports, reader confusion, and invariant requirements. This is embarrassing to recommend because it feels like loss of momentum, but it is probably the highest-value anti-sunk-cost intervention.

Seventh: separate project continuation from project enlargement. The next thing may be a report, a dataset paper, a reproducibility artifact, or a calibration essay. It does not have to be a larger model.

Eighth: make a v2 kill criterion. Example: “No v2 training run may start until the data pipeline can produce a per-example provenance record with source, family, publication year, inclusion status, split assignment, dedup key, and cutoff-risk flag.” Without that, v2 is just v1 with more expensive ambiguity.

4. EXPLICIT THINGS-NOT-TO-DO list

Do not train a 1B, 3B, or larger model now.

Do not run a scaling experiment to rescue the thesis. The current failure mode is measurement repair, not insufficient ambition.

Do not patch the loader again as the main v2 move. Redesign provenance/cache invariants instead.

Do not headline 1.1092 bpb.

Do not describe the model as historically reliable, historically grounded, temporally isolated in a strong sense, or proof of cutoff capability.

Do not use generic benchmarks as credibility filler unless they kill a specific claim. They will mostly invite the wrong comparison class.

Do not add a large prompt gallery. A few examples may help the card, but prompt theater will blur the thesis.

Do not perform full dedup under the label “weekend eval.” Do a leakage smoke test now; reserve real dedup for v2.

Do not release the HF checkpoint with a normal cheerful model card. If the limitations are not unmistakable above the fold, do not release the checkpoint.

Do not start v2 because the 3090 is available.

Do not start v2 because the researcher is tired and wants the work to “be worth it.”

Do not hide the ugly parts. The ugly parts are the artifact.

Do not let “scientific instrument project” become a prestige phrase. It must mean calibration logs, failure modes, invariants, and measurement humility.

5. MY MINIMUM-VIABLE WEEKEND BUNDLE (if different from theirs, justify)

My bundle is narrower than what I suspect the prior panels implied. It is designed to decide release posture, not improve the model.

0. Freeze the claim ledger before running evals.
For each claim, write: “supported,” “weakened,” “killed,” or “not testable at 615M.” This must happen first.

1. Freeze the artifact manifest.
Record checkpoint hash, tokenizer hash, git commit, configs, training log locations, data manifests, eval commands, and exact corpus/accounting numbers. This is non-negotiable.

2. Produce a per-family validation table.
Report bpb by family/source, with flags for single-source train coverage and warmdown wrapping. Aggregate validation should be secondary or appendix-level.

3. Reconfirm the Gutenberg held-out result in context.
Keep the 1.1092 bpb number, but label it precisely: cross-source held-out generalization on a 262k Gutenberg-books prefix from an unseen source, not aggregate validation.

4. Run a split/provenance audit on the eval sets.
For every eval slice used in the report, verify source, family, publication-year rule, split assignment, and whether the source/work/author appears in training. This is more important than adding new evals.

5. Run an exact-overlap leakage smoke test.
At minimum, test eval text against train text for exact or high-confidence near-exact overlap at paragraph/chunk level. This is not full dedup. It is a release-blocking smoke test.

6. Run a cutoff-risk sample audit.
Sample examples that pass the publication-year cutoff and manually classify whether they contain obvious semantic anachronism risk: later editorial matter, modern introductions, OCR metadata, copyright pages, retrospective notes, database wrappers, etc. The goal is not to close the cutoff problem; it is to measure and display it.

7. Reconcile token accounting.
Lock the ~16.18B unique-token figure, the tokens:params ratio of ~26.3:1, and the distinction between unique tokens, repeated training tokens, warmdown tokens, and validation tokens. This prevents accidental Chinchilla overclaiming.

8. Draft the model card before deciding HF release.
The model card should include a “not for” section: not for factual historical QA, not proof of cutoff learning, not aggregate historical competence, not clean time-isolation.

9. Optional only: small qualitative prompt sanity panel.
Use perhaps 20 prompts to illustrate failure modes and card warnings. Do not score it as evidence for the thesis. Do not let it consume the weekend.

The key difference from a broader bundle: I would exclude broad benchmark runs, large qualitative galleries, new held-out source hunts, full dedup, scaling tests, and any retraining. They are either decorative, misleading, or v2 infrastructure.

6. MY OWN OVERALL NEXT-ACTION RECOMMENDATION (one paragraph, naming the FIRST CONCRETE STEP the researcher should take Monday morning)

The next action is closure on v1, not v2 training. Monday morning, the researcher should create a single file called something like R3_claim_ledger_and_release_gates.md and fill it with the exact claims, evidence, pass/fail consequences, and release conditions before touching the GPU. Then run only the minimum weekend-style bundle above, write the report/model card around the resulting decision table, release the GitHub artifact by default, and release the HF checkpoint only if the card is aggressively limitation-forward. After that, impose a cooling-off gate: no v2 training until provenance, split invariants, dedup policy, cutoff-risk accounting, and cache design are repaired on paper and in code. The honorable next move is not a bigger model; it is making the existing artifact impossible to misunderstand.