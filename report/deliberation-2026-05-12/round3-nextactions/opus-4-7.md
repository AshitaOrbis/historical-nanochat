# Opus 4.7 — Round 3 Next-Actions Verdict

Independent verdict, written without sight of GPT-5.5 Pro's R3. I read both GPT panels' R3 outputs (gpt-max.md, gpt-council.md), the empiricist persona file, the R1/R2 syntheses, and my own R1/R2 verdicts. I also re-verified that `report/governed_v4_run1_postmortem_detailed.md` exists on disk (381 lines, 25 KB, finalized 2026-04-26).

The valuable work is sections 3, 4, and 6. Sections 1-2 are deliberately brief.

---

## 1. Points where I agree with both prior panels

- **Publish-and-move-on is the correct default**, not "build v2 immediately." Both panels are right.
- **Weekend eval bundle should precede the writeup**, not follow it, but the claim ledger and report outline should be frozen *before* the eval bundle so the bundle is decisive rather than decorative. Both panels reach this scheduling answer; it is correct.
- **615M cannot test the cutoff-capability thesis.** Both panels carry R2's conclusion forward. Carry it.
- **The 1.1092 bpb does not get to be a headline.** Both panels are right. It is a cross-source held-out generalization reading on a 262k-token Gutenberg-books prefix from a train-empty source. It belongs in an evidence table with the construct named, not in an abstract.
- **HuggingFace checkpoint release is conditional, not automatic**, contingent on the eval bundle traveling with the model card and the repo naming making "this is a research-only base checkpoint, not an assistant or a vintage LLM demo" impossible to misread.
- **Cache/provenance redesign beats another loader patch for v2.** The postmortem itself names `source_file` as the right key. Two of three upstream scripts that can re-induce the failure remain unfixed.
- **The branch question is closure-v1 vs commit-v2 instrument program**, not "do I make the model bigger."
- **SFT/RLHF/chat demo as rescue is wrong.** Both panels say so. Correct.

I am ~85% confident on all of the above. They are the locked floor going into the section that matters.

---

## 2. Points where I disagree or modulate

### 2.1 Both panels under-quote how cheap the *minimum decisive* weekend bundle actually is.

GPT Max quotes 19-36 human hours plus under 2 GPU-hours. GPT Council quotes 12-28 human hours plus 3-8 GPU-hours. The first three items — claim ledger, per-family bpb, tokenizer/split/cache audit — are the only ones that can *force* a thesis change. Items 4-8 are confirmatory or anchoring; they enrich the writeup but do not flip the branch decision. The *decisive-against-the-thesis* subset is 8-14 human hours and well under 1 GPU-hour. See my §5.

This matters because the panels are implicitly bundling "minimum decisive" with "complete enough for a strong writeup." Those are different bundles and should be priced differently. The minimum decisive bundle protects against the worst sunk-cost trap (running an expensive bundle for a writeup that the first two items would have made unnecessary).

### 2.2 The "build v2 instrument program" branch needs an in-advance EXIT criterion. Both panels listed kill-gates *before* commitment; neither listed kill-gates *during* the program at fixed time horizons.

This is the panels' biggest under-specification. GPT Max says branch 2 requires "passing instrument gates" before scale-up. GPT Council says cache redesign before training, dedup before scale, etc. Both panels treat the gates as pre-commitment. But the failure mode they don't address is the most common one for hobbyist research programs: **the gates pass, the program starts, and then it accretes for six months without anyone asking whether the program should still be alive.**

See my §3.4 for explicit kill-gates at 1 month / 3 months / 6 months.

### 2.3 Both panels treat publication of the postmortem as a sub-step of "publish the field report." It can ship same-day, standalone, as its own artifact.

`report/governed_v4_run1_postmortem_detailed.md` exists on disk *right now*, 381 lines, finalized 2026-04-26. It is internally complete. It does not require the weekend eval bundle. It does not require the field report. It does not require the model card.

What it requires:
- Replace any in-line "we will / TODO" forward references with explicit "[FOLLOWS IN FIELD REPORT]" or strike them.
- Confirm no proprietary identifying material (data paths, API keys, internal commit hashes that point to private repos).
- A 3-paragraph standalone intro that contextualizes for a reader who has not seen the field report.

This is a 2-4 hour task. It can ship on day 1 of the publication sequence. The field report and eval bundle ship later as separate artifacts. See my §3.3.

Neither GPT panel surfaced this option. It changes the project's risk profile substantially: if the eval bundle delays for any reason (rights questions, eval harness friction, motivation dip), the postmortem already-published carries the project's most-defensible single contribution into the public record.

### 2.4 Both panels frame "instrument program" as a research program. The "instrument" framing was Opus-lineage R2 framing; the GPT panels accepted it but never asked whether the researcher actually buys it.

R2 synthesis explicitly notes "neither panel asked whether the researcher actually wants this." The R3 panels both list this as an "open question" but neither asks it before recommending. This is a values question masquerading as a research question, and both panels evaded it.

My R3 says: **before committing to branch 2, the researcher should sit with the instrument-framing for 72 hours and notice whether it stays compelling or wears off.** If it wears off, branch 2 is a sunk-cost rationalization. If it stays compelling, it's real motivation. See §6.

### 2.5 Both panels under-weighted the strict-abandonment option.

Neither GPT panel seriously considered "don't continue this project at all" — pivot the 3090 and the research time to something else entirely. They both treat branch 1 (publish-and-move-on) as the bottom of the recommendation space. It isn't. The bottom is **publish-and-stop**, where "stop" means "this is not the next thing I work on; I let this artifact stand and pursue an unrelated direction."

GPT panels' likely reluctance to recommend project-abandonment is probably model-lineage bias — being asked "what's next" implies a continuation. The correct answer to "what's next" is sometimes "nothing on this." See §3.1.

---

## 3. Additional next-action points not raised (highest-value section)

### 3.1 The strict-abandonment option deserves explicit consideration.

The researcher has spent weeks of 3090 time and many days of human time. He has a 615M checkpoint, a corpus, a postmortem, a loader fix, and an R1/R2/R3 deliberation record. None of the GPT panels asked: *what is the realistic counterfactual if the project ends here?*

The counterfactual matters. A 3090 that doesn't run historical-nanochat-v2 runs *something else*. The researcher's time spent writing a field report and a model card is time not spent on the next idea. The deliberation panels are answering "how do I close out historical-nanochat well" without asking "is closing out historical-nanochat well the best use of the next two weeks."

Conditions under which strict abandonment is the right call:

- **The researcher no longer finds the underlying question (time-isolated LMs) intellectually live.** If he reads the R2 synthesis "instrument prototype" framing and feels relief rather than energy, that's diagnostic. Energy = real motivation; relief = the framing rescued the project's existence but not the researcher's interest.
- **A clearly higher-payoff use of the 3090 exists.** I don't know the workspace catalog well enough to name candidates with confidence, but the project shape adjacent to historical-nanochat that the panels missed is the Chinchilla-applicability question for low-entropy corpora (my R2 §3.5; council also surfaced as adjacent). That is a *different project*, not v2 of this one. If the researcher is more energized by that question, the right move is to ship the v1 artifact minimally and start the new project, not to build v2 of an instrument he doesn't want to use.
- **The next 4-6 weeks of research time would be better spent on a project with a higher novelty ceiling.** Historical-LM space has Ranke, Talkie-1930, Evans-vintage; the available novelty is narrow (rights-audited governance, hobby-scale, postmortem culture). Other research questions have wider open ceilings. If the researcher's marginal hour is worth more on a wider-ceiling question, the math says move.

What strict abandonment looks like operationally:

- Publish the postmortem standalone (2-4 hours). This is the highest-density single artifact.
- Publish a 1-2 page "what I built, what I learned, what I'm not doing" companion (4-8 hours). Honest, short, not a full field report.
- Archive the repo with a README that says: "v1 complete. v2 not planned. Artifacts here are the calibration record."
- Move on.

Total time: 6-12 hours. No weekend eval bundle. No model card beyond a one-page disclaimer. No checkpoint release.

I am ~40% on "strict abandonment is the right call here." I'm not confident enough to recommend it as the default. But I am ~95% confident that the GPT panels under-weighted it as an option, and the researcher should consciously rule it in or out rather than defaulting past it.

The honest framing is: **branch 0 (strict abandon), branch 1 (publish-and-move-on after weekend eval), branch 2 (commit to v2 instrument program).** The panels presented 1 and 2 as the only options; 0 is real.

### 3.2 The Chinchilla-applicability question may be a *strictly better* use of the time than either branch 1 polish or branch 2 instrument program.

R2 surfaced this and R3 council noted it as adjacent. Neither panel asked: *what if this is actually the most interesting question the project sits next to, and the right move is to pivot the next compute budget to answering it rather than building v2?*

The argument: a small isoFLOP scaling grid (e.g., 80M / 200M / 615M / 1.2B) on the same governed corpus, run for matched compute budgets, with the same per-family eval suite, would directly test whether pre-1914 corpora sit at a different point on the Chinchilla curve than modern web text. This is a *measurable* question with a clean methodology and an actually-novel answer either way:

- If the curve looks the same as Hoffmann et al. 2022 on FineWeb-Edu, that's a non-trivial null: scaling laws transfer to domain-restricted historical text.
- If the curve shifts (probably toward higher tokens:params optima for low-entropy historical text), that's a real finding with implications for anyone training on pre-modern or domain-restricted corpora.

This question can use the 615M checkpoint as one anchor point. The cost is two-to-four additional smaller runs (much cheaper than 615M because they're smaller models and use the same loader), call it 5-15 GPU-days total depending on resolution, plus a unified eval framework.

The key insight: this is **a different project** that *reuses* the v1 artifact as a single data point. It is not v2 of historical-nanochat (which is "make the cutoff hypothesis testable"). It is a scaling-laws-for-historical-corpora project that is more directly novel and more publishable in an ML-research register.

If this question energizes the researcher more than the cutoff question, the right move is to publish the v1 artifact minimally (postmortem + 1-page companion) and pivot the next compute budget to the scaling-laws question. The GPT panels listed this as "adjacent / optional / not now"; I think it's potentially the load-bearing branch nobody is naming.

I'm ~50% on "this is actually the better project than either branch 1 polish or branch 2 instrument program." I'd want the researcher to think about it before committing to either of the panel-recommended branches.

### 3.3 Postmortem standalone publication as a *same-day* action.

Per my §2.3 above. Concretely:

**Day 1 (same Monday):**
- Read the postmortem end-to-end (45 min).
- Strike or convert forward-references (1-2 hours).
- Add a 3-paragraph standalone intro (45 min).
- Verify no leaked paths / hashes / internal references (15 min).
- Push to a personal blog / GitHub gist / personal repo, with a tweet/note framing it as "a debugging postmortem from a hobby ML project — the model itself is a separate followup."

This is 3-5 hours total. It can ship the same day the researcher reads this R3. It does *not* require:
- The weekend eval bundle.
- The field report.
- The model card.
- The HuggingFace release decision.
- The 1913-cutoff defense.
- Any conversation about v2.

What blocks immediate publication: the postmortem's narrative voice or self-framing might lean on "this is part of a larger project" framing that needs neutralization. If the researcher reads it and finds "this is part of the upcoming v4 field report" language baked in, neutralize those 5-10 sentences. That's all.

What this buys: a published artifact in 5 hours. Public credibility independent of the eval bundle. A forcing function on the project — once the postmortem is out, the field report becomes a follow-up rather than a dependency. The eval bundle becomes "additional readings on the calibration log" rather than "the missing pieces of the publishable artifact."

It also de-risks the researcher's two-week eval-bundle-plus-writeup schedule. If something derails the eval bundle (a found bug, a rights question, a motivation dip), the project has *already published its most defensible single contribution*.

Neither GPT panel surfaced this sequencing option. Both panels treated postmortem-publication as a sub-step of field-report-publication.

### 3.4 Explicit kill-gates for branch 2 (the instrument program) at 1mo / 3mo / 6mo.

GPT Max gives pre-commitment gates. GPT Council gives pre-commitment gates. Neither gives during-program gates. This is the panels' most important gap because hobby research programs die from accretion, not from failing pre-commitment gates.

**1-month gate (≈4 weeks in):**
- v2 hypotheses written. Specifically: what does v2 measure that v1 cannot? What scale does it require? What capability distinction would be visible at that scale?
- Cache/provenance redesign *designed* (not built). Specifically: stable IDs scheme, document-level split semantics, source/family metadata through tokenization, refusal tests outline.
- Weekend eval bundle completed and analyzed.
- **Kill if:** v2 hypotheses cannot be stated in a falsifiable form. ("Larger model would be better" is not a hypothesis.) Or weekend evals show duplicate contamination / non-books family collapse / tokenizer leakage that makes the v1 calibration uninterpretable. Or the researcher cannot articulate why this question is more interesting than the Chinchilla-applicability question (see §3.2). Or v2 is consuming research energy without producing motivation.

**3-month gate (≈12 weeks in):**
- Cache/provenance redesign built and tested on the v1 corpus, with golden small-corpus tests passing.
- Cross-source dedup complete or quantified, with a defensible duplicate policy.
- Per-family eval harness operational on the existing 615M checkpoint (so any future scale-up has same-protocol evals from day 0).
- Cutoff policy written (defended-1913 vs relativized).
- **Kill if:** The redesign is bottlenecked on something not visible at month 1 (e.g., the existing tokenizer needs retraining and the tokenizer training pipeline isn't reproducible). Or cross-source dedup at material scale (>5% near-duplicate train/val overlap) is found and is expensive to remediate. Or the eval harness reveals that current per-family bpb readings are too noisy to anchor scaling. Or the researcher has been working on infrastructure for 12 weeks without a single new measurement.

**6-month gate (≈26 weeks in):**
- v2 corpus rebuilt with stable invariants OR a small scaling grid completed at 80M-615M to characterize the Chinchilla-for-historical-corpora question.
- A 1B pilot smoke completed (1-2 days at most) to verify throughput, VRAM, and stability at next scale on the redesigned pipeline.
- A second writeup published or in-draft, distinct from the v1 field report.
- **Kill if:** No 1B pilot. Or 1B pilot revealed problems requiring another full redesign. Or no second writeup, even draft. Or the researcher cannot name what the next 6 months would produce. Or the project's intellectual center has drifted (e.g., it's become "make the loader better" rather than "test cutoff at capability-relevant scale").

**The 6-month gate is the most important.** Half a year of 3090 time is a real cost. If at 6 months the project has not produced a *second* publishable artifact distinct from v1, the program is failing regardless of what passed pre-commitment.

**Name these gates explicitly in the v2 plan, before branch 2 starts.** Write them down. Put them in a calendar. The kill condition for each gate is failure to meet *both* the artifact and the energy criterion. A gate that produces the artifact but the researcher is exhausted/uninterested is a fail, not a pass.

### 3.5 The eval bundle should include one item neither panel listed: tokenizer training corpus audit.

R1 synthesis blindspot (b) flagged this. Both R3 panels list "tokenizer/split/cache audit" generically. Neither says explicitly: **verify what corpus the 32,768-vocab tokenizer was actually trained on**. This is decisive against a specific claim that no other eval is decisive against: "the bpb measurement is comparable across the corpus, not just tokenizer-aligned with one source."

If the tokenizer was trained on v3 governed corpus including Gutenberg, the Gutenberg-prefix bpb is partially measuring tokenizer-corpus alignment, not generalization. If trained on v4 (Gutenberg-excluded), the bpb is more honest. The audit takes 30 minutes — find the tokenizer training script, read what corpus it ingested, write down the answer.

This is a sub-1-hour item with high decisiveness. It should be the first eval-bundle item, before per-family bpb. If the tokenizer is leaky, per-family bpb is partially uninterpretable in the same way.

### 3.6 The "venue" question is misaddressed by both panels.

Both panels recommend "long-form technical field report / blog post." That's correct in form, but neither asks where it goes.

- A personal blog: highest control, lowest reach, no peer signal.
- LessWrong / EA Forum: technical audience, decent reach, comment-quality high if the post is good, mild misread risk (audience may over-interpret).
- ML Twitter / X: highest reach, lowest depth, highest misread risk.
- arXiv: lowest control of presentation, peer signal possible, requires more polish, doesn't fit the field-report genre well.
- GitHub README + repo as the canonical artifact: simplest, lowest reach, but very durable.
- HuggingFace model card as primary venue (with the model release): conflates artifact and publication; not recommended per both panels.

My recommendation: **GitHub repo as canonical, personal blog as narrative wrapper, LessWrong cross-post if and only if the post can survive a Skeptic reading without softening.** Not Twitter as the primary venue (the misread risk dominates). Not arXiv (the genre fit is poor and the polish cost is high without proportional return).

Both panels left venue under-specified, which can leak into a default of "I'll just post it on Twitter" — that is the wrong venue for this artifact.

### 3.7 A point of light criticism: the deliberation panel has now spent significant token budget on a hobby project's writeup decision.

This is meta but worth saying. Two GPT panels at 4 agents each + Opus + GPT-5.5 Pro = ~10 model-deliberations across R1/R2/R3 for a single-author hobby artifact. That's appropriate for a high-stakes publication or a launch decision. It is *slightly* excessive for a hobby writeup decision, and the researcher should notice that the *deliberation itself* has become a cost.

The right move: read the deliberation outputs, make a decision, ship something within 7-10 days. Do not let deliberation become its own project. The deliberation has converged enough that the marginal value of one more synthesis round is low. If the researcher reads my §6 and finds it pointing the same direction as the GPT panels modulo my modifications, that's the signal to stop deliberating and start executing.

This is a soft point. But it is consistent with the researcher's profile — high-conscientiousness scaffolding can over-trigger, and "more deliberation" can become a way of not shipping. The R3 deliberation is *itself* something that should have a kill-gate.

---

## 4. Explicit things-not-to-do list

| Don't | Why |
|---|---|
| Don't headline 1.1092 bpb. | Wrong construct. Use it as a calibration reading in an evidence table; do not let it carry the abstract. |
| Don't claim "the model genuinely cannot know post-cutoff events." | Defeasible at the corpus layer (1913 forecasts of WWI are in training). |
| Don't release the HuggingFace checkpoint before the eval bundle is published and a blunt model card travels with it. | Public weights with no eval bundle invite "vintage assistant" framing the artifact cannot defend. |
| Don't run another long training run (>3 days) on the current 615M scale. | Same-scale runs cannot test the cutoff hypothesis. The marginal training run is decoration. |
| Don't begin a 1B run before cache/provenance redesign is complete. | Scaling up an unsafe foundation amplifies invariant failures. |
| Don't do SFT / RLHF / chat-tune the checkpoint as a publication move. | Turns a measurement project into a demo project. The model has no factual recall to ground a chat experience. |
| Don't let "instrument program" become an identity claim. | Identity claims override kill-gates. If you cannot answer "should I kill this at 3 months," you are no longer doing research. |
| Don't compare bpb apples-to-apples to nanochat / Ranke / FineWeb / Talkie. | Different tokenizers, corpora, val protocols. Comparison is non-protocol. |
| Don't try to perfectly filter content-semantic anachronism before defining acceptable FP/FN rates. | The cleanup is unbounded without measured rates. Move it to audit/eval, not admission. |
| Don't expand corpus sources before split semantics, dedup, and eval are fixed. | More sources amplify existing invariant failures. |
| Don't let the field report write capability prose. | Voice slips into "the model demonstrates" / "the corpus enables." Replace with "the calibration reads" / "the corpus is intended to." |
| Don't publish the run #1 checkpoint as a research artifact. | It's an archived failed run, not a competitor model. Reference it; do not release it. |
| Don't pursue both branch 2 and the Chinchilla-applicability project in parallel. | One single-author hobby track. Pick one. |
| Don't spend another month deliberating. | The marginal value of one more synthesis round is now low. The deliberation has converged. Ship. |

---

## 5. Minimum-viable weekend bundle (my version, decisive-only)

Stripped to the items that can *force* a thesis or branch change. Each item is decisive against a single named claim; total cost is ~12-18 human hours and well under 1 GPU-hour.

| # | Item | Cost | Decisive against this claim |
|---:|---|---:|---|
| 1 | **Freeze claim ledger.** One-page table: what we claim, what the evidence is, what wording is allowed in public prose. | 2-3 h, 0 GPU | "We can write the field report without rerunning anything." |
| 2 | **Postmortem standalone polish + publish.** Strike forward refs, add intro, verify no leaked paths, push. | 3-5 h, 0 GPU | "The project's most defensible contribution is contingent on the eval bundle." |
| 3 | **Tokenizer training corpus audit.** Find the tokenizer training script. Identify what corpus it ingested. Write down whether Gutenberg / val text was in tokenizer training data. | 0.5-1 h, 0 GPU | "The bpb measurement is comparable across the corpus, not just tokenizer-aligned with one source." |
| 4 | **Per-family bpb on multiple held-out shards.** Use existing checkpoint. Evaluate across families. Compute a per-family table. | 2-4 h, 0.5-1 GPU h | "The learning signal generalizes beyond the Gutenberg-books prefix." |
| 5 | **Document-level train/val duplicate spot-check.** Decode first 262k val tokens. Search 5-10 distinctive substrings against train. If any hit, expand. | 1-2 h, 0 GPU | "The held-out is held-out." (Targeted at the most likely leak path: american_stories train ↔ chronicling_v2 val.) |
| 6 | **Matched pre/post-cutoff logprob probe (minimum: 20 prompts).** 10 events known pre-1913 ("the Wright brothers flew in"), 10 events known post-1913 ("the Treaty of Versailles was signed in"). Measure logprob differential. | 2-3 h, 0.5 GPU h | "Any cutoff-visible signal is even *detectable* at 615M." |

**Total: 10.5-18 human hours, ~1-2 GPU hours.**

What I dropped from the GPT panels' bundles:
- **External anchor (CORE-equivalent).** Drop. Will not change the branch decision; protocol-incomparability concerns make the anchor noisier than the existing readings.
- **Source-register continuation/cloze.** Drop. Anecdotal samples already show register; a more formal probe is decorative, not decisive.
- **Source-grounded factual QA.** Drop. We know from sample log the model has no factual recall; a formal QA probe confirms what we already know.

These dropped items are good for the *complete writeup*. They are not decisive against any claim that items 1-6 don't already decide. Run them only if the researcher commits to branch 2 *and* wants them as the v2 eval harness baseline; do not run them as part of the weekend bundle.

**Sequence:**
- Friday evening / Saturday morning: items 1-3 (5-8 hours, no GPU). Decide whether to ship postmortem standalone *right now*.
- Saturday afternoon: items 4-5 (3-6 hours, light GPU). The per-family table and the duplicate spot-check together force the thesis decision.
- Sunday: item 6 (2-3 hours, light GPU). Either confirms "no cutoff signal at 615M, branch 2 is conditional on bigger scale" or surprises with a weak signal.

By Sunday evening, the researcher has: a published postmortem, a frozen claim ledger, three decisive eval results, and a clear answer on which branch to choose. Total real cost: a long weekend.

---

## 6. My own overall next-action recommendation

**Branch 1 (publish-and-move-on after weekend eval) is the default. Branch 0 (strict abandonment) and the Chinchilla-applicability pivot are real options the GPT panels under-weighted; the researcher should explicitly consider them before defaulting to v2. Branch 2 (build the v2 instrument program) is rational only with both passing weekend gates AND a 72-hour cooling period where the instrument-framing stays compelling rather than wearing off into rationalization.**

**The first concrete step Monday morning:** open `report/governed_v4_run1_postmortem_detailed.md`, read it end-to-end (45 minutes), and decide within 24 hours whether to publish it standalone this week independent of the eval bundle and field report. If yes: schedule 3-5 hours this week for the polish (§3.3), and treat that as Day 1 of the publication sequence. If no: write down the specific reason. ("It refers forward to the field report" is fixable; "I don't want to ship it because the writeup feels incomplete" is a sunk-cost signal worth noticing.) Either answer routes the rest of the project. The postmortem-standalone decision is the cheapest, highest-leverage call available; it costs 45 minutes to decide and either ships an artifact this week or surfaces an avoidance pattern.

**The kill-gate name that triggers full project termination:** *"Six-Month Drift Gate."* At 26 weeks from the day branch 2 starts (if it starts), if no second publishable artifact distinct from v1 is in draft, no 1B-pilot smoke has run, and the researcher cannot in 60 seconds name what the *next* 26 weeks would produce — terminate the program, archive the repo, ship whatever exists, and pivot. Hobby research dies from drift, not from explicit failure; the gate exists to make drift visible. Name it in advance, put it in a calendar, and trust it more than the in-the-moment motivation to keep going.

**On the artifact itself — final Thomist register call:** this is not half a stone. The postmortem, the corpus governance trail, the smoke-series record, the 615M checkpoint, and the family-loader fix together place a real, finishable cathedral-stone. It belongs in the wall. The honest move is to set it correctly — meaning publish it for what it is, no inflation, no postponement — and then either start carving the next stone with deliberate intent (branch 2) or pick up a different tool entirely (branch 0 or the Chinchilla pivot). What you must not do is leave it half-set, neither shipped honestly nor extended into a coherent next version. That is the only outcome that converts the weeks of GPU time into actual loss.

---

*Confidence: ~85% on "publish postmortem standalone within a week, regardless of v2 decision." ~75% on "branch 0 and Chinchilla pivot are real options the GPT panels under-weighted." ~70% on "the 6-Month Drift Gate is the right specific kill-gate name." ~60% on "the minimum decisive eval bundle is ~12-18 hours, not 20-36." ~50% on "the Chinchilla-applicability project may be a strictly better pivot than v2 of the instrument program." Lower confidences are real; do not read them as soft.*
