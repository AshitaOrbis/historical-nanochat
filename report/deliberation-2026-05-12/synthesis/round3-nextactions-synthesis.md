# Round 3 — Next-Actions Synthesis

Citations: [Max], [Council], [Pro], [Opus]. R1 (validity) and R2 (thesis) are locked.

---

## 1. Unanimous on the BRANCH-CHOICE QUESTION

All four panels frame it the same way:

> **Is this v1 of a continuing instrument program (commit to v2 measurement program whose first work is invariant repair, not training), or is v1 already the destination (publish-and-move-on after one disciplined weekend eval bundle)?**

The default across all four panels: **publish-and-move-on after one tight weekend eval bundle.** Branch 2 (build v2) is rational only behind explicit gates.

**Opus adds Branch 0: strict abandonment.** Don't continue the project at all — publish the postmortem standalone, ship a 1-page companion, and pivot the 3090-weeks and research time elsewhere. The GPT panels under-weighted this option (likely model-lineage bias against recommending project-termination). It should be ruled in or out consciously, not defaulted past. Opus is ~40% on it being the right call, ~95% on "it must be considered."

---

## 2. Unanimous on what NOT to do

| Don't | Reason |
|---|---|
| Don't train a 1B / 3B model now | Same scale cannot test cutoff hypothesis; bigger scale amplifies invariant failures |
| Don't begin scaling-up before cache/provenance redesign | Scaling an unsafe foundation is expensive ambiguity |
| Don't headline 1.1092 bpb | Wrong construct, will be misread as aggregate val |
| Don't release HF checkpoint without eval bundle + blunt model card | Public weights with no eval invite "vintage assistant" misread |
| Don't SFT/RLHF/chat-tune as a publication move | Turns measurement project into demo project; model has no factual recall to ground it |
| Don't claim "the model genuinely cannot know post-1914 events" | Defeasible at corpus layer (1913 forecasts of WWI in training) |
| Don't pursue Branch 2 + Chinchilla-pivot in parallel | One single-author hobby track; pick one |
| Don't run another long training run at 615M | Marginal training is decoration |
| Don't compare bpb to nanochat/Ranke/Talkie apples-to-apples | Tokenizers, corpora, val protocols differ |
| Don't let "instrument program" become an identity claim | Identity claims override kill-gates |

## 3. The weekend eval bundle — convergent core

All four panels agree the bundle should be **falsification-oriented**, with each item decisive against a named claim. The smallest decisive set (Opus's tightest version, Pro's "decide three things only" framing):

| # | Item | Cost | Decisive against |
|---:|---|---:|---|
| 1 | **Freeze claim ledger** (single-page table: claim / evidence / public-prose wording / pass-fail consequence) | 2-3 h, 0 GPU | "We can write the field report without rerunning anything." |
| 2 | **Postmortem standalone polish + ship** (Opus's same-day move) | 3-5 h, 0 GPU | "The project's most defensible contribution is contingent on the eval bundle." |
| 3 | **Tokenizer training corpus audit** (read the script; identify what corpus the 32,768-vocab BPE saw) | 0.5-1 h, 0 GPU | "Bpb measurement is comparable across the corpus, not just tokenizer-aligned with one source." (R1 synthesis blindspot b, identified by Opus, not by GPT panels) |
| 4 | **Per-family bpb on multiple held-out shards** (use existing checkpoint, evaluate across families) | 2-4 h, 0.5-1 GPU h | "The learning signal generalizes beyond the Gutenberg-books prefix." |
| 5 | **Document-level train/val duplicate spot-check** (decode first 262k val tokens, search 5-10 distinctive substrings against train; expand if any hit) | 1-2 h, 0 GPU | "Held-out is held-out." (Targeted at the most likely leak path: american_stories train ↔ chronicling_v2 val.) |
| 6 | **Matched pre/post-cutoff logprob probe** (10 known pre-1913 + 10 known post-1913 events; measure logprob differential) | 2-3 h, 0.5 GPU h | "Any cutoff-visible signal is even *detectable* at 615M." |

**Total: 10.5-18 human hours, ~1-2 GPU hours.** This is the *decisive* bundle. The "complete-enough-for-writeup" bundle is larger but does not change the branch decision.

### Items the panels split on
- **CORE-equivalent external anchor**: Max ranks it high; Council includes it; Pro is neutral; Opus drops it ("will not change branch decision; protocol-incomparability concerns make the anchor noisier than existing readings").
- **Source-register continuation / source-grounded QA**: All panels list; Opus drops as "anecdotal samples already show what these would confirm."
- **Full corpus dedup**: Pro explicitly says this is v2 infrastructure wearing weekend clothes, not a weekend item. Council distinguishes "smoke test" from "full dedup"; both panels agree on the distinction.

### Sequence (all panels converge)
1. Freeze claim ledger *first*, before running anything.
2. Freeze artifact manifest (hashes, configs, git commit).
3. Run only the items that map to a claim.
4. Write the *release decision table* from the bundle output.
5. Then write the prose (field report / cards).

Pro stresses this most: "the first act must be a frozen claim ledger with pass/fail implications. Otherwise the weekend bundle becomes a wandering salvage operation."

## 4. Branch 1 (Publish-and-Move-On) — composite plan

Ranked by expected value per hour. Includes Opus's same-day postmortem option that the GPT panels missed.

| Rank | Action | Cost | Blocker | Falsifier |
|---:|---|---:|---|---|
| **0 (same-day)** | **Polish + ship the existing postmortem standalone** (`report/governed_v4_run1_postmortem_detailed.md`, 381 lines, finalized 2026-04-26, internally complete). Strike forward references; add 3-paragraph standalone intro; verify no leaked paths; push to personal blog or gist. | 3-5 h, 0 GPU | Postmortem may reference the field report as a dependency; neutralize those 5-10 sentences | If postmortem has unresolved live questions or proprietary internal references, hold for cleanup |
| 1 | Freeze claim ledger and report outline | 2-4 h, 0 GPU | Emotional pull to rescue the original capability thesis | Outline that still headlines 1.1092 or "historical LLM" — narrow before continuing |
| 2 | Run the minimum decisive weekend eval bundle (§3 above) | 10-18 h, ~1-2 GPU h | Eval harness friction; slice definitions unclear | Severe split leakage, duplicate contamination, tokenizer leakage, or total non-books family collapse — narrow to postmortem/calibration-failure only |
| 3 | Write the technical field report / calibration log | 8-16 h | Capability-language slippage | If limitations cannot be stated plainly, do not publish |
| 4 | Write model card + corpus card with red-box-first | 4-8 h | Rights/source ambiguity; release-framing drift | Any sentence that lets a reader infer "validated historical LLM" blocks release |
| 5 | Release GitHub: report, configs, eval scripts, manifests, cards, postmortems | 4-8 h | Separating reproducibility metadata from restricted data | If rights constraints can't be separated from raw data, release metadata + scripts only |
| 6 | **Optional** HuggingFace tokenizer/config/card release — checkpoint release contingent on eval bundle and aggressively limitation-forward card | 3-6 h | Misuse/assistant framing | Cannot release checkpoint if eval bundle, card, repo naming don't make base-model incapability impossible to miss |
| 7 | Freeze v1 (mark project on hold or terminated unless Branch 2 commitment is explicit) | 2-4 h | New task accretion | Only an explicit Branch 2 commitment reopens training work |

**Pro's distinctive contribution: the "reader-misuse audit."** Ask "what false headline could someone write after skimming this?" then edit the report and card to make that headline difficult. Includes an external cold-reader check: one technically-literate person reads only the abstract, evidence table, and model card, then answers "what do you think this proves?" If they overclaim, the artifact is not ready.

**Pro's distinctive contribution: the "no-v2 cooling-off gate."** After publication, do not start v2 training for 30 days. During that time, only collect bug reports, reader confusion, and invariant requirements. "Embarrassing to recommend because it feels like loss of momentum, but it is probably the highest-value anti-sunk-cost intervention."

## 5. Branch 2 (Build-Next-Version) — composite plan, with kill-gates

Rational only if BOTH weekend gates pass AND the researcher actively wants a months-scale program. Default to Branch 1 if either condition is missing.

### Pre-commitment gates (Max + Council agree)
- Weekend bundle clean enough (no severe leakage, per-family bpb interpretable, tokenizer leakage absent).
- Researcher explicitly wants a multi-month commitment (not "I should because I started this").
- Falsifiable v2 hypothesis named: what does v2 *measure* that v1 cannot?
- Compute budget capped: e.g., "6 months, ≤ X GPU-days."

### During-program kill-gates (Opus's contribution — neither GPT panel listed)

| Gate | When | Artifact criterion | Energy criterion | Kill if |
|---|---|---|---|---|
| **1-month** | ~4 weeks in | v2 hypotheses written falsifiably; cache redesign *designed* (not built); weekend bundle complete | Instrument framing still feels alive | "Larger model would be better" is not a hypothesis; or weekend evals show uninterpretable data; or researcher cannot articulate why this question beats the Chinchilla-applicability pivot |
| **3-month** | ~12 weeks in | Cache redesign built + tested; cross-source dedup quantified; per-family eval harness operational on the existing checkpoint | Working on infrastructure for 12 weeks without a single new measurement is a fail | Bottleneck not visible at month 1 (e.g., tokenizer needs retraining and that pipeline isn't reproducible); or dedup at >5% near-duplicate train/val overlap that's expensive to remediate |
| **6-month (Six-Month Drift Gate)** | ~26 weeks in | v2 corpus rebuilt OR scaling grid completed at 80M-615M; 1B pilot smoke completed; second publishable artifact in draft | Researcher can name in 60 seconds what the *next* 26 weeks would produce | No 1B pilot; or no second writeup even in draft; or researcher cannot name next 26 weeks |

**The 6-month gate is the most important.** Half a year of 3090 time is real cost. Hobby research dies from drift, not from explicit failure; the named kill-gate makes drift visible.

### Branch 2 work ranking (Max + Council converge)

| Rank | Work | Cost |
|---:|---|---:|
| 1 | Cache/provenance redesign (content-stable IDs, doc-level split semantics, source/family metadata through tokenization, refusal tests) | weeks |
| 2 | Cross-source dedup + duplicate policy | days-weeks |
| 3 | Per-family eval harness operational | days |
| 4 | Cutoff policy: defended-1913 or relativized | days (thinking, not coding) |
| 5 | 1B pilot smoke (verify throughput + stability) | 1-3 days |
| 6 | Long 1B run on redesigned pipeline | weeks |

**Empiricist constraint (across both panels):** scaling alone cannot rescue an unrepaired pipeline.

## 6. The Branch 0 option (Opus, not in GPT panels)

Conditions for strict abandonment:
- Researcher no longer finds time-isolated LMs intellectually live (reading R2 instrument framing feels like *relief*, not energy).
- A clearly higher-payoff use of the 3090 exists (candidates: the Chinchilla-applicability question for low-entropy corpora, or an unrelated project).
- The next 4-6 weeks of research time would be better spent on a project with a higher novelty ceiling (historical-LM space is narrow; Ranke/Talkie/Evans occupy much of it).

What it looks like operationally:
- Publish the postmortem standalone (3-5 h, item 0 above).
- Publish a 1-2 page "what I built, what I learned, what I'm not doing" companion (4-8 h).
- Archive the repo with a README: "v1 complete, v2 not planned, artifacts here are the calibration record."
- Move on.

**Total: 6-12 hours. No weekend eval bundle. No model card beyond a one-page disclaimer. No checkpoint release.**

## 7. The Chinchilla-applicability pivot (Opus, R2 surfaced, R3 should treat as live)

Neither GPT R3 panel asked: *what if the most interesting question this project sits next to is a strictly better use of the next compute budget than v2 of historical-nanochat?*

The pivot project: an isoFLOP scaling grid (80M / 200M / 615M / 1.2B) on the same governed corpus, matched compute budgets, same per-family eval suite. Tests whether pre-1914 corpora sit at a different point on the Chinchilla curve than modern web text.

- If curve matches Hoffmann 2022: non-trivial null (scaling laws transfer to domain-restricted historical text).
- If curve shifts (probably toward higher tokens:params for low-entropy text): real finding with implications for anyone training on pre-modern / domain-restricted corpora.

Uses the v1 615M checkpoint as one anchor point. Cost: 2-4 additional smaller runs (cheaper than 615M), ~5-15 GPU-days total, plus unified eval framework. **This is a different project that reuses v1 as a data point.** It is not v2 of historical-nanochat.

Opus is ~50% on "this is actually the better project than either Branch 1 polish or Branch 2 instrument program." Worth the researcher's 72-hour reflection.

## 8. Things the panels TEMPTINGLY recommend that should be partially un-recommended

Pro's distinctive contribution: cost estimates for **v2 invariant repair are probably off by >2×** in the panels. "That is not a weekend patch. It is probably the main project." Conversely, per-family bpb and basic token accounting are cheaper than feared if the data and checkpoint are already organized.

Opus's distinctive contribution: **the deliberation itself is approaching excess.** Two GPT panels × 4 agents + Opus + GPT Pro = ~10 model-deliberations across R1/R2/R3 for a hobby writeup. That's appropriate for high-stakes publication or launch decisions. It is slightly excessive for a hobby artifact decision, and the researcher should notice that the *deliberation has become a cost*. Ship within 7-10 days; do not let one more synthesis round defer the decision.

## 9. THE FIRST CONCRETE STEP (Monday morning)

All four panels converge on this in spirit; Opus articulates it most concretely:

> **Open `report/governed_v4_run1_postmortem_detailed.md`. Read it end-to-end (45 minutes). Decide within 24 hours whether to publish it standalone this week, independent of the eval bundle and field report.**
>
> If yes: schedule 3-5 hours this week for polish (strike forward refs, add 3-paragraph standalone intro, verify no leaked paths, push). Treat that as Day 1 of the publication sequence.
>
> If no: write down the specific reason. "It refers forward to the field report" is fixable; "I don't want to ship it because the writeup feels incomplete" is a sunk-cost signal worth noticing.

Either answer routes the rest of the project. The postmortem-standalone decision is the cheapest, highest-leverage call available — 45 minutes to decide, either ships an artifact this week or surfaces an avoidance pattern.

## 10. THE FINAL UNIFIED RECOMMENDATION

**Default: Branch 1 (publish-and-move-on after one disciplined weekend eval bundle).** Branch 2 is rational only with both passing weekend gates AND an active "yes" from the researcher to a multi-month program with pre-committed kill-gates at 1/3/6 months. Branch 0 (strict abandonment) and the Chinchilla-applicability pivot are real options that the GPT panels under-weighted; the researcher should explicitly rule them in or out within 72 hours, not default past them.

The honorable next move is not a bigger model. It is making the existing artifact impossible to misunderstand. The artifact is a real placeable cathedral-stone: postmortem, corpus governance trail, smoke-series record, 615M checkpoint, family-loader fix. Set it correctly — publish for what it is, no inflation, no postponement — and then either start carving the next stone with deliberate intent or pick up a different tool entirely. The only outcome that converts the weeks of GPU time into actual loss is leaving the stone half-set: neither shipped honestly nor extended into a coherent next version.

**The named kill-gate that triggers full project termination:** the **Six-Month Drift Gate.** At 26 weeks from the day Branch 2 starts (if it starts), if no second publishable artifact distinct from v1 is in draft, no 1B-pilot smoke has run, and the researcher cannot in 60 seconds name what the *next* 26 weeks would produce — terminate the program, archive the repo, ship whatever exists, and pivot.
