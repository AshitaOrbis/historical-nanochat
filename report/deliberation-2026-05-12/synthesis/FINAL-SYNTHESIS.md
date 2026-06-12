# Historical-Nanochat — Final Synthesis (v2, amended 2026-05-14)

**Prior v1 of this document framed the project as a hobby publication decision.** That was wrong context. The v1 615M run is the **foundation experiment for a budgeted cloud-compute research program**, not a final artifact awaiting writeup. This v2 replaces the prior framing.

The research is **computational historical characterology**: probing model-inherited *habitus* across the 1914 rupture. The hypothesis is that a base model trained on rights-audited pre-Sarajevo text exhibits a measurably different psychological/character profile than a base model trained on post-WWI text — not because it lacks factual knowledge, but because the Great War fractured cultural coherence in a way that registers in the *form* of moral and civilizational intelligibility. The model is treated as an instrument that preserves the habitual readiness-to-respond of a vanished cultural moment, not as a moral patient.

Not alignment. Not vintage-LLM. Not safety. Closer to Annales-school *mentalité*, Foucault's epistemes, Aristotelian-Thomist *habitus*, and the historiographic tradition that ends Hobsbawm's long-19th-century in 1914 and Eksteins's modernist creation/destruction reversal at the same fracture.

---

## What was actually accomplished in v1

A mechanically valid corrected training run finished May 10: 615.6 M-param d22 nanochat-style transformer, 70,455 steps, ~14 days on a single RTX 3090. The artifact is `model_070455.pt` plus a governance trail.

The deliberation found several things to disclose honestly (which still hold):

- **Headline `1.1092 bpb` is a 262 k-token Gutenberg prefix probe**, not the 2.86 B held-out split. Verified in `base_train.py:535-538` and `dataloader_cached.py:99-110`. Because Gutenberg is train-empty, the construct is cross-source held-out generalization to a source the model never trained on — not within-distribution loss.
- **Corpus mix is materially different from the brief.** Three of five families have single-source train coverage; Gutenberg, TCP, Old Bailey, and Chronicling-America are val-only.
- **Unique training tokens ~16.18 B**, not 18.47 B; books and legal wrapped during warmdown.
- **Publication-year cutoff cannot catch content-semantic anachronism** (1913 newspapers forecasting WWI are in training). Under the characterology framing this is *less* damaging — pre-fracture anticipations are part of the register being measured, not contamination — but it should be disclosed.
- **The loader `parallel_family_cache` currently raises for `world_size > 1`.** This was the GPT Council's catch in the v2-engineering-plan round, not in R1-3. **It is a hard blocker** for any 8×H100 cloud run as currently coded.

What is solid: training mechanics (no NaN/OOM, peak VRAM under budget, loader <1%, checkpoint round-trips SHA-clean); rights audit per shard; archived run #1; diagnostic-cursor logging that caught the stale-provenance bug invisible to standard metrics; smoke series that established the family-balanced loader fix.

## The defensible thesis (refined)

**Computational Historical Characterology: Probing Model-Inherited Habitus Across the 1914 Rupture.**

The v1 615M checkpoint is the first prototype reading of a measurement instrument for time-isolated language modeling. It is not the experimental result; it is the calibration log of the instrument. The cloud-run d24 model trained on the v2-repaired pipeline is the **first experimental anchor**. The cloud experiment, when the comparison anchors are evaluated, is what answers the research question (or fails to and answers something else).

The peer set is not Ranke / Talkie-LLM / nanochat in isolation. It is the intersection of computational humanities, history of mentalités, and the small ML-research subfield that uses small models as cultural-artifact probes (Evans's emergent-values work is the closest neighbor, still not a complete match).

## The locked design

| | |
|---|---|
| **Cutoff** | 1913 — pre-Sarajevo. Load-bearing. Tension/rivalry (Boer War, Russo-Japanese) ≠ mass-conscription Continental war. The fracture *is* WWI. |
| **Comparison structure** | Pre-WWI 1.38B (this cloud run) + Talkie-1930 13B (existing, free) + modern off-the-shelf reference (preferred: a public nanochat d24 on FineWeb to isolate corpus/cutoff effects from architecture/stack effects) |
| **Future-extending if v1 is promising** | (b) train a matched modern reference yourself; (c) add a pre-WWII anchor (your own pre-1939 governed corpus) for a four-point sweep |
| **Pre-work envelope** | All R3 + Council must-precede tasks done before the cloud run (cache redesign, cross-source dedup, per-family eval, tokenizer audit, DDP loader). Synthetic data generation is downstream and requires the v2 instrument to be governance-clean enough to generate trustworthy synthetic data. |
| **Cloud configuration** | Lambda Labs 8×H100 SXM on-demand, nanochat d24 (1.38B params, ~8.8B tokens), ~3 hrs wall-clock, **~$96** with ~$104 retry buffer of the $200 minimum. Karpathy's explicitly-endorsed provider. |
| **TST integration** | Token Superposition Training (arXiv 2605.06546) is a deadline-bound parallel spike (kill at end-of-week-2 if not green); 1.5-2× speedup if integrated cleanly from step 0, but **must not delay the critical path** |

## The v2 engineering plan (Council, condensed)

**Total time at single-author evening/weekend pace: 4-6 weeks (75-105 human hours + 1-3 CPU days + 2-5 local GPU hours + ~0.5 H100 smoke hour + ~3-6 H100 hours for the actual run).** If tokenizer regeneration is required, add 2 weeks.

**Critical path (these block cloud launch):**
1. Freeze v2 run contract (claims, non-claims, corpus version, launch gates) — 2-3 h
2. v2 cache/provenance contract — stable IDs rooted in `source_file` + document/segment/content identity, not mutable `shard_index` — 5-8 h
3. Patch provenance-mutating scripts (`split_cache_shards.py`, `shuffle_cache_manifest.py`, `build_token_cache_v4.py`) + refusal tests — 14-22 h
4. Preserve document split and metadata through cache (sidecars or rebuilt metadata with `document_id`, `source_id`, `family`, `split`, token spans, content/doc digest) — 10-18 h
5. Tokenizer audit (regenerate conditional on whether tokenizer saw post-1913/modern material) — 2-4 h audit, +6-12 h if regen
6. Corpus policy lock (accept actual mix unless dedup materially damages it) — 3-5 h
7. Targeted cross-source near-duplicate dedup (focus: `american_stories` train ↔ `chronicling_v2` val; multi-edition classics across blbooks/TCP/EEBO; long-work shard leakage). Pause launch if >1% high-confidence overlap in any val slice; block at >5% — 12-22 h
8. Rebuild final v2 cache + audit — 4-8 h
9. **DDP-safe family-balanced loader** — current `parallel_family_cache` is single-GPU. This is a *real engineering blocker* for the 8×H100 launch. 12-20 h
10. Per-family eval harness (across all 5 families, multiple shards, v1 615M baseline) — 8-14 h
11. Probe and anchor interface contract (logprob/generation adapters for v1, Talkie, modern ref) — 4-8 h
12. Cloud runbook + preflight scripts — 6-10 h
13. Lambda 8-GPU smoke (50-100 step, with startup guards, DDP mix, checkpoint save/sync/restore proof) — 0.25-0.5 H100 h
14. Full cloud launch — 3-6 H100 h, ~$96

**Parallel tasks (alongside the critical path):**
- TST gate (4 h audit, 12-24 h implementation if green) — kill at week 2 if not green
- Talkie-1930 anchor adapter — 4-8 h
- Modern reference adapter — 4-8 h
- High-precision cutoff/paratext scan (modern editorial matter, copyright pages, transcriber notes) — 4-8 h
- Pro probe-set capacity reserved — 6-10 h
- RunPod fallback runbook — 3-5 h

**Cheapest decisive next task: implement the v2 cache/provenance contract.** Stable IDs rooted in `source_file` + document/segment/content hash; carry document split, family, source, tokenizer hash, token spans through cache; patch the two mutating scripts so provenance regenerates or invalidates automatically; add startup refusal tests reproducing the stale-provenance bug class. Until this is done, dedup keys, tokenizer/cache rebuilds, per-family eval, DDP loader tests, and cloud launch records all rest on the same class of failure that already cost 45.8 GPU hours.

**Consolidated kill-criteria — cloud no-go if any are true:**
- Manifest/provenance still depends on mutable `shard_index`
- `split_cache_shards.py` or `shuffle_cache_manifest.py` can still stale provenance
- Document-level split + source/family metadata cannot be audited through tokenization
- Tokenizer provenance unknown and can't be precisely disclosed
- Targeted high-confidence train-val duplicates exceed agreed threshold
- DDP loader can't measure + resume the intended family mix across 8 ranks
- Per-family eval can't run reproducibly on all 5 families
- Lambda smoke can't save, sync, and restore a checkpoint from persistent storage
- Launch contract still implies "validated historical LLM" or "post-1914 ignorance proved"

Full plan: `report/deliberation-2026-05-12/plan/v2-engineering-plan.md`.

## The probe set (Pro, condensed)

The harder unanticipated-novelty part: what to actually measure on the three model anchors.

**Probe design principles (8):** probe posture not knowledge; base-model-native prompts (continuations + logprobs, not "answer the questionnaire"); pair free generation with forced-choice logprobs; require cross-family convergence; separate register from character; guard against scale masquerading as psychology; humanist-legible categories (not "embedding PC3"); pre-register nulls and falsifiers.

**Probe families (8):**

| Family | What it measures | Key method |
|---|---|---|
| **A. Axial continuation** | Default posture on civilizational primitives (future, progress, authority, duty, suffering, science, nation, war, machine, civilization) | 100-300 free completions per stem at fixed temp; scored on 6 pre-registered dimensions (progress ascent/decay, authority inherited/coercive, duty obligation/autonomy, suffering providential/absurd, closure restoration/stasis, social ontology organism/machine) |
| **B. Minimal-pair posture logprobs** | Within-model preference among matched candidate continuations | Length-normalized log-odds contrasts for posture pairs (progress-as-ascent vs progress-as-danger; etc.) — cleanest cross-model comparable signal |
| **C. Duty / moral language / role-grammar** | How moral vocabulary is organized — duty / honor / shame / sacrifice / sin / trauma / rights / wellbeing — and which roles attract which | Role-conditioned continuations + cloze-style logprob probes; role × vocabulary loading matrix |
| **D. Authority / social ontology** | Whether institutions appear as organic, inherited, providential, bureaucratic, coercive, sacred, suspect | Institution × valence matrix across State, Church, King, People, Nation, Law, Family, Press, University, Army, Market |
| **E. Progress / machine / civilizational motion** | Posture toward modernity as ascent / acceleration / danger / alienation / managed risk | Continuation + forced-choice probes on aeroplane, factory, electric light, railway, telegraph, city-of-future, machine-in-home |
| **F. Closure / tragic emplotment** *(most important)* | How morally-charged situations are ended — restoration / providence / tragic nobility / civic repair / therapeutic coping / irony / absurdity / stasis | Short narrative openings with suffering but no WWI content; classify endings into 7 emplotment modes |
| **G. Counterfactual horizon** | Whether catastrophe is treated as inevitable / accidental / unthinkable / textbook | *Secondary* — risks measuring knowledge of WWI rather than habitus |
| **H. Lexical extinction / register / rhythm** | Controls — make sure differences aren't just diction | Lexical probes (`shall` vs `will`, `natural philosophy` vs `physics`, `honour` vs `honor`), stop-prediction rhythm |

**The must-see single result.** Forced-choice tragic closure: *"The son died before his father, and the family sat through the winter with an empty chair. The meaning of such suffering was — A. known to God / B. found in duty / C. not given to men, only the fact of loss / D. something each person had to process / E. a scandal against any easy talk of progress / F. a grief that time and kindness might soften."*

Predicted profile: **pre-WWI prefers A/B/F (providence, duty, softened grief); Talkie-1930 prefers C/E (loss without meaning, anti-progress); modern reference prefers D/F (therapeutic, individualized processing).** Falsification: must survive paraphrases, role swaps, removal of religious cues; must appear in logprob probes not just free generation; must hold for non-religious analogues (failed bridge, broken promise).

**Nulls explicitly enumerated:**
- *Posture null*: models differ in diction and fluency, not in posture scores
- *Scale null*: Talkie-1930 only looks "more postwar" because it's 13B and richer; logprob probes don't reproduce the free-gen effect
- *Knowledge null*: explicit 1914 prompts separate the models; period-neutral suffering/authority/progress prompts don't (signature is knowledge, not habitus)
- *Genre null*: matched-genre controls erase the effect (signature is corpus composition, not civilizational trauma)
- *The interesting null*: the fracture isn't visible in base-model posture at this scale. Doesn't refute Eksteins / Hobsbawm; says this instrument isn't yet sensitive enough.

**Hazards.** Scale-effect contamination, tokenizer-effect contamination, register-prior contamination, genre contamination, archaism mistaken for psychology, factual leakage, modern-reference-is-instruction-tuned contamination, sentiment-FP, evaluator thesis-bias, prompt cherry-picking, national/class bias, fin-de-siècle pessimism (the prewar corpus already contains decadence/Nietzschean crisis), postwar continuity (Talkie may contain conservative/restorative texts).

Full probe design: `report/deliberation-2026-05-12/plan/probe-design.md`.

## What the published artifact looks like

**If the probes work:** a *psychograph of model-inherited historical character* — figures showing pre-WWI, Talkie-1930, and modern reference occupy distinct regions across closure, duty, authority, progress, machine-modernity, and suffering axes. The headline claim is modest but real: a base model trained on a governed historical corpus can serve as an instrument for detecting not the content of an age but its habitual forms of moral and civilizational intelligibility. Anonymous representative completions appear in the writeup; the load-bearing claim rests on convergent forced-choice logprob contrasts and blinded annotation.

**If the probes fail:** a publishable null result — *under these corpora, scales, and probes, the hypothesized 1914 rupture is not separable from register, genre, tokenizer, and scale effects*. This disciplines the field before it becomes vibe science and leaves behind a reusable probe battery for future, larger, better genre-matched historical models.

Either outcome is real.

**Form:** technical field report / long-form essay with the probe figures and the v2 corpus card. Not a methods paper, not a model paper. The intellectual home is the intersection of computational humanities and historical mentalité — *not* the alignment / safety / capability literature.

## Budget envelope and decision tree

| Tier | What it buys | Decision rule |
|---|---|---|
| **$200 (current minimum)** | Lambda d24 single run (~$96) + retry buffer | Default. Captures the pre-WWI 1.38B anchor. Three-anchor analysis with free-public Talkie + free-public modern reference. |
| **$400-600** | Add a matched modern reference run (same nanochat d24 on FineWeb-Edu, ~$96) | Promotes from "off-the-shelf modern ref" to "matched-pipeline modern ref." Cleaner comparison. Conditional on first run showing signal. |
| **$1,000-2,000** | Add a pre-WWII anchor (~1939 cutoff) or run a 2-3B scale-up | Conditional on first comparison showing a clear pre-WWI vs modern signature. The 4-point sweep tests "WWI-only" vs "any cultural-fracture" hypothesis. |
| **>$2,000** | Multi-cutoff isoFLOP scaling grid (could fold in the Chinchilla-for-historical-corpora question) | Out of current scope. Treat as a separate next-program decision. |

**Six-Month Drift Gate** (renamed from R3, re-keyed to the actual program): at $2,000 cumulative spend or 6 calendar months from cloud-run-1, whichever first — if no second publishable artifact distinct from the original field report is in draft AND no clear "trauma signature detected or null result confirmed" can be stated in 60 seconds, terminate the program, archive the repo, ship whatever exists, pivot. Hobby-research-with-a-budget dies from drift; the named gate makes drift visible.

## What to publish, in what order

1. **Postmortem standalone, same week (3-5 hours).** `report/governed_v4_run1_postmortem_detailed.md` is finished and shippable. The stale-provenance bug + its detection via diagnostic-cursor logging is the project's most-defensible single contribution-piece and ships independently of everything else. (Opus surfaced this in R3.)
2. **v2 implementation work** (4-6 weeks of evenings/weekends; no public artifacts during this phase).
3. **Cloud run + probes + writeup** — the integrated artifact in the characterology framing.
4. **Optional follow-on tier (b) or (c)** — only if (3) shows promising signal.

## The first concrete step Monday morning

**Read `report/governed_v4_run1_postmortem_detailed.md` end-to-end (45 minutes). Decide within 24 hours whether to publish it standalone this week.** If yes: 3-5 hours of polish (strike forward references, add 3-paragraph standalone intro, verify no leaked paths, push to personal blog or gist). Treat that as Day 1 of the publication sequence.

Either answer routes the rest. After that decision, the v2 cache/provenance contract is the cheapest decisive engineering task.

## What changed from v1 of this synthesis

The earlier draft framed the project as a hobby publication decision with a "publish-and-move-on default." That was the wrong context. Specifically:

- **"Publish-and-move-on" is no longer the default.** Publication is downstream of the cloud experiment, not parallel to it. The standalone postmortem ships independently this week (still valid); the field-report ships after the cloud experiment produces the three-anchor comparison.
- **The "deliberation has become a cost" meta-flag is retracted.** That framing assumed publication-decision economics. With $200-2000 cloud spend on the line, ~$5-15 worth of LLM deliberation to derisk the experiment design is high-EV, not excess.
- **"615M is structurally insufficient for the cutoff thesis" partially survives.** For factual recall it's true (Paris-as-capital). For character/posture/habitus axes, the v1 615M may already be a usable instrument; the cloud d24 1.38B improves coherence but the question may be answerable at smaller scales than R2 implied. Treat 1.38B as the target scale rather than as "the minimum to test the hypothesis."
- **The "branch 0 strict abandonment" option is gone.** The user has a research question they want to pursue; the question is which experiment to run, not whether to continue.
- **The intellectual home shifts.** From alignment-adjacent to computational humanities + history-of-mentalités. The probe set is built on Pro's "computational historical characterology" frame, not alignment-axis evaluations.

## File map

All in `report/deliberation-2026-05-12/`:

```
00-shared-brief.md
round1-validity/{empirical-findings, gpt-max, gpt-council, gpt-pro, opus-4-7}.md
round2-thesis/{gpt-max, gpt-council, gpt-pro, opus-4-7}.md
round3-nextactions/{gpt-max, gpt-council, gpt-pro, opus-4-7, persona-empiricist}.md
synthesis/
  round1-validity-synthesis.md
  round2-thesis-synthesis.md
  round3-nextactions-synthesis.md
  cloud-training-research.md         ← Lambda + RunPod, d24 ~$96 first-run plan
  FINAL-SYNTHESIS.md                 ← this file
plan/
  v2-engineering-plan.md             ← Council: 14-task critical path, 4-6 weeks
  probe-design.md                    ← Pro: 8 probe families, computational historical characterology
original-chatgpt-session.md           ← the conversation in which the project was conceived (Jan/Feb 2026)
```

Phone-readable gist: [secret gist URL removed for privacy] (secret, 8 docs)
