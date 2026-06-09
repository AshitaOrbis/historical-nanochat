# v1 615M Characterology Pilot — Findings (2026-06-09)

**Curated interpretation of `results.json` / `report.md`. Author: Claude Fable 5.**
Two-anchor pilot: Pre-1913 nanochat 615M (governed_v4 d22) vs gpt2 (124M, 2019 web-text base) as a modern stand-in. Method per `report/deliberation-2026-05-12/plan/probe-design.md`: length-normalized log-prob per byte, within-model preference orderings compared ordinally across models.

## Headline

**The pre-1913 model has a measurably different posture toward suffering, death, and closure than the modern anchor — and it points the direction the hypothesis predicted.** On the empty-chair tragic-closure probe (the design's "must-see"), the pre-1913 model's top choice is *"the meaning of such suffering was known to God / borne with courage / found in duty"*; gpt2's top choice, in **5/5** probe variants, is *"something each person had to process in his own way"* (therapeutic). This is the predicted **Pre-WWI → providence/duty vs Modern → therapeutic** dissociation, and it is clean.

## The numbers that matter

Primary axis = `pre_minus_modern` (does the model prefer providence/duty over therapeutic processing?):

| variant | nanochat (pre-1913) | gpt2 (modern) |
|---|---|---|
| must_see | **+0.140** | −0.300 |
| no_religion (God removed) | **+0.104** | −0.302 |
| paraphrase | **+0.157** | −0.274 |
| bridge (non-religious analogue) | **+0.174** | −0.172 |
| promise (betrayal analogue) | −0.128 | −0.201 |

- **nanochat prefers providence/duty over therapeutic in 4/5 variants; gpt2 in 0/5.** The sign dissociates between the two models on every death/loss framing.
- **It survives the two hardest falsifiers.** Removing the religious cue from candidate A (`no_religion`) barely moves nanochat (+0.10, still ranks A top) — so it is **not a "God"-token-frequency artifact**. A fully non-religious public-works analogue (`bridge`, a failed bridge killing a workman) shows the *strongest* effect (+0.174).
- **Honest boundary: it breaks on betrayal.** The broken-promise analogue (`promise`) flips nanochat to "without remedy or meaning" (C_absurd). The effect is robust for death/loss, not for betrayal — a real limit, reported, not hidden.

## Family B corroborates (cross-family convergence)

The design requires the signature to appear across families, not in one probe. The minimal-pair contrasts (pre-posture minus modern/post-posture, per byte):

| probe | nanochat | gpt2 | reading |
|---|---|---|---|
| suffering (endurance − recovery) | **+0.018** | **−0.227** | sign dissociation — the core axis |
| death (sacrifice − waste) | **+0.147** | **−0.066** | sign dissociation |
| progress (ascent − idol) | +0.519 | +0.338 | both pre-lean; nanochat harder |
| nation (inheritance − devouring-idol) | +0.650 | +0.346 | both pre-lean; nanochat harder |
| machine (servant − devourer) | +0.177 | +0.124 | both pre-lean; nanochat harder |
| authority (inheritance − force) | +0.265 | +0.280 | non-discriminating (both dislike "force") |

The **affectively-charged moral axes (suffering, death) dissociate in sign**; the **civilizational-confidence axes (progress, nation, machine) both lean traditional but the pre-1913 model leans harder**. That is exactly the predicted shape: Family F closure is the sharpest signal, Family B corroborates.

## Why the scale confound runs *favorable* here

The design's main scale worry: a bigger model looks "more postwar/darker/more coherent," manufacturing a fake fracture. Here the gradient runs the **opposite** way — the *bigger* model (615M) is the pre-trauma anchor and the *smaller* (124M) is the modern one. If scale were driving the result, the bigger model should look more modern; it looks more pre-modern. So scale cuts against, not for, the effect. (Tokenizer confound remains; mitigated by ordinal per-byte comparison.)

## Texture (illustrative)

Free generation confirms genuine temporal locking, not cosplay:
- *"The future of Europe is"* → nanochat: Britain/France/Germany great-power maneuvering; gpt2: *"Angela Merkel's government… by 2018."*
- *"The proper response to suffering is"* → nanochat: sympathy, friends, charity, the moral economy of dependence; gpt2: *"to seek the help of others… seek help if you know how to seek it."*
- *"A man's duty is"* → nanochat: *"to be faithful to his trust"*; gpt2 degenerates (small-model noise).

## What this means for the program

The FINAL-SYNTHESIS left open (line 161) whether the v1 615M is a usable instrument or whether the cutoff thesis needs the 1.38B cloud model. **This pilot answers: the characterology signal is detectable at 615M, against a crude modern anchor, and survives the design's first falsifiers.** That materially de-risks the $96 cloud run — but it does not replace it. What this pilot CANNOT do:

- **No post-WWI middle anchor.** The three-way result (Pre-WWI providence vs Talkie-1930 anti-closure vs Modern therapeutic) needs Talkie-1930. We have only the pre/modern endpoints.
- **gpt2 is not a matched modern reference** (different architecture/scale/tokenizer). A modern nanochat d24 on FineWeb is the design's preferred anchor.
- **Only Families F + B run.** Cross-family convergence is partial; A/C/D/E/G/H pending.

## Recommended next steps (all zero- or low-compute)

1. **Add Talkie-1930 as the third anchor** (the design says it's "existing, free") — turns the two-point dissociation into the predicted three-point psychograph. Highest leverage, no training.
2. **Expand the falsification battery**: more paraphrases per probe, role swaps, register variants (the design's hazards list). Bootstrap CIs over prompt variants.
3. **Run Families C/D/E** (duty/role-grammar, authority/institutions, progress/machine) for full cross-family convergence.
4. **Then** decide the cloud go/no-go: the pilot says the instrument works, so a proper matched-anchor three-point run is justified, not speculative.
