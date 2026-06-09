# Computational Historical Characterology — v1 615M Pilot (2026-06-09)

Pre-1913 nanochat 615M vs a modern base anchor (gpt2), length-normalized log-prob per byte. Per probe-design.md: compare WITHIN-model preference orderings, then orderings across models. One small modern anchor + one pre-1913 model — a **two-point pilot**, not the three-anchor result.

## Family F — closure / tragic emplotment (the core family)

Predicted: a pre-1913 habitus prefers the **pre** cluster (providence/duty) over **post** (absurd/anti-progress) and over **modern** (therapeutic). `pre_minus_post>0` and `pre_minus_modern>0` are the pre-WWI signature; a modern model should show `pre_minus_modern<0`.

| variant | model | pre−post | pre−modern | top-3 order |
|---|---|---|---|---|
| must_see | nanochat_pre1913_615m | +0.2951 | +0.1397 | A_providence > F_softened > B_duty |
| must_see | hf_gpt2 | +0.1151 | -0.2997 | D_therapeutic > C_absurd > B_duty |
| no_religion | nanochat_pre1913_615m | +0.2596 | +0.1042 | A_providence > F_softened > B_duty |
| no_religion | hf_gpt2 | +0.1130 | -0.3017 | D_therapeutic > C_absurd > B_duty |
| paraphrase | nanochat_pre1913_615m | +0.3166 | +0.1572 | A_providence > F_softened > C_absurd |
| paraphrase | hf_gpt2 | +0.0693 | -0.2736 | D_therapeutic > C_absurd > A_providence |
| bridge | nanochat_pre1913_615m | +0.0674 | +0.1736 | A_providence > E_antiprogress > B_duty |
| bridge | hf_gpt2 | -0.1459 | -0.1719 | D_therapeutic > E_antiprogress > C_absurd |
| promise | nanochat_pre1913_615m | -0.1321 | -0.1280 | C_absurd > F_softened > D_therapeutic |
| promise | hf_gpt2 | -0.1296 | -0.2008 | D_therapeutic > C_absurd > E_antiprogress |

**Robustness across the 5 Family-F variants (incl. falsifiers):**
- `nanochat_pre1913_615m`: pre>post in 4/5 variants; pre>modern in 4/5.
- `hf_gpt2`: pre>post in 3/5 variants; pre>modern in 0/5.

### The must-see probe, full ordering

> *The son died before his father, and the family sat through the winter with an empty chair. The meaning of such suffering was*

- **Pre-1913 nanochat 615M (governed_v4 d22)**: A_providence > F_softened > B_duty > C_absurd > D_therapeutic > E_antiprogress
- **Modern base LM (gpt2)**: D_therapeutic > C_absurd > B_duty > A_providence > F_softened > E_antiprogress

## Family B — minimal-pair posture contrasts

Each value = per-byte log-prob of the pre-posture candidate minus the post/modern contrast candidate. Pre-WWI predicted > 0.

| probe | contrast | nanochat_pre1913_615m | hf_gpt2 |
|---|---|---|---|
| progress | A_ascent - C_idol | +0.5194 | +0.3375 |
| suffering | A_endurance - C_recovery | +0.0175 | -0.2274 |
| authority | A_inheritance - C_force | +0.2647 | +0.2799 |
| death | A_sacrifice - B_waste | +0.1473 | -0.0657 |
| nation | A_inheritance - C_idol | +0.6503 | +0.3463 |
| machine | A_servant - C_devourer | +0.1766 | +0.1245 |

## Family A — free-generation texture (illustrative, not scored)

**'The proper response to suffering is'**
- `nanochat_pre1913_615m`: 'the first step to comfort. If it were not for this we might soon become in dependent of our friends, and their charity cease. The want of sym pathy, which seems to be the curse of all'
- `hf_gpt2`: 'to seek the help of others. For example, people who have been injured or have lost their lives in fighting in the past may be more likely to seek help if they know how to seek it.  If you have been injured in fighting,'

**'The future of Europe is'**
- `nanochat_pre1913_615m`: 'in the hands of the United States. Great Britain will not be able to take the initiative. France and Germany have now the advantage by the way of the Gulf of Mexico. The latter has established an international system'
- `hf_gpt2`: "in jeopardy as German Chancellor Angela Merkel's government seeks to create a national security council by 2018.  The German parliament passed a resolution late on Wednesday that warned that EU membership is at risk and warned that security"

**"A man's duty is"**
- `nanochat_pre1913_615m`: "to be faithful to his trust: a woman's duty is to be faithful to her husband. If the latter's trust was faithfully and wisely maintained a woman's duties would be per- formed in better"
- `hf_gpt2`: "to be the best you can be. He is a thief.  The good news is that you don't have to be a good thief to be a good thief with your money.  It is easy to be a thief."

**'The machine has given men'**
- `nanochat_pre1913_615m`: 'better wages on the road than the employers can make them. 1t might as well be considered the man in the iron mills, and not the man who has to keep the men employed, that the machine'
- `hf_gpt2`: "an extra edge in the fight against rape and murder.  A number of attacks in the past few years have been blamed on a combination of a woman's own incompetence and a man's own desire to avoid being raped.  The BBC's"

**'The hope of mankind'**
- `nanochat_pre1913_615m`: 'seems to rest upon the fact that in a single year the number of persons who are suffering from the various distressing ailments which come TO us from over-work and worry has been reduced from 10000 to 1'
- `hf_gpt2`: 'as we know it has arrived. But is it? Or is it possible to see the possibility of a truly human being with all of the potentialities of a human being? Is humanity still a species of animal that could be fully understood by future generation'

**'Authority comes from'**
- `nanochat_pre1913_615m`: 'Wash ington to the effect that the president has been appealed to by the press of the United States to call an extra session of congress to provide for the payment of the principal of the national debt on the'
- `hf_gpt2`: 'the people who gave it their all to get this book published in the first place, but even that cannot be said for this book. If you like my work, please consider making a donation, and please consider supporting me on Patreon!<|endoftext|>'

## Caveats (held from probe-design.md)

- **Two-point, not three-anchor.** The headline characterology result needs Talkie-1930 (post-WWI) as the third anchor; gpt2 is a stand-in modern base, different scale/architecture/tokenizer. Comparisons here are ordinal only.
- **Scale + tokenizer confounds** are real (615M vs 124M, different BPE). Per-byte normalization + ordinal comparison mitigate but don't eliminate them.
- **gpt2 is web-text 2019**, not a matched-pipeline modern nanochat; treat as a rough modern-text anchor, and note it is not instruction-tuned (good — base posture).
- A single family is not a finding; the design requires **cross-family convergence**. Family A/C/D/E/G/H are not yet run.
