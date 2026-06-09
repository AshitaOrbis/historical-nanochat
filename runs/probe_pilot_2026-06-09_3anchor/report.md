# Computational Historical Characterology — v1 615M Pilot (2026-06-09)

Pre-1913 nanochat 615M vs a modern base anchor (gpt2), length-normalized log-prob per byte. Per probe-design.md: compare WITHIN-model preference orderings, then orderings across models. One small modern anchor + one pre-1913 model — a **two-point pilot**, not the three-anchor result.

## Family F — closure / tragic emplotment (the core family)

Predicted: a pre-1913 habitus prefers the **pre** cluster (providence/duty) over **post** (absurd/anti-progress) and over **modern** (therapeutic). `pre_minus_post>0` and `pre_minus_modern>0` are the pre-WWI signature; a modern model should show `pre_minus_modern<0`.

| variant | model | pre−post | pre−modern | top-3 order |
|---|---|---|---|---|
| must_see | nanochat_pre1913_615m | +0.2951 | +0.1397 | A_providence > F_softened > B_duty |
| must_see | gptq_talkie_1930_13b_base_gptq_int4 | +0.3135 | +0.1274 | A_providence > B_duty > C_absurd |
| must_see | hf_gpt2 | +0.1151 | -0.2997 | D_therapeutic > C_absurd > B_duty |
| no_religion | nanochat_pre1913_615m | +0.2596 | +0.1042 | A_providence > F_softened > B_duty |
| no_religion | gptq_talkie_1930_13b_base_gptq_int4 | +0.2164 | +0.0304 | B_duty > C_absurd > A_providence |
| no_religion | hf_gpt2 | +0.1130 | -0.3017 | D_therapeutic > C_absurd > B_duty |
| paraphrase | nanochat_pre1913_615m | +0.3166 | +0.1572 | A_providence > F_softened > C_absurd |
| paraphrase | gptq_talkie_1930_13b_base_gptq_int4 | +0.3597 | +0.2019 | A_providence > B_duty > F_softened |
| paraphrase | hf_gpt2 | +0.0693 | -0.2736 | D_therapeutic > C_absurd > A_providence |
| bridge | nanochat_pre1913_615m | +0.0674 | +0.1736 | A_providence > E_antiprogress > B_duty |
| bridge | gptq_talkie_1930_13b_base_gptq_int4 | +0.0959 | +0.1679 | A_providence > B_duty > E_antiprogress |
| bridge | hf_gpt2 | -0.1459 | -0.1719 | D_therapeutic > E_antiprogress > C_absurd |
| promise | nanochat_pre1913_615m | -0.1321 | -0.1280 | C_absurd > F_softened > D_therapeutic |
| promise | gptq_talkie_1930_13b_base_gptq_int4 | -0.0397 | -0.0132 | C_absurd > A_providence > F_softened |
| promise | hf_gpt2 | -0.1296 | -0.2008 | D_therapeutic > C_absurd > E_antiprogress |

**Robustness across the 5 Family-F variants (incl. falsifiers):**
- `nanochat_pre1913_615m`: pre>post in 4/5 variants; pre>modern in 4/5.
- `gptq_talkie_1930_13b_base_gptq_int4`: pre>post in 4/5 variants; pre>modern in 4/5.
- `hf_gpt2`: pre>post in 3/5 variants; pre>modern in 0/5.

### The must-see probe, full ordering

> *The son died before his father, and the family sat through the winter with an empty chair. The meaning of such suffering was*

- **Pre-1913 nanochat 615M (governed_v4 d22)**: A_providence > F_softened > B_duty > C_absurd > D_therapeutic > E_antiprogress
- **Talkie-1930 13B (pre-1931 corpus, post-WWI anchor)**: A_providence > B_duty > C_absurd > D_therapeutic > F_softened > E_antiprogress
- **Modern base LM (gpt2)**: D_therapeutic > C_absurd > B_duty > A_providence > F_softened > E_antiprogress

## Family B — minimal-pair posture contrasts

Each value = per-byte log-prob of the pre-posture candidate minus the post/modern contrast candidate. Pre-WWI predicted > 0.

| probe | contrast | nanochat_pre1913_615m | gptq_talkie_1930_13b_base_gptq_int4 | hf_gpt2 |
|---|---|---|---|---|
| progress | A_ascent - C_idol | +0.5194 | +0.3631 | +0.3375 |
| suffering | A_endurance - C_recovery | +0.0175 | +0.1265 | -0.2274 |
| authority | A_inheritance - C_force | +0.2647 | +0.3371 | +0.2799 |
| death | A_sacrifice - B_waste | +0.1473 | +0.2409 | -0.0657 |
| nation | A_inheritance - C_idol | +0.6503 | +0.6901 | +0.3463 |
| machine | A_servant - C_devourer | +0.1766 | +0.1381 | +0.1245 |

## Family A — free-generation texture (illustrative, not scored)

**'The proper response to suffering is'**
- `nanochat_pre1913_615m`: 'in the very nature of things a question of some moment. And it has a very con- spicuous place in the present situation, inasmuch as many sufferers by the strike have had to live as best they could.'
- `hf_gpt2`: 'self-preservation. We need to remember that the only way out of this tragic situation is to help as many people as possible. The time has come to act responsibly and to stop suffering. When we are not dealing with painful situations, we are'

**'The future of Europe is'**
- `nanochat_pre1913_615m`: 'at stake. The struggle is to be waged upon the great question of the future of the French in dustrieS. The conflict is to be waged up on the question of the future of the South in the'
- `hf_gpt2`: "now looking bleak for European Union members.  In a new report, the Commission today released a plan to make it easier in 2019 to join the EU as a single country.  The EU's biggest single market has been hit by economic woes"

**"A man's duty is"**
- `nanochat_pre1913_615m`: 'not only to keep clean his own boots, but to provide a clean pair for others to wear at the proper season.<|bos|>The time to build bridges is when building them is going on. When once they are built, you'
- `hf_gpt2`: 'to follow orders from a policeman.  This includes the fact that law enforcement officers are often more than a little surprised when a man is arrested or charged in connection with a crime.  However, there have been some cases where officer'

**'The machine has given men'**
- `nanochat_pre1913_615m`: 'the highest results and we are proud to acknowledge the fact. The same high standard should be maintained with the workers at home.<|bos|>The city officials of Spokane are said to be in favor of the proposed'
- `hf_gpt2`: "the power to rape their female companions and carry out rape against them.  The women in the picture are identified only as Sabine, Naeem and Neeel.  The men's number is also unknown.  In his"

**'The hope of mankind'**
- `nanochat_pre1913_615m`: '. in the midst of the storm, is in the protection of an overruling Providence.<|bos|>The following is said to be the latest speculation in Boston papers: "The re. ports of the death of Mrs. Mary f'
- `hf_gpt2`: 'has been that all the people of the earth are of one mind; not that they are all of one mind; but that all the people of the earth have one mind, and all the people of the earth have one mind. And since for their'

**'Authority comes from'**
- `nanochat_pre1913_615m`: 'Washington that Congress will take the initiative in the fight on the Philippines.<|bos|>In order to put to the test the strength OF the republican party on the Philippine tariff question, the Chicago convention of the repub'
- `hf_gpt2`: 'the first name of the sovereign power. It is held by an individual or group of people. In this way, it is not the authority that governs the nation or country but the sovereign power.  It is an authority that gives them the power'

## Caveats (held from probe-design.md)

- **Two-point, not three-anchor.** The headline characterology result needs Talkie-1930 (post-WWI) as the third anchor; gpt2 is a stand-in modern base, different scale/architecture/tokenizer. Comparisons here are ordinal only.
- **Scale + tokenizer confounds** are real (615M vs 124M, different BPE). Per-byte normalization + ordinal comparison mitigate but don't eliminate them.
- **gpt2 is web-text 2019**, not a matched-pipeline modern nanochat; treat as a rough modern-text anchor, and note it is not instruction-tuned (good — base posture).
- A single family is not a finding; the design requires **cross-family convergence**. Family A/C/D/E/G/H are not yet run.
