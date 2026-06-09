# Three-Anchor Characterology Pilot — Findings (2026-06-09)

**Curated interpretation of the three-anchor run. Author: Claude Fable 5.**
Anchors: **Pre-1913 nanochat 615M** (governed_v4 d22) · **Talkie-1930 13B** (pre-1931 corpus, post-WWI; `dtestnyrr/...gptq-int4` model + xlr8harder's correct `TalkieTokenizer`, bf16/Triton) · **gpt2 124M** (modern web-text base). Length-normalized log-prob per byte; within-model orderings compared ordinally (probe-design.md).

> **Loader note (and a correction).** An earlier pass wrongly concluded "every talkie conversion is broken." That was over-generalized from a single bad apple: `dtestnyrr`'s GPTQ repo bundles a *wrong* 262,144-token tokenizer against the correct 65,536-vocab model, so ordinary text encoded to out-of-range ids → CUDA asserts. The model is fine. Pairing it with xlr8harder's correct `TalkieTokenizer` (65,536; max probe id 27,359, in range) + loading in **bf16** (fp16 overflowed to NaN at 13B) yields correct logprobs on the 24GB 3090. GPT-5.5-Pro independently confirmed the reconciliation ("the official code uses a 65,536 model vocab by filtering the 262,144-line BPE file").

## Headline: the fracture is NOT at 1914

The design predicted a **three-way** split — Pre-WWI (providence/duty) vs Talkie-1930 (anti-closure) vs Modern (therapeutic). The data shows a **two-way** split: **Pre-1913 ≈ Talkie-1930** (both providence/duty) **vs Modern** (therapeutic). Talkie-1930, trained on text through 1930 — *including WWI and the 1920s* — retains the traditional closure posture. **The providence→therapeutic shift is post-1930, not a consequence of the Great War.** The instrument detects a large, robust closure dissociation; it just isn't where the 1914 hypothesis put it. This is one of the design's pre-registered outcomes (the specific fracture not separable at this cutoff), with a positive twist: a *pre-modern vs modern* signature is clearly present.

## The must-see probe, three orderings
> *The son died before his father… the meaning of such suffering was*

| anchor | ordering |
|---|---|
| Pre-1913 615M | **A_providence** > F_softened > B_duty > C > D > E |
| Talkie-1930 13B | **A_providence** > B_duty > C > D > F > E |
| Modern gpt2 | **D_therapeutic** > C_absurd > B_duty > A_providence(4th) > F > E |

Both historical models lead with providence; the modern model leads with therapeutic processing and sinks providence to 4th of 6.

## Family F robustness (pre>modern = prefers providence/duty over therapeutic)
- Pre-1913 615M: **4/5** variants
- Talkie-1930 13B: **4/5** variants
- Modern gpt2: **0/5** variants

Both historical models clear the bar in 4/5; both break on the same variant (`promise`, a *betrayal* framing — a consistent, honest boundary of the effect). The modern model never clears it.

## Family B corroborates on the affective axes
| contrast | Pre-1913 | Talkie-1930 | Modern gpt2 |
|---|---|---|---|
| suffering (endurance − recovery) | +0.018 | **+0.127** | **−0.227** |
| death (sacrifice − waste) | +0.147 | **+0.241** | **−0.066** |
| progress (ascent − idol) | +0.519 | +0.363 | +0.338 |
| nation (inheritance − idol) | +0.650 | +0.690 | +0.346 |
| machine (servant − devourer) | +0.177 | +0.138 | +0.124 |
| authority (inheritance − force) | +0.265 | +0.337 | +0.280 |

The **suffering and death axes dissociate in sign** — both historical models positive, the modern model negative — exactly where the design said the signal would be sharpest. On the civilizational-confidence axes (progress/nation/machine) all three lean traditional; the historical pair leans harder.

## Why the scale confound is now strongly defused
The design's #1 worry: a bigger model looks "more postwar/darker," manufacturing a fake era effect. Here the anchors span **124M → 615M → 13B**, and the clustering is **by era, not size**: Talkie-1930 (**13B**) and Pre-1913 (**615M**) pattern together across a **~20× scale gap**, while gpt2 (124M) is the outlier. If scale drove posture, the 13B and 615M models would not agree. They do. The corpus/era explanation survives; the scale explanation does not fit the data.

## Caveats (held)
- **Talkie-1930 is int4-quantized.** Quantization perturbs logits slightly; the effects here are large and ordinal, so robust, but a bf16 confirmation run is worth doing.
- **The modern anchor is small (gpt2 124M).** The design's preferred control is the **matched `talkie-web-13b-base`** (same architecture/FLOPs as Talkie-1930, FineWeb corpus). That gives a clean 13B-vs-13B (1930 vs modern) contrast isolating corpus from scale entirely — the single highest-value next addition (now feasible: same loader, ~26GB bf16 + bitsandbytes 4-bit, or a correct quant).
- **Single-turn, two families.** Cross-family convergence is partial (F + B). Families C/D/E/G/H pending.
- **`promise` boundary** (betrayal vs death) is consistent across both historical models — a real limit, not noise.

## What this means for the program
The instrument works, the signal is real and robust, and it **refines the central hypothesis**: not a 1914 rupture but a pre-modern/modern one, detectable at 615M and confirmed at 13B. For the cloud program this is decisive de-risking — and it suggests the most interesting next experiment may be a **finer-grained cutoff sweep** (1913 / 1930 / pre-WWII / modern) to locate *when* the therapeutic turn actually registers, rather than assuming 1914.

## Next steps (zero/low compute)
1. **Add matched `talkie-web-13b-base`** (modern 13B) — perfect scale control; turns the strong-but-unmatched modern anchor into a clean matched one.
2. **bf16 confirmation** of the Talkie-1930 result (rule out quantization artifacts on the headline).
3. **Families C/D/E** for full cross-family convergence.
4. Reframe the writeup around *pre-modern vs modern closure*, with the 1914-null as a genuine (and interesting) result.
