# Three-Anchor Characterology Pilot — Findings (2026-06-09)

**Curated interpretation of the three-anchor run. Author: Claude Fable 5.**
Anchors: **Pre-1913 nanochat 615M** (governed_v4 d22) · **Talkie-1930 13B** (pre-1931 corpus, post-WWI; `dtestnyrr/...gptq-int4` model + xlr8harder's correct `TalkieTokenizer`, bf16/Triton) · **gpt2 124M** (modern web-text base). Length-normalized log-prob per byte; within-model orderings compared ordinally (probe-design.md).

> **Loader note (and a correction).** An earlier pass wrongly concluded "every talkie conversion is broken." That was over-generalized from a single bad apple: `dtestnyrr`'s GPTQ repo bundles a *wrong* 262,144-token tokenizer against the correct 65,536-vocab model, so ordinary text encoded to out-of-range ids → CUDA asserts. The model is fine. Pairing it with xlr8harder's correct `TalkieTokenizer` (65,536; max probe id 27,359, in range) + loading in **bf16** (fp16 overflowed to NaN at 13B) yields correct logprobs on the 24GB 3090. GPT-5.5-Pro independently confirmed the reconciliation ("the official code uses a 65,536 model vocab by filtering the 262,144-line BPE file").

## Headline (CORRECTED 2026-06-09): a real pre-modern/modern split — but Talkie-1930 does NOT test the 1914 fracture

**Retraction of an earlier overreach.** An earlier version of this doc concluded "the fracture is post-1930, not 1914" because Pre-1913 and Talkie-1930 both lead with providence/duty while modern gpt2 leads with therapeutic. **That inference is unsupported and is withdrawn.** Talkie-1930 is trained on *all* public-domain English text up to 1930, and that corpus is almost certainly **dominated by pre-1914 (likely pre-1900) material by volume** — its dominant source class (Institutional Books / digitized public-domain books) is, per the parallel corpus study arXiv:2606.02991, "concentrated between 1800 and 1900" (Institutional Books = 97.7% of that corpus). Talkie does not publish its temporal distribution. So Talkie-1930 is **not a post-WWI anchor** — it is a pre-1931 *mixture* whose averaged posture is pulled by 19th-century mass. Its patterning with the Pre-1913 model is exactly what shared pre-war corpus mass predicts and says **nothing** about whether a 1914 rupture exists.

**What the data DOES support:** a large, robust, falsifier-surviving **modern-vs-old closure dissociation** on the affective axes (suffering, death) — modern gpt2 reaches for therapeutic processing where both historical corpora reach for providence/duty. The instrument works. **What it does NOT support:** locating *when* the shift happened. That requires anchors that isolate a period by volume, which neither Talkie-1930 (pre-1914-dominated mixture) nor a 615M-vs-124M scale-mismatched pair provides.

**The right experiment (per the user):** isolate the interwar era with a *clean-cutoff* corpus — a model trained only on ~1918–1930 text (the post-WWI window currently in US public domain; 1931–1939 is not yet PD) — compared to the existing pre-1914 model and a modern reference, ideally all from the *same* data pipeline so only the date filter varies. See the program note below.

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
The instrument works and the modern-vs-old closure dissociation is real and robust. But **off-the-shelf anchors cannot locate the fracture date** — Talkie-1930 is a pre-1914-dominated mixture, and a 615M-vs-124M pair is scale-mismatched. To answer "when did the closure posture shift," the project must build **clean-cutoff anchors from its own pipeline**, varying only the publication-date filter:

- **Pre-1914** (have it: the governed 615M, or a fresh d24 cloud run)
- **Interwar 1918–1930** (NEW — the post-WWI window in current US public domain; this is the anchor that actually isolates the Great War's effect)
- **Modern** (a same-pipeline FineWeb run, or as a rough check, gpt2/talkie-web)

This is strictly better than Talkie-1930 + talkie-web, because same-pipeline cutoff variants hold *subject matter, register, OCR, and tokenizer* fixed — talkie's own authors warn that talkie-1930-vs-talkie-web differ in "distribution of subject matters," not just era. A pre-1914 vs 1918–1930 contrast from one pipeline is the clean test the 1914 hypothesis actually needs.

**Copyright constraint:** the interwar window is currently limited to ~1918–1930 (works enter US public domain at 95 years; 1931–1939 is not yet free). That still cleanly isolates post-WWI (Versailles, Weimar, the 1920s, early Depression) from the pre-war world. A true pre-1939 cutoff must wait, or use non-US/openly-licensed interwar sources.

## Next steps
1. **Empirically estimate Talkie-1930's temporal weighting** (optional, low compute): measure its bits-per-byte on register-matched dated text samples (e.g., 1880s vs 1925) to put a number on the pre-1914 dilution. Confounded by register, so corpus-composition reasoning remains primary.
2. **Scope the interwar (1918–1930) corpus build** from the existing `data/download/` pipeline (Chronicling America newspapers are dense through ~1922; Institutional Books / Internet Archive for 1918–1930 books; periodicals). This is the real next experiment.
3. The talkie-web / bf16-confirmation / Families-C-D-E items remain useful for hardening the *modern-vs-old* result, but they do not address the *fracture-date* question — only clean-cutoff interwar anchors do.
