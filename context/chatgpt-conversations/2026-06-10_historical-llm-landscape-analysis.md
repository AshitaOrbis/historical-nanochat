# Historical LLM Landscape Analysis

- Source: https://chatgpt.com/c/6a28abc9-8a58-83e8-be5d-48cc09c858e0
- Conversation ID: 6a28abc9-8a58-83e8-be5d-48cc09c858e0
- Created: 2026-06-10T00:11:53.986Z
- Archived: false
- Messages: 2
- Captured: 2026-06-09 (Fable-5 session, CDP backend-api content-search)

---

## USER

Research task: survey the landscape of "vintage" / historical / time-locked language models and assess the experimental value of a 1918–1930 interwar "window" model.

CONTEXT: I run a research project ("historical-nanochat") training time-locked LLMs from scratch on pre-cutoff historical text, to probe "model-inherited habitus" — the inherited posture/character of a historical text-world (how it completes neutral prompts about suffering, duty, authority, progress, closure), NOT factual knowledge. I have a pre-1913 model (615M params, trained on rights-audited pre-1913 English text). My primary interest is a GOOD pre-1914 model, and I believe the most interesting contrast is pre-1914 vs MODERN. A 1918–1930 interwar model is a possible additional anchor but I'm unsure it's worth the cost (extra corpus storage, an extra training run).

PLEASE INVESTIGATE AND REPORT:

1. COMPLETE LANDSCAPE OF EXISTING HISTORICAL/VINTAGE LLMs. Enumerate every one you can find with: name, params, training cutoff, corpus (sources + size in tokens), release date, who made it, availability (open weights? HF repo?), and whether it's a single cumulative cutoff (trained on everything ≤ year X) or a window model. I know of: talkie-1930-13b (talkie-lm; ≤1930; 260B tokens; HF), talkie-web-13b (FineWeb modern twin), GPT-1900 (3.3B, ≤1900), Mr. Chatterbox (340M, ≤1899), TimeCapsuleLLM (1.2B, ≤1875), TypewriterLM / "Pretraining Language Models on Historical Text" arXiv:2606.02991 (≤1913, ~54B-token "TypewriterCorpus", 97.7% Institutional Books, concentrated 1800–1900), Ranke-4B (history-llms, Zurich), and Owain Evans's "vintage LLMs" concept work. Find ALL of them, correct my details, and add any I'm missing. Note especially anything with a cutoff between 1914 and 1939, or any genuine WINDOW (period-isolated, not cumulative) model.

2. THE KEY METHODOLOGICAL POINT: every cumulative cutoff model (everything ≤ year X) is dominated by 19th-century volume (digitized public-domain text concentrates 1800–1900), so its averaged "posture" is pulled by pre-war mass and cannot isolate a later period. Confirm or challenge this. Is anyone aware of it? Has anyone built a true period-WINDOW model (e.g., trained only on 1918–1930)?

3. VALUE OF A 1918–1930 INTERWAR WINDOW MODEL — compared to, and in conjunction with, a pre-1914 model:
   - What does the interwar window add that pre-1914-vs-modern alone cannot show? (e.g., does it let you locate WHEN a posture shift happened vs merely that old≠modern?)
   - Is "pre-1914 vs modern" already a complete, interesting, publishable result on its own for characterology, or does it leave an obvious gap a reviewer would attack?
   - Counterargument: is the interwar model "interesting experiment but not essential," as I suspect? Steelman both sides.
   - Historiographically: is 1918–1930 the right window to isolate the Great War's cultural rupture (Eksteins's Rites of Spring, Hobsbawm's long-19th-century-ends-1914 thesis)? Or is the interesting break elsewhere?

4. PRACTICAL: a clean US-public-domain interwar corpus is limited to ~1918–1930 (1931–1939 not yet PD under the 95-year rule). Roughly how many tokens of clean 1918–1930 English text are realistically sourceable (Chronicling America newspapers, Institutional Books / Internet Archive, periodicals, Caselaw Access Project, patents)? Enough for a ~1–10B-token training run? What's the dominant available source for that specific window?

5. RECOMMENDATION: given my stated priority (a good pre-1914 model; pre-1914 vs modern as the core contrast), is building the 1918–1930 model worth a meaningful extra data-storage + compute cost, or should I focus resources on making the pre-1914 model excellent and the pre-1914-vs-modern contrast rigorous? Give a clear recommendation with reasoning.

Be concrete, cite sources/URLs/HF repos, correct my facts, and prioritize completeness on the model landscape (Q1) and a decisive recommendation (Q5).

---

## ASSISTANT

I’ll verify the named models against primary artifacts, search for adjacent vintage/time-locked projects, distinguish cumulative cutoffs from true window models, and flag missing public details.

---
