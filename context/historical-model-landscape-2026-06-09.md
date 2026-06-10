# Historical / Vintage LM Landscape + Interwar-Model Assessment

**Captured 2026-06-09. Sources: GPT-5.5 Pro deep investigation (req_1781050312424) + Fable-5-session research. For the pre-1914-vs-interwar-vs-modern decision.**

## Bottom line (GPT Pro, concurring with the project's own read)
**No public, clean, English, autoregressive model trained only on 1918–1930 exists.** A true interwar *window* model would be genuinely novel. BUT: **do not make it a co-equal expenditure yet.** The strongest paper is a rigorous **pre-1914 vs modern** characterology study, scoped to "old print-world habitus vs modern model/web/alignment habitus" — NOT "the Great War caused X." Build the pre-1914 model well + a clean modern control + strong evaluation. Treat 1918–1930 as a Phase-2 anchor or a small pilot (350M–615M on 1–3B tokens) only if cheap.

## The landscape (every public model is a CUMULATIVE cutoff unless noted)

| Model | Maker | Params | Cutoff/window | Corpus | Avail | Type |
|---|---|---|---|---|---|---|
| talkie-1930-13b base/it | Radford/Levine/Duvenaud, Apr 2026 | 13B | ≤1930 | 260B tok; books/news/periodicals/journals/patents/law | HF `talkie-lm/*`, Apache-2 | cumulative; largest |
| talkie-web-13b-base | same | 13B | modern | 260B FineWeb, matched arch/FLOPs | HF | modern twin/control |
| TypewriterLM | Luo et al., arXiv 2606.02991 | 7.24B | ≤1913 | TypewriterCorpus 54B tok, **97.68% Institutional Books, concentrated 1800–1900** | `hf.co/typewriter-ai` | cumulative; most relevant to pre-1914 concern |
| **Ranke-4B** | Göttlich/Voth et al. (Zurich/Cologne) | 4B | **1913/1929/1933/1939/1946** | 600B timestamped tok; 4B trained on 80B | forthcoming | cumulative cutoff *family* — the closest peer; has 1929/1933/1939 anchors |
| GPT-1900 | Michael Hla / Machina Mirabilis | 3.29B | ≤1900 | ~22B tok; HathiTrust/IA/BL/US newspapers | HF `mhla/gpt1900-d34-22btok`, MIT | cumulative |
| GPT-1905 | same | 3.29B | ≤1905 | ~40B tok; Institutional Books + American Stories | HF `mhla/gpt1905-d34`, MIT | cumulative; near-Edwardian |
| GPT-1964 | same | 3.29B | **1900–1964** | Inst. Books 1900–22 + American Stories 1900–64 | HF `mhla/gpt1964-d34`, MIT | true broad window but "mostly useless"/newspaper-overtrained |
| Mr. Chatterbox | Trip Venturella | 340M | **1837–1899** | 28K BL books, ~2.93B tok | HF, MIT | **true window** (rights-clean, small/early) |
| TimeCapsuleLLM v2 | Hayk Grigorian | 1.22B | **London 1800–1875** | ~112GB, 136K docs | HF, MIT | **true place+period window** |
| Violet | zakarth | 1.41B | 1800–1899 | Gutenberg/IA/BL | HF, CC0 | window-ish but chat/SFT-contaminated |
| helloLondon | Amit Bahree | 117M/354M | **London 1500–1850** | ~125M tok | GitHub/HF | true window, educational scale |
| MonadGPT | P-C Langlais | 7B | 17th-c flavor | Mistral fine-tune on EEBO/Gallica | HF | **NOT from scratch — modern-base contamination** |
| OCRonos-Vintage | PleIAs | 124M | ≤1955 | 18B tok LoC/IA/HathiTrust | HF | OCR-correction specialist, not general |
| **yours** | — | 615M | ≤1913 | governed pre-1913 | local | cumulative |

**Adjacent (encoder/point-in-time, methodologically relevant, not directly comparable):** histLM / Living-with-Machines BERTs (1760–1900 with TRUE slices 1850–75, 1875–90, 1890–1900 — but encoder-only); StoriesLM/v2 (1900–1963 newspaper encoders, 100B tok); ChronoBERT/ChronoGPT, DatedGPT, NBER point-in-time (modern annual cutoffs); Owain Evans "vintage LLM" concept (no weights).

## The methodological point (confirmed)
Cumulative cutoffs are pulled by earlier mass. TypewriterLM is the direct evidence: 54B tokens, **97.68% Institutional Books, concentrated 1800–1900** — a "≤1913" model is really a late-19th-c book-register model. Reweighting by decade/source helps (Ranke does bucket allocation; StoriesLM uses no-replay yearly updates) but **cumulative cutoff ≠ period window**. A true 1918–1930 model is different in kind: it removes the prewar mass rather than appending postwar text.

## What the interwar window would add (the 4-hypothesis table)
pre-1914-vs-modern shows old≠modern but not *when*. pre-1914 + 1918–1930 + modern distinguishes: long-drift (interwar in between) / Great-War rupture (interwar already shifted from pre-1914) / modern-web-artifact (interwar resembles pre-1914, modern is outlier) / source-artifact. It converts a binary contrast into a trajectory.

## Why pre-1914-vs-modern is publishable alone (and the reviewer attacks to pre-empt)
Coherent claim: "a rights-audited pre-1914 corpus yields systematically different neutral-prompt completions from a matched modern control, on posture not knowledge." Reviewers can't demand an interwar model unless you make WWI causal claims. They WILL attack: books-vs-web? instruction-tuning? archaic-diction-vs-posture? pre-1914-mostly-19th-c? size/tokenizer/tokens/decoding confounds? **These are answered by a rigorous pre-1914 setup + clean controls, not by a weak interwar anchor.**

## Copyright/periodization caveat on 1918–1930
It's "partly a copyright boundary, not a historical one" — omits 1914–18 wartime text, stops before the Depression/fascism years many historians weight heavily. Cleaner historiographic windows: 1900–13 / 1914–18 / 1919–29 / 1929–39. 1918–1930 is the best *legally clean* interwar-ish window (US PD reaches 1930; 1931 opens Jan 1 2027) but not the perfect break.

## Corpus feasibility for 1918–1930
1–3B clean tokens straightforward; 3–10B realistic if newspapers dominate. **Dominant source: American Stories / Chronicling America** (438M structured articles from ~20M scans, pre-1920 concentrated). Complements: Institutional Books/IA (better prose, less volume), periodicals, CAP caselaw (6.5M decisions), patents, government text. For habitus, build a *balanced* corpus (news/books/periodicals/science/law/religion/education) and report natural-volume vs balanced.

## GPT Pro's near-term plan (= the project's path)
1. Make the pre-1914 model excellent — balance by decade/source/OCR/dedup, document provenance.
2. Build a modern twin control — same arch/tokenizer/FLOPs, **base-vs-base** (not base-historical-vs-RLHF-modern).
3. Build the habitus benchmark first — neutral prompts (suffering/duty/authority/progress/closure/sacrifice/...), blinded human ratings + embedding/lexical controls, factual prompts kept separate.
4. Assemble but DON'T train the interwar corpus — metadata/token-counts/buckets/OCR/dedup/rights — preserve the option.
5. Small interwar pilot (350M–615M, 1–3B tok) only if cheap — "is there a stable signal worth scaling?"

**Final judgment (GPT Pro): the 1918–1930 model is an interesting experiment but not essential — essential only if the thesis becomes a timing claim about the Great War.**
