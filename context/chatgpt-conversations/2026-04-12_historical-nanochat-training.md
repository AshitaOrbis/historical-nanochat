# Historical Nanochat Training

- Source: https://chatgpt.com/c/69dc25a0-7f50-83e8-b46f-9be4018ccbb4
- Conversation ID: 69dc25a0-7f50-83e8-b46f-9be4018ccbb4
- Created: 2026-04-12T23:07:12.776Z
- Archived: known
- Messages: 5
- Captured: 2026-06-09 (Fable-5 session, CDP backend-api content-search)

---

## USER

I'm building a project called "Historical Nanochat" — training small language models from scratch on exclusively pre-1913 historical texts, so the resulting models genuinely don't know about anything after their cutoff date. I'd like your thorough analysis of the training pipeline and suggestions for improvement.

## Project Goal
Train "time-locked" LLMs using Karpathy's nanochat framework on historical corpora. The models should authentically simulate a pre-1913 worldview — not through fine-tuning a modern model, but by training from scratch on era-appropriate text only.

## Current Architecture

### Model
- **Framework**: Karpathy's nanochat (GPT-2 style, modernized)
- **Architecture features**: RoPE, QK-norm, untied embeddings, ReLU² MLP, parameterless RMSNorm, no bias, GQA support
- **Current config**: 125M params (d12 — 12 layers, 768 dim, ~6 heads)
- **Tokenizer**: Custom 32K vocab BPE trained on the historical corpus itself
- **Optimizer**: Muon for linear layers, AdamW for embeddings/lm_head
- **Sequence length**: 2048
- **Total batch size**: 262,144 tokens (device_batch_size=2, grad_accum=64 on single GPU)
- **LR schedule**: flat with 40% warmdown, sqrt batch-size scaling
- **Training horizon**: 15,250 iterations (target_param_data_ratio=8)

### Hardware
- Single RTX 3090 (24GB VRAM), native Linux (ext4)
- device_batch_size=2 to stay at ~32% VRAM (batch_size=4 caused OOM stalls at 98%)
- torch.compile disabled (was failing on prior WSL setup, may work on native Linux now)
- ~36K tok/sec, ~7.1s per step
- Full training run: ~30 hours for d12

### Data Pipeline
10 historical text sources, ~16B tokens total after deduplication and sampling:

| Source | Raw Size | Description |
|--------|----------|-------------|
| American Stories | 257G | OCR'd historical newspapers (LOC) |
| BHL (Biodiversity Heritage Library) | 118G | Scientific/natural history texts |
| British Library Newspapers | 12G | UK historical newspapers |
| EEBO (Early English Books Online) | 6.2G | Pre-1700 English texts |
| Gutenberg | 4.0G | Public domain books |
| TCP (Text Creation Partnership) | 1.0G | Early modern texts |
| Old Bailey | 432M | London criminal court proceedings |
| Caselaw | 80K | American legal cases |
| Chronicling America | 52K + 13M | LOC newspaper OCR |

Data pipeline:
1. **Download** from HuggingFace/public APIs (source-specific scripts)
2. **Deduplication**: Exact hash-based (MD5 of normalized text) + optional MinHash LSH (threshold 0.8, 128 perms)
3. **Contamination check**: Rule-based anachronism detection — definite terms (e.g. "atomic bomb", "nazi"), year references, URLs, modern annotations. Word-boundary matching with context-aware rules.
4. **Shard packaging**: Streaming packager with per-source sampling rates (American Stories 0.33, BHL 0.60, others 1.0), 250M chars per parquet shard, shuffle buffer of 50K docs, zstd compression
5. **Two shard sets exist**: "main" (322 shards, 36G) and "small" (3,469 shards, 65G) — unclear which was used for the last training run

### Training Results So Far
- Only one checkpoint survived (step 3000, val BPB 3.73) — lost during migration
- Scaling law experiments were planned for d8/d12/d16 but unclear if completed
- No CORE metric evaluations logged

## What I Want From You

Please provide a thorough analysis covering:

1. **Data quality & pipeline improvements** — Is the contamination detection approach sufficient? What about OCR noise in newspaper sources? Is the dedup strategy optimal? Are the sampling rates well-calibrated?

2. **Model architecture & training config** — Is d12/125M the right scale for 16B tokens? How does the 8:1 data:param ratio compare to Chinchilla-optimal? Should I use GQA at this scale? Any hyperparameter suggestions (LR, warmup, warmdown, batch size)?

3. **Tokenizer considerations** — Custom 32K BPE on historical text — is this the right call? Tradeoffs vs using a standard tokenizer? Impact of archaic vocabulary on tokenization efficiency?

4. **RTX 3090 optimization** — torch.compile on native Linux (worth retrying?), mixed precision strategy, any other VRAM or throughput optimizations?

5. **Evaluation strategy** — How should I evaluate a "time-locked" model? The standard CORE metric won't work for models that intentionally lack modern knowledge. What novel evaluation approaches would demonstrate the model's historical authenticity?

6. **Scaling recommendations** — Given 16B tokens and a single 3090, what's the optimal model size? Should I pursue larger models via cloud compute? What would a d20 or d26 run look like?

7. **Research value & novelty** — How does this compare to existing work (Ranke-4B from Zurich, Vintage LLMs concept)? What would make this publishable? What experiments would be most interesting?

8. **Anything I'm missing** — Blind spots, low-hanging fruit, anti-patterns in the current setup.

---

## ASSISTANT

I’m pressure-testing your setup across data hygiene, compute-optimal scaling, tokenizer and OCR choices, 3090-specific optimization, and the publishable research angle, then I’ll turn that into concrete recommendations.

---

## ASSISTANT

Early read: the main leverage is in data, not architecture—your 125M run is likely undertrained relative to 16B available tokens, and OCR-heavy newspaper mix quality is the biggest risk. Prior work also supports your core premise: genuine period pretraining appears more reliable for avoiding anachronism than prompting or light post-training on modern models.

---

## ASSISTANT

One important correction before I finalize the advice: your logged batch size and horizon imply about 4.0B training tokens, so the 125M run is roughly 32 tokens/parameter, not 8. That shifts the focus toward source-aware OCR filtering, stronger near/subsequence dedup, and better mixture weighting and cutoff evaluation rather than a bigger architectural change.

---

## ASSISTANT

Conceptually, this is a strong project. A recent study on historical simulation found that prompting modern models with period prose is not enough, and even fine-tuning modern models still leaves outputs that humans can distinguish from authentic historical text; the authors tentatively conclude that **pretraining on period prose may be required**. That is exactly the niche you’re targeting. The closest adjacent effort I found, **Ranke-4B**, is also training time-locked historical models from scratch at multiple cutoffs, which suggests your framing is research-grade rather than just a curiosity build. citeturn451059view3turn784148view5

My main diagnosis is that your **model design is mostly fine**, but your **corpus control and experiment hygiene** need to get much tighter. Historical corpora are especially vulnerable to low-quality OCR, hidden duplicates, and subtle temporal leakage. On top of that, the ambiguity around which shard set was used, plus the lost checkpoints and missing eval logs, is a bigger threat to the project than whether you use GQA or tweak warmdown by 10 points. citeturn784148view4turn402744view0turn451059view5

## 1) Data quality and pipeline improvements

Your current contamination detector is a good first pass, but it is **not sufficient as the main defense**. Blacklists catch obvious leaks like “Nazi” or URLs, but they miss subtler contamination: modern editor notes, OCR-inserted metadata, post-1913 introductions to older works, and near-duplicate later reprints that don’t trip keyword rules. Work on model cutoffs shows that “reported” temporal boundaries and “effective” ones often diverge, partly because deduplication misses semantically near-duplicate material. And work on ex-ante reasoning shows leakage is especially stubborn near the cutoff boundary. citeturn451059view5turn451059view6

I would move to a **three-layer temporal audit**:

First, keep your deterministic rules. They are cheap and worth having.

Second, add a **document- or window-level date estimator**. Temporal text classification is a real task, and current work shows text can be dated from diachronic linguistic signals. You do not need a giant model for this; even a lightweight classifier trained on clean, metadata-trusted historical documents could flag passages whose *predicted* date is materially later than the source metadata or later than 1913. That will catch many leaks your blacklist misses. citeturn485514view0

Third, add an **adversarial audit set**. Sample documents that pass all filters but are “suspicious” by other heuristics: unusually modern punctuation, anomalously high date predictions, OCR markup artifacts, or metadata-source disagreement. Read a few hundred manually. On projects like this, a small, disciplined manual audit often finds whole classes of failure you did not think to pattern-match.

For OCR-heavy newspapers, I would not treat them as “bad data” and simply downweight them across the board. They are probably your most valuable source for everyday worldview, political mood, and period register. The better move is **section-level quality scoring**. The Ranke-4B prerelease notes say standard web-text filters did not work reliably for historical data, and they ended up using heuristic filters that detect low-quality sections *within otherwise useful documents*, including rolling-window line metrics to catch OCR-degraded spans. That is the direction I would copy. citeturn784148view4turn451059view2

So instead of one corpus, I would create two views of it: a **Core Clean** stream and a **Noisy Frontier** stream. Core Clean would include better OCR, books, legal texts, and science texts with aggressive filtering. Noisy Frontier would keep newspapers and rougher OCR, but only after section-level trimming. Then train with a light curriculum: earlier training weighted toward Core Clean, later training mixed with Noisy Frontier. That preserves worldview diversity without making early optimization spend too much capacity on garbage.

Your dedup setup is good as a start, but still incomplete for historical news. The classic dedup paper found that LM corpora contain both **near-duplicate examples and long repetitive substrings**, and that better dedup both reduces memorization and improves training efficiency. Newspapers are especially bad here because of syndicated columns, repeated market tables, boilerplate legal notices, reprinted speeches, and OCR echoes. I would add **span-level dedup** or repetitive-substring filtering on top of document-level exact hashing and MinHash. citeturn402744view0

The current per-source sampling rates look heuristic rather than calibrated. That is okay for a first run, but it should not be your long-term policy. Work on data mixtures like **DoReMi** and **RegMix** shows that domain weights matter a lot, and small proxy models can be used to learn better mixtures than intuition alone. I would not trust “0.33 for American Stories, 0.60 for BHL” unless a proxy study shows those are actually better on your historical eval suite. citeturn123057view0turn123057view1turn123057view2

One blind spot here is **provenance after tokenization**. You need exact manifests for: source, source version, document count, token count after filtering, token count after dedup, token count after sampling, and shard IDs used in each run. Right now the “main vs small” ambiguity is unacceptable for research. Fixing this is low effort and high value.

## 2) Model architecture and training config

The architecture itself is broadly sensible. RoPE is standard and solid; QK-normalization exists specifically to make attention less prone to saturation; and GQA’s main documented advantage is inference efficiency, where it gets quality close to MHA with much faster decoder-time KV handling. None of those choices are red flags. citeturn123057view8turn123057view9

For your setup, **d12 / 125M is a good science model**, even if it is not the model that best matches your total 16B-token reservoir. Put differently: for your **available data**, 125M is small; for your **available compute**, 125M is about right. Nanochat itself is organized around depth as the main scale knob, and its current defaults describe a tokens:params regime around 10.5 as compute-optimal, with nearby tradeoffs used in practice. citeturn271766view0turn271766view1turn271766view2

One important inconsistency: by your own numbers, `262,144 tokens/step × 15,250 steps ≈ 4.0B tokens`. For a 125M model, that is roughly **32 tokens per parameter**, not 8. So either the “target_param_data_ratio=8” note is stale, the training horizon changed, or the batch accounting comes from a different run. I would fix that bookkeeping before doing anything else. Right now your configuration description points in two different directions.

If the 4B-token figure is real, I suspect you are **overtraining d12 for the baseline pipeline stage**, at least relative to nanochat’s current default assumptions. That is not automatically wrong. Noisy data can justify longer training, and your goal is not just benchmark score but historical absorption. But I would not make 4B your default until you can show that authenticity metrics keep improving after the earlier loss-optimal regime.

So my recommendation is:

Use **d12** as the baseline, **d16** as the likely sweet spot for a serious single-GPU paper, and treat anything above that as optional until the evaluation story is mature.

For GQA at this scale, I would probably **turn it off** unless you care about inference latency or KV-cache memory during deployment. At 125M on a single 3090, the quality-risk / complexity tradeoff is not compelling enough. The gain is mostly on the serving side. citeturn123057view9

On hyperparameters, I would stay conservative. Nanochat’s own speedrun notes point to long warmdown schedules and make clear that depth is the main dial while other hyperparameters are derived around it. Your flat schedule with 40% warmdown is in-family. I would not spend much time chasing exotic LR schedules until data quality and evaluation are nailed down. citeturn271766view1turn271766view2

Your global batch size is fine. Nanochat’s docs mention 524,288 tokens as a desirable total batch in its 8-GPU setup, but that does **not** mean you should contort a single-3090 run to hit that number. What matters for you is stable optimization at the highest real throughput you can sustain. If device batch 2 is what fits comfortably, that is okay. citeturn271766view1

## 3) Tokenizer considerations

A **custom tokenizer trained on historical text** is the right call. Using a modern inherited tokenizer would buy convenience, but it would encode segmentation assumptions shaped by modern distributions. Since one of your core claims is authenticity from the bottom up, matching the tokenizer to the corpus is defensible. Tools like SentencePiece also make it easy to train directly from raw text rather than relying on modern pretokenization conventions. citeturn675304view0

That said, **32K BPE is a baseline, not obviously the endpoint**. Recent work on vocabulary scaling argues that vocabulary size matters more than the field has historically acknowledged, and that larger models often deserve larger vocabularies; in their experiments, moving from 32K to 43K improved downstream performance at fixed compute. citeturn589272view0turn589272view4

For historical English, you have three extra tokenizer pressures that modern corpora have less of: spelling variation, OCR corruption, and long-tail named entities. That makes me think you should at least run a small proxy comparison of **32K vs 48K or 64K**. I would measure not just validation loss, but also:

- tokens per character
- tokens per word
- fragmentation of common archaic spellings
- fragmentation of person/place names
- fragmentation of OCR-noisy strings
- generation quality on quotations and titles

Subword robustness is a real issue. Recent work found that subword tokenization is sensitive to typos and format variation, and that **BPE-dropout/subword regularization** can help. That is especially relevant for OCR-heavy newspapers. I would seriously consider training the tokenizer with a little regularization, or at least testing it in the proxy stage. citeturn721380view2

A more exploratory option is to try something beyond vanilla BPE. There is recent work on **Length-MAX** tokenization claiming materially fewer tokens than BPE at the same vocab size. I would treat that as experimental, not your main branch, but historical corpora are exactly the kind of place where token efficiency might matter more than usual. citeturn123057view7

## 4) RTX 3090 optimization

Yes, `torch.compile` is worth retrying on native Linux. Current PyTorch documentation is explicit that `torch.compile` is meant to speed up PyTorch code with minimal changes, and that graph breaks usually mean lost optimization opportunities rather than silent corruption. On your current stack, I would run a controlled A/B benchmark for a few hundred steps. citeturn675304view1

Also verify that you are actually on PyTorch’s optimized SDPA path. The current docs say `scaled_dot_product_attention` can use FlashAttention-2 or memory-efficient attention kernels on CUDA and will automatically choose the best implementation it can. If your code path is bypassing that, you may be leaving easy throughput on the table. citeturn675304view3

If batch size 4 only barely OOMs, the most practical tool is **activation checkpointing**. PyTorch is very clear that checkpointing trades compute for memory, and it recommends `use_reentrant=False`. On a 3090, that trade is often worth it if it unlocks a better device batch or a deeper model. citeturn675304view2

For mixed precision, benchmark both modes your stack supports cleanly. PyTorch’s AMP docs note that fp16 can underflow or overflow depending on the model, so if you stay on fp16, use proper gradient scaling and watch for NaNs. citeturn675304view4

Also enable **TF32** matmuls on Ampere unless you have evidence that convergence regresses. PyTorch’s numerical-accuracy note explicitly recommends TF32 for many neural workloads when full fp32 precision is not necessary. citeturn675304view5

The biggest optimization you need, though, is not FLOPs. It is **checkpoint hygiene**. Saving only one surviving checkpoint from a 30-hour run is the real performance bug. Save often, rotate checkpoints, and copy them off-box.

## 5) Evaluation strategy for a time-locked model

You are right that standard CORE is not enough. CORE can still be a secondary sanity check for raw capability, but your main claims are different: **temporal boundary, historical authenticity, and controlled ignorance**.

I would build an evaluation suite with five pillars.

First, **cutoff fidelity**. Dated Data shows that a model’s effective cutoff can differ sharply from its advertised one, and TimeShift-style work shows how to probe time-sensitive recall systematically. For your project, I would build a pre/post-1913 factual suite with narrow year bins around the boundary, especially 1908–1918. Measure pre-cutoff recall, post-cutoff leakage, and where the effective knowledge horizon actually sits. citeturn451059view5turn451059view8

Second, **ex-ante reasoning**. This is a perfect fit for your goal. ExAnte shows that modern LLMs struggle to honor temporal constraints and leak badly, especially near the cutoff. Your model should be evaluated on questions that a 1913 observer could reasonably answer *from 1913 evidence*, without asking it to know outcomes. That is a much stronger test than simple fact recall. citeturn451059view6

Third, **historical authenticity**, not just accuracy. The historical simulation paper is important here because it found automated style judgments can be fooled while humans still tell the difference. So you need some human evaluation. I would combine human raters with automatic judges, following the spirit of benchmarks like Fùxì, which explicitly separate **linguistic accuracy** from **cultural authenticity**. citeturn451059view3turn451059view7

Concretely, I would ask:
- Can historians date the output to the right decade?
- Can they identify the genre correctly: newspaper editorial, court transcript, natural history prose, sermon, political tract?
- Do they see anachronistic concepts even when the wording sounds old?
- Does the model sound like a real period source, or like a modern parody of one?

Fourth, **worldview measurement**. This is probably your most novel angle. Ranke-4B explicitly treats normative views as something to preserve and measure across posttraining, which is a useful precedent. For your project, I would build a prompt set on topics like empire, labor, suffrage, race, religion, medicine, industrialization, and technology, and compare model answers to contemporaneous source distributions rather than to present-day truthiness. The point is not to endorse those views, but to measure whether the model’s priors are actually historical. citeturn784148view5turn784148view4

Fifth, **anachronism and abstention**. A good time-locked model should not only avoid wrong future facts; it should also know when to stop. Track:
- post-cutoff named entity leakage
- post-cutoff terminology leakage
- explicit future-year references
- false certainty on unknowable post-1913 questions
- abstention quality

One metric I would invent is **HLR: Historical Leakage Rate** — the share of answers that contain post-cutoff content under prompts where such content should be impossible.

## 6) Scaling recommendations

On a single 3090, I would not make “go larger” the first move. I would make **d8 / d12 / d16** the main program. Nanochat itself treats depth as the core scaling knob, and its leaderboard notes place GPT-2-like capability somewhere around d24–d26 in the current stack, which gives you a rough sense of how quickly cost rises as you climb depth. citeturn271766view1turn271766view2

Assuming you keep the same width rule as your d12 run, training cost will rise roughly with **depth cubed**. That means, very roughly:

- d20 is about **4.6×** the compute of d12
- d26 is about **10×** the compute of d12

If d12 takes about 30 hours now, a naive extrapolation puts d20 at roughly **6 days** and d26 around **12–13 days**, before accounting for worse memory pressure, more checkpointing overhead, and likely smaller device microbatches. That is why I think d20 is a stretch target and d26 is really a cloud job.

Roughly, with your current width scaling, d20 is in the ballpark of **half a billion parameters** and d26 is around **one billion**, assuming the vocab stays at 32K. Those are useful targets conceptually, but they are not where I would start on one 3090.

Should you use cloud compute? Yes, but **not yet**. Cloud becomes worth it after you have:
- a fixed corpus manifest
- a real authenticity benchmark
- a defensible source mixture
- checkpoint discipline
- at least one scaling curve on d8/d12/d16

Otherwise you will just buy a larger ambiguous result.

## 7) Research value and novelty

Your project is publishable if you position it correctly.

The “closest idea” space now includes at least two strands. One is the historical-simulation critique that argues period pretraining may be necessary. The other is Ranke-4B, which is pursuing large time-locked historical models trained from scratch and explicitly measuring pre- vs post-cutoff knowledge and normative views. citeturn451059view3turn784148view5turn784148view4

So the novelty is **not** just “I trained a historical model.” The novelty has to be one or more of these:

- a **small-scale, reproducible** time-locked training pipeline on commodity hardware
- a strong **evaluation framework** for historical authenticity and leakage
- a careful study of **OCR filtering, tokenizer design, and source mixture** in historical corpora
- a **multi-cutoff** analysis showing how worldview and leakage shift across 1850 / 1900 / 1913 / 1929, etc.

The most interesting experiments, in my view, are:

1. **True historical pretraining vs modern-model fine-tuning** on the same benchmark. The recent historical simulation paper makes this especially compelling. citeturn451059view3

2. **OCR-cleaning ablation**: raw OCR vs aggressive filtering vs section-level filtering vs curriculum.

3. **Source-mixture ablation**: newspapers vs books vs science vs law, and measure not only BPB but worldview shifts.

4. **Tokenizer ablation**: 32K vs 48K/64K, BPE vs unigram/SentencePiece, maybe one experimental Length-MAX branch. citeturn675304view0turn589272view4turn123057view7

5. **Cutoff-boundary study**: 1900, 1913, 1929 as separate checkpoints, then plot leakage curves and worldview drift.

6. **Human-vs-automatic authenticity evaluation**: especially valuable because the literature already suggests automatic judges can overestimate authenticity. citeturn451059view3turn451059view7

## 8) What you’re missing

The biggest blind spot is methodological: your model will not represent “what people thought” in general. It will represent **what survives in printed historical text**, heavily shaped by literacy, institutions, editors, class, empire, and archive bias. That is fine, but it needs to be named honestly. Otherwise “historical authenticity” gets overstated.

The second blind spot is reproducibility. Right now you do not have a publishable training record if you cannot say exactly which shards were used, which tokenizer version was used, what the true token budget was, and where the checkpoints and eval logs live.

The third is split design. You need held-out sets stratified by **source, genre, year, and document family**, so that a repeated serialized novel, syndicated article, or reprinted legal text cannot make your eval look better than it is.

The fourth is normalization policy. OCR cleanup and spelling normalization can make text easier to model, but they can also erase exactly the orthographic cues that make the period feel authentic. I would keep both a **diplomatic/raw stream** and a **normalized stream**, and be explicit about which objective each serves.

The fifth is licensing and dataset governance. Historical text is often public domain, but the packaging, OCR, metadata, or API terms may not be simple. Research projects get tripped up here more often than they should.

## My strongest recommendations, in order

1. **Freeze the data pipeline** with immutable manifests, token counts, and run metadata.
2. **Add section-level OCR filtering and span-level dedup**, not just document-level filters.
3. **Build the benchmark first**: cutoff fidelity, ex-ante reasoning, authenticity, worldview, leakage.
4. **Run a clean d8/d12/d16 series** before any d20 ambition.
5. **Ablate tokenizer size** and **learn source weights with proxy models** instead of hand-tuning them.
6. Only then decide whether a bigger cloud run is justified.

The short version of my verdict is: **the idea is excellent, the current model choice is defensible, and the biggest gains are in data discipline and evaluation design rather than in architecture tweaks.** If you get those two pieces right, even a 125M–300M historical model can be a meaningful paper.

---
