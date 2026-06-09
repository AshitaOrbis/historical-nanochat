# Empirical Findings (R1 / Validity) — verified during Round 1

These are facts confirmed by reading the actual code and data, not panel inferences. Cite these to the rest of the deliberation panel.

## 1. The headline 1.1092 bpb is NOT measured on the 2.86 B source-stratified val set

**Status:** VERIFIED in code.

`scripts/base_train.py` lines 532-547 (the eval block):
```python
val_loader = build_val_loader()
eval_steps = args.eval_tokens // (args.device_batch_size * current_seq_len * ddp_world_size)
with autocast_ctx:
    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
```

With `eval_tokens=262_144`, `device_batch_size=8`, `current_seq_len=1024`, `ddp_world_size=1`:
→ `eval_steps = 32` batches → exactly **262,144 tokens per eval**.

`build_val_loader()` returns a fresh `cached_distributed_data_loader(...)` each call. In `dataloader_cached.py` lines 99-110, when no `resume_state_dict` is supplied, the loader starts at `shard_cursor = owned[0] = 0` and `token_cursor = 0`. The cache is read in deterministic `shard_index` order (line 68). So every val eval reads **the first 262,144 tokens of val/shard at shard_index 0**.

What is val shard 0?
```
filename: shard_001594.bin
source_file: shard_books_general_gutenberg_000001.parquet
docs: 9
tokens: 999,563
family: books_general
source_id: gutenberg
```

Of the 2,881 val shards (2.86 B tokens, source-stratified), the loader only ever touches the first ~26% of one Gutenberg books shard.

**Implication:** the "validation bpb" trajectory is a real learning signal on a fixed ~262k-token Gutenberg books prefix. It is NOT a source-stratified val measure of corpus generalization. The brief's framing ("source-stratified separate 2.86 B val split") is technically true about the *cache on disk* but misleading about what the *training run actually measured*.

This is not a bug — `eval_tokens=262_144` is the configured value, the loader is doing what it's coded to do. It is a **construct-validity gap**: the headline number does not measure the construct the brief implies.

## 2. The val shards are not interleaved by family

**Status:** VERIFIED.

First 5 val shards by index:
- 0, 1, 2, 3: gutenberg / books_general
- 4: tcp / early_modern

To get any newspaper, BHL/science, or CAP/legal val signal at the current `eval_tokens` budget, you'd have to (a) increase `eval_tokens` to roughly 3+ GB to traverse multiple shards, OR (b) interleave the val shards by family at build time, OR (c) write a per-family eval that explicitly picks shards by family and computes bpb separately.

## 3. The val cache itself is governed and source-stratified

**Status:** VERIFIED.

`provenance.json` for split=val has 2,881 entries with `family` and `source_id` per shard. The infrastructure to compute per-family val bpb exists. It just isn't used by `base_train.py`'s default eval loop.

## 4. What "monotone descent" actually looks like

**Status:** VERIFIED partial counter-example.

Sampled val bpb trajectory after the resume:
```
step 10000 → 1.2406
step 13000 → 1.2341  (UP from step 12000's 1.2267)
step 25000 → 1.1868  (UP from 22000's 1.1800)
step 27000 → 1.1829
step 34000 → 1.1745  (UP from 29000's 1.1663)
step 70455 → 1.1092
```

The trend is strongly downward but it is NOT monotone. Calling the curve "monotone descending" overstates; "strongly downward with several short upticks, ending at the minimum" is more accurate. My initial narrative said "monotone or flat-monotone at the end" — that was loose. The honest read is closer to GPT Max's "downward trend with late improvement."

## 5. The 1.1092 number is on a fixed deterministic slice, so the trajectory is a real signal

**Status:** Construct-valid for what it measures.

Because each eval reads the same 262,144 tokens, the val bpb across steps is comparing the model at different points against an identical loss target. The decreasing trajectory is therefore real evidence that the model is improving its modeling of *that specific Gutenberg books prefix* over training. It is not noise. It is just a narrow construct.

## 6. The corpus mix percentages may not match the loader schedule, by design

**Status:** known/disclosed in the launch justification.

Corpus token mix (from §2 of brief):
```
newspapers 37.7 / science 26.9 / books 17.5 / em 9.8 / legal 8.2
```

Loader schedule (12/8/6/3/3 microbatches per step):
```
newspapers 37.5 / science 25.0 / books 18.75 / legal 9.375 / em 9.375
```

These are close but not identical. The launch justification already calls this out as an honest caveat. The model's *effective* training distribution is the schedule, not the inventory. With 18.5B tokens consumed at this schedule, smaller families (legal, em at ~1.7B each) likely cycled multiple times through their families' total token count, while the larger families saw their token-pools incompletely.

To verify: cross-check the final `fam_cursors` from the training log against the per-family total tokens in the cache.

**Verified.** Final fam_cursors at step 70455:
```
newspapers_periodicals: 7183 / 7468 shards
science_technical:      4631 / 5156 shards
books_general:           104 / 2804 shards   ← wrapped
legal_government:        174 / 1627 shards   ← wrapped
early_modern:           1732 / 1871 shards
```

Per-family cycle counts under the schedule:

| family | scheduled tokens | available in cache | cycles | comment |
|---|---:|---:|---:|---|
| newspapers_periodicals | 6.93 B | 7.20 B | 0.96× | partial pass |
| science_technical | 4.62 B | 5.14 B | 0.90× | partial pass |
| books_general | **3.46 B** | **3.34 B** | **1.04×** | full pass + 3.7% second pass |
| legal_government | **1.73 B** | **1.56 B** | **1.11×** | full pass + 10.7% second pass |
| early_modern | 1.73 B | 1.87 B | 0.93× | partial pass |

So `books_general` cycled once and re-trained on its first ~3.7% of shards. `legal_government` cycled once and re-trained on its first ~10.7% of shards. The other three families got partial single passes. This is a mild but real cross-family exposure asymmetry — the smallest family (legal) is most over-exposed.

The 18.5 B "tokens trained" includes ~123 M tokens of literal repeat (books wrap) and ~167 M of literal repeat (legal wrap), so ~290 M tokens (~1.6 %) of the run is on tokens the model has already seen exactly. This is not severe but should be disclosed.

## 7. Sample probes at end of run

Final sample completions printed at step 70455 (verbatim from training log):
```
The capital of France is 100000, 900 francs, and the population
The chemical symbol of gold is the same as that of silver, and the same as that of copper
If yesterday was Friday, then tomorrow will be Friday. If yesterday was Saturday, then to-day will be Saturday.
The planets of the solar system are: Saturn, Jupiter, Mars, Uranus, Ne
The opposite of hot is the best thing to do with the cold. A hot bath is the
My favorite color is a light brown, but the best is a dark brown, with a
If 5*x + 3 = 13, then x is the number of the square root of the number of the square root of the number
```
English fluent. Period register present at the token level. Factual recall, arithmetic, and basic logic absent. This is what a 615M base model with no midtraining looks like — consistent with the construct, but not by itself evidence of "period-appropriate competence" beyond surface forms.

---

Treat these as facts, not opinions, when forming your verdict.
