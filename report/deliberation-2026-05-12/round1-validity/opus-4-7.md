# Opus 4.7 — Round 1 Validity Verdict

Independent Opus check on the GPT Max + GPT Council convergent verdict. I verified every novel claim against primary code, the provenance.json, the source_mix.md, and the training log. Where I converge with the GPT panels I am brief; where I diverge or extend, I show evidence.

---

## 1. Points where I agree with the GPT panels

### 1.1 The narrow mechanical claim survives.
A 615,645,184-parameter d22 transformer completed 70,455 optimizer steps under the repaired `parallel_family_cache` loader, with no NaN/OOM/instability, with checkpoint round-trips clean, with peak VRAM 17.5 GiB, with loader < 1% of step time. The artifact `model_070455.pt` is real. (Both panels; verified in log + meta files.)

### 1.2 The 1.1092 bpb is a fixed-prefix probe, not a held-out corpus measure.
Verified independently. `base_train.py:535-538` rebuilds a fresh val loader each eval, `eval_tokens=262144`, `device_batch_size=8`, `seq_len=1024`, `world_size=1` → exactly 32 batches of 8193 tokens = 262,144 tokens. `cached_distributed_data_loader` starts at sorted `shard_index=0` and `token_off=0` deterministically. Same 262k-token slice probed at every eval. This is the highest-confidence adverse finding in both panels, and I confirm it.

### 1.3 "Monotone descent" is overstated.
The trajectory has multiple local upticks (e.g., 12000→13000, 22000→25000, 29000→34000); "downward trend with final-equals-minimum" is accurate, "monotone descent" is not. (GPT Council, Empiricist; verified.)

### 1.4 Run #1 establishes that ordinary smoke + training-health signals are insufficient for data-integrity claims.
This is the most epistemically load-bearing point in either panel. 46 hours of healthy loss, healthy val bpb, healthy throughput, healthy VRAM, healthy sha-rt — all coexistent with the loader silently reading 16.5% of the cache with 80.8% wrong family tags. Both panels are right to keep weight on this. The loader guards now in place (`coverage < 0.95 raise`, `family_mismatches > 5% raise`) would have caught run #1, but they catch only the specific failure mode of stale shard_index keys against a regenerated manifest. They do not generalize to other manifest-mutation classes.

### 1.5 Effective training distribution ≠ corpus inventory.
The 12/8/6/3/3 schedule, not the 37.7/26.9/17.5/9.8/8.2 corpus mix, governs what the model trained on. Books and legal wrapped once (`family_wrap_counts=1` at step 70000 in `meta_070000.json` and `meta_070455.json`); newspapers, science, early_modern did not. The brief's framing collapses these and reads as more honest than the GPT panels gave it credit for. (Verified in checkpoint metadata.)

### 1.6 Samples are smoke-level evidence at most.
Period-flavor tokens ("francs", "to-day") are not historical competence. The arithmetic / chronology / factual-recall failures in the same sample set are equally diagnostic of "small base model, no midtraining" as of "governed pre-1914 corpus working as intended." Both panels are right to refuse to read the samples as anything stronger.

---

## 2. Points where I disagree or modulate

### 2.1 I think the headline-metric problem is **worse** than either panel framed it — and worse than the empirical-findings.md captures.

Both panels say: "1.1092 bpb is on a fixed 262k-token Gutenberg books prefix; it is a real learning signal on that prefix; it is not an aggregate corpus measure." All true. But the empirical-findings.md presents this primarily as a *narrowness* problem — "small slice, narrow construct, valid for what it measures."

The deeper issue is that **the slice is a held-out source the model never saw in training**. Gutenberg has **1,132 val shards (1.13B tokens, 10,188 docs) and zero train shards**. I confirmed this two ways:
- `data/processed/corpus_1913_v4_balanced_candidate/reports/source_mix.md` lists `gutenberg: 0 train shards / 1 val shard / 1.16B val tokens` (and the launch justification's "books_general share 17.5% (BL Books CC0 + Gutenberg)" is **factually wrong about Gutenberg's contribution to train** — Gutenberg contributed nothing to training).
- `provenance.json` confirms: train cache source_ids are `{american_stories, bhl, bl_newspapers, blbooks, eebo, caselaw_cap_v2}`. Gutenberg, TCP, Old Bailey, Chronicling-America, Chronicling-v2 are **train-empty, val-only**. Only `chronicling_v2` is an intentional held-out source per `split_holdouts.py:FULL_SOURCE_HOLDOUT`.

So 1.1092 bpb is not "narrow in-distribution loss." It is **a cross-source held-out generalization measure** on a source whose distribution the model has never seen, evaluated on a tiny 262k-token prefix.

This cuts both ways:
- **Bad**: The construct is not what the brief implies, AND not what the empirical-findings.md implies. The headline metric is conceptually closer to "OOD perplexity on classic English literature" than "in-distribution governed-corpus loss." Comparing it to nanochat's 0.81 val bpb (in-distribution FineWeb-Edu eval) is even more misleading than the panels already flagged.
- **Surprisingly good (one dimension)**: there is zero document-level train/val leakage on the specific 262k slice, because the model never saw any Gutenberg. So the descent IS real held-out generalization, just to a more demanding (different-source) target than implied.

I am ~85% confident this matters more than the panels weighted it. I'd bet the researcher did not intend Gutenberg to be train-excluded; it looks like a v3-vs-v4 build accident — the V3_GOVERNED `gutenberg` shards may have had `split=val_by_document` rows pre-set by an earlier `split_holdouts.py` run, but `build_v4_balanced.py` doesn't read the `split` column and was working off shard-level whole-shard sampling, so Gutenberg got bucketed val-only by random whole-shard sampling.

### 2.2 I disagree with the panels' implicit acceptance of "books and legal wrapped" as a minor curiosity.

The actual final exposure profile from `meta_070455.json`:

| family | cache shards | cursor | wrap | unique-token exposure |
|---|---:|---:|---:|---:|
| newspapers | 7,468 | 7,183 | 0 | 96.2% of pool (first pass only, did not reach end) |
| science | 5,156 | 4,631 | 0 | 89.8% of pool (first pass only) |
| books | 2,804 | 104 | **1** | 100% of pool + 3.7% re-exposure |
| legal | 1,627 | 174 | **1** | 100% of pool + 10.7% re-exposure |
| early_modern | 1,871 | 1,732 | 0 | 92.6% of pool (first pass only) |

Three implications the panels gloss:

(a) **The 30:1 tokens-to-params Chinchilla claim is on scheduled tokens, not unique tokens.** Unique training tokens ≈ 6.93 + 4.62 + 3.34 + 1.56 + 1.73 = ~16.18B. That's ~26.3:1 unique-tokens-to-params. The "Chinchilla-optimal" framing is off by ~12%.

(b) **Books and legal re-exposure happened during warmdown.** Books wrapped between step 60000 and 70000 (cursor 2476 → 85); warmdown starts at step 42273 and LR is decaying to 0. The re-exposed slice of books/legal contributes much less to the final model than the first-pass tokens. So the model's effective books exposure is closer to "one pass at decaying LR" than "memorized books distribution."

(c) **Re-exposure is on deterministic shard order, not a re-shuffle.** The first 104 blbooks shards consumed twice. If the manifest is shuffled (it was, per the v3→v4→shuffle pipeline), the re-exposed slice is a random subsample of blbooks. If the shuffle was seeded and reproducible, that's fine; if it depended on an environment-state seed, the re-exposure is non-reproducible.

### 2.3 I'd modulate the "governance was repaired for the completed run" claim.

The GPT panels treat the loader guards as repairing the governance class. They do — for the *specific* known failure mode (provenance.json keyed by stale shard_index against a regenerated manifest). But:
- The fix doesn't address the **structural cause**: provenance is still keyed by `shard_index`, a volatile position, not by `source_file`, a content-stable identifier (the postmortem itself names this as the right long-term fix and lists it as a TODO).
- `split_cache_shards.py` and `shuffle_cache_manifest.py` still don't auto-regenerate provenance. The postmortem lists "TODO: bake into the splitter script" — not done. So the same class of bug can recur the next time those scripts are run.
- The fix targets the failure that already happened. It does not target adjacent silent-data-integrity failures: e.g., a future-tense post-1914 doc with `publication_year=NaN` slipping through `unknown_date_action: exclude`; a duplicate document on both sides of train/val; a wrong-family parquet shard.

"Governance fixed for the loader-stale-provenance failure class" is the correct framing. The GPT Council comes close to this with "the known stale-provenance class appears fixed at this layer." GPT Max is a bit looser ("supports 'the known provenance bug was repaired for the completed run'"). I'd be stricter still: **the run was salvaged by adding two assertions; the underlying file-coupling that produced the failure remains, and two of the three upstream scripts that can re-induce it remain unfixed.**

---

## 3. Additional validity issues not raised by either panel

This is the section the deliberation lives or dies on. I have several. Ordered by severity.

### 3.1 No cross-source near-duplicate dedup. Explicitly acknowledged in the code, ignored by both panels.

`data/phase0/README.md:120` says verbatim: *"Cross-source SimHash/MinHash dedup pass (we inherit the legacy deduped/ outputs)"* — listed under **"What's NOT here."**

This is not a theoretical risk; it is a known gap, acknowledged in the data pipeline's own README, never closed. Concrete consequences for this run:

- **Classics in multiple corpora.** A Shakespeare play in blbooks (train) and the same play in TCP (val) → train/val leakage. Same for translated Greek/Latin classics, the King James Bible, Pilgrim's Progress, etc. The fact that Gutenberg is train-excluded reduces this risk for one source; it remains for blbooks/TCP/EEBO overlap.
- **Syndicated newspaper articles.** `american_stories` (train, 7,097 shards) extracts articles from Chronicling America newspaper pages. `chronicling_v2` (val-only, 5 shards) also derives from Chronicling America. If the same article appears in both — likely, since `american_stories` is the article-level extraction and `chronicling_v2` is the page-level version — that is direct train/val document overlap.
- **Multiple BL Books editions.** Same book at different libraries, multiple printings, multiple OCR passes. BL Books is per-page, and the README says verbatim: *"BL Books records are per-page, not per-book. A future dedupe pass would be useful."* So a single book contributes potentially hundreds of pages, with no per-book grouping — meaning shard-level train/val split can put pages 1-200 in train and pages 201-400 in val of the same book.

The val bpb almost certainly underestimates true OOD loss because some val content is near-duplicated in train. I'd put this at ~70% likelihood for `american_stories`/`chronicling_v2` overlap specifically. Neither panel surfaced this beyond "could be a risk."

### 3.2 The two split mechanisms are uncoordinated.

`split_holdouts.py` exists and does a **document-disjoint, deterministic, hash-based** train/val split. It writes a `split` column in-place into the governed parquet shards.

`build_v4_balanced.py` (the script that built the v4 corpus actually used) does NOT read or honor the `split` column. It does its own **whole-shard** split per source (lines 179-226). The docstring at line 182 says "by stable doc-hash" but the code at 196-214 only does whole-shard random sampling per source.

Consequences:
- Train/val isolation depends on **whether documents span shards** at the source level. For sources where one shard = one document (Gutenberg: 9 docs/999K tokens ≈ 110K tokens/doc; BHL: 3 docs/999K tokens), this is fine.
- For sources where shards contain many documents (american_stories: 3,846 docs/shard; blbooks: ~3,000 docs/shard; oldbailey: ~50,000 docs/shard), a single book or article series could have docs distributed across multiple shards by the dedupe pipeline, with no guarantee that all of one book's pages end up on the same side of the split.
- For sources where the same document literally appears twice (multi-edition books, syndicated newspaper articles), **whole-shard splitting does not prevent the document from appearing in both train and val.**

The infrastructure for document-disjoint splitting EXISTS (`split_holdouts.py`) but is not used by the v4 build. Neither panel surfaced this. Either panel could have caught it by reading `build_v4_balanced.py:select_shards_for_budget`.

### 3.3 Date-cutoff enforcement is per-source registry-mediated and has known semantic gaps.

`data/phase0/sources/registry.yaml` defines `cutoff_year: 1914`, `unknown_date_action: exclude`. The `date_audit.py` script enforces this via per-source `date_field_candidates` and `date_confidence_default`.

But the cutoff guarantee is only as strong as the per-source date semantics. Two concrete issues:

(a) **Gutenberg metadata records year-of-original-composition, not publication-date**, as the data_audit module's own docstring warns: *"EXCLUDE records where only ingestion/download date is known (e.g. Gutenberg release date != publication date)."* I verified: shard_gutenberg_000000.parquet has `publication_year` values ranging from **5 to 1913** (year 5 CE is presumably a Roman classic; many records cluster at 1000-1300 for medieval texts). Gutenberg's `publication_year` is the **year of original composition**, not the year of publication or modern release. This is fine for Gutenberg specifically (it's train-excluded), but the same pattern could affect other sources where the metadata field semantics differ from publication date in subtle ways. The trust line is in the registry data, not in the audit code.

(b) **Content semantics ≠ publication date.** The cutoff guarantee is "no document with publication_year >= 1914." It is NOT "no content semantically referencing post-1914 events." A 1913 newspaper article forecasting WWI, a 1910 medical textbook predicting future germ theory, an 1890 prophetic essay about airplane warfare — all in training, all containing post-cutoff-aware content. The anachronism eval problem is structurally unsolvable by date-based filtering alone; it requires content-level filtering, which the pipeline doesn't do. Neither panel surfaced this, though it's central to the project's thesis ("the model genuinely cannot know post-cutoff events").

### 3.4 The corpus mix percentages in the brief diverge from the built corpus.

Brief says:
- newspapers_periodicals 37.7% — sources Chronicling America, etc.
- science_technical 26.9% — BHL
- books_general 17.5% — BL Books + Gutenberg
- early_modern 9.8% — EEBO + TCP subset
- legal_government 8.2% — CAP + Old Bailey

Actual built corpus per `source_mix.md` (train only):
- newspapers: american_stories 38.38% + bl_newspapers 1.89% = **40.27%** (chronicling_america and chronicling_v2 are train-empty)
- science: bhl 20.40%
- books: blbooks 20.13% (**gutenberg train-empty**)
- early_modern: eebo 9.13% (**tcp train-empty**)
- legal: caselaw_cap_v2 10.07% (**oldbailey train-empty**)

The brief's source attribution is wrong for at least 3 families. The brief lists Gutenberg, TCP, Old Bailey, Chronicling-America as **training sources**, but they are **val-only**. The brief's "37.7 / 26.9 / 17.5 / 9.8 / 8.2" mix matches some earlier corpus characterization, but not the actually-trained-on distribution (which is closer to 40.27 / 20.40 / 20.13 / 9.13 / 10.07).

This is not a minor reporting drift. It means the model trained on a substantially **less diverse** corpus than the brief claims, particularly in the books and legal families which now have **single-source coverage** (only blbooks for books; only caselaw_cap_v2 for legal). Single-source family coverage limits generalization claims more than five-family multi-source coverage would.

### 3.5 Smoke #4 passing with broken provenance has a deeper implication than "ordinary metrics insufficient."

Both panels flag the smoke #4 paradox (passed with broken provenance, 46h of healthy training). The deeper implication they miss: **what is the load-bearing guard now?**

The loader guards added post-postmortem (`coverage < 0.95 raise`; `family_mismatches > 5% raise`) are themselves **mechanical checks against the same kind of failure**. They catch the specific class where `provenance.json` has fewer entries than the manifest OR where provenance family-tags disagree with `source_file` parsing.

Failure modes they do NOT catch:
- Provenance and manifest both regenerated correctly, BUT the underlying parquet was the wrong family at ingestion time (e.g., a newspaper shard renamed `shard_books_general_blbooks_999999.parquet` by accident). The family-cross-check uses `source_file` name parsing — if the source_file name is consistent (even if wrong), no mismatch fires.
- Manifest entries pointing to deleted .bin files (memmap would succeed on whatever bytes are at that path, including residual junk).
- Tokenizer mismatch (val tokenized with a different tokenizer than train, producing artificially-low or artificially-high bpb).
- Document-level train/val contamination (the guards check shard-level integrity, not document overlap).

The point isn't that more guards are missing (that's always true). The point is that the answer to "what guarantees this run wasn't silently miscorrupted in some new way" is: **diagnostic cursor logging + the researcher manually noticing the cursor pattern**. That's the same fragile guarantee that almost failed in run #1 (was visible from step 0, noticed at step 10,293). The "guards" are best-understood as documentation of what to check, not as a robust safety net.

### 3.6 Bpb arithmetic on a custom tokenizer is not protocol-comparable, but I'd go further.

GPT Council notes that bpb is not protocol-comparable across tokenizers/corpora. Correct. But the deeper issue: bpb on a 32,768-vocab tokenizer trained on the 1913 corpus, evaluated on Gutenberg books (the original Gutenberg shard 0), is **a measure of how well the model models a corpus the tokenizer was optimized for**. If the tokenizer training set included Gutenberg-style English (which it almost certainly did, since the tokenizer was trained on the 1913 corpus and Gutenberg was probably in the tokenizer-training data even if not the model-training data — would need to verify), then the bpb on a Gutenberg-prefix is mechanically lower than it would be on, say, OCR'd newspaper text where the tokenizer has to spend tokens on OCR noise.

So: 1.1092 bpb on a Gutenberg prefix tells you **the model is good at predicting clean English literary prose**, partially because the tokenizer is also good at compressing clean English literary prose. Comparing 1.1092 to Karpathy nanochat's 0.81 (FineWeb-Edu, GPT-2-style tokenizer) is at least 3 confounds deep.

### 3.7 The deterministic val-shard-0 probe potentially benefits from token-level memorization.

If the first 262k tokens of val/shard_001594.bin contain text that exists (verbatim or near-verbatim) in train — for example, a quoted Shakespeare passage that appears in both Gutenberg val and EEBO/blbooks train — then the val bpb on those tokens benefits from memorization, not just generalization. Without document-level cross-source dedup (§3.1), there is no guarantee this isn't happening.

This is testable: dump the first 262k val tokens, decode them, search for substrings in train. A 10-line script. Not done.

---

## 4. Blindspots shared by both GPT panels

These are the patterns where Opus-lineage training data / judgment may differ from GPT-5.5-lineage. I am ~60-70% confident on each individually; the pattern as a whole is more confident.

### 4.1 Both panels under-investigated the val cache structure.

Both panels stopped after confirming "val shard 0 is Gutenberg books." Neither panel pulled the **per-source train/val provenance counts** that would have revealed Gutenberg's zero-train status. The check is one Python script away from the empirical-findings.md. Both panels treat the val cache as "source-stratified, separate, fine for what it's used for" rather than asking "what sources are actually IN it, and which sources cross train/val?"

### 4.2 Both panels accepted the "wrap" claim without verifying it against checkpoint metadata.

Final family wraps and cursor positions are in `meta_070455.json.dataloader_state_dict`. Either panel could have read this file. Neither did. (Empirical-findings.md proposes this as future verification.) The actual data — wrap=1 for books/legal, wrap=0 for everything else, unique-token coverage of 96.2/89.8/100/100/92.6% — turns out to be qualitatively similar to what the panels assumed, but the panels couldn't verify it.

### 4.3 Both panels treated "governance" as primarily a loader/provenance question.

Both panels exhaustively analyzed loader-side data integrity. Neither panel asked whether the **upstream** parquet shards have the right content. Date-cutoff enforcement, rights audit semantics, document-level dedup, registry-source-mapping coherence, raw-document-to-shard auditability — these are all upstream of the loader and matter equally for "governed corpus" claims. The panels' blindspot is taking source-stratified provenance as evidence of source correctness.

### 4.4 Both panels gave the brief a pass on its tabular precision.

The brief lists "books_general 17.5% — BL Books CC0 + Gutenberg" as a corpus fact. The actual built corpus has Gutenberg contributing zero training tokens. Either panel could have caught this by reading `source_mix.md` (which is a referenced artifact in §7 of the brief). Neither did. This suggests the panels weighted the brief's narrative over the artifacts the brief itself references.

### 4.5 Both panels under-emphasized that the construct being measured by 1.1092 is "OOD generalization to held-out source," not "in-distribution loss."

GPT Max and GPT Council both characterize 1.1092 as a "narrow learning signal on a fixed prefix." That's accurate at the within-run level. But neither panel articulates that the prefix is from a source the model never trained on. The construct is closer to "Pile-Books cross-eval" than to "training-distribution val loss." This changes how to interpret the descent — it's evidence the model learned features that generalize cross-source, which is a STRONGER (not weaker) signal in some senses than within-distribution descent would be.

Both panels treat the headline-metric finding as purely deflationary. It's actually mixed-valence: the construct is more demanding than implied, but the descent on the more-demanding construct is more impressive than within-distribution descent of the same magnitude.

### 4.6 Shared blindspot: anachronism is a content-semantics problem, not a date-filter problem.

Both panels flag "no anachronism eval" as a gap. Neither panel flags that the date-cutoff enforcement strategy itself cannot in principle prevent pre-1914 documents from containing post-1914-relevant content. A 1913 newspaper article anticipating WWI is in training and IS post-cutoff-aware. The thesis ("the model genuinely cannot know post-cutoff events") is **partially defeasible at the corpus level**, not just at the eval level. The content-level pre-filter that would be needed doesn't exist in this pipeline. Neither panel surfaced this.

---

## 5. My own overall validity verdict

**Completed corrected training run, mechanically credible, with a headline metric that measures a different construct than the brief implies AND a corpus that differs in important ways from the brief's tabular description.** The run produced a real artifact: `model_070455.pt`, a 615M-param d22 base transformer trained for 70,455 steps on the repaired `parallel_family_cache` loader, with no NaN/OOM, healthy compute throughput, and a final fixed-prefix val bpb of 1.1092. The GPT panels are right that the headline metric measures a 262k-token Gutenberg prefix rather than the 2.86B held-out split, and they are right that this is a serious construct-validity gap. They are also right that the run is not a validated historical-base-model PoC by any operational definition. Where I diverge: the Gutenberg slice is not just narrow — it is a source the model never saw in training, making 1.1092 a cross-source held-out generalization measure rather than within-distribution loss; the actual trained-on corpus is materially less diverse than the brief claims (single-source coverage in books, legal, science families because Gutenberg, TCP, Old Bailey, Chronicling-America are all train-empty); the pipeline has known unmitigated cross-source near-duplicate exposure that almost certainly leaks document-level content from val into train via syndicated newspaper articles and multi-edition books; the date-cutoff enforcement is publication-year-only and cannot in principle catch pre-1914 content that anticipates post-cutoff events; and the load-bearing safety net against a recurrence of the run #1 class of bug is two assertions in the loader plus the researcher reading diagnostic cursor logs, not a structurally fixed manifest/provenance coupling. The right label is something like: **"mechanically valid corrected run with a real but mis-described learning signal, a less-diverse trained-on corpus than reported, and several construct-validity gaps the panels did not surface."**
