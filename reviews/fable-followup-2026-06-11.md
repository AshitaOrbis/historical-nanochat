# Adversarial Security/Correctness/Privacy Review — `historical-nanochat`

> Archived verbatim. Produced 2026-06-11 by a fresh Claude Fable 5 reviewer that
> was blind to the GPT Pro review and to the set of fixes applied. This review
> ran against the post-fix working tree and was used to catch gaps in the
> remediation (notably `trust_remote_code`, the `***REMOVED***` username leak, and the
> secret-gist URL).

## Summary

The repo's `SECURITY.md` is unusually honest and mostly accurate — the documented P0 (`execution.py`), P1s (`eval()`, `torch.load`, tokenizer pickle), and P2 (`report.py shell=True`) all check out against the code. However, I found **one significant undocumented code-execution sink** that `SECURITY.md` omits despite claiming to enumerate all inherited unsafe patterns, plus **two real privacy leaks in committed files**, and some correctness rot in the entry-point scripts.

---

## P1 — Undocumented RCE sink: `trust_remote_code=True` (documentation gap + supply-chain RCE)

**File:** `probes/harness.py:149` and `probes/harness.py:153`
```python
self.tok = AutoTokenizer.from_pretrained(tokenizer_id or hf_id, trust_remote_code=True)
self.model = AutoModelForCausalLM.from_pretrained(
    hf_id, trust_remote_code=True, device_map="cuda", ...)
```
**Driver:** `probes/run_pilot.py:167-170` hardcodes third-party model `dtestnyrr/talkie-1930-13b-base-gptq-int4` and tokenizer `xlr8harder/talkie-1930-13b-base-tf`.

`trust_remote_code=True` downloads and **executes arbitrary Python** (`modeling_*.py` / `tokenization_*.py`) from those third-party HuggingFace repos at load time — a textbook supply-chain RCE. Running `probes/run_pilot.py` triggers it automatically.

**Why this is a finding beyond the code itself:** `SECURITY.md` opens by promising to state the posture "honestly" and explicitly enumerates "several components [that] inherit known-unsafe patterns" — pickle, `torch.load`, `eval`, `exec`, `shell=True`. It says nothing about `trust_remote_code`. A reader who audited against `SECURITY.md` would conclude the only deserialization risks are checkpoints/tokenizer pickles and would miss that simply running the probe harness executes two strangers' code. That is a real gap between the claimed-complete enumeration and reality.

**Disposition:** Add a §7 to `SECURITY.md` documenting it; ideally pin model revisions by commit hash and gate `trust_remote_code` behind an explicit opt-in flag.

---

## P2 — Privacy: real PII / secret-by-obscurity leaks committed to a public repo

This is a public repo (`origin = https://github.com/AshitaOrbis/historical-nanochat.git`), so these are publicly exposed.

1. **Windows username leak** — `claude.md:31-33`:
   ```
   /mnt/c/Users/***REMOVED***/D-drive-data/historical-nanochat-deduped/
   /mnt/c/Users/***REMOVED***/D-drive-data/historical-nanochat-shards/
   /mnt/c/Users/***REMOVED***/D-drive-data/historical-nanochat-shards-small/
   ```
   `***REMOVED***` is a personal account username. The author clearly cares about this (commit `eecfcf0`: "drop private ChatGPT title manifest from git (privacy)") — this one slipped through.

2. **"Secret" GitHub gist URL leak** — `report/deliberation-2026-05-12/synthesis/FINAL-SYNTHESIS.md:186`:
   ```
   Phone-readable gist: https://gist.github.com/AshitaOrbis/REDACTED-GIST (secret, 8 docs)
   ```
   A GitHub *secret* gist's only protection is its unguessable URL. Committing that URL (self-labeled "secret, 8 docs") into a public repo defeats the protection entirely — anyone reading the repo can open all 8 documents.

**Disposition:** Scrub `***REMOVED***` (replace with `<user>`), and either delete the gist or remove the URL from the committed report. Note: a `git filter-repo`/history rewrite is needed for full remediation since both are in committed history.

*(Lower-severity, not PII: `nanochat/dev/repackage_data_reference.py:37` and `report/.../cloud-training-research.md` contain generic `/home/ubuntu/...` paths — cloud-instance defaults, not personal. No action needed.)*

---

## P2 — `serve.py` exposure hardening

**File:** `serve.py`
- `serve.py:221` — `app.run(host='0.0.0.0', port=5000)` binds the inference API to **all interfaces with no authentication**. Any host on the network can hit `/v1/completions`. For a research box this is a real exposure (resource abuse / unauthenticated model access). Bind to `127.0.0.1` by default.
- `serve.py:98` and `serve.py:166` — `data = request.json` with no guard; a request without a JSON body / wrong content-type yields `None`, and the subsequent `data.get(...)` raises → unhandled 500. Minor DoS / robustness gap.

The checkpoint load here goes through `torch.load` (`checkpoint_manager.py:45`) — that's the documented P1 pickle issue, acceptable under "load only your own checkpoints."

---

## P2/P3 — Correctness: stale/nonexistent hardcoded checkpoints in entry points

- `serve.py:34-35`:
  ```python
  checkpoint_dir = os.path.expanduser("~/.cache/nanochat/base_checkpoints/d12_v1")
  model_data, _, meta_data = load_checkpoint(checkpoint_dir, 15250, ...)
  ```
- `convert_to_hf.py:110-111`: same `d12_v1` / step `15250`.

`claude.md` states **"Checkpoints: None surviving (lost during WSL→native Linux migration)"** and every committed checkpoint/meta is a **d22** model (`base_checkpoints/.../meta_0*.json` → `n_layer: 22`), trained under tags like `governed_v4_d22_r30_parallel_family`. The `d12_v1` step-15250 checkpoint these two scripts hardcode does not exist in this tree, so both scripts fail out-of-the-box. Low security impact, but they're dead entry points referencing a model generation that was superseded.

*(I checked the related concern that `serve.py` doesn't strip the `_orig_mod.` torch.compile prefix that `build_model` strips — this is NOT a bug: `base_train.py:592` saves `orig_model.state_dict()` (uncompiled), so saved checkpoints carry no prefix.)*

---

## P3 — Minor doc/artifact inconsistencies

- `tokenizer/tokenizer.pkl` **is committed** (`git ls-files` confirms), yet `tools/recover_tokenizer_artifacts.py:3-4` states the pickle is "intentionally NOT produced," and `SECURITY.md §4`'s mitigation is literally "delete `tokenizer.pkl` … `get_tokenizer()` will then take the JSON branch." Since `tokenizer.py:422` checks `tokenizer.pkl` **first**, the shipped default still favors the pickle path the docs tell you to avoid. (Practically low-risk because `get_tokenizer()` reads from `NANOCHAT_BASE_DIR`, not the repo's `tokenizer/` dir — but it's a claim/artifact contradiction worth resolving.)
- Compiled `__pycache__/*.pyc` files are committed (`serve.cpython-312.pyc`, etc.) — noise, should be gitignored.

---

## Confirmed-accurate `SECURITY.md` claims (verified against code, no gap)

These I actively tried to break and could not — the documentation is honest:
- **§1 `execution.py` P0** — `reliability_guard()` (`execution.py:134-211`) is monkey-patch-only; `exec(code, exec_globals)` at line 254 with builtins intact. Documented as non-sandbox / WONT-FIX(upstream). Accurate.
- **§2 `engine.py` `eval()` P1** — `use_calculator` (`engine.py:47-80`) char-allowlist + blacklist (`__`, `import`, etc.) + required `.count(`. The classic `__subclasses__` payload is blocked (`__` rejected, `[`/`]` not in allowed set). No working bypass found through the normal path. Accurately rated P1 (brittle blacklist around `eval`).
- **§3 checkpoint `torch.load` P1** — `checkpoint_manager.py:45,50`, `convert_to_hf.py:20`. Accurate.
- **§5 `report.py shell=True` P2** — `run_command` (`report.py:18`) call sites (`report.py:31-39,169,175`) are all hardcoded git strings; `git_patterns` (`report.py:168`) is built from a hardcoded extension list `['py','md','rs','html','toml','sh']`. No untrusted source flows in. Claim of "no current injection source" is accurate.

---

### Recommended priority order
1. Document `trust_remote_code=True` in `SECURITY.md` and pin/flag-gate it (P1).
2. Scrub `***REMOVED***` and the secret-gist URL from committed files + history (P2 privacy).
3. Default `serve.py` to `127.0.0.1` and guard `request.json` (P2).
4. Fix or remove the dead `d12_v1` checkpoint references in `serve.py` / `convert_to_hf.py` (P2/P3).
