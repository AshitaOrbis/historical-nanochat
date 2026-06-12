# Security Model and Threat Boundaries

This document states the security posture of `historical-nanochat` honestly. The
repository is a **research training pipeline** built on top of
[Karpathy's nanochat](https://github.com/karpathy/nanochat). It is designed to be
run by a single researcher on their own machine over **data and artifacts they
control**. It is **not** hardened to safely process adversarial inputs, and
several components inherit known-unsafe patterns directly from upstream nanochat
and from the OpenAI HumanEval evaluation harness.

If you are evaluating this code for a multi-tenant, networked, or
untrusted-input deployment: **don't, without adding the isolation described
below.** The sections that follow describe each issue, whether it can be fixed
without changing what the tool is, and what we did about it.

---

## TL;DR trust assumptions

The pipeline assumes **all of the following are trusted**:

1. **The model-generated code** that the HumanEval task executes (`execute_code`).
2. **Every checkpoint** (`*.pt`) you load — your own or anyone else's.
3. **Every tokenizer artifact** (`tokenizer.pkl`, `token_bytes.pt`) on disk.
4. **The local filesystem and `NANOCHAT_BASE_DIR`** you point the tools at.
5. **Any HuggingFace model/tokenizer the probe harness loads** — `probes/harness.py`
   uses `trust_remote_code=True`, which executes remote repo code at load time.

Violate any of these assumptions and you should expect **arbitrary code
execution on the host, under your UID**. These are inherent properties of the
upstream design, not bugs we can patch away without becoming a different
project. They are documented here rather than papered over with a fake fix.

---

## 1. `execution.py` is NOT a security sandbox (host RCE on untrusted code)

**Severity: P0 — Critical. Disposition: INHERENT + DOCUMENTED / WONT-FIX(upstream).**

`nanochat/nanochat/execution.py` runs model-generated Python (for the HumanEval
coding benchmark) in a subprocess that monkey-patches a handful of dangerous
functions (`os.system`, `os.kill`, `subprocess.Popen`, `shutil.rmtree`, …) and
sets a timeout + memory limit. This is **accident reduction, not adversarial
isolation.** The module's own docstring says so, and the OpenAI HumanEval code it
was adapted from says so.

It is bypassable in multiple independent ways, including:

- **`importlib.reload`** repopulates a patched module from the real
  implementation:
  ```python
  import importlib, os
  importlib.reload(os)
  os.system("id")          # the None-assignment is gone
  ```
- **Unpatched spawn paths**, e.g. `os.posix_spawn(...)`, which the guard never
  touches.
- **`ctypes`** native calls (`ctypes.CDLL(None).system(b"id")`), explicitly
  called out as out-of-scope by the docstring.
- **Builtins are not removed** — `exec(code, {})` still injects `__builtins__`,
  so `import` and the full standard library remain reachable.
- **Network egress is not blocked** (sockets work).

### Why we did not "fix" it in code

`execution.py` is upstream nanochat / HumanEval code (the blob is shared with
`karpathy/nanochat`). Re-implementing a real sandbox in pure Python is not
possible — Python-level filtering is defense-in-depth, never a boundary. The
honest fix is operational, not a code patch:

**If you run the HumanEval task on a model you do not fully trust, run the whole
process inside a real isolation boundary:**

- An unprivileged container or microVM (gVisor, Firecracker, Docker with a
  locked-down profile).
- Read-only root filesystem, `tmpfs` workdir, **no host secrets mounted**.
- **No network** by default.
- `seccomp` / AppArmor / SELinux confinement.
- cgroup limits for CPU / memory / PIDs, scrubbed environment, closed inherited
  FDs, a killable process tree.

The in-process timeout, memory cap, and temp dir reduce *accidental* damage from
a benign-but-buggy model. They do **nothing** against malicious code. Treat
`execute_code` as "runs arbitrary code on this host."

---

## 2. `engine.py` calculator uses `eval()` (footgun, no confirmed live bypass)

**Severity: P1. Disposition: INHERENT + DOCUMENTED (mitigation present upstream).**

`nanochat/nanochat/engine.py` implements a tool-use "calculator" that ultimately
calls:

```python
eval(formula, {"__builtins__": {}}, {})
```

`use_calculator()` prefilters input: math mode allows only digits/arithmetic
punctuation and rejects `**`; string mode allows a restricted character set,
rejects dangerous substrings (`__`, `import`, `exec`, `eval`, `open`,
`globals`, `getattr`, …), and requires a `.count(` call. Through the normal
`Engine.generate → use_calculator → eval_with_timeout` path we found **no
working bypass** in the current filters (the classic
`().__class__.__base__.__subclasses__()` payload is blocked: `__` is rejected and
`[`/`]` are not in the allowed set).

**It remains a P1 because the security boundary is a brittle blacklist wrapped
around a known-dangerous primitive (`eval`).** Any of the following turns it into
expression-level code execution:

- A future "small" expansion of allowed string operations.
- A new caller that invokes `eval_with_timeout()` directly, skipping the filter.
- A missing prefilter in another code path.

**Hardening (recommended if you extend this):** replace `eval_with_timeout` with
a tiny AST interpreter — `ast.parse(expr, mode="eval")` plus an allowlist of
node types (`Constant`, `BinOp` with +−×÷, `UnaryOp`, and an `Attribute` call to
`str.count` only). Reject every other node. We left the upstream code in place
rather than diverge from nanochat; if you depend on this calculator with
untrusted model output, do the AST swap.

---

## 3. Checkpoint loading is unsafe deserialization (`torch.load` pickle RCE)

**Severity: P1. Disposition: INHERENT + DOCUMENTED.**

`nanochat/nanochat/checkpoint_manager.py` (`load_checkpoint`) and the
`convert_to_hf.py` helper load model/optimizer state with:

```python
torch.load(model_path, map_location=device)
```

`torch.load` unpickles by default. A malicious `.pt` file executes arbitrary
code at load time via `__reduce__`:

```python
class Pwn:
    def __reduce__(self):
        return (os.system, ("id",))
torch.save(Pwn(), "model_000001.pt")   # loading this runs `id`
```

In the normal "load only your own checkpoints" workflow this is not exploitable.
It becomes a **practical supply-chain RCE** the moment you load a checkpoint you
downloaded or were sent — which ML practitioners routinely do.

**Why not just add `weights_only=True`?** It is the right direction, but nanochat
checkpoints include optimizer state (Muon/AdamW) and resume metadata that are not
guaranteed to be plain tensor dicts; flipping `weights_only=True` unconditionally
can break `--resume_from_step`. Rather than ship a change that silently breaks
training resume, we document the boundary:

**Only load checkpoints you produced or fully trust.** If you must ingest a
third-party checkpoint:

- Prefer `safetensors` for weights.
- If you stay on `torch.load`, load with `weights_only=True` on a PyTorch
  version that supports it, and schema-check that the result is exactly
  `dict[str, torch.Tensor]` with expected keys/shapes/dtypes.
- Verify a SHA-256 / signature manifest before loading.
- Make untrusted loading an explicit, opt-in flag.

---

## 4. Tokenizer artifacts are unsafe deserialization (`pickle.load` / `torch.load`)

**Severity: P1. Disposition: MITIGATED (JSON fallback exists) + DOCUMENTED.**

`nanochat/nanochat/tokenizer.py` loads `tokenizer.pkl` with `pickle.load` and
`token_bytes.pt` with `torch.load`. `get_tokenizer()` **prefers the pickle**
(`tokenizer.pkl`) for speed and falls back to `tokenizer.json` only if the pickle
is absent. `pickle.load` on attacker-controlled data is arbitrary code execution,
identical in shape to the checkpoint issue above. The artifact directory is
redirectable via `NANOCHAT_BASE_DIR`, widening the attack surface to "anyone who
can write that directory."

**Important — what this repo actually ships:** the active `tokenizer/` directory
in this repository currently contains **only `tokenizer.pkl`** (plus
`tokenizer_manifest.json`), **not** a `tokenizer.json`. Because
`get_tokenizer()` (`nanochat/nanochat/tokenizer.py`) checks for `tokenizer.pkl`
**first**, the default load path for this repo IS the pickle path. The
`tokenizer.pkl` shipped here is the author's own artifact and is trusted; but if
you redirect `NANOCHAT_BASE_DIR` at a directory you do not control, or replace
this pickle, you are loading attacker-controlled pickle data.

**Mitigation:** this fork's loader supports a safe `tokenizer.json` (HuggingFace)
format (`from_directory` in `tokenizer.py` reads `tokenizer.json`). To use the
safe branch, **generate a `tokenizer.json` and remove `tokenizer.pkl`** from the
tokenizer directory you load; `get_tokenizer()` will then fall back to the JSON
branch. Do not point `NANOCHAT_BASE_DIR` at an untrusted or shared directory.
Store `token_bytes` as `.npy` / raw binary with a shape+dtype manifest, or load
with `weights_only=True`.

**Boundary:** treat the tokenizer directory and `NANOCHAT_BASE_DIR` as trusted,
writable only by you. Do not point them at a shared or attacker-influenced
location.

---

## 5. `report.py` uses `shell=True` (no current injection source)

**Severity: P2. Disposition: INHERENT + DOCUMENTED / hardening deferred.**

`nanochat/nanochat/report.py` runs `subprocess.run(cmd, shell=True, …)` via
`run_command()`. All current call sites pass **fixed, hardcoded git command
strings** (`git rev-parse …`, `git status --porcelain`, etc.) or strings built
from a hardcoded extension list. We found **no model-controlled, user-controlled,
or repo-metadata source flowing into `run_command()`**, so this is not
exploitable command injection today.

**Residual risk:** `run_command` is a footgun — the next caller that passes
untrusted text gets shell injection, and it relies on `PATH` lookup for
`git`/`xargs`/`wc` (PATH-hijack in a hostile environment). If you extend
`report.py`, convert `run_command` to take a `list[str]` argv and use
`shell=False`.

---

## 6. `probes/harness.py` loads HuggingFace models with `trust_remote_code=True`

**Severity: P1. Disposition: INHERENT + DOCUMENTED.**

`probes/harness.py` (`GPTQModel`, lines ~149 and ~153) loads third-party
HuggingFace tokenizers and models with `trust_remote_code=True`:

```python
self.tok = AutoTokenizer.from_pretrained(tokenizer_id or hf_id, trust_remote_code=True)
self.model = AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True, ...)
```

`probes/run_pilot.py` (~lines 166–170) hardcodes the model
`dtestnyrr/talkie-1930-13b-base-gptq-int4` and tokenizer
`xlr8harder/talkie-1930-13b-base-tf` as the "modern anchor."

`trust_remote_code=True` **downloads and executes arbitrary Python**
(`modeling_*.py` / `tokenization_*.py`) from those third-party repos at load time.
Running `probes/run_pilot.py` triggers this automatically. This is a
supply-chain code-execution path on the host, identical in impact to the
checkpoint/pickle issues above, but sourced from remote HuggingFace repos rather
than local files.

**Boundary / mitigation:** only run the probe harness against models you trust.
These specific repos are community conversions the author selected deliberately;
if you depend on them, **pin the model revision to a specific commit hash**
(`revision="<sha>"` in `from_pretrained`) so the executed code cannot change
under you, and consider gating `trust_remote_code` behind an explicit opt-in
flag. Do not run `run_pilot.py` with arbitrary or untrusted `hf_id` values.

---

## 7. Data-pipeline scripts — path / resource / network hardening gaps

**Severity: P2. Disposition: DOCUMENTED — local-user tooling.**

The data download/processing scripts use Python APIs (not `shell=True`) and have
no obvious shell-injection sink. They do accept arbitrary local `--output-dir` /
`--input` paths and write fixed filenames there, materialize inputs in memory,
and (for Chronicling America) follow `ocr_url` values from the LOC API with
`requests.get`. None of this is RCE, but if run **as a privileged service or
cron job over attacker-controlled paths** it enables disk-fill, symlink-clobber,
and SSRF-style fetches.

**Boundary:** treat the data pipeline as **local-user tooling**. Do not run it as
root, over untrusted output directories, or against an untrusted network without
adding: workspace-confined paths, symlink rejection / no-follow opens, max
input/document size limits, streaming instead of full materialization, and
allowlisted fetch hosts/schemes.

---

## Reporting

This is a research repository with no production deployment and no formal
disclosure process. If you find an issue, open a GitHub issue. Do **not** assume
any component is safe against adversarial input unless this document says it is —
and it says it is not.
