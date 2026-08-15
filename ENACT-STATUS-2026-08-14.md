# Review enactment status — 2026-08-14

Scope: `SOL-BRIEF-REVIEW-ENACT-2026-08-14.md`, enacted locally on branch
`sol/review-enact-20260814` from workspace `main` at
`830793725deded9ce1592a06a3ac3fa59ead593c`. No push or deployment was
performed.

## 1. Published-product boundary

Commit: `c07143417df0816e5869fedd779b316be97c6015` — `Restore the published historical data workflow`

- Replaced the workspace-only `data` symlink contract with regular, tracked
  downloader and processor source files under `data/download/` and
  `data/process/`.
- Narrowed the root wheel metadata from the nonexistent `scripts` package to
  the shipped `data` package and changed `.gitignore` so source remains tracked
  while generated corpora remain ignored.
- Added `tools/check_public_contract.py`, which rejects README module commands
  and wheel package declarations that lack regular files in the candidate tree.
- RED: `python3 tools/check_public_contract.py` on the untouched tree exited 1
  with 7 violations: five missing README modules and two missing wheel packages.
- GREEN: the same working-tree check passed; the seven previously failing
  historical workflow tests passed (`7 passed in 1.36s`).
- Clean-clone proof: `python3 tools/check_public_contract.py --git-ref HEAD`
  passed, and the check also passed inside a fresh `git archive HEAD`
  extraction at `/tmp/historical-nanochat-final-clone.iGlSzR`.
- Package proof: `uv build --offline` built both the sdist and wheel; wheel
  inspection showed all eight `data/download/*.py` and `data/process/*.py`
  source files.

## 2. Attest-while-skipping controls

Commit: `c752d5edcf94cd7ca667d7e2f96913edf6937403` — `Make training integrity guards fail closed`

- Canary records are keyed by `(after_yield, rank)`, with duplicate pairs,
  out-of-range ranks, incomplete per-yield rank sets, and foreign-rank checks
  rejected instead of silently passing.
- The cached loader rejects unknown dtypes; provenance must be a unique, exact
  manifest-index bijection and must bind to the manifest SHA-256.
- Family-loader resume state now binds loader strategy, exact schedule and
  family set, gradient-accumulation geometry, batch/sequence geometry, cursor
  domains, and cache-manifest SHA-256.
- Added the public `tools/build_cache_provenance.py` generator for valid
  manifest-hash-bound provenance.
- RED: eight focused cases failed against the old code (same-yield two-rank
  collision, three malformed canary-rank shapes, foreign-rank pass, unknown
  dtype, duplicate-provenance false coverage, and changed-schedule resume).
- GREEN: `pytest -q nanochat/tests/test_train_guards.py
  nanochat/tests/test_family_loader.py nanochat/tests/test_design_c.py` reported
  `45 passed, 3 skipped`; the skips explicitly require the private token-cache
  fixture, while all new regression cases use synthetic public fixtures.

## 3. Fail-closed scaling runner

Commit: `84f6f7fff0cca84304f3d158c5676783e6c5abc3` — `Abort scaling sweeps on failed training`

- Added `set -Eeuo pipefail`, preserved the trainer's pipeline exit code, and
  aborts before observation append on trainer failure.
- Requires numeric parameters/iterations/BPB and a `complete_*.json`
  checkpoint sentinel before appending a CSV observation.
- RED: the fake trainer exited 23, while the old wrapper exited 0, appended
  three observations, and printed `Scaling Runs Complete!`.
- GREEN: `pytest -q nanochat/tests/test_scaling_runner.py` reported `3 passed`:
  exit 23 propagates with no row/banner; success without a sentinel is rejected;
  valid metrics plus a sentinel append exactly one row. `bash -n
  run_scaling_3090.sh` passed.

## 4. CORS

Commit: `ff50b3d3ba7c6b0ce3c36d2a18aa2f262d12e640` — `Remove wildcard CORS from loopback chat`

- Removed `CORSMiddleware` and the wildcard origin/method/header grant from
  `nanochat/scripts/chat_web.py`; the same-origin UI needs no CORS grant.
- Preserved both loopback defaults: `chat_web.py --host` remains
  `127.0.0.1`, and `serve.py` remains `SERVE_HOST=127.0.0.1` by default.
- RED: the two-service regression check reported one pass (`serve.py`) and one
  failure (`chat_web.py`).
- GREEN: `pytest -q nanochat/tests/test_loopback_cors.py` reported `2 passed`;
  both service modules also passed `py_compile`.

## 5. Temporal-ignorance wording

Commit: `43029fc473bba07ceb7541e786fce2af571bcc43` — `Qualify temporal ignorance claims`

- Replaced the absolute ignorance guarantee with a corpus-filtering objective,
  an empirical measurement requirement, and explicit residual-contamination
  classes.
- Relabeled the cutoff table as target excluded knowledge that must be
  evaluated.
- RED: the public-claims test failed on the original absolute wording.
- GREEN: `pytest -q nanochat/tests/test_public_claims.py` reported `1 passed`;
  a direct search found none of the removed guarantee phrases in `README.md`.

## Verification hygiene and final checks

Commit: `0f0faaacbd7417f55ce2a212a8b2be0a6d933474` — `Isolate the curriculum smoke test process group`

- The full suite exposed pre-existing cross-test leakage of `RANK`,
  `LOCAL_RANK`, and `WORLD_SIZE` into the curriculum subprocess. The test now
  scrubs those variables and initializes the single-rank Gloo group required
  by Muon, matching the production collective contract.
- Full suite (outside the filesystem/network/socket sandbox): `89 passed, 3
  skipped, 14 warnings in 5.70s`. The three skips are the explicitly private
  token-cache fixture tests; warnings are upstream Torch JIT deprecations.
- `python -m compileall -q data nanochat/nanochat nanochat/scripts
  nanochat/tests tools serve.py`: exit 0.
- `git diff --check`: exit 0.
- Final wheel/sdist build: exit 0.

## Publication preparation and limits

- The external allowlist already contained the restored `data` paths and
  `tools/build_cache_provenance.py`; it was extended with this branch's public
  contract check and focused test files.
- Read-only allowlist audit against this isolated worktree: 233 entries, zero
  symlink targets. This resolves the named `data/download/__init__.py` symlink
  rejection for the branch.
- The same audit found three pre-existing allowlist entries from the separate,
  unmerged 2026-08-13 review branch that are absent here:
  `nanochat/nanochat/artifact_guard.py`,
  `nanochat/tests/test_artifact_contract.py`, and
  `tokenizer/token_bytes.npy`. They were not silently deleted or folded in
  because today's brief fences off the already-documented §3/§4 work. The
  orchestrator must reconcile that prior branch when it integrates this branch
  and before using the real publisher.
- The real `publish.sh historical-nanochat` dry run was not used as evidence:
  its configured source is the dirty main checkout, not this isolated branch,
  so it would test different bytes. No `--push`, push, deploy, or publication
  operation was run.

Goal achieved
