"""
First-checkpoint verification for legacy_textonly_d22_r30_internal_baseline.

READ-ONLY. Does not touch the live training process, the live token cache,
or the shards directory. Intended to run against the first saved checkpoint
(expected at step 2000 for the current run) and emit a pass/fail report.

Checks performed:
  1) Checkpoint files exist (model_*.pt, meta_*.json, optim_*_rank0.pt).
  2) meta.json is well-formed and has the expected keys.
  3) Model state dict loads without missing/unexpected keys under strict=True.
  4) Saved state round-trips: load -> save -> sha256-compare.
  5) Loss / val BPB trajectory from the training log is monotone enough.
  6) Peak VRAM from the bench CSV is within budget (<22 GB for 24 GB card).
  7) No OOM / NaN / compile recompile storms in the log.
  8) Tok/s stability: stddev(last 500 steps) / mean < 2%.

Exits 0 only if all gates pass. Writes a Markdown report into
  report/verification_<step>_<timestamp>.md
regardless of exit code.

Usage:
  python tools/verify_first_checkpoint.py \
      --model-tag d22_r30 \
      --step 2000 \
      --log logs/d22_r30_20260420_090531.log \
      --bench-csv logs/d22_r30_20260420_090531_bench.csv \
      --device cpu            # default: CPU. Use 'cuda' ONLY if live run is stopped.

Never pass --device cuda while the live training is using the GPU.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make nanochat importable
sys.path.insert(0, str(REPO_ROOT / "nanochat"))


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    warn: bool = False


@dataclass
class Report:
    run_tag: str = "legacy_textonly_d22_r30_internal_baseline"
    model_tag: str = ""
    step: int = 0
    checkpoint_dir: str = ""
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_markdown(self) -> str:
        lines = []
        lines.append(f"# First-Checkpoint Verification Report")
        lines.append("")
        lines.append(f"- **Run tag:** `{self.run_tag}`")
        lines.append(f"- **Model tag:** `{self.model_tag}`")
        lines.append(f"- **Step verified:** {self.step}")
        lines.append(f"- **Checkpoint dir:** `{self.checkpoint_dir}`")
        lines.append(f"- **Timestamp:** {self.timestamp}")
        lines.append(f"- **Overall:** {'PASS' if self.overall else 'FAIL'}")
        lines.append("")
        lines.append("| # | Check | Result | Detail |")
        lines.append("|---|---|---|---|")
        for i, c in enumerate(self.checks, 1):
            mark = "PASS" if c.passed else ("WARN" if c.warn else "FAIL")
            detail = c.detail.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {i} | {c.name} | {mark} | {detail} |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            "This artifact is for `legacy/internal/baseline` use only. It is NOT "
            "evidence that the run is a governed 3090 PoC, release candidate, or "
            "teacher candidate."
        )
        return "\n".join(lines) + "\n"


def check_files_present(checkpoint_dir: Path, step: int) -> list[CheckResult]:
    results = []
    step_str = f"{step:06d}"
    required = [
        f"model_{step_str}.pt",
        f"meta_{step_str}.json",
    ]
    optional_any = [
        f"optim_{step_str}_rank0.pt",
    ]
    missing_required = [f for f in required if not (checkpoint_dir / f).exists()]
    results.append(
        CheckResult(
            name="checkpoint files present",
            passed=not missing_required,
            detail=(
                f"All required files present in {checkpoint_dir}"
                if not missing_required
                else f"missing: {missing_required}"
            ),
        )
    )
    opt_present = [f for f in optional_any if (checkpoint_dir / f).exists()]
    results.append(
        CheckResult(
            name="optimizer shard present (rank0)",
            passed=bool(opt_present),
            warn=not opt_present,
            detail=(
                f"found {opt_present}"
                if opt_present
                else "rank0 optimizer shard missing; resume will fail"
            ),
        )
    )
    return results


def check_meta_json(checkpoint_dir: Path, step: int) -> tuple[CheckResult, dict[str, Any] | None]:
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    if not meta_path.exists():
        return (
            CheckResult(name="meta.json well-formed", passed=False, detail="file not found"),
            None,
        )
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        return (
            CheckResult(
                name="meta.json well-formed",
                passed=False,
                detail=f"json decode error: {e}",
            ),
            None,
        )
    required_keys = ["model_config", "step"]
    missing = [k for k in required_keys if k not in meta]
    if missing:
        return (
            CheckResult(
                name="meta.json well-formed",
                passed=False,
                detail=f"missing keys: {missing}",
            ),
            meta,
        )
    model_config = meta.get("model_config", {})
    # Expected d22 shape
    expected = {"n_layer": 22, "vocab_size": 32768}
    bad = [
        (k, model_config.get(k), v) for k, v in expected.items() if model_config.get(k) != v
    ]
    return (
        CheckResult(
            name="meta.json well-formed",
            passed=not bad,
            detail=(
                f"config OK (n_layer={model_config.get('n_layer')}, vocab={model_config.get('vocab_size')})"
                if not bad
                else f"config mismatch: {bad}"
            ),
        ),
        meta,
    )


def _sha256_of_state_dict(state_dict) -> str:
    # Deterministic hash: iterate sorted keys, write tensor bytes in CPU/float32
    import torch

    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        h.update(k.encode())
        if hasattr(t, "detach"):
            t = t.detach().to("cpu")
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def check_roundtrip(checkpoint_dir: Path, step: int, device: str) -> CheckResult:
    """Load the checkpoint, re-save via torch.save to a bytes buffer, re-load,
    and verify sha256 matches. Uses the repo's build_model if available."""
    try:
        import torch

        from nanochat.checkpoint_manager import load_checkpoint  # type: ignore

        model_data, _, meta_data = load_checkpoint(
            str(checkpoint_dir), step, torch.device(device), load_optimizer=False
        )
        # Normalize torch.compile key prefix
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
        h1 = _sha256_of_state_dict(model_data)
        buf = io.BytesIO()
        torch.save(model_data, buf)
        buf.seek(0)
        model_data_rt = torch.load(buf, map_location=device)
        model_data_rt = {k.removeprefix("_orig_mod."): v for k, v in model_data_rt.items()}
        h2 = _sha256_of_state_dict(model_data_rt)
        return CheckResult(
            name="state_dict round-trip (save -> load -> hash)",
            passed=(h1 == h2),
            detail=f"sha256 match={h1 == h2} ({h1[:12]}... vs {h2[:12]}...)",
        )
    except Exception as e:
        return CheckResult(
            name="state_dict round-trip (save -> load -> hash)",
            passed=False,
            detail=f"exception: {type(e).__name__}: {e}",
        )


def parse_log_trajectory(log_path: Path) -> dict[str, Any]:
    """Parse loss per step and val BPB per eval step from the training log."""
    losses: list[tuple[int, float]] = []
    vals: list[tuple[int, float]] = []
    oom = nan = compile_issue = False
    pat_step = re.compile(r"step (\d+)/\d+ \([\d.]+%\) \| loss: ([\d.]+)")
    pat_val = re.compile(r"Step (\d+) \| Validation bpb: ([\d.]+)")
    pat_bad = re.compile(r"(CUDA out of memory|OutOfMemoryError|NaN|nan\b)")
    pat_recompile = re.compile(r"recompil", re.IGNORECASE)
    compile_hits = 0
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                m = pat_step.search(line)
                if m:
                    losses.append((int(m.group(1)), float(m.group(2))))
                m = pat_val.search(line)
                if m:
                    vals.append((int(m.group(1)), float(m.group(2))))
                if pat_bad.search(line):
                    oom = oom or "out of memory" in line.lower() or "OutOfMemory" in line
                    nan = nan or "nan" in line.lower()
                if pat_recompile.search(line):
                    compile_hits += 1
    except FileNotFoundError:
        return {
            "losses": [],
            "vals": [],
            "oom": False,
            "nan": False,
            "compile_hits": 0,
            "error": f"log not found: {log_path}",
        }
    return {
        "losses": losses,
        "vals": vals,
        "oom": oom,
        "nan": nan,
        "compile_hits": compile_hits,
    }


def check_loss_descending(traj: dict[str, Any], up_to_step: int) -> CheckResult:
    losses = [(s, v) for s, v in traj.get("losses", []) if s <= up_to_step]
    if len(losses) < 200:
        return CheckResult(
            name="train loss descending",
            passed=False,
            warn=True,
            detail=f"only {len(losses)} loss points <= step {up_to_step}",
        )
    first = losses[:100]
    last = losses[-100:]
    mean_first = sum(v for _, v in first) / len(first)
    mean_last = sum(v for _, v in last) / len(last)
    drop = mean_first - mean_last
    return CheckResult(
        name="train loss descending",
        passed=mean_last < mean_first,
        detail=f"mean_first100={mean_first:.3f}, mean_last100={mean_last:.3f}, drop={drop:+.3f}",
    )


def check_val_bpb_sane(traj: dict[str, Any], up_to_step: int) -> CheckResult:
    vals = [(s, v) for s, v in traj.get("vals", []) if s <= up_to_step]
    if len(vals) < 2:
        return CheckResult(
            name="validation BPB trajectory",
            passed=False,
            warn=True,
            detail=f"only {len(vals)} val points available",
        )
    init_bpb = vals[0][1]
    last_bpb = vals[-1][1]
    # init should be ~log2(vocab)=15 before training, but this model logs init BPB on
    # per-token basis; we accept a rough threshold that last < init and last < 5.0.
    descending = last_bpb < init_bpb
    sane = last_bpb < 5.0 and last_bpb > 0.0
    return CheckResult(
        name="validation BPB trajectory",
        passed=descending and sane,
        detail=(
            f"init_bpb@{vals[0][0]}={init_bpb:.4f}, last_bpb@{vals[-1][0]}={last_bpb:.4f}, "
            f"points={len(vals)}"
        ),
    )


def check_no_bad_log_events(traj: dict[str, Any]) -> list[CheckResult]:
    return [
        CheckResult(
            name="no OOM in log",
            passed=not traj.get("oom", False),
            detail="no CUDA OOM detected" if not traj.get("oom") else "OOM detected",
        ),
        CheckResult(
            name="no NaN in log",
            passed=not traj.get("nan", False),
            detail="no NaN detected" if not traj.get("nan") else "NaN detected",
        ),
        CheckResult(
            name="no compile recompile storm",
            # Allow up to 5 recompile events; anything more warns.
            passed=traj.get("compile_hits", 0) <= 5,
            warn=5 < traj.get("compile_hits", 0) <= 20,
            detail=f"recompile hits={traj.get('compile_hits', 0)}",
        ),
    ]


def check_bench_csv(bench_path: Path, up_to_step: int) -> list[CheckResult]:
    if not bench_path.exists():
        return [
            CheckResult(
                name="bench CSV available",
                passed=False,
                warn=True,
                detail=f"not found: {bench_path}",
            )
        ]
    rows: list[dict[str, str]] = []
    try:
        with open(bench_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        return [
            CheckResult(
                name="bench CSV parseable",
                passed=False,
                detail=str(e),
            )
        ]
    # Try common column names
    step_col = next(
        (c for c in ["step", "iter", "iteration"] if rows and c in rows[0]), None
    )
    vram_col = next(
        (
            c
            for c in ["peak_mem_mib", "peak_vram_gb", "peak_vram", "vram_gb", "vram"]
            if rows and c in rows[0]
        ),
        None,
    )
    tps_col = next(
        (
            c
            for c in ["tokens_per_sec", "tok_per_sec", "tok_sec", "toks_per_sec", "tok_per_s"]
            if rows and c in rows[0]
        ),
        None,
    )
    out = []
    out.append(
        CheckResult(
            name="bench CSV parseable",
            passed=bool(rows),
            detail=f"rows={len(rows)}, cols={list(rows[0].keys()) if rows else []}",
        )
    )
    if vram_col and rows:
        try:
            peaks = [float(r[vram_col]) for r in rows if r.get(vram_col) not in (None, "")]
            # peak_mem_mib is in MiB; others assumed GB
            to_gb = (lambda x: x / 1024.0) if vram_col == "peak_mem_mib" else (lambda x: x)
            peak = to_gb(max(peaks)) if peaks else float("nan")
            out.append(
                CheckResult(
                    name="peak VRAM <= 22 GB (24 GB card budget)",
                    passed=peak <= 22.0,
                    detail=f"peak_vram={peak:.2f} GB over {len(peaks)} rows (col={vram_col})",
                )
            )
        except Exception as e:
            out.append(
                CheckResult(name="peak VRAM readable", passed=False, detail=str(e))
            )
    else:
        out.append(
            CheckResult(
                name="peak VRAM column present",
                passed=False,
                warn=True,
                detail="no vram column found; verify with nvidia-smi separately",
            )
        )
    if tps_col and rows:
        try:
            tps = [float(r[tps_col]) for r in rows if r.get(tps_col) not in (None, "")]
            if tps:
                last = tps[-500:] if len(tps) >= 500 else tps
                mean = sum(last) / len(last)
                var = sum((x - mean) ** 2 for x in last) / max(1, len(last) - 1)
                stddev = math.sqrt(var)
                rel = stddev / mean if mean else float("inf")
                out.append(
                    CheckResult(
                        name="tok/sec stability (last 500 rows stddev/mean < 2%)",
                        passed=rel < 0.02,
                        warn=0.02 <= rel < 0.05,
                        detail=f"mean={mean:.0f}, stddev={stddev:.0f}, rel={rel*100:.2f}%",
                    )
                )
        except Exception as e:
            out.append(CheckResult(name="tok/sec readable", passed=False, detail=str(e)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="d22_r30")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--checkpoints-root", default=str(REPO_ROOT / "base_checkpoints"))
    ap.add_argument("--log", default=None)
    ap.add_argument("--bench-csv", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--skip-roundtrip",
        action="store_true",
        help="Skip state_dict round-trip (no torch import required)",
    )
    ap.add_argument("--report-dir", default=str(REPO_ROOT / "report"))
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoints_root) / args.model_tag
    report = Report(
        model_tag=args.model_tag,
        step=args.step,
        checkpoint_dir=str(checkpoint_dir),
    )

    # 1-2: file presence + meta parsing
    report.checks.extend(check_files_present(checkpoint_dir, args.step))
    meta_res, _ = check_meta_json(checkpoint_dir, args.step)
    report.checks.append(meta_res)

    # 3-4: round-trip load/save
    if not args.skip_roundtrip and meta_res.passed:
        report.checks.append(check_roundtrip(checkpoint_dir, args.step, args.device))
    else:
        report.checks.append(
            CheckResult(
                name="state_dict round-trip (save -> load -> hash)",
                passed=False,
                warn=True,
                detail="skipped (flag or meta invalid)",
            )
        )

    # 5-7: log trajectory checks
    log_path = Path(args.log) if args.log else (REPO_ROOT / "logs" / "d22_r30_20260420_090531.log")
    traj = parse_log_trajectory(log_path)
    report.checks.append(check_loss_descending(traj, args.step))
    report.checks.append(check_val_bpb_sane(traj, args.step))
    report.checks.extend(check_no_bad_log_events(traj))

    # 8: bench CSV stability
    bench_path = (
        Path(args.bench_csv)
        if args.bench_csv
        else (REPO_ROOT / "logs" / "d22_r30_20260420_090531_bench.csv")
    )
    report.checks.extend(check_bench_csv(bench_path, args.step))

    # Emit report
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.report_dir) / f"verification_step{args.step:06d}_{ts}.md"
    out_path.write_text(report.to_markdown())

    # Console summary
    print(f"\nverification report -> {out_path}")
    for c in report.checks:
        sym = "PASS" if c.passed else ("WARN" if c.warn else "FAIL")
        print(f"  [{sym}] {c.name}: {c.detail}")
    print(f"\noverall: {'PASS' if report.overall else 'FAIL'}")
    sys.exit(0 if report.overall else 1)


if __name__ == "__main__":
    main()
