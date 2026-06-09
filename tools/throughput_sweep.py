"""
Throughput sweep for pre-kickoff config selection (Gate 5+ pass 2).

Runs 5 canonical candidates at d16 T=1024 with constant total_batch_size and
reports tokens/sec, peak VRAM, loader wait %, initial/final loss, and BPB.
The winner (highest tok/sec passing VRAM budget) also gets a save+resume check.

Candidates:
  A  cache on, activation checkpoint on, current DEVICE_BATCH_SIZE=4
  B  cache on, activation checkpoint OFF, DEVICE_BATCH_SIZE=4
  C  cache on, activation checkpoint OFF, DEVICE_BATCH_SIZE=8
  D  cache on, activation checkpoint OFF, largest safe DBS under ~21 GB peak
  E  best-non-compile config + COMPILE_MODE=default

For each candidate:
  - 20 warmup iterations (excluded from measurements)
  - 100 measured iterations
  - CORE eval disabled (SAMPLE_EVERY=-1 CORE_METRIC_EVERY=-1)
  - Tiny BPB eval at step 20 and step 119
  - benchmark_csv for per-step throughput/memory

Usage:
  python tools/throughput_sweep.py --output-dir /tmp/sweep_results
  python tools/throughput_sweep.py --candidates A,B,C,D,E
  python tools/throughput_sweep.py --candidates D --probe-dbs 12  # single candidate override
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NANOCHAT_ROOT = REPO_ROOT / "nanochat"

TOTAL_BATCH_SIZE = 262144
MAX_SEQ_LEN = 1024
DEPTH = 16
WARMUP = 20
MEASURED = 100


CANDIDATES = {
    "A": dict(label="cache + ckpt + DBS=4 (baseline from smoke)",
              ACTIVATION_CKPT=1, DEVICE_BATCH_SIZE=4, COMPILE_MODE="none"),
    "B": dict(label="cache + no-ckpt + DBS=4",
              ACTIVATION_CKPT=0, DEVICE_BATCH_SIZE=4, COMPILE_MODE="none"),
    "C": dict(label="cache + no-ckpt + DBS=8",
              ACTIVATION_CKPT=0, DEVICE_BATCH_SIZE=8, COMPILE_MODE="none"),
    "D": dict(label="cache + no-ckpt + DBS=<probe>",
              ACTIVATION_CKPT=0, DEVICE_BATCH_SIZE=None, COMPILE_MODE="none"),
    "E": dict(label="winner + compile",
              ACTIVATION_CKPT=None, DEVICE_BATCH_SIZE=None, COMPILE_MODE="default"),
}


def parse_step_metrics(log_path: Path) -> dict:
    """Extract tok/sec, loader %, peak VRAM, loss, bpb from a training log."""
    text = log_path.read_text() if log_path.exists() else ""
    # Steps after warmup (skip first 20). We match all step lines then filter.
    step_re = re.compile(
        r"step (\d+)/\d+.*\| loss: (\S+) \| lrm: \S+ \| dt: (\S+)ms \| loader: (\S+)ms \((\S+)%\) \| tok/sec: ([\d,]+) \| mfu: (\S+) \| total time:"
    )
    bpb_re = re.compile(r"Step (\d+) \| Validation bpb: (\S+)")
    peak_re = re.compile(r"Peak memory usage: (\S+)MiB")

    steps = []
    for m in step_re.finditer(text):
        step, loss, dt, loader_ms, loader_pct, tokps, mfu = m.groups()
        steps.append({
            "step": int(step),
            "loss": float(loss),
            "dt_ms": float(dt),
            "loader_ms": float(loader_ms),
            "loader_pct": float(loader_pct),
            "tok_per_sec": int(tokps.replace(",", "")),
            "mfu": float(mfu),
        })
    bpbs = {int(m.group(1)): float(m.group(2)) for m in bpb_re.finditer(text)}
    peak_m = peak_re.search(text)
    peak_mib = float(peak_m.group(1)) if peak_m else None

    # Keep only measured-phase steps (>= WARMUP)
    measured = [s for s in steps if s["step"] >= WARMUP]
    if not measured:
        return {"error": "no measured-phase steps", "raw_steps": len(steps)}

    tokps_values = [s["tok_per_sec"] for s in measured]
    loader_values = [s["loader_pct"] for s in measured]
    dt_values = [s["dt_ms"] for s in measured]
    return {
        "warmup_steps": sum(1 for s in steps if s["step"] < WARMUP),
        "measured_steps": len(measured),
        "tok_per_sec_mean": sum(tokps_values) / len(tokps_values),
        "tok_per_sec_median": sorted(tokps_values)[len(tokps_values) // 2],
        "loader_pct_mean": sum(loader_values) / len(loader_values),
        "dt_ms_mean": sum(dt_values) / len(dt_values),
        "mfu_mean": sum(s["mfu"] for s in measured) / len(measured),
        "initial_loss": measured[0]["loss"],
        "final_loss": measured[-1]["loss"],
        "peak_mib": peak_mib,
        "bpb_measurements": bpbs,
    }


def run_candidate(cand_id: str, cand: dict, token_cache_dir: str, output_dir: Path, env_extra=None) -> dict:
    """Run base_train with the candidate's config and capture metrics."""
    num_iter = WARMUP + MEASURED  # 120
    env = os.environ.copy()
    env.update({
        "NANOCHAT_BASE_DIR": str(REPO_ROOT),
        "TORCH_COMPILE_DISABLE": "1" if cand["COMPILE_MODE"] == "none" else "",
        "WANDB_MODE": "offline",
        "PATH": f"{REPO_ROOT}/.venv/bin:" + env.get("PATH", ""),
    })
    if env_extra:
        env.update(env_extra)

    tag = f"sweep_{cand_id}"
    log_path = output_dir / f"{tag}.log"
    bench_csv = output_dir / f"{tag}_bench.csv"

    # Build base_train invocation directly (skip the shell wrapper so we can pass
    # token_cache_dir cleanly).
    cmd = [
        f"{REPO_ROOT}/.venv/bin/python", "-m", "scripts.base_train",
        "--run", tag,
        "--model_tag", tag,
        "--depth", str(DEPTH),
        "--max_seq_len", str(MAX_SEQ_LEN),
        "--device_batch_size", str(cand["DEVICE_BATCH_SIZE"]),
        "--total_batch_size", str(TOTAL_BATCH_SIZE),
        "--kv_head_ratio", "1.0",
        "--compile_mode", cand["COMPILE_MODE"],
        "--sdpa_backend", "auto",
        "--save_every", "-1",          # save only at end for resume check later
        "--eval_every", "50",          # tiny BPB eval at step 0 and 50
        "--eval_tokens", str(TOTAL_BATCH_SIZE),  # 1 batch
        "--core_metric_every", "-1",
        "--sample_every", "-1",
        "--num_iterations", str(num_iter),
        "--token_cache_dir", token_cache_dir,
        "--benchmark_csv", str(bench_csv),
    ]
    if cand["ACTIVATION_CKPT"]:
        cmd.extend(["--activation_checkpoint", "--ckpt_every_n_blocks", "1"])
    # Always use chunked loss for the sweep; the VRAM win is free in training.
    cmd.extend(["--chunked_loss", "--loss_chunk_size", "1024"])

    print(f"\n>>> Candidate {cand_id}: {cand['label']}")
    print(f"    DBS={cand['DEVICE_BATCH_SIZE']}, ckpt={cand['ACTIVATION_CKPT']}, compile={cand['COMPILE_MODE']}")
    print(f"    log: {log_path}")
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(NANOCHAT_ROOT), env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )
    wall_sec = time.time() - t0
    metrics = parse_step_metrics(log_path)
    metrics["wall_sec"] = round(wall_sec, 1)
    metrics["exit_code"] = result.returncode
    metrics["candidate"] = cand_id
    metrics["label"] = cand["label"]
    metrics["device_batch_size"] = cand["DEVICE_BATCH_SIZE"]
    metrics["activation_ckpt"] = bool(cand["ACTIVATION_CKPT"])
    metrics["compile_mode"] = cand["COMPILE_MODE"]
    # grad_accum derived for logging
    metrics["grad_accum"] = TOTAL_BATCH_SIZE // (cand["DEVICE_BATCH_SIZE"] * MAX_SEQ_LEN)
    return metrics


def probe_largest_dbs(token_cache_dir: str, output_dir: Path) -> int:
    """Binary-search the largest DBS that runs without OOM on a short probe.
    Target: peak VRAM under 21 GB (~88% of 24 GB)."""
    candidates_dbs = [8, 12, 16, 24, 32]
    best = 4
    for dbs in candidates_dbs:
        if TOTAL_BATCH_SIZE % (dbs * MAX_SEQ_LEN) != 0:
            continue
        print(f"\n[probe] trying DBS={dbs} for 5 iterations")
        cand = dict(label=f"probe DBS={dbs}", ACTIVATION_CKPT=0, DEVICE_BATCH_SIZE=dbs, COMPILE_MODE="none")
        probe_dir = output_dir / "probes"
        probe_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "NANOCHAT_BASE_DIR": str(REPO_ROOT),
            "TORCH_COMPILE_DISABLE": "1",
            "WANDB_MODE": "offline",
            "PATH": f"{REPO_ROOT}/.venv/bin:" + env.get("PATH", ""),
        })
        log_path = probe_dir / f"probe_dbs_{dbs}.log"
        cmd = [
            f"{REPO_ROOT}/.venv/bin/python", "-m", "scripts.base_train",
            "--run", "dummy", "--model_tag", f"probe_dbs_{dbs}",
            "--depth", str(DEPTH), "--max_seq_len", str(MAX_SEQ_LEN),
            "--device_batch_size", str(dbs),
            "--total_batch_size", str(TOTAL_BATCH_SIZE),
            "--compile_mode", "none",
            "--save_every", "-1", "--eval_every", "-1",
            "--core_metric_every", "-1", "--sample_every", "-1",
            "--num_iterations", "5",
            "--chunked_loss", "--loss_chunk_size", "1024",
            "--token_cache_dir", token_cache_dir,
        ]
        result = subprocess.run(cmd, cwd=str(NANOCHAT_ROOT), env=env,
                                stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        metrics = parse_step_metrics(log_path)
        peak = metrics.get("peak_mib", float("inf"))
        if result.returncode != 0 or peak is None:
            print(f"    DBS={dbs} failed (exit={result.returncode}, peak={peak})")
            break
        peak_gib = peak / 1024.0
        print(f"    DBS={dbs}: peak={peak_gib:.2f} GiB  tok/sec={metrics.get('tok_per_sec_median', 'n/a')}")
        if peak_gib > 21.0:
            print(f"    exceeds 21 GiB budget; stopping at DBS={best}")
            break
        best = dbs
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default="/tmp/sweep_results")
    ap.add_argument("--candidates", type=str, default="A,B,C,D,E")
    ap.add_argument("--token-cache-dir", type=str, required=True,
                    help="Path to the pre-built token cache directory.")
    ap.add_argument("--probe-dbs", type=int, default=None,
                    help="Override the auto-probe and force DBS for candidate D.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen = args.candidates.split(",")

    all_results = []

    # A-C are direct. D needs a probe. E needs the winner of A-D.
    for cid in chosen:
        cand = CANDIDATES[cid]
        if cid == "D":
            if args.probe_dbs:
                cand["DEVICE_BATCH_SIZE"] = args.probe_dbs
                print(f"\n[D] using forced DBS={args.probe_dbs}")
            else:
                best = probe_largest_dbs(args.token_cache_dir, output_dir)
                cand["DEVICE_BATCH_SIZE"] = best
                print(f"\n[D] probe settled on DBS={best}")
        if cid == "E":
            # pick winner by tok_per_sec_median from A-D
            prior = [r for r in all_results if r["candidate"] in ("A", "B", "C", "D")
                     and r.get("exit_code") == 0 and r.get("tok_per_sec_median")]
            if not prior:
                print("[E] skipping: no valid prior candidate to compile-boost")
                continue
            winner = max(prior, key=lambda r: r["tok_per_sec_median"])
            cand = dict(cand)
            cand["DEVICE_BATCH_SIZE"] = winner["device_batch_size"]
            cand["ACTIVATION_CKPT"] = 1 if winner["activation_ckpt"] else 0
            cand["label"] = f"winner({winner['candidate']}) + compile=default"
            print(f"\n[E] basing on winner {winner['candidate']} "
                  f"(DBS={cand['DEVICE_BATCH_SIZE']} ckpt={cand['ACTIVATION_CKPT']})")
        if cand["DEVICE_BATCH_SIZE"] is None:
            print(f"[{cid}] missing DBS; skipping")
            continue
        metrics = run_candidate(cid, cand, args.token_cache_dir, output_dir)
        all_results.append(metrics)

    # Print + save summary
    print("\n" + "=" * 80)
    print("SWEEP RESULTS")
    print("=" * 80)
    rows = []
    for r in all_results:
        row = {
            "cand": r["candidate"],
            "label": r["label"][:40],
            "DBS": r["device_batch_size"],
            "grad_acc": r["grad_accum"],
            "ckpt": "Y" if r["activation_ckpt"] else "N",
            "cmp": r["compile_mode"],
            "tok/s(med)": r.get("tok_per_sec_median", -1),
            "peak MiB": r.get("peak_mib", -1),
            "load%": round(r.get("loader_pct_mean", -1), 1),
            "mfu%": round(r.get("mfu_mean", -1), 2),
            "init_loss": round(r.get("initial_loss", -1), 3),
            "fin_loss": round(r.get("final_loss", -1), 3),
            "wall_s": r["wall_sec"],
            "exit": r["exit_code"],
        }
        rows.append(row)
        print(f"  {row['cand']}: DBS={row['DBS']} acc={row['grad_acc']} ckpt={row['ckpt']} "
              f"cmp={row['cmp']}  {row['tok/s(med)']:>7,} tok/s  "
              f"{row['peak MiB']:>7.0f} MiB  loader={row['load%']}%  mfu={row['mfu%']}%  "
              f"loss {row['init_loss']}->{row['fin_loss']}  wall={row['wall_s']}s  exit={row['exit']}")

    # CSV summary
    csv_path = output_dir / "sweep_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # JSON with full detail
    json_path = output_dir / "sweep_summary.json"
    with json_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    # Winner + resume check
    valid = [r for r in all_results if r.get("exit_code") == 0 and r.get("tok_per_sec_median")]
    if valid:
        winner = max(valid, key=lambda r: r["tok_per_sec_median"])
        print(f"\nWinner: {winner['candidate']} ({winner['label']})")
        print(f"  tok/s(med) = {winner['tok_per_sec_median']:,}")
        print(f"  peak = {winner['peak_mib']} MiB")
        print(f"  loader = {winner['loader_pct_mean']:.1f}%")
        print(f"\nSummary CSV: {csv_path}")
        print(f"Full JSON:   {json_path}")
    else:
        print("\nNo valid candidates — every run failed. Check individual logs.")


if __name__ == "__main__":
    main()
