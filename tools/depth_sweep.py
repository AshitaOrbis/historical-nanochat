"""
Depth sweep at the winning config (E) for the real-run kickoff decision.

For each depth in {18, 20, 22, 24}, benchmark with the proven good base config
(cache on, compile=default, chunked_loss, no activation checkpoint, DBS=8).
On OOM or non-zero exit, fall back in order: DBS=4 no ckpt → DBS=4 + ckpt.
Record tok/s, peak VRAM, loss, loader wait, and derive wall-clock estimates
for ratio=20 (Chinchilla-optimal) and ratio=40 (2x Chinchilla).

Usage:
  python tools/depth_sweep.py --token-cache-dir /home/user/historical-nanochat/data/token_cache_v2 --output-dir /tmp/depth_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NANOCHAT_ROOT = REPO_ROOT / "nanochat"

TOTAL_BATCH_SIZE = 262144
MAX_SEQ_LEN = 1024
WARMUP = 20
MEASURED = 100
NUM_ITER = WARMUP + MEASURED  # 120


STEP_RE = re.compile(
    r"step (\d+)/\d+.*\| loss: (\S+) \| lrm: \S+ \| dt: (\S+)ms \| loader: (\S+)ms \((\S+)%\) \| tok/sec: ([\d,]+) \| mfu: (\S+) \| total time:"
)
BPB_RE = re.compile(r"Step (\d+) \| Validation bpb: (\S+)")
PEAK_RE = re.compile(r"Peak memory usage: (\S+)MiB")
PARAMS_RE = re.compile(r"Number of parameters: ([\d,]+) \(scaling: ([\d,]+)\)")
FLOPS_RE = re.compile(r"Estimated FLOPs per token: (\S+)")
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory|cudaErrorMemoryAllocation", re.IGNORECASE)


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text() if log_path.exists() else ""
    steps = []
    for m in STEP_RE.finditer(text):
        step, loss, dt, _, loader_pct, tokps, mfu = m.groups()
        steps.append({
            "step": int(step),
            "loss": float(loss),
            "dt_ms": float(dt),
            "loader_pct": float(loader_pct),
            "tok_per_sec": int(tokps.replace(",", "")),
            "mfu": float(mfu),
        })
    bpbs = {int(m.group(1)): float(m.group(2)) for m in BPB_RE.finditer(text)}
    peak_m = PEAK_RE.search(text)
    params_m = PARAMS_RE.search(text)
    flops_m = FLOPS_RE.search(text)

    measured = [s for s in steps if s["step"] >= WARMUP]
    result = {
        "raw_steps": len(steps),
        "measured_steps": len(measured),
        "peak_mib": float(peak_m.group(1)) if peak_m else None,
        "num_params": int(params_m.group(1).replace(",", "")) if params_m else None,
        "num_scaling_params": int(params_m.group(2).replace(",", "")) if params_m else None,
        "flops_per_token": float(flops_m.group(1)) if flops_m else None,
        "bpb_measurements": bpbs,
        "oom": bool(OOM_RE.search(text)),
    }
    if measured:
        toks = [s["tok_per_sec"] for s in measured]
        loaders = [s["loader_pct"] for s in measured]
        dts = [s["dt_ms"] for s in measured]
        mfus = [s["mfu"] for s in measured]
        result.update({
            "tok_per_sec_mean": sum(toks) / len(toks),
            "tok_per_sec_median": sorted(toks)[len(toks) // 2],
            "tok_per_sec_p10": sorted(toks)[max(0, int(len(toks) * 0.1))],
            "loader_pct_mean": sum(loaders) / len(loaders),
            "dt_ms_mean": sum(dts) / len(dts),
            "mfu_mean": sum(mfus) / len(mfus),
            "initial_loss": measured[0]["loss"],
            "final_loss": measured[-1]["loss"],
        })
    else:
        result.update({
            "tok_per_sec_mean": None, "tok_per_sec_median": None, "tok_per_sec_p10": None,
            "loader_pct_mean": None, "dt_ms_mean": None, "mfu_mean": None,
            "initial_loss": None, "final_loss": None,
        })
    return result


def run_config(tag: str, depth: int, dbs: int, ckpt: bool, compile_mode: str,
               token_cache_dir: str, output_dir: Path,
               resume_from: int | None = None,
               save_every: int = -1,
               eval_every: int = -1,
               num_iter: int = NUM_ITER) -> dict:
    log_path = output_dir / f"{tag}.log"
    bench_csv = output_dir / f"{tag}_bench.csv"
    env = os.environ.copy()
    env.update({
        "NANOCHAT_BASE_DIR": str(REPO_ROOT),
        "TORCH_COMPILE_DISABLE": "1" if compile_mode == "none" else "",
        "WANDB_MODE": "offline",
        "PATH": f"{REPO_ROOT}/.venv/bin:" + env.get("PATH", ""),
    })
    assert TOTAL_BATCH_SIZE % (dbs * MAX_SEQ_LEN) == 0, (
        f"total_batch ({TOTAL_BATCH_SIZE}) must be divisible by dbs*T ({dbs * MAX_SEQ_LEN})"
    )

    cmd = [
        f"{REPO_ROOT}/.venv/bin/python", "-m", "scripts.base_train",
        "--run", tag, "--model_tag", tag,
        "--depth", str(depth), "--max_seq_len", str(MAX_SEQ_LEN),
        "--device_batch_size", str(dbs),
        "--total_batch_size", str(TOTAL_BATCH_SIZE),
        "--kv_head_ratio", "1.0",
        "--compile_mode", compile_mode,
        "--sdpa_backend", "auto",
        "--save_every", str(save_every),
        "--eval_every", str(eval_every),
        "--eval_tokens", str(TOTAL_BATCH_SIZE),
        "--core_metric_every", "-1",
        "--sample_every", "-1",
        "--num_iterations", str(num_iter),
        "--token_cache_dir", token_cache_dir,
        "--benchmark_csv", str(bench_csv),
        "--chunked_loss", "--loss_chunk_size", "1024",
    ]
    if ckpt:
        cmd.extend(["--activation_checkpoint", "--ckpt_every_n_blocks", "1"])
    if resume_from is not None and resume_from >= 0:
        cmd.extend(["--resume_from_step", str(resume_from)])

    print(f"\n>>> {tag}: d{depth} DBS={dbs} ckpt={ckpt} compile={compile_mode} iters={num_iter}"
          + (f" (resume from {resume_from})" if resume_from is not None else ""))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(NANOCHAT_ROOT), env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    wall = time.time() - t0
    parsed = parse_log(log_path)
    parsed.update({
        "tag": tag,
        "depth": depth,
        "device_batch_size": dbs,
        "activation_ckpt": ckpt,
        "compile_mode": compile_mode,
        "grad_accum": TOTAL_BATCH_SIZE // (dbs * MAX_SEQ_LEN),
        "wall_sec": round(wall, 1),
        "exit_code": result.returncode,
    })
    return parsed


def try_depth(depth: int, token_cache_dir: str, output_dir: Path) -> dict | None:
    """Run depth with DBS=8 first; fall back to DBS=4 then DBS=4+ckpt on OOM."""
    fallbacks = [
        (8, False, "default"),
        (4, False, "default"),
        (4, True, "default"),
    ]
    for dbs, ckpt, compile_mode in fallbacks:
        tag = f"d{depth}_dbs{dbs}_ckpt{int(ckpt)}"
        result = run_config(tag, depth, dbs, ckpt, compile_mode, token_cache_dir, output_dir)
        if result["exit_code"] == 0 and result.get("tok_per_sec_median"):
            return result
        print(f"    {tag}: exit={result['exit_code']} oom={result.get('oom')}  tok/s={result.get('tok_per_sec_median')}")
        if result.get("oom"):
            print(f"    OOM, trying next fallback")
            continue
        # non-OOM failure — record and return
        return result
    return result


def estimate_wall_hours(num_scaling_params: int, tok_per_sec: int, ratio: float) -> float:
    """Wall-clock hours for a full training run at the given Chinchilla ratio."""
    target_tokens = ratio * num_scaling_params
    return target_tokens / tok_per_sec / 3600.0


def save_resume_check(winner: dict, token_cache_dir: str, output_dir: Path) -> dict:
    """For the winning depth/config, run 35 iters saving at step 25, then resume
    and run another 5 iters. Verifies the full round-trip at real depth."""
    depth = winner["depth"]
    dbs = winner["device_batch_size"]
    ckpt = winner["activation_ckpt"]
    compile_mode = winner["compile_mode"]

    tag_save = f"d{depth}_save_rt"
    # clean any prior checkpoints
    ckpt_dir = REPO_ROOT / "base_checkpoints" / tag_save
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    print(f"\n[SAVE/RESUME CHECK] winner d{depth} DBS={dbs} ckpt={ckpt}")
    save_result = run_config(
        tag_save, depth, dbs, ckpt, compile_mode, token_cache_dir, output_dir,
        num_iter=35, save_every=25, eval_every=-1,
    )
    if save_result["exit_code"] != 0:
        return {"status": "save_failed", "detail": save_result}

    resume_result = run_config(
        tag_save, depth, dbs, ckpt, compile_mode, token_cache_dir, output_dir,
        num_iter=30, save_every=-1, eval_every=29,
        resume_from=25,
    )
    status = "pass" if resume_result["exit_code"] == 0 and resume_result.get("final_loss") is not None else "resume_failed"
    return {
        "status": status,
        "save": {k: save_result[k] for k in ("exit_code", "wall_sec", "peak_mib", "final_loss")},
        "resume": {k: resume_result[k] for k in ("exit_code", "wall_sec", "peak_mib", "final_loss")},
        "resume_bpb": resume_result.get("bpb_measurements"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache-dir", required=True)
    ap.add_argument("--output-dir", default="/tmp/depth_sweep")
    ap.add_argument("--depths", default="18,20,22,24")
    ap.add_argument("--skip-save-resume", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    depths = [int(d) for d in args.depths.split(",")]

    results: list[dict] = []
    for depth in depths:
        r = try_depth(depth, args.token_cache_dir, output_dir)
        if r is None:
            print(f"[d{depth}] all fallbacks failed")
            continue
        if r.get("tok_per_sec_median"):
            ratio20 = estimate_wall_hours(r["num_scaling_params"], r["tok_per_sec_median"], 20)
            ratio40 = estimate_wall_hours(r["num_scaling_params"], r["tok_per_sec_median"], 40)
            r["wall_hours_ratio20"] = round(ratio20, 1)
            r["wall_hours_ratio40"] = round(ratio40, 1)
        results.append(r)

    # Print summary
    print("\n" + "=" * 100)
    print("DEPTH SWEEP RESULTS  (d{depth} @ T=1024, total_batch={TBS}, cache on, compile=default)".format(
        depth="", TBS=TOTAL_BATCH_SIZE))
    print("=" * 100)
    hdr = f"{'depth':>5} {'DBS':>4} {'ckpt':>5} {'params':>12} {'tok/s':>9} {'peak MiB':>9} {'load%':>6} {'mfu%':>6} {'loss':>14} {'r=20 h':>8} {'r=40 h':>8} {'exit':>5}"
    print(hdr)
    for r in results:
        params = r.get("num_scaling_params")
        tokps = r.get("tok_per_sec_median")
        peak = r.get("peak_mib")
        load = r.get("loader_pct_mean")
        mfu = r.get("mfu_mean")
        il = r.get("initial_loss")
        fl = r.get("final_loss")
        r20 = r.get("wall_hours_ratio20")
        r40 = r.get("wall_hours_ratio40")
        loss_str = f"{il:.2f}->{fl:.2f}" if il is not None else "n/a"
        print(f"{r['depth']:>5} {r['device_batch_size']:>4} {'Y' if r['activation_ckpt'] else 'N':>5}"
              f" {params:>12,} {tokps:>9,} {peak:>9.0f} {load:>6.2f} {mfu:>6.2f}"
              f" {loss_str:>14} {r20:>8.1f} {r40:>8.1f} {r['exit_code']:>5}")

    # Pick winner by ratio=20 wall time heuristic: best params × tok/s balance.
    # We favor highest params that still fits the budget. Apply the decision rule:
    #   d20 if >= 18k tok/s AND exit==0 → default serious run
    #   d24 if >= 12k tok/s AND exit==0 → ambitious 2-week run
    valid = [r for r in results if r["exit_code"] == 0 and r.get("tok_per_sec_median")]
    winner = None
    if valid:
        by_depth = {r["depth"]: r for r in valid}
        d24 = by_depth.get(24)
        d22 = by_depth.get(22)
        d20 = by_depth.get(20)
        d18 = by_depth.get(18)
        # Decision cascade
        if d24 and d24["tok_per_sec_median"] >= 12000:
            winner = d24
            pick_reason = "d24 >= 12k tok/s: ambitious path"
        elif d22 and d22["tok_per_sec_median"] >= 15000:
            winner = d22
            pick_reason = "d22 >= 15k tok/s: middle path"
        elif d20 and d20["tok_per_sec_median"] >= 18000:
            winner = d20
            pick_reason = "d20 >= 18k tok/s: default path"
        elif d18:
            winner = d18
            pick_reason = "d18 fallback"
        else:
            winner = max(valid, key=lambda r: r["num_scaling_params"] * r["tok_per_sec_median"])
            pick_reason = "best params×throughput among runs"
        print(f"\nPick: d{winner['depth']}  ({pick_reason})")
        print(f"  tok/s(med) = {winner['tok_per_sec_median']:,}")
        print(f"  peak VRAM  = {winner['peak_mib']} MiB")
        print(f"  wall time  = {winner['wall_hours_ratio20']}h (ratio=20), {winner['wall_hours_ratio40']}h (ratio=40)")
    else:
        print("\nNo valid candidates.")

    # JSON and CSV summaries
    json_path = output_dir / "depth_sweep.json"
    csv_path = output_dir / "depth_sweep.csv"
    with json_path.open("w") as f:
        json.dump({"results": results, "winner": winner}, f, indent=2)
    if results:
        fields = ["depth", "device_batch_size", "activation_ckpt", "compile_mode",
                  "num_scaling_params", "tok_per_sec_median", "peak_mib",
                  "loader_pct_mean", "mfu_mean", "initial_loss", "final_loss",
                  "wall_hours_ratio20", "wall_hours_ratio40", "exit_code"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
    print(f"\nSummary JSON: {json_path}")
    print(f"Summary CSV:  {csv_path}")

    # Save/resume check for the winner
    if winner and not args.skip_save_resume:
        rt = save_resume_check(winner, args.token_cache_dir, output_dir)
        rt_path = output_dir / "save_resume_result.json"
        with rt_path.open("w") as f:
            json.dump(rt, f, indent=2)
        print(f"\nSave/resume check: {rt['status']}")
        print(f"  {rt_path}")


if __name__ == "__main__":
    main()
