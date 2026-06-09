"""Live loader-wait% monitor for the running training process.

Tails the training log and computes rolling mean of loader% over a sliding
window of the most recent N log lines. Writes a JSONL record every `interval`
seconds to `logs/loader_wait_monitor.jsonl`.

This is a non-blocking watchdog for the Phase-0-lite full-corpus scans.
If rolling loader% exceeds --threshold, it prints a WARN line so the
orchestrator (or a human tailing the monitor log) can respond.

Usage:
  python tools/monitor_loader_wait.py \
      --log logs/d22_r30_20260420_090531.log \
      --out logs/loader_wait_monitor.jsonl \
      --window 120 --interval 60 --threshold 2.0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import time
from collections import deque
from pathlib import Path

PAT_STEP = re.compile(r"step (\d+)/\d+ \([\d.]+%\) \| loss: ([\d.]+).*?loader: ([\d.]+)ms \(([\d.]+)%\) \| tok/sec: ([\d,]+)")


def tail_last_n(path: Path, n: int) -> list[str]:
    """Return the last N non-empty lines from the file."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return []
    if size == 0:
        return []
    # Read last ~2 MB which should contain > 120 log lines
    read_bytes = min(size, 2 * 1024 * 1024)
    with open(path, "rb") as f:
        f.seek(size - read_bytes)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-n:]


def parse_lines(lines: list[str]) -> list[dict]:
    out = []
    for ln in lines:
        m = PAT_STEP.search(ln)
        if not m:
            continue
        out.append(
            {
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "loader_ms": float(m.group(3)),
                "loader_pct": float(m.group(4)),
                "tok_sec": int(m.group(5).replace(",", "")),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--window", type=int, default=120, help="rolling window of log lines")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between samples")
    ap.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="warn if rolling loader_pct > threshold",
    )
    ap.add_argument("--max-samples", type=int, default=10_000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    last_step = -1
    samples = 0
    start = _dt.datetime.now(_dt.timezone.utc)
    print(f"[monitor] start {start.isoformat()}  watching {args.log}")
    print(f"[monitor] threshold={args.threshold}% window={args.window} interval={args.interval}s")

    with open(args.out, "a", buffering=1) as fout:
        while samples < args.max_samples:
            lines = tail_last_n(args.log, args.window)
            parsed = parse_lines(lines)
            if not parsed:
                time.sleep(args.interval)
                continue
            mean_loader_pct = sum(p["loader_pct"] for p in parsed) / len(parsed)
            mean_tok_sec = sum(p["tok_sec"] for p in parsed) / len(parsed)
            mean_loss = sum(p["loss"] for p in parsed) / len(parsed)
            latest = parsed[-1]
            rec = {
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "window_n": len(parsed),
                "step_latest": latest["step"],
                "step_delta_since_last_sample": latest["step"] - last_step if last_step >= 0 else 0,
                "loader_pct_mean": round(mean_loader_pct, 3),
                "loader_pct_max": round(max(p["loader_pct"] for p in parsed), 3),
                "tok_sec_mean": round(mean_tok_sec, 1),
                "loss_mean": round(mean_loss, 4),
                "warn": mean_loader_pct > args.threshold,
            }
            fout.write(json.dumps(rec) + "\n")
            tag = "WARN" if rec["warn"] else "ok"
            print(
                f"[monitor {tag}] step={latest['step']} "
                f"loader_mean={mean_loader_pct:.2f}% "
                f"loader_max={rec['loader_pct_max']:.2f}% "
                f"tok/s={rec['tok_sec_mean']:.0f}",
                flush=True,
            )
            last_step = latest["step"]
            samples += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
