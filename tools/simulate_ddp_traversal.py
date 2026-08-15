"""Design-C Tier-1 gate: CPU-only, torch-free simulation of the exact 8-rank
DDP traversal of the cached sequential loader over a baked shard ordering.

Replays the loader's consumption arithmetic (needed = B*T+1 tokens per yield,
shard striping by ordering position mod world_size, wraparound, consumed-cursor
resume semantics) using only manifest metadata — never the .bin files — and
asserts the launch-gate invariants:

  - per-rank owned shard sets are disjoint, union == full manifest, counts ±1
  - zero wraparound within the chosen horizon (no repetition)
  - global (union-across-ranks) consumed family histogram within tolerance of
    the corpus mix  (per-rank imbalance is reported, NOT asserted)
  - deterministic replay (identical trace digest on re-run)
  - deterministic resume (restart from any saved per-rank state reproduces the
    identical continuation)
  - train/val source disjointness
  - unseen-tail composition report (the not-quite-one-epoch remainder)
  - canary table: (step, rank) -> expected consumed (shard, offset) for runtime
    assertion during training

Exit code 0 = all gates pass. Usage:

    python tools/simulate_ddp_traversal.py \
        --cache-dir .../token_cache_v4_balanced_candidate/train \
        --val-cache-dir .../token_cache_v4_balanced_candidate/val \
        --world-size 8 --device-batch 32 --seq-len 2048 \
        --out runs/tier1_report.json --canary-out runs/canaries.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RankSim:
    """Exact consumption model of one rank of the sequential cached loader."""

    def __init__(self, positions: list[int], tokens: list[int], families: list[str],
                 needed: int, start_ptr: int = 0, start_off: int = 0):
        # positions: full-list ordering positions owned by this rank (ascending)
        self.positions = positions
        self.tokens = tokens          # token count per owned shard (same order)
        self.families = families      # family per owned shard (same order)
        self.needed = needed
        self.ptr = start_ptr          # index into owned lists
        self.off = start_off          # token offset within current owned shard
        self.wraps = 0
        self.family_tokens: dict[str, int] = {}
        self.consumed = 0

    @classmethod
    def from_state(cls, positions, tokens, families, needed, position, offset):
        """Init from a consumed-state (full-list position, offset)."""
        ptr = positions.index(position)
        return cls(positions, tokens, families, needed, ptr, offset)

    def step(self) -> tuple[int, int]:
        """One yield: consume `needed` tokens. Returns consumed state after."""
        n = self.needed
        while n > 0:
            avail = self.tokens[self.ptr] - self.off
            if avail <= 0:
                self._advance()
                continue
            take = min(n, avail)
            fam = self.families[self.ptr]
            self.family_tokens[fam] = self.family_tokens.get(fam, 0) + take
            self.off += take
            self.consumed += take
            n -= take
            if self.off >= self.tokens[self.ptr]:
                self._advance()
        return self.positions[self.ptr], self.off

    def _advance(self):
        self.ptr += 1
        self.off = 0
        if self.ptr >= len(self.positions):
            self.ptr = 0
            self.wraps += 1


def load_ordering(cache_dir: str):
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    ordering_path = os.path.join(cache_dir, "shard_ordering.json")
    if not os.path.exists(ordering_path):
        raise SystemExit(f"GATE FAIL: no shard_ordering.json in {cache_dir} (run tools/build_balanced_ordering.py)")
    with open(ordering_path) as f:
        doc = json.load(f)
    actual_manifest_sha = sha256_file(manifest_path)
    if doc["manifest_sha256"] != actual_manifest_sha:
        raise SystemExit(
            "GATE FAIL: shard_ordering.json is STALE — manifest_sha256 mismatch "
            f"(ordering has {doc['manifest_sha256'][:16]}..., manifest is {actual_manifest_sha[:16]}...). "
            "Rebuild the ordering."
        )
    order = doc["order"]
    meta = doc["shards"]
    if len(set(order)) != len(order):
        raise SystemExit("GATE FAIL: duplicate filenames in ordering")
    if set(order) != set(meta.keys()):
        raise SystemExit("GATE FAIL: ordering order/shards keys mismatch")
    return doc, order, meta


def build_ranks(order, meta, world_size, needed):
    ranks = []
    for r in range(world_size):
        positions = [p for p in range(len(order)) if p % world_size == r]
        tokens = [meta[order[p]]["tokens"] for p in positions]
        families = [meta[order[p]]["family"] for p in positions]
        ranks.append(RankSim(positions, tokens, families, needed))
    return ranks


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--val-cache-dir", default=None)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--device-batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=-1,
                    help="optimizer steps to simulate (-1 = max zero-wrap horizon)")
    ap.add_argument("--tolerance-pp", type=float, default=1.0,
                    help="max abs deviation (percentage points) of global consumed family share vs corpus share")
    ap.add_argument("--canaries", type=int, default=12)
    ap.add_argument("--resume-extra", type=int, default=16,
                    help="yields to verify after each resume test point")
    ap.add_argument("--out", default=None)
    ap.add_argument("--canary-out", default=None)
    args = ap.parse_args(argv)

    cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    W = args.world_size
    needed = args.device_batch * args.seq_len + 1
    doc, order, meta = load_ordering(cache_dir)
    N = len(order)
    failures: list[str] = []
    report: dict = {"cache_dir": cache_dir, "world_size": W, "needed_per_yield": needed,
                    "num_shards": N, "ordering_sha256": doc["order_sha256"], "checks": {}}

    # --- ownership partition -------------------------------------------------
    counts = [len(range(r, N, W)) for r in range(W)]
    if max(counts) - min(counts) > 1:
        failures.append(f"per-rank shard counts differ by >1: {counts}")
    all_positions = set()
    for r in range(W):
        owned = set(range(r, N, W))
        if all_positions & owned:
            failures.append(f"rank {r} overlaps another rank's shards")
        all_positions |= owned
    if all_positions != set(range(N)):
        failures.append("union of rank shard sets != full manifest")
    report["checks"]["ownership"] = {"per_rank_counts": counts, "disjoint_and_complete": not failures}

    # --- horizon --------------------------------------------------------------
    ranks = build_ranks(order, meta, W, needed)
    rank_tokens = [sum(r.tokens) for r in ranks]
    max_yields_no_wrap = min(t // needed for t in rank_tokens)
    recommended_steps = max_yields_no_wrap // args.grad_accum
    yields_per_rank = (args.steps if args.steps > 0 else recommended_steps) * args.grad_accum
    if args.steps > 0 and args.steps > recommended_steps:
        failures.append(f"requested steps {args.steps} exceed zero-wrap horizon {recommended_steps}")
    report["rank_tokens"] = rank_tokens
    report["max_optimizer_steps_no_wrap"] = recommended_steps
    report["simulated_optimizer_steps"] = yields_per_rank // args.grad_accum
    report["recommended_num_iterations"] = recommended_steps

    # --- main traversal + canaries + trace digest -----------------------------
    canary_steps = sorted(set(
        max(1, round((i + 1) * (yields_per_rank / max(1, args.canaries))))
        for i in range(args.canaries)))
    canaries = []
    digest = hashlib.sha256()
    resume_points = sorted(set(p for p in (7, yields_per_rank // 2, yields_per_rank - args.resume_extra - 1)
                               if 0 < p < yields_per_rank))
    saved_states: dict[int, list[tuple[int, int]]] = {}

    for y in range(1, yields_per_rank + 1):
        states = [rk.step() for rk in ranks]
        if y in canary_steps:
            for r, (pos, off) in enumerate(states):
                canaries.append({"after_yield": y, "optimizer_step": y // args.grad_accum,
                                 "rank": r, "shard": order[pos], "position": pos, "offset": off})
                digest.update(f"{y}:{r}:{pos}:{off}".encode())
        if y in resume_points:
            saved_states[y] = list(states)

    for rk in ranks:
        if rk.wraps:
            failures.append(f"wraparound occurred (rank wraps={rk.wraps}) within horizon — repetition")

    # --- global family histogram ----------------------------------------------
    corpus_family: dict[str, int] = {}
    for fn in order:
        m = meta[fn]
        corpus_family[m["family"]] = corpus_family.get(m["family"], 0) + m["tokens"]
    corpus_total = sum(corpus_family.values())
    consumed_family: dict[str, int] = {}
    for rk in ranks:
        for fam, t in rk.family_tokens.items():
            consumed_family[fam] = consumed_family.get(fam, 0) + t
    consumed_total = sum(consumed_family.values())
    fam_check = {}
    for fam in sorted(corpus_family):
        corpus_share = corpus_family[fam] / corpus_total
        consumed_share = consumed_family.get(fam, 0) / consumed_total
        delta_pp = (consumed_share - corpus_share) * 100
        fam_check[fam] = {"corpus_share": round(corpus_share, 6),
                          "consumed_share": round(consumed_share, 6),
                          "delta_pp": round(delta_pp, 4)}
        if abs(delta_pp) > args.tolerance_pp:
            failures.append(f"global family drift: {fam} delta {delta_pp:+.3f}pp > {args.tolerance_pp}pp")
    report["checks"]["global_family_histogram"] = fam_check
    report["checks"]["per_rank_family_shares"] = [
        {fam: round(t / max(1, rk.consumed), 4) for fam, t in sorted(rk.family_tokens.items())}
        for rk in ranks]

    # --- unseen tail composition ----------------------------------------------
    tail = {fam: corpus_family[fam] - consumed_family.get(fam, 0) for fam in corpus_family}
    report["checks"]["unseen_tail"] = {
        "total_tokens": corpus_total - consumed_total,
        "fraction_of_corpus": round((corpus_total - consumed_total) / corpus_total, 6),
        "by_family": tail,
    }

    # --- deterministic replay ---------------------------------------------------
    ranks2 = build_ranks(order, meta, W, needed)
    digest2 = hashlib.sha256()
    for y in range(1, yields_per_rank + 1):
        states = [rk.step() for rk in ranks2]
        if y in canary_steps:
            for r, (pos, off) in enumerate(states):
                digest2.update(f"{y}:{r}:{pos}:{off}".encode())
    if digest.hexdigest() != digest2.hexdigest():
        failures.append("deterministic replay FAILED: trace digests differ")
    report["checks"]["deterministic_replay"] = digest.hexdigest() == digest2.hexdigest()
    report["trace_digest"] = digest.hexdigest()

    # --- resume determinism -----------------------------------------------------
    resume_ok = True
    for y0, states in saved_states.items():
        # uninterrupted continuation
        base = build_ranks(order, meta, W, needed)
        for _ in range(y0):
            for rk in base:
                rk.step()
        cont_a = [[rk.step() for rk in base] for _ in range(args.resume_extra)]
        # resumed continuation from saved consumed states
        resumed = []
        for r in range(W):
            positions = list(range(r, N, W))
            tokens = [meta[order[p]]["tokens"] for p in positions]
            families = [meta[order[p]]["family"] for p in positions]
            pos, off = states[r]
            resumed.append(RankSim.from_state(positions, tokens, families, needed, pos, off))
        cont_b = [[rk.step() for rk in resumed] for _ in range(args.resume_extra)]
        if cont_a != cont_b:
            resume_ok = False
            failures.append(f"resume divergence from yield {y0}")
    report["checks"]["deterministic_resume"] = {"points": list(saved_states), "ok": resume_ok}

    # --- train/val disjointness ---------------------------------------------------
    if args.val_cache_dir:
        val_manifest = os.path.join(os.path.abspath(os.path.expanduser(args.val_cache_dir)),
                                    "cache_manifest.json")
        with open(val_manifest) as f:
            val_doc = json.load(f)
        with open(os.path.join(cache_dir, "cache_manifest.json")) as f:
            train_manifest = json.load(f)
        train_by_base = {os.path.basename(e.get("source_file", "")): e for e in train_manifest["shards"]}
        val_by_base = {os.path.basename(e.get("source_file", "")): e for e in val_doc["shards"]}
        path_overlap = ({e.get("source_file", "") for e in train_manifest["shards"]}
                        & {e.get("source_file", "") for e in val_doc["shards"]})
        base_overlap = set(train_by_base) & set(val_by_base)
        # The v4 packager restarts shard numbering per split, so bare basename
        # collisions are expected; identical (docs, tokens) under the same name
        # is the strong same-file signal and DOES fail the gate.
        suspicious = [b for b in base_overlap
                      if (train_by_base[b].get("docs"), train_by_base[b].get("tokens"))
                      == (val_by_base[b].get("docs"), val_by_base[b].get("tokens"))]
        if path_overlap:
            failures.append(f"train/val source_file overlap: {len(path_overlap)} files")
        if suspicious:
            failures.append(
                f"train/val same-basename shards with identical (docs,tokens): {len(suspicious)} "
                f"(e.g. {suspicious[:3]}) — likely the same file in both splits")
        report["checks"]["train_val_disjoint"] = {
            "path_overlap": len(path_overlap),
            "basename_collisions": len(base_overlap),
            "identical_stat_collisions": len(suspicious),
            "note": "bare basename collisions are benign (per-split numbering); "
                    "identical (docs,tokens) pairs fail the gate",
        }

    # --- outputs -------------------------------------------------------------------
    report["failures"] = failures
    report["gate"] = "PASS" if not failures else "FAIL"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    if args.canary_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.canary_out)) or ".", exist_ok=True)
        with open(args.canary_out, "w") as f:
            json.dump({"ordering_sha256": doc["order_sha256"], "needed_per_yield": needed,
                       "world_size": W, "grad_accum": args.grad_accum,
                       "canaries": canaries}, f, indent=1)

    print(f"GATE {report['gate']}  steps={report['simulated_optimizer_steps']:,} "
          f"(zero-wrap max={recommended_steps:,})  digest={report['trace_digest'][:16]}")
    for fam, c in fam_check.items():
        print(f"  {fam:24s} corpus={c['corpus_share']:.4f} consumed={c['consumed_share']:.4f} "
              f"delta={c['delta_pp']:+.3f}pp")
    t = report["checks"]["unseen_tail"]
    print(f"  unseen tail: {t['total_tokens']:,} tokens ({t['fraction_of_corpus']*100:.3f}% of corpus)")
    for msg in failures:
        print(f"  FAIL: {msg}", file=sys.stderr)
    return report


if __name__ == "__main__":
    rep = run()
    sys.exit(0 if rep["gate"] == "PASS" else 1)
