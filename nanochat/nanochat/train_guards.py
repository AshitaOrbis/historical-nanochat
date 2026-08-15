"""Pure-dict runtime guards for base_train (Sol P0-6/P0-7): canary loading and
per-yield checking, expected-resolved-config assertions, and step-list parsing.
Kept free of torch/GPU state so every branch is unit-testable on CPU.
"""
from __future__ import annotations

import json


def parse_steps_list(spec: str | None) -> set[int]:
    """'954,1907, 3815' -> {954, 1907, 3815}. Empty/None -> empty set."""
    if not spec:
        return set()
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        step = int(part)
        if step <= 0:
            raise ValueError(f"step list entries must be positive, got {step!r}")
        out.add(step)
    return out


def load_canaries(path: str, *, needed: int, world_size: int, grad_accum: int,
                  ordering_sha256: str | None) -> dict[tuple[int, int], dict]:
    """Load and validate a canary file; return {(after_yield, rank): canary}.

    Refuses on any config mismatch — a canary set generated for a different
    DBS/world/ordering would either never fire or fire falsely (P0-7). This is
    also what makes the DBS-16 OOM fallback a *new attempt*: the DBS-32 canary
    file fails the `needed` check instead of silently mis-asserting (P1-4).
    """
    with open(path) as f:
        doc = json.load(f)
    if doc.get("needed_per_yield") != needed:
        raise RuntimeError(
            f"canary file {path}: needed_per_yield={doc.get('needed_per_yield')} "
            f"but this run yields {needed} tokens (device_batch_size*seq_len+1). "
            "Regenerate canaries for this batch geometry (Tier-1) before training.")
    if doc.get("world_size") != world_size:
        raise RuntimeError(
            f"canary file {path}: world_size={doc.get('world_size')} != runtime {world_size}")
    if doc.get("grad_accum") != grad_accum:
        raise RuntimeError(
            f"canary file {path}: grad_accum={doc.get('grad_accum')} != runtime {grad_accum}")
    if ordering_sha256 is not None and doc.get("ordering_sha256") != ordering_sha256:
        raise RuntimeError(
            f"canary file {path}: ordering_sha256 {str(doc.get('ordering_sha256'))[:16]}… "
            f"does not match the loaded shard ordering {ordering_sha256[:16]}… — "
            "stale canaries would assert the wrong traversal.")
    canaries = doc.get("canaries") or []
    if not canaries:
        raise RuntimeError(f"canary file {path}: no canaries listed")
    by_yield_rank: dict[tuple[int, int], dict] = {}
    ranks_by_yield: dict[int, set[int]] = {}
    for index, canary in enumerate(canaries):
        if not isinstance(canary, dict):
            raise RuntimeError(f"canary file {path}: canary {index} must be an object")
        try:
            after_yield = int(canary["after_yield"])
            rank = int(canary["rank"])
            position = int(canary["position"])
            offset = int(canary["offset"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"canary file {path}: canary {index} has invalid required coordinates"
            ) from exc
        if after_yield <= 0 or position < 0 or offset < 0:
            raise RuntimeError(
                f"canary file {path}: canary {index} coordinates must be non-negative "
                "and after_yield must be positive"
            )
        if not 0 <= rank < world_size:
            raise RuntimeError(
                f"canary file {path}: canary {index} rank={rank} is outside "
                f"runtime ranks 0..{world_size - 1}"
            )
        key = (after_yield, rank)
        if key in by_yield_rank:
            raise RuntimeError(
                f"canary file {path}: duplicate canary for "
                f"after_yield={after_yield}, rank={rank}"
            )
        by_yield_rank[key] = canary
        ranks_by_yield.setdefault(after_yield, set()).add(rank)

    expected_ranks = set(range(world_size))
    for after_yield, ranks in ranks_by_yield.items():
        missing = expected_ranks - ranks
        if missing:
            raise RuntimeError(
                f"canary file {path}: after_yield={after_yield} is missing "
                f"rank records {sorted(missing)}"
            )
    return by_yield_rank


def check_canary(canary: dict, state: dict, rank: int) -> str | None:
    """Compare a canary against the consumed loader state after its yield.
    Returns None on pass, or a human-readable mismatch description.
    Canary coords: position = position in the BAKED ORDERING (not manifest
    shard_index), offset = token offset of the consumed cursor after the yield.
    """
    canary_rank = int(canary.get("rank", -1))
    if canary_rank != rank:
        return f"canary rank {canary_rank} != runtime rank {rank}"
    cur = (state or {}).get("per_rank", {}).get(str(rank))
    if cur is None:
        return f"no per_rank[{rank}] cursor in loader state"
    mismatches = []
    if cur.get("shard_idx") != canary["position"]:
        mismatches.append(
            f"ordering_position {cur.get('shard_idx')} != expected {canary['position']}")
    if cur.get("token_off") != canary["offset"]:
        mismatches.append(
            f"token_off {cur.get('token_off')} != expected {canary['offset']}")
    ident = (state or {}).get("identity") or {}
    if ident.get("filename") and canary.get("shard") and ident["filename"] != canary["shard"]:
        mismatches.append(f"filename {ident['filename']} != expected {canary['shard']}")
    return "; ".join(mismatches) or None


def assert_expected_resolved(expected: dict, resolved: dict,
                             float_rtol: float = 1e-6) -> list[str]:
    """Compare a nested expected-config dict against runtime-resolved values.
    Returns the list of checked paths; raises AssertionError listing EVERY
    mismatch (not just the first) so a bad launch surfaces completely.
    Floats compare with relative tolerance; lists elementwise; dicts recurse.
    Keys present in `expected` but absent from `resolved` are mismatches.
    """
    failures: list[str] = []
    checked: list[str] = []

    def close(a, b) -> bool:
        if isinstance(a, float) or isinstance(b, float):
            try:
                a, b = float(a), float(b)
            except (TypeError, ValueError):
                return False
            denom = max(abs(a), abs(b), 1e-30)
            return abs(a - b) / denom <= float_rtol
        return a == b

    def walk(exp, res, path):
        if isinstance(exp, dict):
            if not isinstance(res, dict):
                failures.append(f"{path}: expected mapping, resolved {type(res).__name__}")
                return
            for k, v in exp.items():
                if k.startswith("_"):    # underscore keys are commentary
                    continue
                if k not in res:
                    failures.append(f"{path}.{k}: MISSING from resolved config")
                else:
                    walk(v, res[k], f"{path}.{k}")
        elif isinstance(exp, list):
            if not isinstance(res, (list, tuple)) or len(res) != len(exp):
                failures.append(f"{path}: length/type mismatch (expected {exp!r}, resolved {res!r})")
                return
            for i, (e, r) in enumerate(zip(exp, res)):
                walk(e, r, f"{path}[{i}]")
        else:
            checked.append(path)
            if not close(exp, res):
                failures.append(f"{path}: expected {exp!r}, resolved {res!r}")

    walk(expected, resolved, "$")
    if failures:
        raise AssertionError(
            "resolved-config assertions FAILED:\n  " + "\n  ".join(failures))
    return checked
