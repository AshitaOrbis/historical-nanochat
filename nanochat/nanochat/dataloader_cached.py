"""
Cached-tokens dataloader that reads the binary token streams produced by
`scripts/build_token_cache.py`.

Contract matches `dataloader.tokenizing_distributed_data_loader_with_state`:
yields `(inputs, targets, state_dict)` per call; `state_dict` carries a
dict with enough info to resume roughly where we left off.

Layout expected in `cache_dir`:
    cache_manifest.json     # global, includes dtype + shard list
    shard_00000.bin         # raw uint16 or uint32 token stream
    shard_00000.meta.json   # { tokens, docs, source_file }
    ...

Note: cached tokens are already BOS-prefixed per document (builder does this).
The loader treats the whole file as a flat token stream and slices (B*T+1)
contiguous tokens per iteration, same packing logic as the parquet path.
"""

import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from nanochat.common import get_dist_info


def _load_manifest(cache_dir: str) -> dict:
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"cache_manifest.json not found in {cache_dir}. "
            "Run `python -m scripts.build_token_cache --input-dir ... --output-dir ...` first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def _dtype_from_str(name: str):
    # whitelist: only dtypes our builder uses.
    return {"uint16": np.uint16, "uint32": np.uint32}.get(name, np.uint16)


def cached_distributed_data_loader_with_state(
    B: int,
    T: int,
    split: str,
    device: str = "cuda",
    cache_dir: str = None,
    resume_state_dict: dict = None,
):
    """
    Infinite loader over mmap'd token cache files.

    DDP sharding is per-shard (rank r reads shards where shard_idx % world_size == r)
    with an in-shard offset so we don't overlap. This is simpler than the parquet
    loader's row-group striping and slightly wastes some tokens at shard boundaries,
    but for cached data that overhead is negligible.
    """
    assert split in ("train", "val", "all"), "split must be 'train' | 'val' | 'all'"
    assert cache_dir is not None, "cache_dir is required"

    manifest = _load_manifest(cache_dir)
    dtype = _dtype_from_str(manifest["dtype"])
    shard_entries = sorted(manifest["shards"], key=lambda e: e["shard_index"])

    # Split semantics:
    #   "train" - legacy: all shards except last (last is treated as val)
    #   "val"   - legacy: only the last shard
    #   "all"   - v4+: use every shard in this cache dir; callers supply
    #             train and val as SEPARATE cache dirs.
    if split == "train":
        shard_entries = shard_entries[:-1] if len(shard_entries) > 1 else shard_entries
    elif split == "val":
        shard_entries = shard_entries[-1:]
    # "all": leave shard_entries as-is
    if not shard_entries:
        raise RuntimeError(f"No shards for split={split} in {cache_dir}")

    _, rank, _, world_size = get_dist_info()
    needed = B * T + 1

    # Sharding by shard-index means each rank needs at least one shard to own.
    # Fail loudly instead of infinite-looping on ranks that'd own nothing.
    owned = [i for i in range(len(shard_entries)) if i % world_size == rank]
    if not owned:
        raise RuntimeError(
            f"Rank {rank} owns no shards in the cached loader "
            f"(world_size={world_size}, num_shards={len(shard_entries)} for split={split}). "
            "Rebuild the cache with more shards (one per rank minimum), or use the "
            "parquet dataloader which shards at row-group granularity."
        )

    # Per-rank resume: state_dict stores {rank: (shard_idx, token_off)} so each rank
    # restores its OWN cursor without getting confused by another rank's state.
    per_rank_state = (resume_state_dict or {}).get("per_rank", {})
    rank_key = str(rank)
    if rank_key in per_rank_state:
        shard_cursor = per_rank_state[rank_key].get("shard_idx", owned[0])
        token_cursor = per_rank_state[rank_key].get("token_off", 0)
    else:
        # Back-compat: if an older single-state dict is passed, use it only if
        # the saved shard_idx is owned by this rank; otherwise start from this
        # rank's first owned shard.
        saved_shard = (resume_state_dict or {}).get("shard_idx", owned[0])
        shard_cursor = saved_shard if saved_shard % world_size == rank else owned[0]
        token_cursor = (resume_state_dict or {}).get("token_off", 0) if shard_cursor == saved_shard else 0

    token_buffer = deque()
    use_cuda = str(device).startswith("cuda")

    def shard_path(entry):
        # manifest stored either the raw filename or the full path; normalize.
        fn = entry.get("filename") or f"shard_{entry['shard_index']:05d}.bin"
        return os.path.join(cache_dir, fn)

    while True:
        while len(token_buffer) < needed:
            if shard_cursor >= len(shard_entries):
                # Loop back to the start for multi-epoch training (parquet loader does the same).
                shard_cursor = 0
                token_cursor = 0
            entry = shard_entries[shard_cursor]
            # Rank-shard ownership: only rank (shard_cursor % world_size) reads this shard.
            if shard_cursor % world_size != rank:
                shard_cursor += 1
                token_cursor = 0
                continue

            mm = np.memmap(shard_path(entry), dtype=dtype, mode="r")
            total = mm.shape[0]
            if token_cursor >= total:
                shard_cursor += 1
                token_cursor = 0
                continue

            # Stream the shard into the buffer. We chunk by 1M tokens to avoid
            # blowing out memory on very large shards.
            chunk = 1_000_000
            end = min(total, token_cursor + chunk)
            token_buffer.extend(mm[token_cursor:end].tolist())
            token_cursor = end
            if token_cursor >= total:
                shard_cursor += 1
                token_cursor = 0

        # Pop B*T+1 tokens for this iteration.
        ids = [token_buffer.popleft() for _ in range(needed)]
        scratch = torch.tensor(ids, dtype=torch.long, pin_memory=use_cuda)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        # Per-rank state so each rank's resume cursor is independent.
        state = {
            "per_rank": {str(rank): {"shard_idx": shard_cursor, "token_off": token_cursor}},
            # Also include the legacy keys for backwards compat with older checkpoints.
            "shard_idx": shard_cursor,
            "token_off": token_cursor,
        }
        yield inputs, targets, state


def cached_distributed_data_loader(*args, **kwargs):
    """Helper that drops the state_dict, mirroring the parquet loader's 2-yield variant."""
    for inputs, targets, _ in cached_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


# -----------------------------------------------------------------------------
# Parallel family-balanced cached dataloader (Path B).
#
# Each optimizer step accumulates gradients over `grad_accum_steps` microbatches.
# This loader draws each microbatch from a specific family according to a
# deterministic schedule, so every optimizer step sees a stable mix of source
# families instead of many consecutive steps pinned to one family.
#
# Requires a `provenance.json` in the cache_dir (written by
# data/phase0/process/build_token_cache_v4.py) that maps every shard to a
# family. Raises if provenance is missing.
# -----------------------------------------------------------------------------

# Default family schedule for 32-microbatch optimizer steps (DBS=8 @ total_batch=262144).
# Sum must equal grad_accum_steps.
DEFAULT_FAMILY_SCHEDULE = [
    ("newspapers_periodicals", 12),
    ("science_technical", 8),
    ("books_general", 6),
    ("legal_government", 3),
    ("early_modern", 3),
]


def _load_family_shard_lists(cache_dir: str) -> dict[str, list[dict]]:
    """Partition manifest shards by family. Requires provenance.json."""
    prov_path = os.path.join(os.path.dirname(cache_dir.rstrip("/")), "provenance.json")
    # provenance.json lives one level above (parent has train/ and val/ subdirs)
    if not os.path.exists(prov_path):
        raise FileNotFoundError(
            f"parallel_family_cache requires provenance.json at {prov_path}; "
            "build it via `python -m data.phase0.process.build_token_cache_v4`."
        )
    with open(prov_path) as f:
        prov = json.load(f)
    # Determine which split this cache_dir corresponds to (train or val)
    split_key = None
    for k in ("train", "val"):
        if cache_dir.rstrip("/").endswith("/" + k):
            split_key = k
            break
    if split_key is None:
        raise ValueError(
            f"cache_dir {cache_dir} does not look like .../train or .../val "
            "(parallel_family_cache needs the v4 two-dir layout)."
        )
    split_info = prov.get("splits", {}).get(split_key, {})
    per_shard = split_info.get("per_shard", [])
    if not per_shard:
        raise RuntimeError(
            f"provenance.json has no per_shard entries for split={split_key}."
        )
    manifest = _load_manifest(cache_dir)
    manifest_by_idx = {s["shard_index"]: s for s in manifest["shards"]}
    by_family: dict[str, list[dict]] = {}
    matched = 0
    skipped = 0
    family_mismatches = 0
    KNOWN_FAMILIES = (
        "books_general", "newspapers_periodicals", "legal_government",
        "science_technical", "early_modern",
    )

    def _family_from_source_file(source_file: str) -> str | None:
        name = source_file.split("/")[-1]
        if name.startswith("shard_"):
            name = name[6:]
        for f in KNOWN_FAMILIES:
            if name.startswith(f + "_"):
                return f
        return None

    for rec in per_shard:
        sidx = rec["shard_index"]
        fam_provenance = rec["family"]
        mentry = manifest_by_idx.get(sidx)
        if mentry is None:
            skipped += 1
            continue
        # Sanity: verify provenance family matches the source_file's family.
        fam_from_source = _family_from_source_file(mentry.get("source_file", ""))
        if fam_from_source and fam_from_source != fam_provenance:
            family_mismatches += 1
        by_family.setdefault(fam_provenance, []).append(mentry)
        matched += 1

    coverage = matched / max(1, len(manifest["shards"]))
    if coverage < 0.95:
        raise RuntimeError(
            f"parallel_family_cache: provenance.json covers only "
            f"{coverage*100:.1f}% of manifest shards ({matched}/{len(manifest['shards'])}). "
            "Regenerate provenance via "
            "`python -m data.phase0.process.build_token_cache_v4 --skip-train --skip-val`."
        )
    if family_mismatches > 0:
        mismatch_pct = family_mismatches / max(1, matched) * 100
        if mismatch_pct > 5.0:
            raise RuntimeError(
                f"parallel_family_cache: {family_mismatches} of {matched} shards have a "
                f"provenance family that does not match the source_file family "
                f"({mismatch_pct:.1f}%). Provenance is stale — regenerate via "
                "`python -m data.phase0.process.build_token_cache_v4 --skip-train --skip-val`."
            )
    # Stabilize ordering by shard_index for reproducibility
    for fam in by_family:
        by_family[fam].sort(key=lambda e: e["shard_index"])
    return by_family


def cached_family_balanced_data_loader_with_state(
    B: int,
    T: int,
    split: str,
    device: str = "cuda",
    cache_dir: str = None,
    grad_accum_steps: int = 32,
    family_schedule: list[tuple[str, int]] = None,
    resume_state_dict: dict = None,
):
    """Family-balanced cached dataloader.

    Yields (inputs, targets, state) like the sequential loader.

    The `state` dict carries:
      - loader_strategy: "parallel_family_cache"
      - microbatch_index: position within the current optimizer step (0..grad_accum_steps-1)
      - family_cursors: { family: shard_idx_within_family }
      - family_token_cursors: { family: token_offset_within_current_shard }
      - family_wrap_counts: { family: N_wraps }
      - family_schedule: the schedule used (so resume can validate it)
    """
    assert cache_dir is not None, "cache_dir is required"
    if family_schedule is None:
        family_schedule = list(DEFAULT_FAMILY_SCHEDULE)
    if sum(count for _, count in family_schedule) != grad_accum_steps:
        raise ValueError(
            f"family_schedule microbatch count {sum(c for _,c in family_schedule)} != "
            f"grad_accum_steps {grad_accum_steps}. Fix the schedule."
        )

    _, rank, _, world_size = get_dist_info()
    if world_size > 1:
        raise NotImplementedError(
            "parallel_family_cache currently supports world_size=1 only. "
            "For DDP, families would need per-rank striping."
        )

    manifest = _load_manifest(cache_dir)
    dtype = _dtype_from_str(manifest["dtype"])
    family_shards = _load_family_shard_lists(cache_dir)
    missing = [fam for fam, _ in family_schedule if not family_shards.get(fam)]
    if missing:
        raise RuntimeError(
            f"family_schedule lists families with no shards in provenance: {missing}"
        )

    # Flatten schedule into per-microbatch family labels
    schedule_flat: list[str] = []
    for fam, count in family_schedule:
        schedule_flat.extend([fam] * count)
    assert len(schedule_flat) == grad_accum_steps

    # Resume or init per-family cursors
    resume = resume_state_dict or {}
    family_cursors: dict[str, int] = {fam: 0 for fam, _ in family_schedule}
    family_token_cursors: dict[str, int] = {fam: 0 for fam, _ in family_schedule}
    family_wrap_counts: dict[str, int] = {fam: 0 for fam, _ in family_schedule}
    microbatch_index = 0
    if resume.get("loader_strategy") == "parallel_family_cache":
        family_cursors.update(resume.get("family_cursors", {}))
        family_token_cursors.update(resume.get("family_token_cursors", {}))
        family_wrap_counts.update(resume.get("family_wrap_counts", {}))
        microbatch_index = resume.get("microbatch_index", 0) % grad_accum_steps

    needed = B * T + 1
    use_cuda = str(device).startswith("cuda")

    # Cache memmaps per (family, local_idx) so we don't re-open the same file
    # on every yield.
    memmap_cache: dict[tuple[str, int], "np.memmap"] = {}

    def shard_path(entry):
        fn = entry.get("filename") or f"shard_{entry['shard_index']:05d}.bin"
        return os.path.join(cache_dir, fn)

    def _get_memmap(fam: str, local_idx: int):
        key = (fam, local_idx)
        mm = memmap_cache.get(key)
        if mm is None:
            entry = family_shards[fam][local_idx]
            mm = np.memmap(shard_path(entry), dtype=dtype, mode="r")
            memmap_cache[key] = mm
        return mm

    def _read_family(fam: str, n: int) -> list[int]:
        """Read exactly n tokens from the family's current shard, advancing
        the cursor. Spans shard boundaries if needed. Deterministic."""
        out: list[int] = []
        while len(out) < n:
            local_idx = family_cursors[fam] % len(family_shards[fam])
            mm = _get_memmap(fam, local_idx)
            total = int(mm.shape[0])
            token_off = family_token_cursors[fam]
            if token_off >= total:
                # advance to next shard within this family
                family_cursors[fam] += 1
                if family_cursors[fam] >= len(family_shards[fam]):
                    family_cursors[fam] = 0
                    family_wrap_counts[fam] += 1
                family_token_cursors[fam] = 0
                continue
            want = min(n - len(out), total - token_off)
            out.extend(mm[token_off:token_off + want].tolist())
            family_token_cursors[fam] = token_off + want
            if family_token_cursors[fam] >= total:
                family_cursors[fam] += 1
                if family_cursors[fam] >= len(family_shards[fam]):
                    family_cursors[fam] = 0
                    family_wrap_counts[fam] += 1
                family_token_cursors[fam] = 0
        return out

    while True:
        fam = schedule_flat[microbatch_index]
        ids = _read_family(fam, needed)
        scratch = torch.tensor(ids, dtype=torch.long, pin_memory=use_cuda)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)

        next_mb = (microbatch_index + 1) % grad_accum_steps
        state = {
            "loader_strategy": "parallel_family_cache",
            "microbatch_index": next_mb,
            "current_microbatch_family": fam,
            "family_cursors": dict(family_cursors),
            "family_token_cursors": dict(family_token_cursors),
            "family_wrap_counts": dict(family_wrap_counts),
            "family_schedule": [[f, n] for f, n in family_schedule],
        }
        yield inputs, targets, state
        microbatch_index = next_mb


def cached_family_balanced_data_loader(*args, **kwargs):
    """Helper that drops the state dict, mirroring the sequential variant."""
    for inputs, targets, _ in cached_family_balanced_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
