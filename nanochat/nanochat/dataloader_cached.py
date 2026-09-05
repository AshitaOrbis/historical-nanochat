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

import hashlib
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from nanochat.common import get_dist_info


def _maybe_apply_shard_ordering(cache_dir: str, shard_entries: list[dict]):
    """Design C: if a baked `shard_ordering.json` exists in cache_dir, reorder
    shard_entries to match it. The ordering is a sidecar (manifest/provenance
    are never modified) and carries the manifest's sha256 so a stale ordering
    after a cache rebuild refuses to load instead of silently mis-serving.
    Returns (entries, ordering_doc_or_None)."""
    ordering_path = os.path.join(cache_dir, "shard_ordering.json")
    if not os.path.exists(ordering_path):
        return shard_entries, None
    with open(ordering_path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise RuntimeError(
            f"shard_ordering.json in {cache_dir} must be a version 1 JSON object"
        )
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    with open(manifest_path, "rb") as f:
        manifest_sha = hashlib.sha256(f.read()).hexdigest()
    if doc.get("manifest_sha256") != manifest_sha:
        raise RuntimeError(
            f"shard_ordering.json in {cache_dir} is STALE: it was baked against "
            f"manifest sha256 {str(doc.get('manifest_sha256'))[:16]}... but the current "
            f"cache_manifest.json is {manifest_sha[:16]}... — rebuild the ordering "
            "(tools/build_balanced_ordering.py) or remove it."
        )
    by_name = {}
    for e in shard_entries:
        fn = e.get("filename") or f"shard_{e['shard_index']:05d}.bin"
        by_name[fn] = e
    order = doc.get("order")
    if not isinstance(order, list) or any(
        not isinstance(name, str) or not name for name in order
    ):
        raise RuntimeError(
            f"shard_ordering.json in {cache_dir} requires a non-empty filename list"
        )
    if len(order) != len(shard_entries) or set(order) != set(by_name.keys()):
        raise RuntimeError(
            f"shard_ordering.json in {cache_dir} does not cover the manifest exactly "
            f"(ordering has {len(order)} names, manifest has {len(shard_entries)} shards)."
        )
    derived_order_sha256 = hashlib.sha256("\n".join(order).encode()).hexdigest()
    if doc.get("order_sha256") != derived_order_sha256:
        raise RuntimeError(
            f"shard_ordering.json in {cache_dir} has an invalid order_sha256: "
            f"stored={doc.get('order_sha256')!r}, derived={derived_order_sha256}. "
            "The traversal identity must be derived from the ordered shard filenames."
        )
    return [by_name[n] for n in order], doc


def load_verified_shard_ordering(
    cache_dir: str, *, require_vocab_size: bool = False
) -> dict | None:
    """Return the ordering document only after independently verifying its hash."""
    manifest = _load_manifest(cache_dir, require_vocab_size=require_vocab_size)
    entries = sorted(manifest["shards"], key=lambda entry: entry["shard_index"])
    _entries, ordering_doc = _maybe_apply_shard_ordering(cache_dir, entries)
    return ordering_doc


def _load_manifest(cache_dir: str, *, require_vocab_size: bool = False) -> dict:
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"cache_manifest.json not found in {cache_dir}. "
            "Run `python -m scripts.build_token_cache --input-dir ... --output-dir ...` first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: cache manifest must be a JSON object")
    if manifest.get("format_version") != 1:
        raise ValueError(
            f"{manifest_path}: format_version must be 1; rebuild or migrate this cache"
        )
    byte_order = manifest.get("byte_order")
    dtype = _dtype_from_str(manifest.get("dtype"), byte_order)
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"{manifest_path}: shards must be a list")

    vocab_size = manifest.get("vocab_size")
    if require_vocab_size and vocab_size is None:
        raise ValueError(
            f"{manifest_path}: strict cache schema requires vocab_size"
        )
    if vocab_size is not None and (
        not isinstance(vocab_size, int) or isinstance(vocab_size, bool)
        or vocab_size <= 0
    ):
        raise ValueError(f"{manifest_path}: vocab_size must be a positive integer")
    tokenizer_identity = manifest.get("tokenizer_identity")
    if tokenizer_identity is not None:
        if not isinstance(tokenizer_identity, dict):
            raise ValueError(f"{manifest_path}: tokenizer_identity must be an object")
        identity_vocab_size = tokenizer_identity.get("vocab_size")
        if vocab_size is not None and identity_vocab_size != vocab_size:
            raise ValueError(
                f"{manifest_path}: vocab_size={vocab_size} does not match "
                f"tokenizer_identity.vocab_size={identity_vocab_size!r}"
            )

    itemsize = dtype.itemsize
    seen_indices = set()
    seen_filenames = set()
    cache_root = Path(cache_dir).resolve()
    for position, entry in enumerate(shards):
        if not isinstance(entry, dict):
            raise ValueError(f"{manifest_path}: shard entry {position} must be an object")
        shard_index = entry.get("shard_index")
        tokens = entry.get("tokens")
        if not isinstance(shard_index, int) or shard_index in seen_indices:
            raise ValueError(
                f"{manifest_path}: shard_index must be a unique integer (entry {position})"
            )
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(
                f"{manifest_path}: shard {shard_index} requires a positive integer tokens field"
            )
        seen_indices.add(shard_index)
        filename = entry.get("filename") or f"shard_{shard_index:05d}.bin"
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"{manifest_path}: shard {shard_index} filename must be a non-empty string"
            )
        if filename in seen_filenames:
            raise ValueError(
                f"{manifest_path}: duplicate shard filename {filename!r}"
            )
        seen_filenames.add(filename)
        resolved_shard_path = (cache_root / filename).resolve()
        if cache_root not in resolved_shard_path.parents:
            raise ValueError(
                f"{manifest_path}: shard {shard_index} filename escapes the cache directory"
            )
        shard_path = str(resolved_shard_path)
        try:
            actual_bytes = os.path.getsize(shard_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{manifest_path}: shard {shard_index} file is missing: {shard_path}"
            ) from exc
        expected_bytes = tokens * itemsize
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"{shard_path}: byte length {actual_bytes} != manifest tokens {tokens} "
                f"* dtype itemsize {itemsize} ({expected_bytes})"
            )
        if "bytes" in entry and entry["bytes"] != expected_bytes:
            raise RuntimeError(
                f"{manifest_path}: shard {shard_index} declares bytes={entry['bytes']} "
                f"but tokens*dtype.itemsize={expected_bytes}"
            )
        if vocab_size is not None:
            # A bounded first/middle/last sample catches width/version corruption
            # without scanning multi-hundred-GB caches during startup.
            window = min(tokens, 256)
            starts = sorted({0, max(0, tokens // 2 - window // 2), tokens - window})
            with open(shard_path, "rb") as shard_file:
                for start in starts:
                    shard_file.seek(start * itemsize)
                    raw = shard_file.read(window * itemsize)
                    sampled = np.frombuffer(raw, dtype=dtype)
                    if sampled.size and int(sampled.max()) >= vocab_size:
                        raise RuntimeError(
                            f"{shard_path}: sampled token id {int(sampled.max())} is "
                            f"outside declared vocab_size={vocab_size}"
                        )
    return manifest


def _dtype_from_str(name: str, byte_order: str):
    """Return an explicitly little-endian cache dtype or reject the schema.

    Union of both review enactments: 0814 required rejecting unknown dtype
    names; 0813 additionally required an explicit little-endian byte order.
    Both rejection paths are kept.
    """
    if byte_order != "little":
        raise ValueError(
            f"cache byte_order must be 'little', got {byte_order!r}"
        )
    dtypes = {"uint16": np.dtype("<u2"), "uint32": np.dtype("<u4")}
    if name not in dtypes:
        raise ValueError(
            f"cache dtype must be one of {sorted(dtypes)}, got {name!r}"
        )
    return dtypes[name]


def cached_distributed_data_loader_with_state(
    B: int,
    T: int,
    split: str,
    device: str = "cuda",
    cache_dir: str = None,
    resume_state_dict: dict = None,
    strict_manifest_schema: bool = False,
):
    """
    Infinite loader over mmap'd token cache files.

    DDP sharding is per-shard (rank r reads shards where shard position % world_size
    == r) with an in-shard offset so we don't overlap. This is simpler than the
    parquet loader's row-group striping and slightly wastes some tokens at shard
    boundaries, but for cached data that overhead is negligible.

    Design C: if cache_dir contains a baked `shard_ordering.json` (see
    tools/build_balanced_ordering.py), shards are served in that stratified order
    instead of shard_index order (split="all" only; fails loudly otherwise).

    Resume state uses CONSUMED-cursor semantics: the per-rank (shard_idx, token_off)
    in the state dict points at the next token the *trainer* has not consumed, not
    at the loader's read-ahead position — resuming reproduces the identical token
    stream (older checkpoints saved the read-ahead cursor and could silently skip
    up to ~1M buffered tokens per rank on restart).
    """
    if split not in ("train", "val", "all"):
        raise ValueError("split must be 'train' | 'val' | 'all'")
    if cache_dir is None:
        raise ValueError("cache_dir is required")

    manifest = _load_manifest(
        cache_dir, require_vocab_size=strict_manifest_schema
    )
    dtype = _dtype_from_str(manifest["dtype"], manifest["byte_order"])
    shard_entries = sorted(manifest["shards"], key=lambda e: e["shard_index"])

    # Split semantics:
    #   "train" - legacy: all shards except last (last is treated as val)
    #   "val"   - legacy: only the last shard
    #   "all"   - v4+: use every shard in this cache dir; callers supply
    #             train and val as SEPARATE cache dirs.
    if split in ("train", "val") and len(shard_entries) < 2:
        raise RuntimeError(
            f"Legacy cached split={split!r} requires at least two shards so validation "
            "cannot alias training data. Supply a separate validation cache and use "
            "split='all', or rebuild the cache with at least two shards."
        )
    if split == "train":
        shard_entries = shard_entries[:-1]
    elif split == "val":
        shard_entries = shard_entries[-1:]
    # "all": leave shard_entries as-is
    if not shard_entries:
        raise RuntimeError(f"No shards for split={split} in {cache_dir}")

    # Design-C stratified ordering (sidecar). Only meaningful for the v4 two-dir
    # layout where the whole dir is one split; refuse ambiguous legacy splits.
    if split == "all":
        shard_entries, _ordering_doc = _maybe_apply_shard_ordering(cache_dir, shard_entries)
    elif os.path.exists(os.path.join(cache_dir, "shard_ordering.json")):
        raise RuntimeError(
            f"{cache_dir} has a shard_ordering.json but split={split!r}: the baked "
            "ordering is only valid with split='all' (v4 layout). Refusing to guess."
        )

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

    def shard_path(entry):
        # manifest stored either the raw filename or the full path; normalize.
        fn = entry.get("filename") or f"shard_{entry['shard_index']:05d}.bin"
        return os.path.join(cache_dir, fn)

    # Consumed-cursor bookkeeping: `segments` records which (position, offset,
    # length) spans have been loaded into token_buffer but not yet handed to the
    # trainer. The consumed cursor (what resume must restore) is the head of this
    # queue; the loaded cursor (shard_cursor/token_cursor) runs ahead of it by up
    # to one read chunk.
    _itemsize = np.dtype(dtype).itemsize

    def _entry_num_tokens(e):
        t = e.get("tokens")
        if t is None:  # legacy manifests without token counts: derive from file size
            t = os.path.getsize(shard_path(e)) // _itemsize
        return t

    entry_tokens = [_entry_num_tokens(e) for e in shard_entries]
    ordered_filenames = [
        entry.get("filename") or f"shard_{entry['shard_index']:05d}.bin"
        for entry in shard_entries
    ]
    traversal_sha256 = hashlib.sha256("\n".join(ordered_filenames).encode()).hexdigest()
    with open(os.path.join(cache_dir, "cache_manifest.json"), "rb") as manifest_file:
        manifest_sha256 = hashlib.sha256(manifest_file.read()).hexdigest()
    resume_contract = {
        "version": 1,
        "loader_strategy": "sequential_cache",
        "cache_manifest_sha256": manifest_sha256,
        "shard_ordering_sha256": (
            _ordering_doc["order_sha256"] if split == "all" and _ordering_doc else None
        ),
        "traversal_sha256": traversal_sha256,
        "split": split,
        "batch_size": B,
        "sequence_length": T,
        "world_size": world_size,
    }

    # Resume coordinates are meaningful only under the exact cache/order/geometry
    # that produced them. The contract is embedded per rank so DDP state merging
    # cannot accidentally discard it.
    rank_key = str(rank)
    if resume_state_dict is not None:
        if not isinstance(resume_state_dict, dict):
            raise RuntimeError("sequential resume state must be an object")
        per_rank_state = resume_state_dict.get("per_rank")
        if not isinstance(per_rank_state, dict) or rank_key not in per_rank_state:
            raise RuntimeError(
                f"sequential resume is missing the bound per_rank[{rank}] cursor"
            )
        rank_state = per_rank_state[rank_key]
        if not isinstance(rank_state, dict):
            raise RuntimeError(f"sequential resume per_rank[{rank}] must be an object")
        saved_contract = rank_state.get("resume_contract")
        if saved_contract != resume_contract:
            mismatches = [
                key for key, expected in resume_contract.items()
                if not isinstance(saved_contract, dict) or saved_contract.get(key) != expected
            ]
            raise RuntimeError(
                f"sequential resume contract does not match runtime: {mismatches}"
            )
        top_contract = resume_state_dict.get("resume_contract")
        if top_contract is not None and top_contract != resume_contract:
            raise RuntimeError("sequential resume top-level contract does not match runtime")
        shard_cursor = rank_state.get("shard_idx")
        token_cursor = rank_state.get("token_off")
        if not isinstance(shard_cursor, int) or not 0 <= shard_cursor < len(shard_entries):
            raise RuntimeError(
                f"sequential resume shard cursor {shard_cursor!r} is outside the dataset"
            )
        if shard_cursor % world_size != rank:
            raise RuntimeError(
                f"sequential resume shard cursor {shard_cursor} is not owned by rank {rank}"
            )
        if not isinstance(token_cursor, int) or not 0 <= token_cursor < entry_tokens[shard_cursor]:
            raise RuntimeError(
                f"sequential resume token cursor {token_cursor!r} is outside shard "
                f"length {entry_tokens[shard_cursor]}"
            )
        expected_identity = {
            "manifest_shard_index": shard_entries[shard_cursor]["shard_index"],
            "filename": ordered_filenames[shard_cursor],
        }
        saved_identity = rank_state.get("identity")
        if not isinstance(saved_identity, dict) or any(
            saved_identity.get(key) != value for key, value in expected_identity.items()
        ):
            raise RuntimeError(
                "sequential resume cursor does not match its exact saved shard identity"
            )
    else:
        shard_cursor = owned[0]
        token_cursor = 0

    token_buffer = deque()
    use_cuda = str(device).startswith("cuda")
    segments: deque = deque()  # each item: [position, offset, remaining]

    def _normalize(pos, off):
        # Canonical coordinates of the next unconsumed token: an owned position
        # with off < shard length (skip foreign/finished shards, wrap at end).
        while True:
            if pos >= len(shard_entries):
                pos, off = 0, 0
                continue
            if pos % world_size != rank:
                pos, off = pos + 1, 0
                continue
            if off >= entry_tokens[pos]:
                pos, off = pos + 1, 0
                continue
            return pos, off

    def _consumed_state():
        if segments:
            pos, off, _ = segments[0]
            return _normalize(pos, off)
        return _normalize(shard_cursor, token_cursor)

    def _advance_consumed(n):
        while n > 0:
            seg = segments[0]
            take = n if n < seg[2] else seg[2]
            seg[1] += take
            seg[2] -= take
            n -= take
            if seg[2] == 0:
                segments.popleft()

    while True:
        while len(token_buffer) < needed:
            if shard_cursor >= len(shard_entries):
                # Loop back to the start for multi-epoch training (parquet loader does the same).
                shard_cursor = 0
                token_cursor = 0
            entry = shard_entries[shard_cursor]
            # Rank-shard ownership: only rank (position % world_size) reads this shard.
            if shard_cursor % world_size != rank:
                shard_cursor += 1
                token_cursor = 0
                continue

            mm = np.memmap(shard_path(entry), dtype=dtype, mode="r")
            total = mm.shape[0]
            if total != entry_tokens[shard_cursor]:
                raise RuntimeError(
                    f"{shard_path(entry)}: on-disk token count {total} != manifest "
                    f"tokens {entry_tokens[shard_cursor]} — cache and manifest disagree."
                )
            if token_cursor >= total:
                shard_cursor += 1
                token_cursor = 0
                continue

            # Stream the shard into the buffer. We chunk by 1M tokens to avoid
            # blowing out memory on very large shards.
            chunk = 1_000_000
            end = min(total, token_cursor + chunk)
            segments.append([shard_cursor, token_cursor, end - token_cursor])
            token_buffer.extend(mm[token_cursor:end].tolist())
            token_cursor = end
            if token_cursor >= total:
                shard_cursor += 1
                token_cursor = 0

        # Pop B*T+1 tokens for this iteration.
        ids = [token_buffer.popleft() for _ in range(needed)]
        _advance_consumed(needed)
        scratch = torch.tensor(ids, dtype=torch.long, pin_memory=use_cuda)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        # Per-rank CONSUMED state so each rank's resume reproduces the exact
        # unconsumed continuation (read-ahead tokens still in the buffer are
        # re-read on resume, never skipped).
        c_pos, c_off = _consumed_state()
        c_entry = shard_entries[c_pos]
        rank_identity = {
            "manifest_shard_index": c_entry["shard_index"],
            "filename": c_entry.get("filename") or f"shard_{c_entry['shard_index']:05d}.bin",
        }
        state = {
            "resume_contract": resume_contract,
            "per_rank": {str(rank): {
                "shard_idx": c_pos,
                "token_off": c_off,
                "identity": rank_identity,
                "resume_contract": resume_contract,
            }},
            # Also include the legacy keys for backwards compat with older checkpoints.
            "shard_idx": c_pos,
            "token_off": c_off,
            # Explicit identity fields (Sol P0-7): shard_idx above is a POSITION
            # IN THE BAKED ORDERING, not a manifest shard index. Diagnostics and
            # provenance must key on manifest_shard_index; conflating the two
            # produced false "provenance sane" evidence.
            "identity": {
                "ordering_position": c_pos,
                **rank_identity,
                "token_off": c_off,
            },
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


KNOWN_FAMILIES = (
    "books_general", "newspapers_periodicals", "legal_government",
    "science_technical", "early_modern",
)


def _load_family_shard_lists(cache_dir: str) -> dict[str, list[dict]]:
    """Partition manifest shards by family after exact, hash-bound provenance validation."""
    prov_path = os.path.join(os.path.dirname(cache_dir.rstrip("/")), "provenance.json")
    # provenance.json lives one level above (parent has train/ and val/ subdirs)
    if not os.path.exists(prov_path):
        raise FileNotFoundError(
            f"parallel_family_cache requires provenance.json at {prov_path}; "
            "build it via `python tools/build_cache_provenance.py --cache-root <root>`."
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
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    with open(manifest_path, "rb") as manifest_file:
        manifest_sha256 = hashlib.sha256(manifest_file.read()).hexdigest()
    bound_sha256 = split_info.get("manifest_sha256")
    if bound_sha256 != manifest_sha256:
        raise RuntimeError(
            f"parallel_family_cache: provenance for split={split_key} is not bound "
            f"to this manifest (expected manifest_sha256={manifest_sha256}, "
            f"got {bound_sha256!r}); regenerate provenance"
        )

    manifest_by_idx = {s["shard_index"]: s for s in manifest["shards"]}
    by_family: dict[str, list[dict]] = {}
    seen_indices: set[int] = set()

    def _family_and_source_from_source_file(
        source_file: str,
    ) -> tuple[str | None, str | None]:
        if not isinstance(source_file, str) or not source_file:
            return None, None
        stem = Path(source_file).stem
        if stem.startswith("shard_"):
            stem = stem[6:]
        suffix = stem.rsplit("_", 1)
        if len(suffix) == 2 and len(suffix[1]) == 6 and suffix[1].isdigit():
            stem = suffix[0]
        for family in KNOWN_FAMILIES:
            prefix = family + "_"
            if stem.startswith(prefix):
                return family, stem[len(prefix):]
        return None, None

    expected_total_tokens = 0
    expected_total_docs = 0
    expected_per_source_tokens: dict[str, int] = {}
    expected_per_family_tokens: dict[str, int] = {}

    for rec in per_shard:
        if not isinstance(rec, dict):
            raise RuntimeError("parallel_family_cache: every provenance record must be an object")
        sidx = rec.get("shard_index")
        fam_provenance = rec.get("family")
        if not isinstance(sidx, int):
            raise RuntimeError("parallel_family_cache: provenance shard_index must be an integer")
        if sidx in seen_indices:
            raise RuntimeError(
                f"parallel_family_cache: duplicate provenance record for shard_index={sidx}"
            )
        seen_indices.add(sidx)
        mentry = manifest_by_idx.get(sidx)
        if mentry is None:
            raise RuntimeError(
                f"parallel_family_cache: provenance contains extra shard_index={sidx} "
                "that is absent from the manifest"
            )
        if fam_provenance not in KNOWN_FAMILIES:
            raise RuntimeError(
                f"parallel_family_cache: shard_index={sidx} has unknown family {fam_provenance!r}"
            )
        source_file = mentry.get("source_file", "")
        fam_from_source, source_id = _family_and_source_from_source_file(source_file)
        if fam_from_source is None:
            raise RuntimeError(
                f"parallel_family_cache: cannot derive a known family from source_file "
                f"for shard_index={sidx}: {source_file!r}"
            )
        if fam_from_source != fam_provenance:
            raise RuntimeError(
                f"parallel_family_cache: shard_index={sidx} provenance family "
                f"{fam_provenance!r} disagrees with source_file family {fam_from_source!r}"
            )
        for count_name in ("docs", "tokens"):
            manifest_count = mentry.get(count_name)
            provenance_count = rec.get(count_name)
            if (
                not isinstance(manifest_count, int)
                or isinstance(manifest_count, bool)
                or manifest_count < 0
            ):
                raise RuntimeError(
                    f"parallel_family_cache: manifest shard_index={sidx} requires a "
                    f"non-negative integer {count_name} count"
                )
            if (
                not isinstance(provenance_count, int)
                or isinstance(provenance_count, bool)
                or provenance_count != manifest_count
            ):
                raise RuntimeError(
                    f"parallel_family_cache: shard_index={sidx} {count_name} count "
                    f"differs: manifest={manifest_count!r}, "
                    f"provenance={provenance_count!r}"
                )
        if rec.get("source_id") != source_id:
            raise RuntimeError(
                f"parallel_family_cache: shard_index={sidx} source_id differs: "
                f"manifest-derived={source_id!r}, provenance={rec.get('source_id')!r}"
            )
        shard_tokens = mentry["tokens"]
        expected_total_tokens += shard_tokens
        expected_total_docs += mentry["docs"]
        expected_per_source_tokens[source_id] = (
            expected_per_source_tokens.get(source_id, 0) + shard_tokens
        )
        expected_per_family_tokens[fam_from_source] = (
            expected_per_family_tokens.get(fam_from_source, 0) + shard_tokens
        )
        by_family.setdefault(fam_provenance, []).append(mentry)

    manifest_indices = set(manifest_by_idx)
    missing_indices = manifest_indices - seen_indices
    if missing_indices:
        raise RuntimeError(
            f"parallel_family_cache: provenance is missing {len(missing_indices)} manifest "
            f"shards, including {sorted(missing_indices)[:5]}; regenerate provenance"
        )
    manifest_aggregates = {
        "total_tokens": manifest.get("total_tokens"),
        "total_docs": manifest.get("total_docs"),
    }
    derived_aggregates = {
        "total_tokens": expected_total_tokens,
        "total_docs": expected_total_docs,
    }
    if manifest_aggregates != derived_aggregates:
        raise RuntimeError(
            "parallel_family_cache: manifest aggregate counts do not equal its "
            f"per-shard counts: manifest={manifest_aggregates!r}, "
            f"derived={derived_aggregates!r}"
        )
    expected_summary = {
        **derived_aggregates,
        "per_source_tokens": expected_per_source_tokens,
        "per_family_tokens": expected_per_family_tokens,
        "per_family_share": {
            family: (tokens / expected_total_tokens if expected_total_tokens else 0)
            for family, tokens in expected_per_family_tokens.items()
        },
    }
    for field, expected in expected_summary.items():
        if split_info.get(field) != expected:
            raise RuntimeError(
                f"parallel_family_cache: provenance {field} does not match manifest "
                f"counts: expected={expected!r}, got={split_info.get(field)!r}"
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
    strict_manifest_schema: bool = False,
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

    manifest = _load_manifest(
        cache_dir, require_vocab_size=strict_manifest_schema
    )
    dtype = _dtype_from_str(manifest["dtype"], manifest["byte_order"])
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    with open(manifest_path, "rb") as manifest_file:
        manifest_sha256 = hashlib.sha256(manifest_file.read()).hexdigest()
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
    resume_supplied = resume_state_dict is not None
    if resume_supplied and not isinstance(resume_state_dict, dict):
        raise RuntimeError("family resume state must be an object")
    resume = resume_state_dict if resume_supplied else {}
    expected_schedule = [[fam, count] for fam, count in family_schedule]
    expected_families = {fam for fam, _count in family_schedule}
    if resume_supplied:
        if resume.get("loader_strategy") != "parallel_family_cache":
            raise RuntimeError(
                "resume loader_strategy does not match parallel_family_cache"
            )
        if resume.get("family_schedule") != expected_schedule:
            raise RuntimeError(
                f"resume family_schedule {resume.get('family_schedule')!r} does not "
                f"match runtime family_schedule {expected_schedule!r}"
            )
        expected_contract = {
            "grad_accum_steps": grad_accum_steps,
            "batch_size": B,
            "sequence_length": T,
            "cache_manifest_sha256": manifest_sha256,
        }
        for key, expected in expected_contract.items():
            if resume.get(key) != expected:
                raise RuntimeError(
                    f"resume {key}={resume.get(key)!r} does not match runtime {expected!r}"
                )
        for cursor_key in (
            "family_cursors", "family_token_cursors", "family_wrap_counts"
        ):
            cursor_map = resume.get(cursor_key)
            if not isinstance(cursor_map, dict) or set(cursor_map) != expected_families:
                raise RuntimeError(
                    f"resume {cursor_key} family set does not match runtime families"
                )
            if any(not isinstance(value, int) or value < 0 for value in cursor_map.values()):
                raise RuntimeError(
                    f"resume {cursor_key} values must be non-negative integers"
                )
        saved_microbatch = resume.get("microbatch_index")
        if not isinstance(saved_microbatch, int) or not 0 <= saved_microbatch < grad_accum_steps:
            raise RuntimeError(
                "resume microbatch_index must be an integer within grad_accum_steps"
            )
        for fam in expected_families:
            family_cursor = resume["family_cursors"][fam]
            if family_cursor >= len(family_shards[fam]):
                raise RuntimeError(
                    f"resume family cursor for {fam}={family_cursor} is outside "
                    f"{len(family_shards[fam])} shards"
                )
            selected_shard = family_shards[fam][family_cursor]
            shard_tokens = selected_shard["tokens"]
            token_cursor = resume["family_token_cursors"][fam]
            if token_cursor >= shard_tokens:
                raise RuntimeError(
                    f"resume token cursor for {fam}={token_cursor} is outside "
                    f"selected shard length {shard_tokens}; shard boundaries use "
                    "the canonical next-shard, offset-zero representation"
                )
    family_cursors: dict[str, int] = {fam: 0 for fam, _ in family_schedule}
    family_token_cursors: dict[str, int] = {fam: 0 for fam, _ in family_schedule}
    family_wrap_counts: dict[str, int] = {fam: 0 for fam, _ in family_schedule}
    microbatch_index = 0
    if resume_supplied:
        family_cursors.update(resume.get("family_cursors", {}))
        family_token_cursors.update(resume.get("family_token_cursors", {}))
        family_wrap_counts.update(resume.get("family_wrap_counts", {}))
        microbatch_index = resume["microbatch_index"]

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
            local_idx = family_cursors[fam]
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
            "family_schedule": expected_schedule,
            "grad_accum_steps": grad_accum_steps,
            "batch_size": B,
            "sequence_length": T,
            "cache_manifest_sha256": manifest_sha256,
        }
        yield inputs, targets, state
        microbatch_index = next_mb


def cached_family_balanced_data_loader(*args, **kwargs):
    """Helper that drops the state dict, mirroring the sequential variant."""
    for inputs, targets, _ in cached_family_balanced_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
