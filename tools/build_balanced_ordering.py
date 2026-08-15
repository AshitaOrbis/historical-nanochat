"""Design-C offline bake: build a family-stratified, seeded, deterministic shard
ordering for the sequential DDP cached loader.

Reads the final token cache (v4 two-dir layout) plus its provenance.json,
validates them against each other and against the files on disk (refuse-to-start
guards, per the governed_v4 run-1 postmortem), and writes a sidecar
`shard_ordering.json` into the cache dir. The existing cache_manifest.json and
provenance.json are never modified.

The ordering algorithm is a token-mass stride interleave over per-family seeded
shuffles: each family's shards are spread as evenly as possible across the whole
sequence by token mass, so striping positions mod world_size hands every rank a
near-iid draw of the global family mix (alias-proof; no rigid block pattern).

Pure stdlib, CPU-only, deterministic. Usage:

    python tools/build_balanced_ordering.py \
        --cache-dir ~/historical-nanochat/data/token_cache_v4_balanced_candidate/train \
        --seed 1913
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

KNOWN_FAMILIES = (
    "books_general", "newspapers_periodicals", "legal_government",
    "science_technical", "early_modern",
)

DTYPE_BYTES = {"uint16": 2, "uint32": 4}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_from_source_file(source_file: str) -> str | None:
    name = source_file.split("/")[-1]
    if name.startswith("shard_"):
        name = name[6:]
    for fam in KNOWN_FAMILIES:
        if name.startswith(fam + "_"):
            return fam
    return None


def load_inputs(cache_dir: str, provenance_path: str):
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"REFUSE: no cache_manifest.json in {cache_dir}")
    if not os.path.exists(provenance_path):
        raise SystemExit(f"REFUSE: no provenance.json at {provenance_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(provenance_path) as f:
        provenance = json.load(f)
    return manifest, provenance, manifest_path


def resolve_split(cache_dir: str) -> str:
    for k in ("train", "val"):
        if cache_dir.rstrip("/").endswith("/" + k):
            return k
    raise SystemExit(
        f"REFUSE: cache_dir {cache_dir} does not end in /train or /val "
        "(need the v4 two-dir layout to pick the provenance split)."
    )


def validate(cache_dir: str, manifest: dict, provenance: dict, split: str,
             check_files: bool = True) -> list[dict]:
    """Refuse-to-start guards. Returns manifest entries joined with family."""
    shards = manifest["shards"]
    per_shard = provenance.get("splits", {}).get(split, {}).get("per_shard", [])
    errors = []

    if len(per_shard) != len(shards):
        errors.append(
            f"entry-count mismatch: manifest has {len(shards)} shards, "
            f"provenance split '{split}' has {len(per_shard)}"
        )

    prov_by_idx = {r["shard_index"]: r for r in per_shard}
    seen_names: dict[str, int] = {}
    dtype_bytes = DTYPE_BYTES.get(manifest.get("dtype", "uint16"), 2)
    joined = []
    family_mismatches = 0
    missing_in_prov = 0

    for entry in shards:
        sidx = entry["shard_index"]
        fn = entry.get("filename") or f"shard_{sidx:05d}.bin"
        if fn in seen_names:
            errors.append(f"duplicate filename {fn} (shard_index {sidx} and {seen_names[fn]})")
        seen_names[fn] = sidx

        rec = prov_by_idx.get(sidx)
        if rec is None:
            missing_in_prov += 1
            continue
        fam = rec["family"]
        fam_src = family_from_source_file(entry.get("source_file", ""))
        if fam_src is not None and fam_src != fam:
            family_mismatches += 1
        if rec.get("tokens") is not None and rec["tokens"] != entry.get("tokens"):
            errors.append(f"{fn}: token count differs manifest={entry.get('tokens')} provenance={rec['tokens']}")

        if check_files:
            path = os.path.join(cache_dir, fn)
            if not os.path.exists(path):
                errors.append(f"{fn}: file missing on disk")
            else:
                actual = os.path.getsize(path)
                expected = entry.get("bytes", entry["tokens"] * dtype_bytes)
                if actual != expected:
                    errors.append(f"{fn}: size {actual} != expected {expected}")

        joined.append({
            "filename": fn,
            "shard_index": sidx,
            "family": fam,
            "tokens": entry["tokens"],
            "source_file": entry.get("source_file", ""),
        })

    if missing_in_prov:
        errors.append(f"{missing_in_prov} manifest shards missing from provenance (require 100% coverage)")
    if family_mismatches:
        errors.append(f"{family_mismatches} shards with provenance family != source_file family (require 0)")

    if errors:
        for e in errors[:40]:
            print(f"REFUSE: {e}", file=sys.stderr)
        if len(errors) > 40:
            print(f"REFUSE: ... and {len(errors) - 40} more", file=sys.stderr)
        raise SystemExit(f"REFUSE: {len(errors)} validation error(s); no ordering written.")
    return joined


def stratified_order(joined: list[dict], seed: int) -> list[dict]:
    """Per-family seeded shuffle, then token-mass stride interleave."""
    by_family: dict[str, list[dict]] = defaultdict(list)
    for e in joined:
        by_family[e["family"]].append(e)

    keyed = []
    for fam in sorted(by_family):
        entries = sorted(by_family[fam], key=lambda e: e["shard_index"])
        rng = random.Random(f"{seed}:{fam}")
        rng.shuffle(entries)
        total = sum(e["tokens"] for e in entries)
        cum = 0
        for e in entries:
            ideal = (cum + e["tokens"] / 2.0) / total  # fraction of family mass, in (0,1)
            keyed.append((ideal, fam, e["shard_index"], e))
            cum += e["tokens"]
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in keyed]


def evenness_report(ordered: list[dict]) -> dict:
    """Max/mean token-gap between consecutive same-family shards, per family."""
    total = sum(e["tokens"] for e in ordered)
    positions: dict[str, list[int]] = defaultdict(list)
    cum = 0
    for e in ordered:
        positions[e["family"]].append(cum)
        cum += e["tokens"]
    rep = {}
    for fam, pos in positions.items():
        fam_tokens = sum(e["tokens"] for e in ordered if e["family"] == fam)
        expected_gap = total / max(1, len(pos))
        gaps = [b - a for a, b in zip(pos, pos[1:])]
        rep[fam] = {
            "shards": len(pos),
            "tokens": fam_tokens,
            "share": round(fam_tokens / total, 6),
            "max_gap_tokens": max(gaps) if gaps else 0,
            "mean_gap_tokens": round(sum(gaps) / len(gaps), 1) if gaps else 0,
            "expected_gap_tokens": round(expected_gap, 1),
        }
    return rep


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, help=".../train cache dir (v4 layout)")
    ap.add_argument("--provenance", default=None, help="default: <cache-dir>/../provenance.json")
    ap.add_argument("--seed", type=int, default=1913)
    ap.add_argument("--out", default=None, help="default: <cache-dir>/shard_ordering.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing ordering file")
    ap.add_argument("--skip-file-checks", action="store_true",
                    help="skip on-disk existence/size checks (tests only)")
    args = ap.parse_args(argv)

    cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    provenance_path = args.provenance or os.path.join(os.path.dirname(cache_dir.rstrip("/")), "provenance.json")
    out_path = args.out or os.path.join(cache_dir, "shard_ordering.json")
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(f"REFUSE: {out_path} exists (use --force to overwrite)")

    split = resolve_split(cache_dir)
    manifest, provenance, manifest_path = load_inputs(cache_dir, provenance_path)
    joined = validate(cache_dir, manifest, provenance, split, check_files=not args.skip_file_checks)
    ordered = stratified_order(joined, args.seed)

    order_names = [e["filename"] for e in ordered]
    order_sha = hashlib.sha256("\n".join(order_names).encode()).hexdigest()
    doc = {
        "version": 1,
        "algorithm": "family_stratified_token_stride_v1",
        "seed": args.seed,
        "split": split,
        "cache_dir": cache_dir,
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_sha256": sha256_file(provenance_path),
        "num_shards": len(ordered),
        "total_tokens": sum(e["tokens"] for e in ordered),
        "family_report": evenness_report(ordered),
        "order_sha256": order_sha,
        "order": order_names,
        "shards": {e["filename"]: {"tokens": e["tokens"], "family": e["family"],
                                   "shard_index": e["shard_index"]} for e in ordered},
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, out_path)

    print(f"Wrote {out_path}")
    print(f"  shards={doc['num_shards']}  tokens={doc['total_tokens']:,}  seed={args.seed}")
    print(f"  order_sha256={order_sha}")
    print(f"  file_sha256={sha256_file(out_path)}  <- record in launch manifest")
    for fam, rep in sorted(doc["family_report"].items()):
        print(f"  {fam:24s} share={rep['share']:.4f} shards={rep['shards']:6d} "
              f"max_gap={rep['max_gap_tokens']:,} (expected~{rep['expected_gap_tokens']:,})")
    return out_path


if __name__ == "__main__":
    main()
