#!/usr/bin/env python3
"""Build exact, manifest-hash-bound family provenance for token caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


KNOWN_FAMILIES = (
    "books_general",
    "newspapers_periodicals",
    "legal_government",
    "science_technical",
    "early_modern",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_family_source(source_file: str) -> tuple[str, str]:
    stem = Path(source_file).stem
    if stem.startswith("shard_"):
        stem = stem[6:]
    stem = re.sub(r"_\d{6}$", "", stem)
    for family in KNOWN_FAMILIES:
        prefix = family + "_"
        if stem.startswith(prefix):
            return family, stem[len(prefix):]
    raise ValueError(f"cannot derive a known family from source_file={source_file!r}")


def build_provenance(cache_root) -> dict:
    cache_root = Path(cache_root)
    document = {"cache_root": str(cache_root), "splits": {}}
    for split in ("train", "val"):
        manifest_path = cache_root / split / "cache_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise ValueError(f"{manifest_path}: shards must be a list")
        per_shard = []
        per_source = Counter()
        per_family = Counter()
        seen_indices = set()
        for entry in shards:
            shard_index = entry.get("shard_index")
            if not isinstance(shard_index, int) or shard_index in seen_indices:
                raise ValueError(
                    f"{manifest_path}: shard_index values must be unique integers"
                )
            seen_indices.add(shard_index)
            family, source_id = split_family_source(entry.get("source_file", ""))
            tokens = entry.get("tokens")
            if not isinstance(tokens, int) or tokens < 0:
                raise ValueError(
                    f"{manifest_path}: shard {shard_index} requires integer tokens"
                )
            per_source[source_id] += tokens
            per_family[family] += tokens
            per_shard.append({
                "shard_index": shard_index,
                "source_id": source_id,
                "family": family,
                "docs": entry.get("docs"),
                "tokens": tokens,
            })
        total_tokens = sum(per_family.values())
        document["splits"][split] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "total_tokens": total_tokens,
            "total_docs": manifest.get("total_docs", 0),
            "per_source_tokens": dict(per_source),
            "per_family_tokens": dict(per_family),
            "per_family_share": {
                key: (value / total_tokens if total_tokens else 0)
                for key, value in per_family.items()
            },
            "per_shard": per_shard,
        }
    if not document["splits"]:
        raise FileNotFoundError(
            f"no train/cache_manifest.json or val/cache_manifest.json under {cache_root}"
        )
    output_path = cache_root / "provenance.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(document, indent=2) + "\n")
    temporary_path.replace(output_path)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    document = build_provenance(args.cache_root)
    print(f"wrote {Path(args.cache_root) / 'provenance.json'}")
    for split, info in document["splits"].items():
        print(
            f"{split}: {len(info['per_shard'])} shards, "
            f"manifest_sha256={info['manifest_sha256']}"
        )


if __name__ == "__main__":
    main()
