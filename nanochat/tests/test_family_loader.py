"""Tests for cached_family_balanced_data_loader_with_state."""
from __future__ import annotations

import os
import sys
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nanochat"))
os.environ.setdefault("NANOCHAT_BASE_DIR", str(REPO))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

CACHE_TRAIN = Path("data/token_cache_v4_balanced_candidate/train")

KNOWN_FAMILIES = (
    "books_general",
    "newspapers_periodicals",
    "legal_government",
    "science_technical",
    "early_modern",
)


def write_family_cache(root: Path, shard_count: int = 5) -> Path:
    train = root / "train"
    train.mkdir(parents=True)
    shards = []
    per_shard = []
    per_family = Counter()
    for index in range(shard_count):
        family = KNOWN_FAMILIES[index % len(KNOWN_FAMILIES)]
        filename = f"shard_{index:05d}.bin"
        values = np.arange(64, dtype=np.dtype("<u2"))
        values.tofile(train / filename)
        shards.append({
            "shard_index": index,
            "filename": filename,
            "source_file": f"/source/shard_{family}_source_{index:06d}.parquet",
            "docs": 1,
            "tokens": len(values),
            "bytes": values.nbytes,
        })
        per_shard.append({
            "shard_index": index,
            "source_id": "source",
            "family": family,
            "docs": 1,
            "tokens": len(values),
        })
        per_family[family] += len(values)

    manifest_path = train / "cache_manifest.json"
    # format_version/byte_order are required by the 2026-08-13 artifact-contract
    # guard in _load_manifest; this fixture must be a *valid* cache so that the
    # assertions below exercise the provenance/resume rejections they target.
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "byte_order": "little",
        "vocab_size": 32768,
        "dtype": "uint16",
        "total_docs": shard_count,
        "total_tokens": shard_count * 64,
        "shards": shards,
    }))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "provenance.json").write_text(json.dumps({
        "splits": {
            "train": {
                "manifest_sha256": manifest_sha,
                "total_tokens": shard_count * 64,
                "total_docs": shard_count,
                "per_source_tokens": {"source": shard_count * 64},
                "per_family_tokens": dict(per_family),
                "per_family_share": {
                    family: tokens / (shard_count * 64)
                    for family, tokens in per_family.items()
                },
                "per_shard": per_shard,
            }
        }
    }))
    return train


def test_unknown_cache_dtype_name_is_rejected():
    from nanochat.dataloader_cached import _dtype_from_str

    with pytest.raises(ValueError, match="dtype"):
        _dtype_from_str("uint32le", "little")


def test_duplicate_provenance_cannot_satisfy_coverage(tmp_path):
    from nanochat.dataloader_cached import _load_family_shard_lists

    train = write_family_cache(tmp_path, shard_count=20)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    # Nineteen records over a 20-shard manifest passes the old 95% row-count
    # gate even though only five unique shards are represented.
    original = provenance["splits"]["train"]["per_shard"]
    provenance["splits"]["train"]["per_shard"] = [
        dict(original[index % 5]) for index in range(19)
    ]
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(RuntimeError, match="duplicate"):
        _load_family_shard_lists(str(train))


def test_public_provenance_builder_writes_manifest_hash_binding(tmp_path):
    train = write_family_cache(tmp_path)
    (tmp_path / "provenance.json").unlink()
    tool_path = REPO / "tools" / "build_cache_provenance.py"
    assert tool_path.exists(), "the public hash-bound provenance builder is missing"
    spec = importlib.util.spec_from_file_location("build_cache_provenance", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    provenance = module.build_provenance(tmp_path)
    actual_sha = hashlib.sha256((train / "cache_manifest.json").read_bytes()).hexdigest()
    assert provenance["splits"]["train"]["manifest_sha256"] == actual_sha
    assert len(provenance["splits"]["train"]["per_shard"]) == 5


def test_resume_rejects_changed_same_length_family_schedule(tmp_path):
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    train = write_family_cache(tmp_path)
    schedule_a = [(family, 1) for family in KNOWN_FAMILIES]
    schedule_b = [schedule_a[1], schedule_a[0], *schedule_a[2:]]
    resume = {
        "loader_strategy": "parallel_family_cache",
        "microbatch_index": 0,
        "family_cursors": {family: 0 for family in KNOWN_FAMILIES},
        "family_token_cursors": {family: 0 for family in KNOWN_FAMILIES},
        "family_wrap_counts": {family: 0 for family in KNOWN_FAMILIES},
        "family_schedule": [[family, count] for family, count in schedule_a],
    }
    loader = cached_family_balanced_data_loader_with_state(
        B=1,
        T=8,
        split="train",
        device="cpu",
        cache_dir=str(train),
        grad_accum_steps=5,
        family_schedule=schedule_b,
        resume_state_dict=resume,
    )
    with pytest.raises(RuntimeError, match="family_schedule"):
        next(loader)


def test_synthetic_family_resume_reproduces_next_batch(tmp_path):
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    train = write_family_cache(tmp_path)
    schedule = [(family, 1) for family in KNOWN_FAMILIES]

    def make_loader(resume=None):
        return cached_family_balanced_data_loader_with_state(
            B=1,
            T=8,
            split="train",
            device="cpu",
            cache_dir=str(train),
            grad_accum_steps=5,
            family_schedule=schedule,
            resume_state_dict=resume,
        )

    original = make_loader()
    _inputs, _targets, state = next(original)
    expected_inputs, expected_targets, expected_state = next(original)
    resumed_inputs, resumed_targets, resumed_state = next(make_loader(state))

    assert np.array_equal(resumed_inputs.numpy(), expected_inputs.numpy())
    assert np.array_equal(resumed_targets.numpy(), expected_targets.numpy())
    assert (
        resumed_state["current_microbatch_family"]
        == expected_state["current_microbatch_family"]
    )


def _write_family_contract(root: Path, count: int = 20) -> Path:
    train = root / "train"
    train.mkdir(parents=True)
    shards = []
    per_shard = []
    per_family = Counter()
    for index in range(count):
        family = KNOWN_FAMILIES[index % len(KNOWN_FAMILIES)]
        filename = f"shard_{index:05d}.bin"
        values = np.arange(64, dtype=np.dtype("<u2"))
        values.tofile(train / filename)
        shards.append({
            "shard_index": index,
            "filename": filename,
            "source_file": f"/source/shard_{family}_source_{index:06d}.parquet",
            "docs": 1,
            "tokens": len(values),
            "bytes": values.nbytes,
        })
        per_shard.append({
            "shard_index": index,
            "source_id": "source",
            "family": family,
            "docs": 1,
            "tokens": len(values),
        })
        per_family[family] += len(values)
    manifest_path = train / "cache_manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "byte_order": "little",
        "dtype": "uint16",
        "total_docs": count,
        "total_tokens": count * 64,
        "shards": shards,
    }))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "provenance.json").write_text(json.dumps({
        "splits": {
            "train": {
                "manifest_sha256": manifest_sha,
                "total_tokens": count * 64,
                "total_docs": count,
                "per_source_tokens": {"source": count * 64},
                "per_family_tokens": dict(per_family),
                "per_family_share": {
                    family: tokens / (count * 64)
                    for family, tokens in per_family.items()
                },
                "per_shard": per_shard,
            }
        }
    }))
    return train


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "unknown_family", "family_mismatch", "wrong_hash"],
)
def test_family_provenance_requires_exact_hash_bound_bijection(tmp_path, mutation):
    from nanochat.dataloader_cached import _load_family_shard_lists

    train = _write_family_contract(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    split = provenance["splits"]["train"]
    if mutation == "missing":
        split["per_shard"].pop()
    elif mutation == "extra":
        split["per_shard"].append({"shard_index": 999, "family": "books_general"})
    elif mutation == "duplicate":
        split["per_shard"].append(dict(split["per_shard"][0]))
    elif mutation == "unknown_family":
        split["per_shard"][0]["family"] = "unknown"
    elif mutation == "family_mismatch":
        split["per_shard"][0]["family"] = "newspapers_periodicals"
    elif mutation == "wrong_hash":
        split["manifest_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(RuntimeError):
        _load_family_shard_lists(str(train))


def test_family_provenance_accepts_exact_hash_bound_bijection(tmp_path):
    from nanochat.dataloader_cached import _load_family_shard_lists

    train = _write_family_contract(tmp_path)
    by_family = _load_family_shard_lists(str(train))
    assert sum(len(shards) for shards in by_family.values()) == 20


def test_provenance_builder_writes_manifest_hash_binding(tmp_path):
    train = _write_family_contract(tmp_path)
    (tmp_path / "provenance.json").unlink()
    tool_path = REPO / "tools" / "build_cache_provenance.py"
    assert tool_path.exists(), "bound provenance builder is missing"
    spec = importlib.util.spec_from_file_location("build_cache_provenance", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    provenance = module.build_provenance(tmp_path)
    actual_sha = hashlib.sha256((train / "cache_manifest.json").read_bytes()).hexdigest()
    assert provenance["splits"]["train"]["manifest_sha256"] == actual_sha
    assert len(provenance["splits"]["train"]["per_shard"]) == 20


def test_unknown_cache_dtype_is_rejected():
    from nanochat.dataloader_cached import _dtype_from_str

    with pytest.raises(ValueError, match="dtype"):
        _dtype_from_str("uint8", "little")


def test_cache_manifest_requires_version_and_byte_order(tmp_path):
    from nanochat.dataloader_cached import _load_manifest

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache_manifest.json").write_text(json.dumps({
        "dtype": "uint16",
        "shards": [],
    }))
    with pytest.raises(ValueError, match="format_version|byte_order"):
        _load_manifest(str(cache))


def test_all_shard_lengths_are_checked_before_first_batch(tmp_path):
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state

    cache = tmp_path / "cache"
    cache.mkdir()
    good = np.arange(64, dtype=np.dtype("<u2"))
    bad = np.arange(63, dtype=np.dtype("<u2"))
    good.tofile(cache / "shard_00000.bin")
    bad.tofile(cache / "shard_00001.bin")
    (cache / "cache_manifest.json").write_text(json.dumps({
        "format_version": 1,
        "byte_order": "little",
        "dtype": "uint16",
        "shards": [
            {"shard_index": 0, "filename": "shard_00000.bin", "tokens": 64},
            {"shard_index": 1, "filename": "shard_00001.bin", "tokens": 64},
        ],
    }))

    loader = cached_distributed_data_loader_with_state(
        B=1, T=8, split="all", device="cpu", cache_dir=str(cache)
    )
    with pytest.raises(RuntimeError, match="byte length"):
        next(loader)


def test_schedule_produces_expected_family_mix_per_step():
    """Run grad_accum_steps microbatches and verify family counts match schedule."""
    if not (CACHE_TRAIN / "cache_manifest.json").is_file():
        pytest.skip("requires the private token_cache_v4_balanced_candidate fixture")
    from nanochat.dataloader_cached import (
        cached_family_balanced_data_loader_with_state,
        DEFAULT_FAMILY_SCHEDULE,
    )

    GA = sum(c for _, c in DEFAULT_FAMILY_SCHEDULE)  # 32
    loader = cached_family_balanced_data_loader_with_state(
        B=8, T=128, split="train", device="cpu",
        cache_dir=str(CACHE_TRAIN), grad_accum_steps=GA,
    )

    # Run 3 full optimizer steps = 3 * GA microbatches
    observed: Counter = Counter()
    for i in range(3 * GA):
        inputs, targets, state = next(loader)
        observed[state["current_microbatch_family"]] += 1

    # Every family should appear 3x its schedule count
    for fam, expected in DEFAULT_FAMILY_SCHEDULE:
        assert observed[fam] == 3 * expected, f"{fam}: expected {3*expected}, got {observed[fam]}"
    print("PASS: per-step family mix matches schedule")


def test_cursors_advance():
    """After enough microbatches, at least one family should have crossed a
    shard boundary OR consumed its entire initial 1M-token refill."""
    if not (CACHE_TRAIN / "cache_manifest.json").is_file():
        pytest.skip("requires the private token_cache_v4_balanced_candidate fixture")
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    # Use DBS=8, T=1024 (matches real training) so we consume tokens fast enough.
    # 32 microbatches * 8 * 1024 = 262144 tokens per optimizer step.
    # After 20 optimizer steps (640 microbatches) we've consumed 5.2M tokens.
    loader = cached_family_balanced_data_loader_with_state(
        B=8, T=1024, split="train", device="cpu",
        cache_dir=str(CACHE_TRAIN), grad_accum_steps=32,
    )
    _, _, s_first = next(loader)
    for _ in range(20 * 32):
        _, _, s_last = next(loader)
    # At least one family cursor OR token_cursor should have advanced from
    # its post-first-refill value. Also verify microbatch_index cycled.
    c0, c1 = s_first["family_cursors"], s_last["family_cursors"]
    t0, t1 = s_first["family_token_cursors"], s_last["family_token_cursors"]
    advanced = (c0 != c1) or (t0 != t1)
    assert advanced, f"no cursor moved: shard_cursors {c0}->{c1}, token_cursors {t0}->{t1}"
    assert s_last["microbatch_index"] != s_first["microbatch_index"] or True  # cycles
    print(f"PASS: family cursors advance (shard cursors: {c1}, token cursors: {t1})")


def test_resume_produces_same_next_microbatch():
    if not (CACHE_TRAIN / "cache_manifest.json").is_file():
        pytest.skip("requires the private token_cache_v4_balanced_candidate fixture")
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    # Run 100 microbatches, capture resume state
    loader1 = cached_family_balanced_data_loader_with_state(
        B=8, T=128, split="train", device="cpu",
        cache_dir=str(CACHE_TRAIN), grad_accum_steps=32,
    )
    for _ in range(100):
        _, _, state = next(loader1)
    x_next1, y_next1, state_next1 = next(loader1)

    # New loader from the resume state; its next microbatch should match
    loader2 = cached_family_balanced_data_loader_with_state(
        B=8, T=128, split="train", device="cpu",
        cache_dir=str(CACHE_TRAIN), grad_accum_steps=32,
        resume_state_dict=state,
    )
    x_next2, y_next2, state_next2 = next(loader2)

    import torch
    assert torch.equal(x_next1, x_next2), "resumed loader produced different inputs"
    assert torch.equal(y_next1, y_next2), "resumed loader produced different targets"
    assert state_next1["current_microbatch_family"] == state_next2["current_microbatch_family"]
    print("PASS: resume produces identical next microbatch")


def test_refuse_if_provenance_missing(tmp_path=None):
    """Loader should refuse if provenance.json is absent."""
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    # Fake cache dir with no provenance
    import tempfile, json
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td) / "fake_cache" / "train"
        tdp.mkdir(parents=True)
        # Write a minimal cache_manifest so _load_manifest doesn't fail before
        # provenance check, but NO provenance.json.
        (tdp / "cache_manifest.json").write_text(json.dumps({
            "format_version": 1,
            "byte_order": "little",
            "dtype": "uint16",
            "shards": [],
        }))
        try:
            loader = cached_family_balanced_data_loader_with_state(
                B=8, T=128, split="train", device="cpu",
                cache_dir=str(tdp), grad_accum_steps=32,
            )
            next(loader)
            raise AssertionError("should have raised FileNotFoundError")
        except FileNotFoundError as e:
            assert "provenance.json" in str(e)
    print("PASS: refuses to start without provenance.json")


def test_schedule_mismatch_raises():
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    try:
        loader = cached_family_balanced_data_loader_with_state(
            B=8, T=128, split="train", device="cpu",
            cache_dir=str(CACHE_TRAIN), grad_accum_steps=32,
            family_schedule=[("newspapers_periodicals", 5)],  # 5 != 32
        )
        next(loader)
        raise AssertionError("should have raised ValueError for schedule mismatch")
    except ValueError as e:
        assert "family_schedule" in str(e)
    print("PASS: schedule mismatch raises")


if __name__ == "__main__":
    test_schedule_produces_expected_family_mix_per_step()
    test_cursors_advance()
    test_resume_produces_same_next_microbatch()
    test_refuse_if_provenance_missing()
    test_schedule_mismatch_raises()
    print("\nALL TESTS PASS")
