"""Burn-day regressions for fail-closed training provenance and identity gates.

All fixtures are tiny and synthetic. No real checkpoint or corpus is opened.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nanochat"))


def _write_cache(root: Path, *, shard_count: int = 2, tokens: int = 64,
                 vocab_size: int = 256) -> tuple[Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_index in range(shard_count):
        filename = f"shard_{shard_index:05d}.bin"
        values = np.arange(tokens, dtype=np.dtype("<u2")) % vocab_size
        values.tofile(root / filename)
        shards.append({
            "shard_index": shard_index,
            "filename": filename,
            "source_file": (
                f"/source/shard_books_general_source_{shard_index:06d}.parquet"
            ),
            "docs": 1,
            "tokens": int(values.size),
            "bytes": int(values.nbytes),
        })
    manifest = {
        "format_version": 1,
        "byte_order": "little",
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "total_docs": shard_count,
        "total_tokens": shard_count * tokens,
        "shards": shards,
    }
    (root / "cache_manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def _write_family_cache(root: Path) -> Path:
    train, manifest = _write_cache(root / "train", shard_count=1, tokens=32)
    manifest_sha = hashlib.sha256((train / "cache_manifest.json").read_bytes()).hexdigest()
    (root / "provenance.json").write_text(json.dumps({
        "splits": {
            "train": {
                "manifest_sha256": manifest_sha,
                "total_tokens": 32,
                "total_docs": 1,
                "per_source_tokens": {"source": 32},
                "per_family_tokens": {"books_general": 32},
                "per_family_share": {"books_general": 1.0},
                "per_shard": [
                    {"shard_index": manifest["shards"][0]["shard_index"],
                     "source_id": "source", "family": "books_general",
                     "docs": 1, "tokens": 32},
                ],
            },
        },
    }))
    return train


def _identity(tokenizer_sha: str = "a" * 64) -> dict:
    return {
        "contract_version": 2,
        "build_config_sha256": "c" * 64,
        "tokenizer_sha256": tokenizer_sha,
        "token_bytes_sha256": "b" * 64,
        "vocab_size": 8,
        "bos_id": 7,
        "token_bytes_dtype": "int32",
        "token_bytes_shape": [8],
    }


def test_ordering_hash_is_derived_instead_of_self_attested(tmp_path):
    from nanochat.dataloader_cached import _maybe_apply_shard_ordering

    cache, manifest = _write_cache(tmp_path / "cache")
    ordered_names = [entry["filename"] for entry in reversed(manifest["shards"])]
    stale_names = [entry["filename"] for entry in manifest["shards"]]
    manifest_sha = hashlib.sha256((cache / "cache_manifest.json").read_bytes()).hexdigest()
    (cache / "shard_ordering.json").write_text(json.dumps({
        "version": 1,
        "manifest_sha256": manifest_sha,
        "order": ordered_names,
        "order_sha256": hashlib.sha256("\n".join(stale_names).encode()).hexdigest(),
    }))

    with pytest.raises(RuntimeError, match="order_sha256"):
        _maybe_apply_shard_ordering(str(cache), manifest["shards"])


def test_manifest_rejects_sampled_token_outside_declared_vocab(tmp_path):
    from nanochat.dataloader_cached import _load_manifest

    cache, manifest = _write_cache(tmp_path / "cache", shard_count=1, tokens=8,
                                   vocab_size=8)
    values = np.arange(8, dtype=np.dtype("<u2"))
    values[-1] = 99
    values.tofile(cache / manifest["shards"][0]["filename"])

    with pytest.raises(RuntimeError, match="vocab_size|token id"):
        _load_manifest(str(cache))


def test_manifest_rejects_duplicate_filenames(tmp_path):
    from nanochat.dataloader_cached import _load_manifest

    cache, manifest = _write_cache(tmp_path / "cache")
    manifest["shards"][1]["filename"] = manifest["shards"][0]["filename"]
    (cache / "cache_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filename|duplicate"):
        _load_manifest(str(cache))


def test_strict_manifest_rejects_missing_vocab_identity(tmp_path):
    from nanochat.dataloader_cached import _load_manifest

    cache, manifest = _write_cache(tmp_path / "cache", shard_count=1)
    manifest.pop("vocab_size")
    (cache / "cache_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="vocab_size|tokenizer_identity"):
        _load_manifest(str(cache), require_vocab_size=True)


def test_sequential_resume_is_bound_to_batch_geometry(tmp_path):
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state

    cache, _ = _write_cache(tmp_path / "cache")
    first = cached_distributed_data_loader_with_state(
        B=1, T=4, split="all", device="cpu", cache_dir=str(cache)
    )
    _, _, state = next(first)
    resumed = cached_distributed_data_loader_with_state(
        B=2, T=4, split="all", device="cpu", cache_dir=str(cache),
        resume_state_dict=state,
    )
    with pytest.raises(RuntimeError, match="batch_size|resume contract"):
        next(resumed)


def test_sequential_empty_resume_state_is_not_treated_as_fresh(tmp_path):
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state

    cache, _ = _write_cache(tmp_path / "cache")
    resumed = cached_distributed_data_loader_with_state(
        B=1, T=4, split="all", device="cpu", cache_dir=str(cache),
        resume_state_dict={},
    )
    with pytest.raises(RuntimeError, match="resume|per_rank"):
        next(resumed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shard_idx", -1),
        ("shard_idx", 2),
        ("token_off", -1),
        ("token_off", 64),
    ],
)
def test_sequential_resume_rejects_noncanonical_cursor(tmp_path, field, value):
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state

    cache, _ = _write_cache(tmp_path / "cache")
    loader = cached_distributed_data_loader_with_state(
        B=1, T=4, split="all", device="cpu", cache_dir=str(cache)
    )
    _, _, state = next(loader)
    state["per_rank"]["0"][field] = value
    resumed = cached_distributed_data_loader_with_state(
        B=1, T=4, split="all", device="cpu", cache_dir=str(cache),
        resume_state_dict=state,
    )
    with pytest.raises(RuntimeError, match="cursor|shard|token"):
        next(resumed)


@pytest.mark.parametrize(
    "cursor_key,value,error",
    [
        ("family_cursors", -1, "non-negative"),
        ("family_cursors", 1, "family cursor"),
        ("family_token_cursors", -1, "non-negative"),
        ("family_token_cursors", 32, "token cursor"),
    ],
)
def test_family_resume_rejects_out_of_domain_cursor(
    tmp_path, cursor_key, value, error,
):
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    train = _write_family_cache(tmp_path)
    schedule = [("books_general", 1)]
    loader = cached_family_balanced_data_loader_with_state(
        B=1, T=3, split="all", device="cpu", cache_dir=str(train),
        grad_accum_steps=1, family_schedule=schedule,
    )
    _, _, state = next(loader)
    state[cursor_key]["books_general"] = value
    resumed = cached_family_balanced_data_loader_with_state(
        B=1, T=3, split="all", device="cpu", cache_dir=str(train),
        grad_accum_steps=1, family_schedule=schedule, resume_state_dict=state,
    )
    with pytest.raises(RuntimeError, match=error):
        next(resumed)


def test_family_empty_resume_state_is_not_treated_as_fresh(tmp_path):
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    train = _write_family_cache(tmp_path)
    resumed = cached_family_balanced_data_loader_with_state(
        B=1, T=3, split="all", device="cpu", cache_dir=str(train),
        grad_accum_steps=1, family_schedule=[("books_general", 1)],
        resume_state_dict={},
    )
    with pytest.raises(RuntimeError, match="resume|loader_strategy"):
        next(resumed)


def _canary_doc() -> dict:
    return {
        "ordering_sha256": "a" * 64,
        "needed_per_yield": 5,
        "world_size": 1,
        "grad_accum": 1,
        "canaries": [
            {"after_yield": 1, "rank": 0, "position": 0, "offset": 5},
        ],
    }


def test_canary_gate_rejects_missing_runtime_order_identity(tmp_path):
    from nanochat.train_guards import load_canaries

    path = tmp_path / "canaries.json"
    path.write_text(json.dumps(_canary_doc()))
    with pytest.raises(RuntimeError, match="ordering_sha256|ordering identity"):
        load_canaries(
            str(path), needed=5, world_size=1, grad_accum=1,
            ordering_sha256=None,
        )


def test_canary_gate_rejects_absent_file(tmp_path):
    from nanochat.train_guards import load_canaries

    with pytest.raises(FileNotFoundError, match="NOT_RUN"):
        load_canaries(
            str(tmp_path / "absent.json"), needed=5, world_size=1, grad_accum=1,
            ordering_sha256="a" * 64,
        )


def test_canary_gate_binds_file_to_runtime_run_id(tmp_path):
    from nanochat.train_guards import load_canaries

    assert "run_id" in inspect.signature(load_canaries).parameters
    path = tmp_path / "canaries.json"
    document = _canary_doc()
    document["run_id"] = "run-a"
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="run_id"):
        load_canaries(
            str(path), needed=5, world_size=1, grad_accum=1,
            ordering_sha256="a" * 64, run_id="run-b",
        )


def test_expected_config_gate_rejects_zero_checked_leaves():
    from nanochat.train_guards import assert_expected_resolved

    with pytest.raises(RuntimeError, match="zero|empty|checked"):
        assert_expected_resolved({}, {})


def test_expected_config_gate_rejects_partial_schema():
    from nanochat.train_guards import assert_expected_resolved

    partial = {"model": {"vocab_size": 8}}
    with pytest.raises(RuntimeError, match="schema|required|missing"):
        assert_expected_resolved(partial, partial, require_full_schema=True)


def test_expected_config_gate_rejects_empty_identity_input():
    from nanochat.train_guards import assert_expected_resolved

    expected = json.loads(
        (REPO / "runs/launch_2026-07-06/expected_resolved_d26.json").read_text()
    )
    expected["data"]["ordering_order_sha256"] = None
    with pytest.raises(RuntimeError, match="identity|SHA|NOT_RUN|non-empty"):
        assert_expected_resolved(expected, expected, require_full_schema=True)


def test_resume_requires_a_future_canary():
    import nanochat.train_guards as guards

    require_future_canary = getattr(guards, "require_future_canary", None)
    assert require_future_canary is not None, "future-canary guard is not implemented"
    with pytest.raises(RuntimeError, match="future canary"):
        require_future_canary({1, 2}, consumed_yields=2)


def test_raw_loader_refuses_single_shard_before_returning_generator(monkeypatch):
    import nanochat.dataloader as dataloader

    monkeypatch.setattr(dataloader, "list_parquet_files", lambda data_dir=None: ["one.parquet"])
    with pytest.raises(RuntimeError, match="at least two|validation"):
        dataloader.tokenizing_distributed_data_loader_with_state(
            B=1, T=4, split="train", device="cpu", parquet_dir="unused"
        )


@pytest.mark.parametrize("split", ["train", "val"])
def test_cached_legacy_split_refuses_single_shard(tmp_path, split):
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state

    cache, _ = _write_cache(tmp_path / "cache", shard_count=1)
    loader = cached_distributed_data_loader_with_state(
        B=1, T=4, split=split, device="cpu", cache_dir=str(cache)
    )
    with pytest.raises(RuntimeError, match="at least two|validation cache"):
        next(loader)


class _FakeTokenizer:
    def get_vocab_size(self):
        return 8

    def get_bos_token_id(self):
        return 7


class _FakeModel:
    def __init__(self, config):
        self.config = config

    def to_empty(self, **_kwargs):
        return self

    def init_weights(self):
        return None

    def load_state_dict(self, *_args, **_kwargs):
        return None

    def eval(self):
        return self

    def train(self):
        return self


def test_central_model_loader_rejects_checkpoint_tokenizer_mismatch(monkeypatch):
    import nanochat.checkpoint_manager as manager

    runtime_identity = _identity()
    checkpoint_identity = _identity(tokenizer_sha="c" * 64)
    monkeypatch.setattr(
        manager,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            {}, None,
            {"model_config": {"vocab_size": 8},
             "artifact_identity": checkpoint_identity},
        ),
    )
    monkeypatch.setattr(manager, "GPTConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(manager, "GPT", _FakeModel)
    monkeypatch.setattr(manager, "get_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(
        manager, "get_token_bytes", lambda device="cpu": torch.arange(8, dtype=torch.int32),
        raising=False,
    )
    monkeypatch.setattr(
        manager, "validate_tokenizer_artifacts",
        lambda *_args, **_kwargs: runtime_identity,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="identity|tokenizer"):
        manager.build_model("unused", 1, torch.device("cpu"), "eval")


def test_direct_checkpoint_load_rejects_tokenizer_mismatch(tmp_path, monkeypatch):
    import nanochat.checkpoint_manager as manager

    runtime_identity = _identity()
    checkpoint_identity = _identity(tokenizer_sha="c" * 64)
    torch.save({}, tmp_path / "model_000001.pt")
    (tmp_path / "meta_000001.json").write_text(json.dumps({
        "model_config": {"vocab_size": 8},
        "artifact_identity": checkpoint_identity,
    }))
    monkeypatch.setattr(manager, "get_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(
        manager, "get_token_bytes", lambda device="cpu": torch.arange(8, dtype=torch.int32)
    )
    monkeypatch.setattr(
        manager, "validate_tokenizer_artifacts", lambda *_args: runtime_identity
    )

    with pytest.raises(RuntimeError, match="identity|tokenizer"):
        manager.load_checkpoint(str(tmp_path), 1, "cpu")


def test_direct_checkpoint_load_rejects_model_vocab_mismatch(tmp_path, monkeypatch):
    import nanochat.checkpoint_manager as manager

    runtime_identity = _identity()
    torch.save({}, tmp_path / "model_000001.pt")
    (tmp_path / "meta_000001.json").write_text(json.dumps({
        "model_config": {"vocab_size": 9},
        "artifact_identity": runtime_identity,
    }))
    monkeypatch.setattr(manager, "get_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(
        manager, "get_token_bytes", lambda device="cpu": torch.arange(8, dtype=torch.int32)
    )
    monkeypatch.setattr(
        manager, "validate_tokenizer_artifacts", lambda *_args: runtime_identity
    )

    with pytest.raises(RuntimeError, match="vocab_size"):
        manager.load_checkpoint(str(tmp_path), 1, "cpu")


def test_checkpoint_save_propagates_validated_parent_identity(tmp_path, monkeypatch):
    import nanochat.checkpoint_manager as manager

    identity = _identity()
    monkeypatch.setattr(manager, "_ACTIVE_ARTIFACT_IDENTITY", identity, raising=False)
    manager.save_checkpoint(
        str(tmp_path), 1, {"weight": torch.zeros(1)}, None,
        {"step": 1, "model_config": {"vocab_size": 8}}, rank=0,
    )
    metadata = json.loads((tmp_path / "meta_000001.json").read_text())
    assert metadata["artifact_identity"] == identity
