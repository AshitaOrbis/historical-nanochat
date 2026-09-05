"""R2 regressions for controls that previously validated only their wrappers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nanochat"))
sys.path.insert(0, str(REPO / "tools"))


def _write_counted_family_cache(root: Path) -> Path:
    train = root / "train"
    train.mkdir(parents=True)
    values = np.arange(8, dtype=np.dtype("<u2"))
    values.tofile(train / "shard_00000.bin")
    manifest = {
        "format_version": 1,
        "byte_order": "little",
        "dtype": "uint16",
        "vocab_size": 256,
        "total_docs": 3,
        "total_tokens": 8,
        "shards": [
            {
                "shard_index": 0,
                "filename": "shard_00000.bin",
                "source_file": "/source/shard_books_general_source_000000.parquet",
                "docs": 3,
                "tokens": 8,
                "bytes": 16,
            }
        ],
    }
    manifest_path = train / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "provenance.json").write_text(json.dumps({
        "splits": {
            "train": {
                "manifest_sha256": manifest_sha256,
                "total_tokens": 8,
                "total_docs": 3,
                "per_source_tokens": {"source": 8},
                "per_family_tokens": {"books_general": 8},
                "per_family_share": {"books_general": 1.0},
                "per_shard": [
                    {
                        "shard_index": 0,
                        "source_id": "source",
                        "family": "books_general",
                        "docs": 3,
                        "tokens": 9,
                    }
                ],
            }
        }
    }))
    return train


def test_provenance_rejects_valid_manifest_hash_with_altered_token_count(tmp_path):
    from nanochat.dataloader_cached import _load_family_shard_lists

    train = _write_counted_family_cache(tmp_path)
    with pytest.raises(RuntimeError, match="token|count"):
        _load_family_shard_lists(str(train))


def _write_simulator_cache(root: Path) -> Path:
    train = root / "train"
    train.mkdir(parents=True)
    shards = []
    order = []
    ordering_shards = {}
    for shard_index in range(2):
        filename = f"shard_{shard_index:05d}.bin"
        order.append(filename)
        entry = {
            "shard_index": shard_index,
            "filename": filename,
            "source_file": f"/source/shard_books_general_source_{shard_index:06d}.parquet",
            "docs": 1,
            "tokens": 20,
            "bytes": 40,
        }
        shards.append(entry)
        ordering_shards[filename] = {
            "shard_index": shard_index,
            "family": "books_general",
            "tokens": 20,
        }
    manifest_path = train / "cache_manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "byte_order": "little",
        "dtype": "uint16",
        "total_docs": 2,
        "total_tokens": 40,
        "shards": shards,
    }))
    (train / "shard_ordering.json").write_text(json.dumps({
        "version": 1,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "order": order,
        "order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        "shards": ordering_shards,
    }))
    return train


def test_canary_producer_emits_run_id_and_legacy_files_report_not_run(tmp_path):
    import simulate_ddp_traversal as simulator
    from nanochat.train_guards import load_canaries

    train = _write_simulator_cache(tmp_path)
    canary_path = tmp_path / "new-canaries.json"
    report = simulator.run([
        "--cache-dir", str(train),
        "--world-size", "1",
        "--device-batch", "1",
        "--seq-len", "2",
        "--steps", "1",
        "--canaries", "1",
        "--run-id", "bd-nano-loader-r2-20260823",
        "--canary-out", str(canary_path),
    ])
    assert report["gate"] == "PASS"
    generated = json.loads(canary_path.read_text())
    assert generated["run_id"] == "bd-nano-loader-r2-20260823"
    loaded = load_canaries(
        str(canary_path), needed=3, world_size=1, grad_accum=1,
        ordering_sha256=generated["ordering_sha256"],
        run_id="bd-nano-loader-r2-20260823",
    )
    assert loaded.status == "PASS"

    legacy_paths = [
        REPO / "runs/design_c_tier1_2026-07-02/canaries.json",
        REPO / "runs/design_c_tier1_2026-07-02/canaries_world1.json",
        REPO / "runs/launch_2026-07-06/canaries_world1_reverify.json",
    ]
    for legacy_path in legacy_paths:
        legacy = json.loads(legacy_path.read_text())
        outcome = load_canaries(
            str(legacy_path),
            needed=legacy["needed_per_yield"],
            world_size=legacy["world_size"],
            grad_accum=legacy["grad_accum"],
            ordering_sha256=legacy["ordering_sha256"],
            run_id="bd-nano-loader-r2-20260823",
            allow_legacy_missing_run_id=True,
        )
        assert outcome.status == "NOT_RUN"
        assert "run_id" in outcome.reason
        assert not outcome


def test_raw_loader_rejects_out_of_domain_resume_parquet_index(
    monkeypatch,
):
    import nanochat.dataloader as dataloader

    monkeypatch.setattr(
        dataloader,
        "list_parquet_files",
        lambda data_dir=None: ["zero.parquet", "one.parquet", "validation.parquet"],
    )
    for resume_pq_idx in (-1, 2):
        with pytest.raises(RuntimeError, match="pq_idx|parquet|resume"):
            dataloader.tokenizing_distributed_data_loader_with_state(
                B=1,
                T=4,
                split="train",
                device="cpu",
                parquet_dir="unused",
                resume_state_dict={"pq_idx": resume_pq_idx, "rg_idx": 0},
            )


class _FakeTokenizer:
    def get_vocab_size(self):
        return 8

    def get_bos_token_id(self):
        return 7


def _write_tokenizer_bundle(base_dir: Path, *, max_chars: int) -> torch.Tensor:
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_dir / "tokenizer.pkl"
    token_bytes_path = tokenizer_dir / "token_bytes.npy"
    tokenizer_path.write_bytes(b"same-tokenizer-output")
    values = np.arange(8, dtype=np.int32)
    np.save(token_bytes_path, values, allow_pickle=False)
    (tokenizer_dir / "tokenizer_manifest.json").write_text(json.dumps({
        "script_version": "tokenizer-builder-2",
        "git_commit": "a" * 40,
        "training": {"max_chars": max_chars, "doc_cap": 10_000},
        "inputs": {"corpus_dir": "data/shards"},
        "tokenizer": {
            "vocab_size": 8,
            "special_tokens": {"<|bos|>": 7},
        },
        "outputs": {
            "tokenizer_pkl": "tokenizer/tokenizer.pkl",
            "sha256_tokenizer_pkl": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "token_bytes_npy": "tokenizer/token_bytes.npy",
            "sha256_token_bytes_npy": hashlib.sha256(token_bytes_path.read_bytes()).hexdigest(),
        },
    }))
    return torch.from_numpy(values.copy())


def test_tokenizer_identity_binds_build_configuration(tmp_path):
    from nanochat.artifact_guard import (
        validate_identity_binding,
        validate_tokenizer_artifacts,
    )

    token_bytes = _write_tokenizer_bundle(tmp_path, max_chars=1_000_000)
    first = validate_tokenizer_artifacts(tmp_path, _FakeTokenizer(), token_bytes)
    _write_tokenizer_bundle(tmp_path, max_chars=2_000_000)
    second = validate_tokenizer_artifacts(tmp_path, _FakeTokenizer(), token_bytes)

    assert first["build_config_sha256"] != second["build_config_sha256"]
    with pytest.raises(RuntimeError, match="build_config_sha256"):
        validate_identity_binding("checkpoint", first, second)
