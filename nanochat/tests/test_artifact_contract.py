from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    def __init__(self, vocab_size: int = 8, bos_id: int = 7):
        self.vocab_size = vocab_size
        self.bos_id = bos_id

    def get_vocab_size(self):
        return self.vocab_size

    def get_bos_token_id(self):
        return self.bos_id


def _guard_module():
    spec = importlib.util.find_spec("nanochat.artifact_guard")
    return importlib.import_module("nanochat.artifact_guard") if spec else None


def _write_artifacts(base_dir: Path):
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    tokenizer_path = tokenizer_dir / "tokenizer.pkl"
    token_bytes_path = tokenizer_dir / "token_bytes.npy"
    tokenizer_path.write_bytes(b"trusted-tokenizer-fixture")
    values = np.arange(8, dtype=np.int32)
    values[-1] = 0
    np.save(token_bytes_path, values, allow_pickle=False)
    manifest = {
        "tokenizer": {
            "format": "test",
            "vocab_size": 8,
            "special_tokens": {"<|bos|>": 7},
        },
        "outputs": {
            "tokenizer_pkl": "tokenizer/tokenizer.pkl",
            "sha256_tokenizer_pkl": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "token_bytes_npy": "tokenizer/token_bytes.npy",
            "sha256_token_bytes_npy": hashlib.sha256(token_bytes_path.read_bytes()).hexdigest(),
        },
    }
    manifest_path = tokenizer_dir / "tokenizer_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, torch.from_numpy(values.copy())


def test_safe_token_bytes_loader_uses_non_pickle_numpy(tmp_path, monkeypatch):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    expected = np.arange(8, dtype=np.int32)
    np.save(tokenizer_dir / "token_bytes.npy", expected, allow_pickle=False)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    from nanochat.tokenizer import get_token_bytes

    actual = get_token_bytes(device="cpu")
    assert actual.dtype == torch.int32
    assert actual.tolist() == expected.tolist()


def test_tokenizer_artifact_contract_accepts_exact_bundle(tmp_path):
    guard = _guard_module()
    assert guard is not None, "nanochat.artifact_guard is missing"
    _, token_bytes = _write_artifacts(tmp_path)
    identity = guard.validate_tokenizer_artifacts(
        tmp_path, FakeTokenizer(), token_bytes
    )
    assert identity["vocab_size"] == 8
    assert identity["bos_id"] == 7
    assert identity["token_bytes_dtype"] == "int32"
    assert identity["token_bytes_shape"] == [8]


@pytest.mark.parametrize(
    "mutation",
    ["missing_manifest", "malformed_manifest", "missing_field", "tokenizer_hash", "bytes_hash",
     "vocab", "bos", "dtype", "shape"],
)
def test_tokenizer_artifact_contract_fails_closed(tmp_path, mutation):
    guard = _guard_module()
    assert guard is not None, "nanochat.artifact_guard is missing"
    manifest_path, token_bytes = _write_artifacts(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    tokenizer = FakeTokenizer()
    if mutation == "missing_manifest":
        manifest_path.unlink()
    elif mutation == "malformed_manifest":
        manifest_path.write_text("{broken")
    elif mutation == "missing_field":
        del manifest["outputs"]["sha256_tokenizer_pkl"]
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "tokenizer_hash":
        manifest["outputs"]["sha256_tokenizer_pkl"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "bytes_hash":
        manifest["outputs"]["sha256_token_bytes_npy"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "vocab":
        tokenizer.vocab_size = 9
    elif mutation == "bos":
        tokenizer.bos_id = 6
    elif mutation == "dtype":
        token_bytes = token_bytes.to(torch.int64)
    elif mutation == "shape":
        token_bytes = token_bytes.reshape(2, 4)

    with pytest.raises((FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError)):
        guard.validate_tokenizer_artifacts(tmp_path, tokenizer, token_bytes)


def test_checkpoint_and_cache_identity_are_mandatory(tmp_path):
    guard = _guard_module()
    assert guard is not None, "nanochat.artifact_guard is missing"
    identity = {
        "contract_version": 1,
        "tokenizer_sha256": "a" * 64,
        "token_bytes_sha256": "b" * 64,
        "vocab_size": 8,
        "bos_id": 7,
        "token_bytes_dtype": "int32",
        "token_bytes_shape": [8],
    }
    with pytest.raises(RuntimeError):
        guard.validate_identity_binding("checkpoint", None, identity)
    with pytest.raises(RuntimeError):
        guard.validate_identity_binding(
            "checkpoint", {**identity, "tokenizer_sha256": "c" * 64}, identity
        )

    cache = tmp_path / "cache"
    cache.mkdir()
    manifest_path = cache / "cache_manifest.json"
    manifest_path.write_text(json.dumps({"format_version": 1, "byte_order": "little"}))
    with pytest.raises(RuntimeError):
        guard.validate_cache_tokenizer_identity(cache, identity)
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "byte_order": "little",
        "tokenizer_identity": identity,
    }))
    assert guard.validate_cache_tokenizer_identity(cache, identity) == identity


def test_repository_bundle_and_docs_match_published_contract():
    readme = (REPO / "README.md").read_text()
    manifest = json.loads((REPO / "tokenizer" / "tokenizer_manifest.json").read_text())
    token_bytes_path = REPO / manifest["outputs"]["token_bytes_npy"]
    assert token_bytes_path.exists()
    values = np.load(token_bytes_path, allow_pickle=False)
    assert values.dtype == np.int32
    assert values.ndim == 1
    assert hashlib.sha256(token_bytes_path.read_bytes()).hexdigest() == manifest["outputs"]["sha256_token_bytes_npy"]
    assert "token_bytes_pt" not in json.dumps(manifest)

    for relative in [
        "data/download/gutenberg_download.py",
        "data/download/oldbailey_download.py",
        "data/download/chronicling_download.py",
        "data/download/caselaw_download.py",
        "data/process/contamination_check.py",
        "data/process/shard_packager.py",
    ]:
        assert (REPO / relative).is_file(), f"published workflow source is missing: {relative}"
    assert 'packages = ["data"]' in (REPO / "pyproject.toml").read_text()

    assert "genuinely don't know" not in readme
    assert "ensuring genuine temporal ignorance" not in readme
    assert "publication-date metadata" in readme
    assert "must be measured" in readme
    assert "semantic anachronisms" in readme


def test_base_train_uses_strict_identity_gate_by_default():
    source = (REPO / "nanochat" / "scripts" / "base_train.py").read_text()
    assert "validate_tokenizer_artifacts(" in source
    assert "validate_cache_tokenizer_identity(" in source
    assert 'validate_identity_binding("checkpoint"' in source
    assert '"artifact_identity": artifact_identity' in source
