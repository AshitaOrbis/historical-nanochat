"""bq-1943: checkpoint verification must actually attempt a strict model load.

check_roundtrip() in tools/verify_first_checkpoint.py only proves a dict
serializes and deserializes to itself -- an empty dict or a dict containing
one made-up key round-trips just fine and reported passed=True. The module's
own docstring promises a separate check ("Model state dict loads without
missing/unexpected keys under strict=True") that was never implemented
anywhere in the file. check_strict_model_load() is the fix under test here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location(
        "historical_nanochat_verify_first_checkpoint",
        REPO / "tools" / "verify_first_checkpoint.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass._is_type() needs this in sys.modules
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier_module()


class _FakeTokenizer:
    def get_vocab_size(self):
        return 32

    def get_bos_token_id(self):
        return 7


def _identity() -> dict:
    return {
        "contract_version": 2,
        "build_config_sha256": "c" * 64,
        "tokenizer_sha256": "a" * 64,
        "token_bytes_sha256": "b" * 64,
        "vocab_size": 32,
        "bos_id": 7,
        "token_bytes_dtype": "int32",
        "token_bytes_shape": [32],
    }


TINY_CONFIG = {
    "sequence_len": 16,
    "vocab_size": 32,
    "n_layer": 1,
    "n_head": 2,
    "n_kv_head": 2,
    "n_embd": 8,
}


def _write_checkpoint(checkpoint_dir: Path, step: int, model_state: dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, checkpoint_dir / f"model_{step:06d}.pt")
    (checkpoint_dir / f"meta_{step:06d}.json").write_text(json.dumps({
        "model_config": TINY_CONFIG,
        "artifact_identity": _identity(),
    }))


def _patch_identity_gate(monkeypatch):
    import nanochat.checkpoint_manager as manager
    identity = _identity()
    monkeypatch.setattr(manager, "get_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(
        manager, "get_token_bytes", lambda device="cpu": torch.arange(32, dtype=torch.int32)
    )
    monkeypatch.setattr(manager, "validate_tokenizer_artifacts", lambda *_a, **_k: identity)


def _real_tiny_gpt_state_dict() -> dict:
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(**TINY_CONFIG)
    model = GPT(config)
    return model.state_dict()


@pytest.mark.parametrize("mutation", ["empty", "wrong_weight_name"])
def test_strict_load_rejects_non_model_states(tmp_path, monkeypatch, mutation):
    """bq-1943 trigger: model_002000.pt containing an empty dict, or a dict
    containing only not_a_model_weight, with parseable metadata."""
    _patch_identity_gate(monkeypatch)
    checkpoint_dir = tmp_path / "ckpt"
    if mutation == "empty":
        model_state = {}
    else:
        model_state = {"not_a_model_weight": torch.zeros(3)}
    _write_checkpoint(checkpoint_dir, 1, model_state)

    result = VERIFIER.check_strict_model_load(checkpoint_dir, 1, "cpu")

    assert result.passed is False, (
        f"strict load must FAIL for a non-model state ({mutation}), got passed=True: {result.detail}"
    )


def test_strict_load_accepts_a_genuine_matching_state_dict(tmp_path, monkeypatch):
    """Control: a real GPT(TINY_CONFIG).state_dict() must pass, proving the
    check isn't just failing everything."""
    _patch_identity_gate(monkeypatch)
    checkpoint_dir = tmp_path / "ckpt"
    _write_checkpoint(checkpoint_dir, 1, _real_tiny_gpt_state_dict())

    result = VERIFIER.check_strict_model_load(checkpoint_dir, 1, "cpu")

    assert result.passed is True, result.detail


def test_roundtrip_check_alone_is_fooled_by_an_empty_state(tmp_path, monkeypatch):
    """Documents the pre-existing gap check_strict_model_load closes: the
    round-trip check by itself passes on an empty dict, which is exactly the
    false assurance bq-1943 reports."""
    _patch_identity_gate(monkeypatch)
    checkpoint_dir = tmp_path / "ckpt"
    _write_checkpoint(checkpoint_dir, 1, {})

    roundtrip = VERIFIER.check_roundtrip(checkpoint_dir, 1, "cpu")
    strict = VERIFIER.check_strict_model_load(checkpoint_dir, 1, "cpu")

    assert roundtrip.passed is True
    assert strict.passed is False
