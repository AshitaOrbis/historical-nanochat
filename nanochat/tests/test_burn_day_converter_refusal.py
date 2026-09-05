"""The legacy converter must fail before emitting a mislabeled HF artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
CONVERTER = REPO / "convert_to_hf.py"


def load_converter_module():
    spec = importlib.util.spec_from_file_location("historical_nanochat_converter", CONVERTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_refuses_incompatible_gpt2_export_before_writing(tmp_path):
    converter = load_converter_module()
    output_dir = tmp_path / "hf-model"

    with pytest.raises(RuntimeError) as exc_info:
        converter.convert_nanochat_to_hf("missing-checkpoint", 1, str(output_dir))

    message = str(exc_info.value)
    assert "not GPT-2-compatible" in message
    for architecture_feature in (
        "RoPE",
        "QK normalization",
        "parameter-free RMSNorm",
        "ReLU-squared",
        "GQA",
        "untied embeddings",
    ):
        assert architecture_feature in message
    assert "logits parity" in message
    assert not output_dir.exists()
