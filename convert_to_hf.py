#!/usr/bin/env python3
"""Refuse the legacy, architecture-incompatible Hugging Face export path."""

from __future__ import annotations


class UnsupportedConversionError(RuntimeError):
    """Raised when a requested export cannot preserve the native model."""


UNSUPPORTED_MESSAGE = (
    "historical-nanochat is not GPT-2-compatible: the native model uses RoPE, "
    "QK normalization, parameter-free RMSNorm, ReLU-squared activations, optional "
    "GQA, bias-free projections, and untied embeddings. The former converter "
    "mislabelled these weights as GPT2LMHeadModel and could emit a corrupt artifact. "
    "Export is disabled until a matching Transformers config/model and complete "
    "tokenizer export pass strict loading with zero missing keys and deterministic "
    "native-vs-export logits parity."
)


def convert_nanochat_to_hf(checkpoint_dir: str, step: int, output_dir: str):
    """Refuse conversion before reading a checkpoint or creating output files."""

    del checkpoint_dir, step, output_dir
    raise UnsupportedConversionError(UNSUPPORTED_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(f"ERROR: {UNSUPPORTED_MESSAGE}")
