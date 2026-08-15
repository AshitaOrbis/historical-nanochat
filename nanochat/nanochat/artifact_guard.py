"""Fail-closed tokenizer, cache, and checkpoint identity validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


IDENTITY_KEYS = (
    "contract_version",
    "tokenizer_sha256",
    "token_bytes_sha256",
    "vocab_size",
    "bos_id",
    "token_bytes_dtype",
    "token_bytes_shape",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(mapping: dict, key: str, location: str):
    if key not in mapping:
        raise RuntimeError(f"tokenizer manifest is missing required field {location}.{key}")
    return mapping[key]


def _artifact_path(base_dir: Path, relative: str, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"tokenizer manifest field {field} must be a non-empty path")
    base_dir = base_dir.resolve()
    path = (base_dir / relative).resolve()
    if path != base_dir and base_dir not in path.parents:
        raise RuntimeError(f"tokenizer manifest field {field} escapes the base directory")
    if not path.is_file():
        raise FileNotFoundError(f"required tokenizer artifact is missing: {path}")
    return path


def validate_tokenizer_artifacts(base_dir, tokenizer, token_bytes: torch.Tensor) -> dict:
    """Validate the complete tokenizer bundle and return its canonical identity."""
    base_dir = Path(base_dir)
    manifest_path = base_dir / "tokenizer" / "tokenizer_manifest.json"
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise RuntimeError("tokenizer manifest must be a JSON object")
    tokenizer_doc = _required(manifest, "tokenizer", "root")
    outputs = _required(manifest, "outputs", "root")
    if not isinstance(tokenizer_doc, dict) or not isinstance(outputs, dict):
        raise RuntimeError("tokenizer manifest tokenizer and outputs fields must be objects")

    manifest_vocab = _required(tokenizer_doc, "vocab_size", "tokenizer")
    special_tokens = _required(tokenizer_doc, "special_tokens", "tokenizer")
    if not isinstance(special_tokens, dict):
        raise RuntimeError("tokenizer.special_tokens must be an object")
    manifest_bos = _required(special_tokens, "<|bos|>", "tokenizer.special_tokens")

    if "tokenizer_pkl" in outputs:
        tokenizer_path_field = "tokenizer_pkl"
        tokenizer_sha_field = "sha256_tokenizer_pkl"
    elif "tokenizer_json" in outputs:
        tokenizer_path_field = "tokenizer_json"
        tokenizer_sha_field = "sha256_tokenizer_json"
    else:
        raise RuntimeError(
            "tokenizer manifest outputs must declare tokenizer_pkl or tokenizer_json"
        )
    tokenizer_path = _artifact_path(
        base_dir,
        _required(outputs, tokenizer_path_field, "outputs"),
        f"outputs.{tokenizer_path_field}",
    )
    token_bytes_path = _artifact_path(
        base_dir,
        _required(outputs, "token_bytes_npy", "outputs"),
        "outputs.token_bytes_npy",
    )
    expected_tokenizer_sha = _required(outputs, tokenizer_sha_field, "outputs")
    expected_token_bytes_sha = _required(outputs, "sha256_token_bytes_npy", "outputs")
    actual_tokenizer_sha = sha256_file(tokenizer_path)
    actual_token_bytes_sha = sha256_file(token_bytes_path)
    if actual_tokenizer_sha != expected_tokenizer_sha:
        raise RuntimeError(
            "tokenizer artifact SHA-256 mismatch: "
            f"manifest={expected_tokenizer_sha}, actual={actual_tokenizer_sha}"
        )
    if actual_token_bytes_sha != expected_token_bytes_sha:
        raise RuntimeError(
            "token-byte artifact SHA-256 mismatch: "
            f"manifest={expected_token_bytes_sha}, actual={actual_token_bytes_sha}"
        )

    runtime_vocab = tokenizer.get_vocab_size()
    runtime_bos = tokenizer.get_bos_token_id()
    if manifest_vocab != runtime_vocab:
        raise RuntimeError(
            f"tokenizer vocab mismatch: manifest={manifest_vocab}, runtime={runtime_vocab}"
        )
    if manifest_bos != runtime_bos:
        raise RuntimeError(
            f"tokenizer BOS mismatch: manifest={manifest_bos}, runtime={runtime_bos}"
        )
    if not isinstance(token_bytes, torch.Tensor):
        raise RuntimeError("token bytes must load as a torch.Tensor")
    if token_bytes.dtype != torch.int32:
        raise RuntimeError(
            f"token bytes dtype must be torch.int32, got {token_bytes.dtype}"
        )
    if token_bytes.ndim != 1:
        raise RuntimeError(
            f"token bytes must be one-dimensional, got shape {list(token_bytes.shape)}"
        )
    if int(token_bytes.shape[0]) != runtime_vocab:
        raise RuntimeError(
            f"token bytes length {int(token_bytes.shape[0])} != tokenizer vocab {runtime_vocab}"
        )

    return {
        "contract_version": 1,
        "tokenizer_sha256": actual_tokenizer_sha,
        "token_bytes_sha256": actual_token_bytes_sha,
        "vocab_size": runtime_vocab,
        "bos_id": runtime_bos,
        "token_bytes_dtype": "int32",
        "token_bytes_shape": [int(token_bytes.shape[0])],
    }


def validate_identity_binding(label: str, candidate, expected: dict) -> dict:
    """Require an exact artifact identity in a cache or checkpoint."""
    if not isinstance(candidate, dict):
        raise RuntimeError(f"{label} is missing required tokenizer artifact identity")
    missing = [key for key in IDENTITY_KEYS if key not in candidate]
    if missing:
        raise RuntimeError(f"{label} tokenizer identity is missing fields: {missing}")
    normalized = {key: candidate[key] for key in IDENTITY_KEYS}
    if normalized != expected:
        mismatches = [
            key for key in IDENTITY_KEYS if normalized.get(key) != expected.get(key)
        ]
        raise RuntimeError(
            f"{label} tokenizer identity does not match runtime artifacts: {mismatches}"
        )
    return normalized


def validate_cache_tokenizer_identity(cache_dir, expected: dict) -> dict:
    manifest_path = Path(cache_dir) / "cache_manifest.json"
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{manifest_path}: cache manifest must be a JSON object")
    return validate_identity_binding(
        f"cache manifest {manifest_path}", manifest.get("tokenizer_identity"), expected
    )
