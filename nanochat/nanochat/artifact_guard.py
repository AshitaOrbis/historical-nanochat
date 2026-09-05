"""Fail-closed tokenizer, cache, and checkpoint identity validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


IDENTITY_KEYS = (
    "contract_version",
    "build_config_sha256",
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


def _build_config_sha256(manifest: dict) -> str:
    """Hash only reproducibility-relevant build inputs, not timestamps/timing."""
    training = manifest.get("training")
    inputs = manifest.get("inputs")
    if training is not None and not isinstance(training, dict):
        raise RuntimeError("tokenizer manifest training field must be an object")
    if inputs is not None and not isinstance(inputs, dict):
        raise RuntimeError("tokenizer manifest inputs field must be an object")
    script_version = manifest.get("script_version")
    if script_version is not None and (
        not isinstance(script_version, str) or not script_version
    ):
        raise RuntimeError(
            "tokenizer manifest script_version must be a non-empty string or null"
        )
    git_commit = manifest.get("git_commit")
    if git_commit is not None and (
        not isinstance(git_commit, str) or not git_commit
    ):
        raise RuntimeError(
            "tokenizer manifest git_commit must be a non-empty string or null"
        )
    training = training or {}
    build_config = {
        "script_version": script_version,
        "git_commit": git_commit,
        "training": {
            "max_chars": training.get("max_chars"),
            "doc_cap": training.get("doc_cap"),
        },
        "inputs": inputs,
    }
    canonical = json.dumps(
        build_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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

    tokenizer_dir = base_dir.resolve() / "tokenizer"
    if "tokenizer_pkl" in outputs:
        tokenizer_path_field = "tokenizer_pkl"
        tokenizer_sha_field = "sha256_tokenizer_pkl"
        canonical_tokenizer_name = "tokenizer.pkl"
    elif "tokenizer_json" in outputs:
        tokenizer_path_field = "tokenizer_json"
        tokenizer_sha_field = "sha256_tokenizer_json"
        canonical_tokenizer_name = "tokenizer.json"
    else:
        raise RuntimeError(
            "tokenizer manifest outputs must declare tokenizer_pkl or tokenizer_json"
        )
    tokenizer_path = _artifact_path(
        base_dir,
        _required(outputs, tokenizer_path_field, "outputs"),
        f"outputs.{tokenizer_path_field}",
    )
    # get_tokenizer() never reads the manifest: it always loads from the fixed
    # canonical filename for the declared format. If the manifest points somewhere
    # else, the file we hash below is not the file that will actually be loaded.
    canonical_tokenizer_path = (tokenizer_dir / canonical_tokenizer_name).resolve()
    if tokenizer_path != canonical_tokenizer_path:
        raise RuntimeError(
            f"outputs.{tokenizer_path_field} must resolve to the canonical loader "
            f"path {canonical_tokenizer_path}, got {tokenizer_path}: get_tokenizer() "
            "only ever reads the fixed canonical filename, so a manifest pointing "
            "elsewhere never describes the artifact that is actually loaded"
        )
    if canonical_tokenizer_name == "tokenizer.json" and (tokenizer_dir / "tokenizer.pkl").is_file():
        # get_tokenizer() prefers tokenizer.pkl unconditionally when it exists on
        # disk, regardless of which format the manifest declares. A stale pickle
        # left over from an earlier build would silently be loaded instead of the
        # JSON tokenizer this manifest attests.
        raise RuntimeError(
            "a stale tokenizer.pkl exists alongside a tokenizer_json manifest entry; "
            "get_tokenizer() prefers tokenizer.pkl unconditionally and would load a "
            "different tokenizer than the one this manifest attests. Remove the stale "
            "pickle or regenerate the manifest to describe it."
        )
    token_bytes_path = _artifact_path(
        base_dir,
        _required(outputs, "token_bytes_npy", "outputs"),
        "outputs.token_bytes_npy",
    )
    # get_token_bytes() likewise always reads the canonical token_bytes.npy path;
    # an alternate manifest-declared byte table is never the table actually loaded.
    canonical_token_bytes_path = (tokenizer_dir / "token_bytes.npy").resolve()
    if token_bytes_path != canonical_token_bytes_path:
        raise RuntimeError(
            f"outputs.token_bytes_npy must resolve to the canonical loader path "
            f"{canonical_token_bytes_path}, got {token_bytes_path}: get_token_bytes() "
            "always reads the fixed canonical filename regardless of manifest content"
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
        "contract_version": 2,
        "build_config_sha256": _build_config_sha256(manifest),
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
