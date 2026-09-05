"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from nanochat.artifact_guard import (
    IDENTITY_KEYS,
    validate_identity_binding,
    validate_tokenizer_artifacts,
)
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
_ACTIVE_ARTIFACT_IDENTITY = None


def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _atomic_torch_save(obj, path):
    """torch.save to a temp file, then atomic rename. A crash or a concurrent
    sidecar sync can never observe a partially-written file under its final name."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_complete_sentinel(checkpoint_dir, step, world_size, has_optimizer):
    """Write complete_<step>.json AFTER verifying every expected file of this
    checkpoint exists. Sidecar sync tooling must only copy checkpoints that have
    this sentinel; resume should prefer the newest COMPLETE step."""
    expected = [f"model_{step:06d}.pt", f"meta_{step:06d}.json"]
    if has_optimizer:
        expected += [f"optim_{step:06d}_rank{r:d}.pt" for r in range(world_size)]
    files = {}
    for name in expected:
        path = os.path.join(checkpoint_dir, name)
        if not os.path.exists(path):
            raise RuntimeError(f"checkpoint step {step} incomplete: missing {name}; not writing sentinel")
        files[name] = os.path.getsize(path)
    sentinel_path = os.path.join(checkpoint_dir, f"complete_{step:06d}.json")
    tmp = sentinel_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"step": step, "world_size": world_size, "files": files}, f, indent=2)
    os.replace(tmp, sentinel_path)
    logger.info(f"Checkpoint step {step} complete: {sentinel_path}")


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    global _ACTIVE_ARTIFACT_IDENTITY
    meta_data = dict(meta_data)
    # Utility-only sentinel tests may omit model_config. Every real model
    # checkpoint must either declare a complete identity (base training) or
    # inherit the identity validated by the central parent-model load path
    # (mid/SFT/RL phase transitions).
    if "model_config" in meta_data:
        candidate_identity = meta_data.get("artifact_identity")
        if _ACTIVE_ARTIFACT_IDENTITY is not None:
            if candidate_identity is None:
                meta_data["artifact_identity"] = dict(_ACTIVE_ARTIFACT_IDENTITY)
            else:
                validate_identity_binding(
                    "checkpoint being written",
                    candidate_identity,
                    _ACTIVE_ARTIFACT_IDENTITY,
                )
        else:
            if not isinstance(candidate_identity, dict):
                raise RuntimeError(
                    "checkpoint metadata is missing required tokenizer artifact identity"
                )
            missing = [key for key in IDENTITY_KEYS if key not in candidate_identity]
            if missing:
                raise RuntimeError(
                    f"checkpoint tokenizer identity is missing fields: {missing}"
                )
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        _atomic_torch_save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        os.replace(tmp_meta, meta_path)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        _atomic_torch_save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    # All files land atomically; now mark the checkpoint complete. Under DDP,
    # wait for every rank's optimizer shard before rank 0 writes the sentinel.
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        torch.distributed.barrier()
    if rank == 0:
        _write_complete_sentinel(checkpoint_dir, step, world_size, optimizer_data is not None)

def save_bf16_snapshot(checkpoint_dir, step, model_state):
    """Model-only bf16 trajectory snapshot (Sol P0-7): ~2 GiB for d26 vs ~12 GiB
    for a full checkpoint. NOT resumable (no optimizer state) — these exist so
    warmdown-start and the other trajectory marks survive without retaining
    full checkpoints. Rank 0 only; atomic."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    snap = {
        k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v.detach())
        for k, v in model_state.items()
    }
    path = os.path.join(checkpoint_dir, f"snapshot_bf16_{step:06d}.pt")
    _atomic_torch_save(snap, path)
    logger.info(f"Saved bf16 model snapshot to: {path}")
    return path


def _validate_loaded_checkpoint_identity(meta_data):
    """Bind every direct checkpoint consumer to the validated tokenizer bundle."""
    global _ACTIVE_ARTIFACT_IDENTITY
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device="cpu")
    runtime_identity = validate_tokenizer_artifacts(
        get_base_dir(), tokenizer, token_bytes
    )
    validate_identity_binding(
        "checkpoint", meta_data.get("artifact_identity"), runtime_identity
    )
    model_config = meta_data.get("model_config")
    if not isinstance(model_config, dict):
        raise RuntimeError("checkpoint metadata is missing required model_config")
    model_vocab_size = model_config.get("vocab_size")
    if model_vocab_size != runtime_identity["vocab_size"]:
        raise RuntimeError(
            f"checkpoint model_config.vocab_size={model_vocab_size!r} does not match "
            f"validated tokenizer vocab_size={runtime_identity['vocab_size']}"
        )
    _ACTIVE_ARTIFACT_IDENTITY = dict(runtime_identity)
    return runtime_identity


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load and validate metadata first. A missing/mismatched identity aborts
    # before checkpoint tensors are deserialized or exposed to any caller.
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    _validate_loaded_checkpoint_identity(meta_data)
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    global _ACTIVE_ARTIFACT_IDENTITY
    if phase not in ("train", "eval"):
        raise ValueError(f"Invalid phase: {phase}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    device_type = device.type if hasattr(device, "type") else str(device).split(":", 1)[0]
    if device_type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")

    # The checkpoint/tokenizer join is centralized here so inference, evaluation,
    # and every downstream training phase receive the same mandatory gate.
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device="cpu")
    runtime_identity = validate_tokenizer_artifacts(
        get_base_dir(), tokenizer, token_bytes
    )
    validate_identity_binding(
        "checkpoint", meta_data.get("artifact_identity"), runtime_identity
    )
    runtime_vocab_size = tokenizer.get_vocab_size()
    if runtime_vocab_size != model_config_kwargs["vocab_size"]:
        raise RuntimeError(
            f"checkpoint model vocab_size={model_config_kwargs['vocab_size']} does not "
            f"match validated tokenizer vocab_size={runtime_vocab_size}"
        )
    _ACTIVE_ARTIFACT_IDENTITY = dict(runtime_identity)

    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Prefer the newest step with a completion sentinel (written after every
    # rank's file landed). Fall back to the legacy highest model_*.pt scan for
    # checkpoint dirs that predate the sentinel, with a warning.
    sentinels = glob.glob(os.path.join(checkpoint_dir, "complete_*.json"))
    if sentinels:
        return int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in sentinels))
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    log0(f"WARNING: no complete_*.json sentinels in {checkpoint_dir}; "
         "falling back to highest model_*.pt (cannot verify the checkpoint is complete)")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
