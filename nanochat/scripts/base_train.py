"""
Train model. From root directory of the project, run as:

python -m scripts.base_train.py

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20

3090-friendly memory knobs: --activation_checkpoint --chunked_loss --max_seq_len=1024 --device_batch_size=4
"""

import csv
import json
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.dataloader_cached import (
    cached_distributed_data_loader,
    cached_distributed_data_loader_with_state,
    cached_family_balanced_data_loader_with_state,
)
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head_dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max_seq_len", type=int, default=2048, help="max context length")
parser.add_argument("--kv_head_ratio", type=float, default=1.0, help="GQA: n_kv_head / n_head (1.0 = MHA, 0.5/0.25 = GQA)")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num_iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target_flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target_param_data_ratio", type=int, default=8, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device_batch_size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total_batch_size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding_lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding_lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--matrix_lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--adam_beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown_ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--max_steps", type=int, default=-1,
                    help="Early-terminate training at this step while keeping the LR schedule keyed off "
                         "--num_iterations (or --target_param_data_ratio). Useful for smoke tests that "
                         "simulate long-run dynamics without sitting through warmdown.")
parser.add_argument("--diagnostic_logging", action="store_true",
                    help="Log grad norm, param norm, per-group LR, shard provenance each step.")
parser.add_argument("--final_lr_frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume_from_step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Memory / throughput knobs (3090-friendly)
parser.add_argument("--activation_checkpoint", action="store_true", help="enable activation checkpointing on transformer blocks")
parser.add_argument("--ckpt_every_n_blocks", type=int, default=1, help="checkpoint every Nth block (1 = all blocks)")
parser.add_argument("--chunked_loss", action="store_true", help="compute LM-head logits and CE in chunks to reduce peak VRAM")
parser.add_argument("--loss_chunk_size", type=int, default=1024, help="chunk size (flattened tokens) for chunked loss")
parser.add_argument("--compile_mode", type=str, default="default",
                    choices=["default", "reduce-overhead", "max-autotune", "none"],
                    help="torch.compile mode ('none' disables). Respects TORCH_COMPILE_DISABLE env var.")
parser.add_argument("--sdpa_backend", type=str, default="auto",
                    choices=["auto", "flash", "efficient", "math"],
                    help="Preferred SDPA backend (auto = let PyTorch decide).")
# Sequence-length curriculum: train most steps at max_seq_len, optionally bump near the end.
parser.add_argument("--seq_len_late", type=int, default=-1,
                    help="Late-stage sequence length (-1 = disabled). Must be <= rotary cache (10x max_seq_len by default).")
parser.add_argument("--seq_len_late_frac", type=float, default=0.8,
                    help="Fraction of training after which to switch to seq_len_late (default: 0.8).")
# Data
parser.add_argument("--parquet_dir", type=str, default=None,
                    help="Directory of pretraining parquet shards. Overrides NANOCHAT_PARQUET_DIR.")
parser.add_argument("--token_cache_dir", type=str, default=None,
                    help="Directory of pre-tokenized .bin shards built by scripts/build_token_cache.py. "
                         "When set, training uses the cached loader and tokenization cost is zero.")
parser.add_argument("--val_cache_dir", type=str, default=None,
                    help="Separate val-split cache dir. If set, val loader reads from this dir "
                         "with split='all'; train reads from --token_cache_dir with split='all'. "
                         "Otherwise the legacy convention is used: last shard of --token_cache_dir "
                         "is val, all others are train.")
parser.add_argument("--loader_strategy", default="sequential_cache",
                    choices=["sequential_cache", "parallel_family_cache"],
                    help="How the cached loader serves microbatches. sequential_cache is the "
                         "legacy behavior (consume shards end-to-end). parallel_family_cache "
                         "uses per-family cursors and a fixed per-step family schedule so each "
                         "optimizer step contains a balanced mix of source families.")
# Evaluation
parser.add_argument("--eval_every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval_tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core_metric_every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core_metric_max_per_task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample_every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save_every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Observability
parser.add_argument("--benchmark_csv", type=str, default=None,
                    help="Optional CSV path to append per-step throughput/memory metrics (rank 0 only).")
# Output
parser.add_argument("--model_tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# SDPA backend selection + diagnostics (CUDA only).
# Uses the legacy torch.backends.cuda.enable_*_sdp API for process-global preference.
# This is advisory: PyTorch still falls back if the requested kernel can't handle the inputs.
if device_type == "cuda":
    backends = torch.backends.cuda
    if args.sdpa_backend == "flash":
        backends.enable_flash_sdp(True); backends.enable_mem_efficient_sdp(False); backends.enable_math_sdp(False)
    elif args.sdpa_backend == "efficient":
        backends.enable_flash_sdp(False); backends.enable_mem_efficient_sdp(True); backends.enable_math_sdp(False)
    elif args.sdpa_backend == "math":
        backends.enable_flash_sdp(False); backends.enable_mem_efficient_sdp(False); backends.enable_math_sdp(True)
    # Report the effective enabled set so the log captures what actually got selected.
    try:
        enabled = []
        if backends.flash_sdp_enabled(): enabled.append("flash")
        if backends.mem_efficient_sdp_enabled(): enabled.append("efficient")
        if backends.math_sdp_enabled(): enabled.append("math")
        cap = torch.cuda.get_device_capability()
        print0(f"SDPA backend request: {args.sdpa_backend}  |  enabled kernels: {enabled}  |  device: sm_{cap[0]}{cap[1]}")
    except Exception as _sdpa_err:
        print0(f"SDPA backend request: {args.sdpa_backend}  |  diagnostics unavailable ({_sdpa_err})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# ---- Startup sanity assertions (Gate 4) ------------------------------------
# Catch misconfigurations BEFORE spending compute on a corrupt state. Cheap,
# noisy on failure, quiet on success. Fail-fast beats silent-wrong.
assert int(token_bytes.shape[0]) == vocab_size, (
    f"token_bytes length ({int(token_bytes.shape[0])}) != tokenizer vocab_size ({vocab_size}). "
    "Regenerate token_bytes.pt via tools/recover_tokenizer_artifacts.py or scripts.tok_train."
)
_bos_id = tokenizer.get_bos_token_id()
print0(f"BOS token id: {_bos_id}")
# If a tokenizer manifest exists next to the tokenizer files, surface its SHA for reproducibility.
# We re-derive the base_dir here (the full `base_dir = get_base_dir()` happens lower in the script).
_tok_manifest_path = os.path.join(get_base_dir(), "tokenizer", "tokenizer_manifest.json")
if os.path.exists(_tok_manifest_path):
    try:
        import json as _json
        _manifest = _json.loads(open(_tok_manifest_path).read())
        _manifest_bos = _manifest.get("tokenizer", {}).get("special_tokens", {}).get("[BOS]")
        if _manifest_bos is not None and _manifest_bos != _bos_id:
            print0(
                f"WARNING: tokenizer manifest expects BOS id={_manifest_bos} but runtime BOS id={_bos_id}. "
                "Proceeding, but verify the tokenizer directory isn't stale."
            )
        _manifest_vocab = _manifest.get("tokenizer", {}).get("vocab_size")
        if _manifest_vocab is not None and _manifest_vocab != vocab_size:
            print0(
                f"WARNING: tokenizer manifest vocab_size ({_manifest_vocab}) != runtime ({vocab_size})."
            )
        _manifest_sha = _manifest.get("outputs", {}).get("sha256_token_bytes_pt", "<none>")
        print0(f"Tokenizer manifest: token_bytes sha256={_manifest_sha[:16]}...  format={_manifest.get('tokenizer', {}).get('format', 'unknown')}")
    except Exception as _mf_err:
        print0(f"Tokenizer manifest present but unreadable ({_mf_err}); continuing without manifest check")
else:
    print0(f"(No tokenizer_manifest.json at {_tok_manifest_path} — skipping manifest SHA print)")

# Model kwargs are derived from the desired depth of the model
num_layers = args.depth
model_dim = args.depth * args.aspect_ratio
def find_num_heads(model_dim, target_head_dim):
    # Find num_heads that divides model_dim evenly, with head_dim closest to target.
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1
num_heads = find_num_heads(model_dim, args.head_dim)

# GQA: n_kv_head = round(n_head * kv_head_ratio), clamped to a divisor of n_head.
def pick_kv_heads(n_head, ratio):
    target = max(1, round(n_head * ratio))
    # Find the largest divisor of n_head that is <= target (standard GQA constraint).
    for candidate in range(min(target, n_head), 0, -1):
        if n_head % candidate == 0:
            return candidate
    return 1

num_kv_heads = pick_kv_heads(num_heads, args.kv_head_ratio)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}  (kv_head_ratio={args.kv_head_ratio})")

# Sequence-length curriculum: precompute rotary cache large enough for the late stage too.
if args.seq_len_late > 0 and args.seq_len_late > args.max_seq_len:
    rotary_seq_len_for_init = args.seq_len_late
    print0(f"Seq-length curriculum: start T={args.max_seq_len}, switch to T={args.seq_len_late} at {args.seq_len_late_frac:.0%}")
else:
    rotary_seq_len_for_init = args.max_seq_len

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size={args.total_batch_size} must be divisible by "
    f"device_batch_size*max_seq_len*world_size = {args.device_batch_size}*{args.max_seq_len}*{ddp_world_size} = {world_tokens_per_fwdbwd}"
)
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard
    # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,} (reference: {reference_batch_size:,})")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights.
# NB: sequence_len is what the rotary cache is sized for (10x inside GPT.__init__).
model_config_kwargs = dict(sequence_len=rotary_seq_len_for_init, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    # All tensors are created as meta tensors (they have shape/dtype but no data)
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device) # All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # All tensors get initialized
# Wire runtime memory knobs (not stored in GPTConfig so saved configs stay portable).
model.use_activation_checkpoint = bool(args.activation_checkpoint)
model.checkpoint_every_n_blocks = max(1, int(args.ckpt_every_n_blocks))
model.use_chunked_loss = bool(args.chunked_loss)
model.loss_chunk_size = max(1, int(args.loss_chunk_size))
print0(f"Memory knobs: activation_checkpoint={model.use_activation_checkpoint} "
       f"(every_n={model.checkpoint_every_n_blocks}), chunked_loss={model.use_chunked_loss} "
       f"(chunk_size={model.loss_chunk_size})")

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    # Vocab-shape sanity: the checkpoint's wte + lm_head must match the tokenizer we're
    # about to feed tokens from. If vocabs drift (e.g. someone swaps tokenizer.json without
    # retraining), strict=True below would error unhelpfully — preempt with a clearer message.
    _ckpt_wte_shape = tuple(model_data["transformer.wte.weight"].shape)
    _ckpt_lm_shape = tuple(model_data["lm_head.weight"].shape)
    _runtime_wte_shape = tuple(model.transformer.wte.weight.shape)
    if _ckpt_wte_shape != _runtime_wte_shape:
        raise RuntimeError(
            f"Resume vocab-shape mismatch: checkpoint wte={_ckpt_wte_shape}, runtime wte={_runtime_wte_shape}. "
            f"Tokenizer vocab is {vocab_size}. The checkpoint was trained with a different tokenizer; "
            "either find the matching tokenizer, or start a fresh run (remove --resume_from_step)."
        )
    print0(f"Resume vocab check OK: wte={_ckpt_wte_shape}, lm_head={_ckpt_lm_shape}")
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# torch.compile: opt-out via env var or --compile_mode=none; fall back gracefully if compilation errors.
compile_disabled = os.environ.get("TORCH_COMPILE_DISABLE") or args.compile_mode == "none"
if compile_disabled:
    print0(f"torch.compile disabled (env={bool(os.environ.get('TORCH_COMPILE_DISABLE'))}, mode={args.compile_mode})")
else:
    try:
        model = torch.compile(model, mode=args.compile_mode, dynamic=False) # inputs never change shape, so dynamic=False is safe
        print0(f"torch.compile enabled (mode={args.compile_mode})")
    except Exception as e:
        print0(f"Warning: torch.compile failed ({e}), continuing without compilation")
num_params = sum(p.numel() for p in model.parameters())
num_scaling_params = orig_model.num_scaling_params()
print0(f"Number of parameters: {num_params:,} (scaling: {num_scaling_params:,})")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio (use scaling params per Kaplan et al.)
    target_tokens = args.target_param_data_ratio * num_scaling_params
    num_iterations = target_tokens // args.total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = args.total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {args.total_batch_size * num_iterations / num_scaling_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=args.weight_decay,
    adam_betas=adam_betas,
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]

current_seq_len = args.max_seq_len

def build_train_loader(B, T, resume_state_dict):
    if args.token_cache_dir:
        if args.loader_strategy == "parallel_family_cache":
            return cached_family_balanced_data_loader_with_state(
                B, T, split="all", device=device,
                cache_dir=args.token_cache_dir,
                grad_accum_steps=grad_accum_steps,
                resume_state_dict=resume_state_dict,
            )
        # sequential_cache: legacy behavior (v4 layout uses split='all' when val_cache_dir is set)
        split = "all" if args.val_cache_dir else "train"
        return cached_distributed_data_loader_with_state(
            B, T, split=split, device=device,
            cache_dir=args.token_cache_dir,
            resume_state_dict=resume_state_dict,
        )
    return tokenizing_distributed_data_loader_with_state(
        B, T, split="train", device=device,
        resume_state_dict=resume_state_dict,
        parquet_dir=args.parquet_dir,
    )

def build_val_loader_fn(B, T):
    if args.token_cache_dir:
        if args.val_cache_dir:
            # v4: explicit val cache dir, use all shards in it
            return lambda: cached_distributed_data_loader(
                B, T, split="all", device=device, cache_dir=args.val_cache_dir,
            )
        return lambda: cached_distributed_data_loader(
            B, T, split="val", device=device, cache_dir=args.token_cache_dir,
        )
    return lambda: tokenizing_distributed_data_loader(
        B, T, split="val", device=device, parquet_dir=args.parquet_dir,
    )

train_loader = build_train_loader(args.device_batch_size, current_seq_len, dataloader_resume_state_dict)
build_val_loader = build_val_loader_fn(args.device_batch_size, current_seq_len)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Seq-length curriculum: returns the T we should use at this step.
def get_seq_len_for_step(step_idx):
    if args.seq_len_late <= 0:
        return args.max_seq_len
    switch_at = int(round(num_iterations * args.seq_len_late_frac))
    return args.seq_len_late if step_idx >= switch_at else args.max_seq_len

# -----------------------------------------------------------------------------
# Benchmark CSV (append-only, rank 0 only). Writes a header if the file is new.
bench_writer = None
bench_file = None
if args.benchmark_csv and master_process:
    new_file = not os.path.exists(args.benchmark_csv)
    bench_file = open(args.benchmark_csv, "a", newline="")
    bench_writer = csv.writer(bench_file)
    if new_file:
        bench_writer.writerow([
            "step", "depth", "seq_len", "device_batch_size", "grad_accum_steps",
            "tokens_per_sec", "dt_ms", "mfu_pct_h100", "peak_mem_mib",
            "loader_wait_ms", "loader_pct",
            "activation_ckpt", "chunked_loss", "compile_mode", "timestamp",
        ])

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Print LR schedule markers at startup (helps diagnostic analysis)
if master_process:
    _warmup_iters = round(args.warmup_ratio * num_iterations)
    _warmdown_iters = round(args.warmdown_ratio * num_iterations)
    _peak_start = _warmup_iters
    _peak_end = num_iterations - _warmdown_iters
    print0(
        f"[lr-schedule] warmup=[0..{_warmup_iters}] peak=[{_peak_start}..{_peak_end}] "
        f"warmdown=[{_peak_end}..{num_iterations}] total={num_iterations} "
        f"max_steps={args.max_steps}  (peak LR duration = {_peak_end - _peak_start} steps)"
    )

# -----------------------------------------------------------------------------
# Load shuffled manifest for shard-provenance lookup (diagnostic logging only)
_shard_provenance_lookup = {}
if args.diagnostic_logging and args.token_cache_dir:
    try:
        _mf_path = os.path.join(args.token_cache_dir, "cache_manifest.json")
        if os.path.exists(_mf_path):
            with open(_mf_path) as _f:
                _mf = json.load(_f)
            for _e in _mf.get("shards", []):
                _src = _e.get("source_file", "")
                _name = _src.split("/")[-1]
                _fam = "unknown"
                for _f2 in ("books_general", "newspapers_periodicals", "legal_government",
                            "science_technical", "early_modern"):
                    if _name.startswith(f"shard_{_f2}_"):
                        _fam = _f2
                        break
                _shard_provenance_lookup[_e["shard_index"]] = {
                    "source_file": _name, "family": _fam,
                }
        print0(f"[provenance] loaded shard-family lookup for {len(_shard_provenance_lookup)} shards")
    except Exception as _prov_err:
        print0(f"[provenance] WARN: could not load shard provenance: {_prov_err}")

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    # Early termination for smoke tests (LR schedule keyed off num_iterations, runtime stops at max_steps)
    if args.max_steps > 0 and step >= args.max_steps:
        last_step = True
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Seq-length curriculum: switch T mid-training if requested. Only rebuilds the loader
    # at the transition so downstream iterations don't pay the cost per step.
    target_seq_len = get_seq_len_for_step(step)
    if target_seq_len != current_seq_len:
        print0(f"Step {step:05d} | seq-len curriculum: switching T {current_seq_len} -> {target_seq_len}")
        current_seq_len = target_seq_len
        new_tokens_per_fwdbwd = args.device_batch_size * current_seq_len * ddp_world_size
        assert args.total_batch_size % new_tokens_per_fwdbwd == 0, (
            f"total_batch_size={args.total_batch_size} not divisible by new per-step tokens={new_tokens_per_fwdbwd}; "
            "pick a seq_len_late that divides evenly."
        )
        grad_accum_steps = args.total_batch_size // new_tokens_per_fwdbwd
        print0(f"  new grad_accum_steps: {grad_accum_steps}")
        train_loader = build_train_loader(args.device_batch_size, current_seq_len, dataloader_state_dict)
        build_val_loader = build_val_loader_fn(args.device_batch_size, current_seq_len)
        x, y, dataloader_state_dict = next(train_loader)  # prime the new loader

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * current_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "current_seq_len": current_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    loader_wait_s = 0.0  # cumulative time the GPU-side loop spent blocked on next(train_loader)
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        # Time the loader fetch. This under-reports true loader cost when the loader
        # runs on a background thread, but captures the GPU-visible stall when the
        # loader can't keep up.
        _tl0 = time.time()
        x, y, dataloader_state_dict = next(train_loader) # prefetch next batch while GPU is busy
        loader_wait_s += time.time() - _tl0
    # Optional diagnostic: grad norm, param norm, NaN/inf detection BEFORE opt.step
    diag = None
    if args.diagnostic_logging:
        with torch.no_grad():
            grad_sq = 0.0
            param_sq = 0.0
            nan_inf_grad_params = 0
            finite_grad_params = 0
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        nan_inf_grad_params += 1
                    else:
                        finite_grad_params += 1
                    grad_sq += p.grad.detach().pow(2).sum().item()
                param_sq += p.detach().pow(2).sum().item()
            grad_norm = grad_sq ** 0.5
            param_norm = param_sq ** 0.5
            loss_finite = bool(torch.isfinite(train_loss).item())
        diag = {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "upd_over_param": grad_norm / max(param_norm, 1e-9),
            "nan_inf_grad_params": nan_inf_grad_params,
            "finite_grad_params": finite_grad_params,
            "loss_finite": loss_finite,
        }
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    loader_pct = 100.0 * loader_wait_s / dt if dt > 0 else 0.0
    # Raw vs EMA'd train loss
    _train_loss_raw = float(train_loss.item())
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f}"
        f" | dt: {dt * 1000:.2f}ms | loader: {loader_wait_s*1000:.1f}ms ({loader_pct:.1f}%)"
        f" | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m{eta_str}"
    )
    if args.diagnostic_logging and diag is not None:
        # Per-group LR snapshot
        _grp_lrs = []
        for _opt in optimizers:
            for _g in _opt.param_groups:
                _grp_lrs.append(_g.get("lr", 0.0))
        # Source-family identification: parallel_family_cache puts family on the
        # state dict directly; sequential_cache uses shard_idx + provenance lookup.
        _state = dataloader_state_dict or {}
        if _state.get("loader_strategy") == "parallel_family_cache":
            _fam = _state.get("current_microbatch_family", "?")
            _cursors = _state.get("family_cursors", {})
            _src = f"fam_cursors={_cursors}"
        else:
            _shard_idx = _state.get("shard_idx", -1)
            _prov = _shard_provenance_lookup.get(_shard_idx, {})
            _fam = _prov.get("family", "?")
            _src = f"shard_idx={_shard_idx} src={_prov.get('source_file','?')}"
        print0(
            f"[diag] step {step:05d} | raw_loss: {_train_loss_raw:.4f} | ema_loss: {debiased_smooth_loss:.4f}"
            f" | grad_norm: {diag['grad_norm']:.4f} | param_norm: {diag['param_norm']:.3e}"
            f" | upd/param: {diag['upd_over_param']:.3e}"
            f" | nan_inf_grad_params: {diag['nan_inf_grad_params']} | loss_finite: {diag['loss_finite']}"
            f" | lrs: {['{:.5f}'.format(x) for x in _grp_lrs]}"
            f" | fam: {_fam} | {_src}"
        )
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/loader_wait_s": loader_wait_s,
            "train/loader_pct": loader_pct,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        wandb_run.log(log_data)
    if bench_writer is not None and step >= 5 and step % 25 == 0:
        bench_writer.writerow([
            step, args.depth, current_seq_len, args.device_batch_size, grad_accum_steps,
            tok_per_sec, f"{dt*1000:.2f}", f"{mfu:.3f}",
            f"{get_max_memory() / 1024 / 1024:.2f}",
            f"{loader_wait_s*1000:.2f}", f"{loader_pct:.2f}",
            int(args.activation_checkpoint), int(args.chunked_loss), args.compile_mode,
            time.strftime("%Y-%m-%dT%H:%M:%S"),
        ])
        bench_file.flush()

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": args.total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
if bench_file is not None:
    bench_file.close()
wandb_run.finish() # wandb run finish
compute_cleanup()
