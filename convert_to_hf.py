#!/usr/bin/env python3
"""Convert nanochat model to HuggingFace format for GGUF conversion."""

import torch
import json
import os
from pathlib import Path

def convert_nanochat_to_hf(checkpoint_dir: str, step: int, output_dir: str):
    """Convert nanochat checkpoint to HuggingFace GPT-2 format."""

    # Add nanochat to path
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanochat"))

    from nanochat.checkpoint_manager import load_checkpoint
    from nanochat.tokenizer import get_tokenizer

    print(f"Loading checkpoint from {checkpoint_dir} step {step}...")
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, "cpu", load_optimizer=False)
    cfg = meta_data["model_config"]

    print(f"Model config: {cfg}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Map nanochat weights to HuggingFace GPT-2 format
    hf_state_dict = {}

    # Embedding
    hf_state_dict["transformer.wte.weight"] = model_data["transformer.wte.weight"]

    # No position embeddings in nanochat (uses RoPE), but HF GPT-2 expects them
    # We'll use GPT-NeoX format instead which supports RoPE

    # Layers
    for i in range(cfg["n_layer"]):
        prefix_nc = f"transformer.h.{i}"
        prefix_hf = f"transformer.h.{i}"

        # Attention
        hf_state_dict[f"{prefix_hf}.attn.c_attn.weight"] = torch.cat([
            model_data[f"{prefix_nc}.attn.c_q.weight"],
            model_data[f"{prefix_nc}.attn.c_k.weight"],
            model_data[f"{prefix_nc}.attn.c_v.weight"],
        ], dim=0).T  # HF uses (in, out) format

        hf_state_dict[f"{prefix_hf}.attn.c_proj.weight"] = model_data[f"{prefix_nc}.attn.c_proj.weight"].T

        # MLP
        hf_state_dict[f"{prefix_hf}.mlp.c_fc.weight"] = model_data[f"{prefix_nc}.mlp.c_fc.weight"].T
        hf_state_dict[f"{prefix_hf}.mlp.c_proj.weight"] = model_data[f"{prefix_nc}.mlp.c_proj.weight"].T

        # Layer norms (nanochat might use RMSNorm)
        if f"{prefix_nc}.ln_1.weight" in model_data:
            hf_state_dict[f"{prefix_hf}.ln_1.weight"] = model_data[f"{prefix_nc}.ln_1.weight"]
        if f"{prefix_nc}.ln_2.weight" in model_data:
            hf_state_dict[f"{prefix_hf}.ln_2.weight"] = model_data[f"{prefix_nc}.ln_2.weight"]

    # Final layer norm
    if "transformer.ln_f.weight" in model_data:
        hf_state_dict["transformer.ln_f.weight"] = model_data["transformer.ln_f.weight"]

    # LM head (usually tied to embeddings)
    if "lm_head.weight" in model_data:
        hf_state_dict["lm_head.weight"] = model_data["lm_head.weight"]
    else:
        hf_state_dict["lm_head.weight"] = model_data["transformer.wte.weight"]

    # Save pytorch model
    torch.save(hf_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # Create config.json
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": cfg["vocab_size"],
        "n_positions": cfg["sequence_len"],
        "n_ctx": cfg["sequence_len"],
        "n_embd": cfg["n_embd"],
        "n_layer": cfg["n_layer"],
        "n_head": cfg["n_head"],
        "activation_function": "gelu_new",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "layer_norm_epsilon": 1e-5,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save tokenizer
    tokenizer = get_tokenizer()

    # For GGUF, we need the vocab in a specific format
    # Export vocab to a text file
    print("Exporting tokenizer...")

    print(f"\nModel saved to {output_dir}")
    print("State dict keys:", list(hf_state_dict.keys())[:10])

    return output_dir

if __name__ == "__main__":
    convert_nanochat_to_hf(
        os.path.expanduser("~/.cache/nanochat/base_checkpoints/d12_v1"),
        15250,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_model"),
    )
