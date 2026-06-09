"""
Sample probe harness for legacy_textonly_d22_r30_internal_baseline.

Runs a fixed prompt set (tools/sample_probe_prompts.yaml) against a specific
checkpoint and writes outputs under report/baseline_samples/step_<step>/ with
every file header-stamped "legacy/internal/baseline — NOT a governed release".

GPU contention note: the live training run holds the 3090. Do NOT run this
harness on --device cuda while training is active. Either:
  (a) run with --device cpu (slow but safe), or
  (b) schedule runs for brief windows when the trainer is paused, or
  (c) wait until the run is stopped, then sample each preserved checkpoint.

Usage:
  python tools/sample_probe.py \
      --model-tag d22_r30 \
      --step 5000 \
      --device cpu \
      --max-new-tokens 128 \
      --temperature 0.8 \
      --top-k 40

The default per-prompt output includes the prompt, the continuation, and
generation params. All files are labeled with the run tag and cannot be
rebranded without losing that tag.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nanochat"))

RUN_TAG = "legacy_textonly_d22_r30_internal_baseline"
LABEL_BANNER = (
    "==============================================================\n"
    f" RUN TAG: {RUN_TAG}\n"
    " CLASSIFICATION: legacy / internal / baseline\n"
    " NOT a governed 3090 PoC, NOT a release candidate, NOT a teacher.\n"
    " Shard schema is text-only; source/date/rights provenance absent.\n"
    " Any fact produced below is uncalibrated against the 1913 cutoff.\n"
    "==============================================================\n"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        # Minimal fallback parser for this file's subset of YAML
        return _naive_yaml_parse(path)
    with open(path) as f:
        return yaml.safe_load(f)


def _naive_yaml_parse(path: Path) -> dict[str, Any]:
    """Tiny YAML-ish parser for sample_probe_prompts.yaml. Prefers PyYAML if
    present; this fallback is just to avoid adding a hard dependency."""
    # Strongly recommend installing pyyaml; fallback is best-effort.
    raise RuntimeError(
        "PyYAML is required for sample_probe.py. Install with: pip install pyyaml"
    )


def _write_sample(
    out_dir: Path,
    category: str,
    prompt_id: str,
    prompt: str,
    continuation: str,
    params: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{category}__{prompt_id}.txt"
    body = (
        LABEL_BANNER
        + "\n"
        + f"category: {category}\n"
        + f"prompt_id: {prompt_id}\n"
        + f"params: {json.dumps(params)}\n"
        + "------ PROMPT ------\n"
        + prompt.rstrip()
        + "\n------ CONTINUATION (UNCALIBRATED) ------\n"
        + continuation.rstrip()
        + "\n"
    )
    fn.write_text(body)
    return fn


def _write_index(out_dir: Path, entries: list[dict[str, Any]], model_meta: dict[str, Any]):
    idx_path = out_dir / "index.json"
    idx = {
        "run_tag": RUN_TAG,
        "classification": "legacy/internal/baseline",
        "model_meta": model_meta,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "entries": entries,
    }
    idx_path.write_text(json.dumps(idx, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="d22_r30")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--checkpoints-root", default=str(REPO_ROOT / "base_checkpoints"))
    ap.add_argument("--prompts", default=str(REPO_ROOT / "tools" / "sample_probe_prompts.yaml"))
    ap.add_argument(
        "--out-root", default=str(REPO_ROOT / "report" / "baseline_samples")
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not load model; emit PROMPT-ONLY files (useful before first checkpoint)",
    )
    args = ap.parse_args()

    if args.device == "cuda":
        print(
            "WARNING: --device cuda may contend with the live training process.\n"
            "         Confirm training is paused or the run has stopped before proceeding."
        )

    prompts_doc = _load_yaml(Path(args.prompts))
    categories = prompts_doc.get("categories", {})
    assert categories, "no categories found in prompts file"

    checkpoint_dir = Path(args.checkpoints_root) / args.model_tag
    out_dir = Path(args.out_root) / f"step_{args.step:06d}"

    model_meta: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "model_tag": args.model_tag,
        "step": args.step,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
    }

    if args.dry_run:
        entries = []
        for cat_name, cat in categories.items():
            for p in cat.get("prompts", []):
                path = _write_sample(
                    out_dir,
                    category=cat_name,
                    prompt_id=p["id"],
                    prompt=p["text"],
                    continuation="[DRY RUN — no model loaded. Use without --dry-run once a checkpoint exists.]",
                    params={"mode": "dry_run"},
                )
                entries.append({"file": str(path.relative_to(out_dir.parent.parent)), "prompt_id": p["id"]})
        _write_index(out_dir, entries, model_meta | {"mode": "dry_run"})
        print(f"dry-run wrote {len(entries)} stubs to {out_dir}")
        return

    # Load model and tokenizer
    import torch  # type: ignore

    from nanochat.checkpoint_manager import build_model  # type: ignore

    device = torch.device(args.device)
    model, tokenizer, meta = build_model(str(checkpoint_dir), args.step, device, phase="eval")
    model_meta["model_config"] = meta.get("model_config", {})

    entries = []
    bos_id = tokenizer.encode_special("<|bos|>") if hasattr(tokenizer, "encode_special") else None

    for cat_name, cat in categories.items():
        for p in cat.get("prompts", []):
            prompt_text = p["text"]
            # Encode; prepend bos if available
            ids = tokenizer.encode(prompt_text)
            if bos_id is not None and (not ids or ids[0] != bos_id):
                ids = [bos_id] + list(ids)
            gen = list(
                model.generate(
                    ids,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=args.seed,
                )
            )
            try:
                cont = tokenizer.decode(gen)
            except Exception:
                cont = f"[decode error; raw ids: {gen[:50]}...]"
            path = _write_sample(
                out_dir,
                category=cat_name,
                prompt_id=p["id"],
                prompt=prompt_text,
                continuation=cont,
                params={
                    "device": args.device,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                },
            )
            entries.append({"file": str(path.relative_to(out_dir.parent.parent)), "prompt_id": p["id"]})

    _write_index(out_dir, entries, model_meta)
    print(f"wrote {len(entries)} sample files under {out_dir}")


if __name__ == "__main__":
    main()
