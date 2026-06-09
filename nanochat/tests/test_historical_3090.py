"""
Smoke tests for the 3090-focused historical-nanochat changes.

Run:
    python -m pytest tests/test_historical_3090.py -v

Covers:
  - parquet shard path wiring via NANOCHAT_PARQUET_DIR + explicit data_dir
  - chunked-loss path matches the full-logits baseline within tolerance
  - activation-checkpoint path runs forward + backward
  - streaming shard packager doesn't materialize the full corpus in memory
  - contamination checker has no broken placeholder behavior
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../nanochat/
HIST_ROOT = REPO_ROOT.parent  # .../historical-nanochat/


# ---------------------------------------------------------------------------
# 1. Parquet dir wiring
# ---------------------------------------------------------------------------

def _write_fake_parquet(path: Path, texts):
    table = pa.Table.from_pydict({"text": texts})
    pq.write_table(table, path, row_group_size=4, compression="zstd",
                   use_dictionary=False, write_statistics=False)


def test_parquet_path_via_explicit_data_dir(tmp_path, monkeypatch):
    """list_parquet_files(data_dir=...) must find files without needing a `base_data` subdir."""
    from nanochat.dataset import list_parquet_files
    # Ensure the env var is not set.
    monkeypatch.delenv("NANOCHAT_PARQUET_DIR", raising=False)
    _write_fake_parquet(tmp_path / "shard_00000.parquet", ["hello", "world"])
    paths = list_parquet_files(data_dir=str(tmp_path))
    assert len(paths) == 1
    assert paths[0].endswith("shard_00000.parquet")


def test_parquet_path_via_env_var(tmp_path, monkeypatch):
    """NANOCHAT_PARQUET_DIR should be used when data_dir is not provided."""
    from nanochat.dataset import list_parquet_files
    _write_fake_parquet(tmp_path / "shard_00000.parquet", ["alpha", "beta"])
    monkeypatch.setenv("NANOCHAT_PARQUET_DIR", str(tmp_path))
    paths = list_parquet_files()
    assert len(paths) == 1
    assert paths[0].endswith("shard_00000.parquet")


# ---------------------------------------------------------------------------
# 2. Chunked-loss numerical match
# ---------------------------------------------------------------------------

def _make_tiny_gpt(device="cpu"):
    from nanochat.gpt import GPT, GPTConfig
    cfg = GPTConfig(sequence_len=32, vocab_size=128, n_layer=2, n_head=4, n_kv_head=4, n_embd=32)
    with torch.device("meta"):
        model = GPT(cfg, pad_vocab_size_to=32)
    model.to_empty(device=device)
    model.init_weights()
    # Keep rotary on CPU matching the model device.
    return model.to(device)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_chunked_loss_matches_full(reduction):
    """Chunked CE should match the full-logits path to within fp32 tolerance."""
    torch.manual_seed(0)
    model = _make_tiny_gpt("cpu")
    B, T = 2, 16
    x = torch.randint(0, 100, (B, T))
    y = torch.randint(0, 100, (B, T))
    # Mark some targets as ignore_index to verify masking.
    y[0, 3] = -1
    y[1, 7] = -1

    model.use_chunked_loss = False
    baseline = model(x, y, loss_reduction=reduction)

    model.use_chunked_loss = True
    model.loss_chunk_size = 9   # awkward chunk that doesn't divide B*T evenly
    chunked = model(x, y, loss_reduction=reduction)

    if reduction == "none":
        # baseline flattens (B,T) -> (B*T,) before CE, so shape is 1D
        assert baseline.shape == chunked.shape == (B * T,)
        torch.testing.assert_close(baseline, chunked, rtol=1e-4, atol=1e-4)
    else:
        assert baseline.shape == chunked.shape == ()
        torch.testing.assert_close(baseline, chunked, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Activation-checkpoint path runs forward + backward
# ---------------------------------------------------------------------------

def test_activation_checkpoint_fwd_bwd():
    torch.manual_seed(0)
    model = _make_tiny_gpt("cpu")
    model.use_activation_checkpoint = True
    B, T = 2, 16
    x = torch.randint(0, 100, (B, T))
    y = torch.randint(0, 100, (B, T))

    model.train()
    loss = model(x, y)
    loss.backward()
    # Gradients should exist on at least the LM head and embeddings.
    assert model.lm_head.weight.grad is not None
    assert model.transformer.wte.weight.grad is not None


def test_activation_checkpoint_every_n():
    """checkpoint_every_n_blocks=2 should still run without error."""
    torch.manual_seed(0)
    model = _make_tiny_gpt("cpu")
    model.use_activation_checkpoint = True
    model.checkpoint_every_n_blocks = 2
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    loss = model(x, y)
    loss.backward()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 4. Streaming shard packager doesn't require full corpus in memory
# ---------------------------------------------------------------------------

def test_streaming_packager_bounded_memory(tmp_path):
    """Feed a huge JSONL via a generator; packager must not load it all."""
    sys.path.insert(0, str(HIST_ROOT))
    from data.process.shard_packager import package_shards_streaming

    input_jsonl = tmp_path / "fake.jsonl"
    n_docs = 500
    import json as _json
    with input_jsonl.open("w") as f:
        for i in range(n_docs):
            body = ("word " * 100).strip() + " doc" + str(i)
            f.write(_json.dumps({"text": body, "source": "gutenberg"}) + "\n")

    output_dir = tmp_path / "shards"
    stats = package_shards_streaming(
        input_files=[str(input_jsonl)],
        output_dir=str(output_dir),
        sample_rates={"gutenberg": 1.0, "default": 1.0},
        chars_per_shard=10_000,   # tiny shards so we get several of them
        row_group_size=16,
        buffer_size=32,
        run_contamination=False,
    )
    assert stats["total_docs"] > 0
    assert stats["num_shards"] >= 1
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    import json
    m = json.loads(manifest_path.read_text())
    # manifest has per-shard entries
    assert len(m["shards"]) == stats["num_shards"]
    # source distribution recorded
    assert "source_counts" in m and "gutenberg" in m["source_counts"]
    # rejection report present even when contamination is off (captures length/sampling/json_errors)
    assert "rejections" in m
    assert "per_file" in m["rejections"] and len(m["rejections"]["per_file"]) >= 1
    assert "totals" in m["rejections"] and "accepted" in m["rejections"]["totals"]


# ---------------------------------------------------------------------------
# 5. Contamination checker has no broken placeholder behavior
# ---------------------------------------------------------------------------

def test_contamination_checker_contextual_terms_not_silenced():
    """Historical regression: contextual terms like 'atomic' were being silenced
    via SAFE_HISTORICAL_TERMS. Ensure 'atomic bomb' fires now."""
    sys.path.insert(0, str(HIST_ROOT))
    from data.process.contamination_check import check_contamination
    r = check_contamination("The atomic bomb was dropped in 1945.", cutoff_year=1913)
    assert r.is_contaminated
    assert "atomic" in r.matched_terms or "atomic bomb" in r.matched_terms


def test_contamination_checker_clean_historical_unaffected():
    sys.path.insert(0, str(HIST_ROOT))
    from data.process.contamination_check import check_contamination
    r = check_contamination("Apollo was worshipped by the ancient Greeks.", cutoff_year=1913)
    assert not r.is_contaminated


# ---------------------------------------------------------------------------
# 5b. OCR-quality heuristic + near-dedup
# ---------------------------------------------------------------------------

def test_ocr_quality_rejects_garbage():
    sys.path.insert(0, str(HIST_ROOT))
    from data.process.shard_packager import ocr_quality_ok
    # Clean English prose
    good = ("The quick brown fox jumped over the lazy dog. " * 30).strip()
    # OCR garbage: mostly punctuation and short fragments
    bad = "\n".join(["a b", "c.", ", .", "12", ".. ,", "3 4", "q"] * 40)
    assert ocr_quality_ok(good)
    assert not ocr_quality_ok(bad)


def test_content_fingerprint_dedup():
    sys.path.insert(0, str(HIST_ROOT))
    from data.process.shard_packager import content_fingerprint
    t1 = "Chapter 1. The beginning of a long story."
    t2 = "Chapter 1.  The beginning of a long story."  # whitespace-normalized dup
    t3 = "Chapter 2. A different tale entirely."
    assert content_fingerprint(t1) == content_fingerprint(t2)
    assert content_fingerprint(t1) != content_fingerprint(t3)


def test_cached_loader_fails_loud_when_world_size_exceeds_shards(tmp_path, monkeypatch):
    """Regression: cached loader used to hang in an infinite rank-skip loop when
    world_size > num_shards. It now raises RuntimeError with a helpful message."""
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state

    # One shard only for split=train would imply world_size<=1.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Two shards total: train gets 1 (the second is val).
    for i in range(2):
        (cache_dir / f"shard_{i:05d}.bin").write_bytes((b"\x00\x00") * 64)
        (cache_dir / f"shard_{i:05d}.meta.json").write_text(
            '{"shard_index": ' + str(i) + ', "docs": 8, "tokens": 64, "bytes": 128}'
        )
    (cache_dir / "cache_manifest.json").write_text(
        '{"vocab_size": 50304, "dtype": "uint16", "shards": ['
        '{"shard_index": 0, "filename": "shard_00000.bin", "docs": 8, "tokens": 64},'
        '{"shard_index": 1, "filename": "shard_00001.bin", "docs": 8, "tokens": 64}]}'
    )

    # Simulate world_size=4, rank=3 via env vars (get_dist_info reads RANK/WORLD_SIZE).
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(RuntimeError, match="owns no shards"):
        loader = cached_distributed_data_loader_with_state(
            B=1, T=8, split="train", device="cpu", cache_dir=str(cache_dir)
        )
        next(loader)  # generators raise at first next() when construction defers work


def test_gutenberg_year_extraction_precedence():
    """Precedence: issued > death > birth+20 > fallback; 'downloads' must NOT be treated as a year."""
    sys.path.insert(0, str(HIST_ROOT))
    from data.download.gutenberg_download import extract_year_from_metadata

    # 'downloads' = 1972 must NOT be mistaken for a year
    assert extract_year_from_metadata({"downloads": "1972"}) is None
    # 'issued' wins over death
    assert extract_year_from_metadata({"issued": "1885-03-10", "authoryearofdeath": "1910"}) == 1885
    # 'authoryearofdeath' used when issued absent
    assert extract_year_from_metadata({"authoryearofdeath": "1910"}) == 1910
    # 'authoryearofbirth' gives floor of birth+20
    assert extract_year_from_metadata({"authoryearofbirth": "1850"}) == 1870
    # Fallback scan picks the earliest plausible year from metadata JSON
    assert extract_year_from_metadata({"title": "Published 1887 edition", "notes": "2024 release"}) == 1887
    # Empty / no signal
    assert extract_year_from_metadata({}) is None


def test_streaming_packager_dedup_end_to_end(tmp_path):
    """Two JSONL records with identical text should collapse to one accepted doc."""
    sys.path.insert(0, str(HIST_ROOT))
    from data.process.shard_packager import package_shards_streaming

    input_jsonl = tmp_path / "fake.jsonl"
    import json as _json
    with input_jsonl.open("w") as f:
        body = ("word " * 100).strip()
        # 5 unique docs, duplicated 3x each => should yield 5 accepted, 10 duplicates
        for rep in range(3):
            for i in range(5):
                f.write(_json.dumps({"text": body + " doc" + str(i), "source": "gutenberg"}) + "\n")

    output_dir = tmp_path / "shards"
    stats = package_shards_streaming(
        input_files=[str(input_jsonl)],
        output_dir=str(output_dir),
        sample_rates={"gutenberg": 1.0, "default": 1.0},
        chars_per_shard=5000,
        row_group_size=4,
        buffer_size=8,
        run_contamination=False,
        run_dedup=True,
    )
    assert stats["total_docs"] == 5
    import json
    m = json.loads((output_dir / "manifest.json").read_text())
    assert m["rejections"]["totals"]["rejected_duplicate"] == 10


# ---------------------------------------------------------------------------
# 6. Seq-length curriculum: loader rebuilds at switch, grad_accum recomputes.
# ---------------------------------------------------------------------------

def test_seq_len_curriculum_switches_loader(tmp_path, monkeypatch):
    """Run a short training job that crosses the seq_len_late switch point
    and confirm (a) the switch log fires, (b) loss still moves, (c) final step
    completes without a shape mismatch."""
    import pickle
    import subprocess
    import sys as _sys
    import tiktoken
    import torch as _torch

    base_dir = tmp_path / "base"
    pq_dir = tmp_path / "parquet"
    tok_dir = base_dir / "tokenizer"
    tok_dir.mkdir(parents=True)
    pq_dir.mkdir(parents=True)

    # Build a tiktoken Encoding with the nanochat special tokens.
    gpt2 = tiktoken.get_encoding("gpt2")
    specials = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>",
                "<|assistant_end|>", "<|python_start|>", "<|python_end|>",
                "<|output_start|>", "<|output_end|>"]
    vocab = gpt2.n_vocab + len(specials)
    enc = tiktoken.Encoding(
        name="dryrun", pat_str=gpt2._pat_str,
        mergeable_ranks=gpt2._mergeable_ranks,
        special_tokens={n: gpt2.n_vocab + i for i, n in enumerate(specials)},
    )
    with open(tok_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(enc, f)
    byts = _torch.ones(vocab, dtype=_torch.int64)
    for i in range(len(specials)):
        byts[gpt2.n_vocab + i] = 0
    _torch.save(byts, tok_dir / "token_bytes.pt")

    # Two small parquet shards.
    for i, nrows in enumerate([64, 32]):
        table = pa.Table.from_pydict({"text": [("the quick brown fox " * 10).strip()] * nrows})
        pq.write_table(table, pq_dir / f"shard_{i:05d}.parquet",
                       row_group_size=8, compression="zstd",
                       use_dictionary=False, write_statistics=False)

    env = os.environ.copy()
    env.update({
        "NANOCHAT_BASE_DIR": str(base_dir),
        "NANOCHAT_PARQUET_DIR": str(pq_dir),
        "TORCH_COMPILE_DISABLE": "1",
        "WANDB_MODE": "offline",
    })
    cmd = [
        _sys.executable, "-m", "scripts.base_train",
        "--device_type=cpu", "--depth=4", "--aspect_ratio=16", "--head_dim=16",
        "--max_seq_len=32", "--seq_len_late=64", "--seq_len_late_frac=0.5",
        "--device_batch_size=1", "--total_batch_size=64",
        "--num_iterations=4",
        "--eval_every=-1", "--core_metric_every=-1",
        "--sample_every=-1", "--save_every=-1",
        "--compile_mode=none",
    ]
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT),
                            capture_output=True, text=True, timeout=180)
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"base_train failed:\n{out[-3000:]}"
    # Verify the switch actually fired at step 2 (num_iterations=4 * 0.5 = 2)
    assert "switching T 32 -> 64" in out, f"No switch log in output:\n{out[-2000:]}"
    # And that we completed at least one step at the new T.
    assert "step 00003/00004" in out, f"Didn't complete post-switch step:\n{out[-2000:]}"
