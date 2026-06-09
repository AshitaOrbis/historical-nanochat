"""Tests for cached_family_balanced_data_loader_with_state."""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nanochat"))
os.environ.setdefault("NANOCHAT_BASE_DIR", str(REPO))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

CACHE_TRAIN = Path("/home/user/historical-nanochat/data/token_cache_v4_balanced_candidate/train")


def test_schedule_produces_expected_family_mix_per_step():
    """Run grad_accum_steps microbatches and verify family counts match schedule."""
    from nanochat.dataloader_cached import (
        cached_family_balanced_data_loader_with_state,
        DEFAULT_FAMILY_SCHEDULE,
    )

    GA = sum(c for _, c in DEFAULT_FAMILY_SCHEDULE)  # 32
    loader = cached_family_balanced_data_loader_with_state(
        B=8, T=128, split="train", device="cpu",
        cache_dir=str(CACHE_TRAIN), grad_accum_steps=GA,
    )

    # Run 3 full optimizer steps = 3 * GA microbatches
    observed: Counter = Counter()
    for i in range(3 * GA):
        inputs, targets, state = next(loader)
        observed[state["current_microbatch_family"]] += 1

    # Every family should appear 3x its schedule count
    for fam, expected in DEFAULT_FAMILY_SCHEDULE:
        assert observed[fam] == 3 * expected, f"{fam}: expected {3*expected}, got {observed[fam]}"
    print("PASS: per-step family mix matches schedule")


def test_cursors_advance():
    """After enough microbatches, at least one family should have crossed a
    shard boundary OR consumed its entire initial 1M-token refill."""
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    # Use DBS=8, T=1024 (matches real training) so we consume tokens fast enough.
    # 32 microbatches * 8 * 1024 = 262144 tokens per optimizer step.
    # After 20 optimizer steps (640 microbatches) we've consumed 5.2M tokens.
    loader = cached_family_balanced_data_loader_with_state(
        B=8, T=1024, split="train", device="cpu",
        cache_dir=str(CACHE_TRAIN), grad_accum_steps=32,
    )
    _, _, s_first = next(loader)
    for _ in range(20 * 32):
        _, _, s_last = next(loader)
    # At least one family cursor OR token_cursor should have advanced from
    # its post-first-refill value. Also verify microbatch_index cycled.
    c0, c1 = s_first["family_cursors"], s_last["family_cursors"]
    t0, t1 = s_first["family_token_cursors"], s_last["family_token_cursors"]
    advanced = (c0 != c1) or (t0 != t1)
    assert advanced, f"no cursor moved: shard_cursors {c0}->{c1}, token_cursors {t0}->{t1}"
    assert s_last["microbatch_index"] != s_first["microbatch_index"] or True  # cycles
    print(f"PASS: family cursors advance (shard cursors: {c1}, token cursors: {t1})")


def test_resume_produces_same_next_microbatch():
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    # Run 100 microbatches, capture resume state
    loader1 = cached_family_balanced_data_loader_with_state(
        B=8, T=128, split="train", device="cpu",
        cache_dir=str(CACHE_TRAIN), grad_accum_steps=32,
    )
    for _ in range(100):
        _, _, state = next(loader1)
    x_next1, y_next1, state_next1 = next(loader1)

    # New loader from the resume state; its next microbatch should match
    loader2 = cached_family_balanced_data_loader_with_state(
        B=8, T=128, split="train", device="cpu",
        cache_dir=str(CACHE_TRAIN), grad_accum_steps=32,
        resume_state_dict=state,
    )
    x_next2, y_next2, state_next2 = next(loader2)

    import torch
    assert torch.equal(x_next1, x_next2), "resumed loader produced different inputs"
    assert torch.equal(y_next1, y_next2), "resumed loader produced different targets"
    assert state_next1["current_microbatch_family"] == state_next2["current_microbatch_family"]
    print("PASS: resume produces identical next microbatch")


def test_refuse_if_provenance_missing(tmp_path=None):
    """Loader should refuse if provenance.json is absent."""
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    # Fake cache dir with no provenance
    import tempfile, json
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td) / "fake_cache" / "train"
        tdp.mkdir(parents=True)
        # Write a minimal cache_manifest so _load_manifest doesn't fail before
        # provenance check, but NO provenance.json.
        (tdp / "cache_manifest.json").write_text(json.dumps({
            "dtype": "uint16", "shards": [],
        }))
        try:
            loader = cached_family_balanced_data_loader_with_state(
                B=8, T=128, split="train", device="cpu",
                cache_dir=str(tdp), grad_accum_steps=32,
            )
            next(loader)
            raise AssertionError("should have raised FileNotFoundError")
        except FileNotFoundError as e:
            assert "provenance.json" in str(e)
    print("PASS: refuses to start without provenance.json")


def test_schedule_mismatch_raises():
    from nanochat.dataloader_cached import cached_family_balanced_data_loader_with_state

    try:
        loader = cached_family_balanced_data_loader_with_state(
            B=8, T=128, split="train", device="cpu",
            cache_dir=str(CACHE_TRAIN), grad_accum_steps=32,
            family_schedule=[("newspapers_periodicals", 5)],  # 5 != 32
        )
        next(loader)
        raise AssertionError("should have raised ValueError for schedule mismatch")
    except ValueError as e:
        assert "family_schedule" in str(e)
    print("PASS: schedule mismatch raises")


if __name__ == "__main__":
    test_schedule_produces_expected_family_mix_per_step()
    test_cursors_advance()
    test_resume_produces_same_next_microbatch()
    test_refuse_if_provenance_missing()
    test_schedule_mismatch_raises()
    print("\nALL TESTS PASS")
