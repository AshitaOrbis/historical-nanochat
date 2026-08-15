"""Design-C tests: bake script, CPU traversal simulator, ordering-aware sequential
loader, consumed-cursor exact resume, DDP trace equivalence, checkpoint sentinel.

Runs entirely on CPU with a small synthetic token cache in tmp_path.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nanochat"))
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault("NANOCHAT_BASE_DIR", str(REPO))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import build_balanced_ordering as bake
import simulate_ddp_traversal as sim

B, T = 2, 16
NEEDED = B * T + 1  # 33


def token_value(shard_index: int, position: int) -> int:
    return (shard_index * 97 + position) % 65536


def build_synthetic_cache(root: Path, families=(("alpha", 7), ("beta", 5), ("gamma", 3)),
                          seed=7) -> tuple[Path, Path]:
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True)
    val.mkdir()
    rng = random.Random(seed)
    shards, per_shard = [], []
    idx = 0
    for fam, n in families:
        for _ in range(n):
            tokens = rng.randrange(200, 400)
            fn = f"shard_{idx:05d}.bin"
            arr = np.array([token_value(idx, p) for p in range(tokens)], dtype=np.uint16)
            arr.tofile(train / fn)
            shards.append({"shard_index": idx, "filename": fn,
                           "source_file": f"/src/train/shard_{fam}_src_{idx:06d}.parquet",
                           "tokens": tokens, "bytes": tokens * 2})
            per_shard.append({"shard_index": idx, "family": fam, "tokens": tokens})
            idx += 1
    (train / "cache_manifest.json").write_text(json.dumps({
        "format_version": 1,
        "byte_order": "little",
        "dtype": "uint16",
        "shards": shards,
    }))

    vshards, vper = [], []
    for j in range(2):
        tokens = 100
        fn = f"shard_{j:05d}.bin"
        np.full(tokens, j, dtype=np.uint16).tofile(val / fn)
        vshards.append({"shard_index": j, "filename": fn,
                        "source_file": f"/src/val/shard_alpha_vsrc_{j:06d}.parquet",
                        "tokens": tokens, "bytes": tokens * 2})
        vper.append({"shard_index": j, "family": "alpha", "tokens": tokens})
    (val / "cache_manifest.json").write_text(json.dumps({
        "format_version": 1,
        "byte_order": "little",
        "dtype": "uint16",
        "shards": vshards,
    }))
    (root / "provenance.json").write_text(json.dumps(
        {"splits": {"train": {"per_shard": per_shard}, "val": {"per_shard": vper}}}))
    return train, val


def bake_ordering(train: Path, seed=1913) -> Path:
    return Path(bake.main(["--cache-dir", str(train), "--seed", str(seed), "--force"]))


def collect_ids(loader, n_yields):
    """Reconstruct the exact consumed token stream and states from n yields."""
    ids, states = [], []
    for _ in range(n_yields):
        inputs, targets, state = next(loader)
        ids.extend(inputs.flatten().tolist())
        ids.append(int(targets.flatten()[-1]))
        states.append(state)
    return ids, states


# ---------------------------------------------------------------- bake script

def test_bake_deterministic(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    p1 = bake_ordering(train, seed=1913)
    doc1 = json.loads(p1.read_text())
    p2 = bake_ordering(train, seed=1913)
    doc2 = json.loads(p2.read_text())
    assert doc1["order"] == doc2["order"]
    assert doc1["order_sha256"] == doc2["order_sha256"]
    doc3 = json.loads(bake_ordering(train, seed=42).read_text())
    assert doc3["order"] != doc1["order"]
    # stratified: every prefix of the ordering contains multiple families quickly
    fams_in_first_5 = {doc1["shards"][n]["family"] for n in doc1["order"][:5]}
    assert len(fams_in_first_5) >= 2


def test_bake_refuses_count_mismatch(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    prov_path = tmp_path / "provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["splits"]["train"]["per_shard"] = prov["splits"]["train"]["per_shard"][:-1]
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(SystemExit):
        bake_ordering(train)


def test_bake_refuses_family_mismatch(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    mpath = train / "cache_manifest.json"
    manifest = json.loads(mpath.read_text())
    # provenance says 'alpha' family; make source_file claim a KNOWN family so the
    # cross-check fires (bake validates provenance-vs-source_file family agreement)
    manifest["shards"][0]["source_file"] = "/src/train/shard_books_general_src_000000.parquet"
    mpath.write_text(json.dumps(manifest))
    prov_path = tmp_path / "provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["splits"]["train"]["per_shard"][0]["family"] = "newspapers_periodicals"
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(SystemExit):
        bake_ordering(train)


def test_bake_refuses_bad_file_size(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    fn = json.loads((train / "cache_manifest.json").read_text())["shards"][3]["filename"]
    with open(train / fn, "ab") as f:
        f.write(b"\x00\x00")  # 1 extra token on disk
    with pytest.raises(SystemExit):
        bake_ordering(train)


# ----------------------------------------------------------------- simulator

def test_simulator_gate_passes(tmp_path):
    train, val = build_synthetic_cache(tmp_path)
    bake_ordering(train)
    report = sim.run(["--cache-dir", str(train), "--val-cache-dir", str(val),
                      "--world-size", "4", "--device-batch", str(B), "--seq-len", str(T),
                      "--tolerance-pp", "6.0", "--resume-extra", "4"])
    assert report["gate"] == "PASS", report["failures"]
    assert report["checks"]["deterministic_replay"] is True
    assert report["checks"]["deterministic_resume"]["ok"] is True
    assert report["recommended_num_iterations"] > 0
    assert report["checks"]["unseen_tail"]["total_tokens"] >= 0


def test_simulator_refuses_stale_ordering(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    bake_ordering(train)
    mpath = train / "cache_manifest.json"
    manifest = json.loads(mpath.read_text())
    mpath.write_text(json.dumps(manifest, indent=1))  # semantically same, different bytes
    with pytest.raises(SystemExit, match="STALE"):
        sim.run(["--cache-dir", str(train), "--world-size", "4",
                 "--device-batch", str(B), "--seq-len", str(T)])


def test_simulator_refuses_duplicate_in_ordering(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    opath = bake_ordering(train)
    doc = json.loads(opath.read_text())
    doc["order"][1] = doc["order"][0]
    opath.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="duplicate|mismatch"):
        sim.run(["--cache-dir", str(train), "--world-size", "4",
                 "--device-batch", str(B), "--seq-len", str(T)])


# ------------------------------------------------------- loader: ordering

def test_loader_respects_baked_ordering(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    opath = bake_ordering(train)
    doc = json.loads(opath.read_text())
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state
    loader = cached_distributed_data_loader_with_state(
        B=B, T=T, split="all", device="cpu", cache_dir=str(train))
    n_yields = 12
    ids, _ = collect_ids(loader, n_yields)
    # expected: concatenation of shards in baked order
    name_to_idx = {n: doc["shards"][n]["shard_index"] for n in doc["order"]}
    expected = []
    for n in doc["order"]:
        sidx = name_to_idx[n]
        expected.extend(token_value(sidx, p) for p in range(doc["shards"][n]["tokens"]))
        if len(expected) >= n_yields * NEEDED:
            break
    assert ids == expected[:n_yields * NEEDED]


def test_loader_refuses_stale_ordering(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    bake_ordering(train)
    mpath = train / "cache_manifest.json"
    mpath.write_text(json.dumps(json.loads(mpath.read_text()), indent=1))
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state
    with pytest.raises(RuntimeError, match="STALE"):
        next(cached_distributed_data_loader_with_state(
            B=B, T=T, split="all", device="cpu", cache_dir=str(train)))


def test_loader_refuses_ordering_with_legacy_split(tmp_path):
    train, _ = build_synthetic_cache(tmp_path)
    bake_ordering(train)
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state
    with pytest.raises(RuntimeError, match="split"):
        next(cached_distributed_data_loader_with_state(
            B=B, T=T, split="train", device="cpu", cache_dir=str(train)))


# ------------------------------------------- loader: consumed-cursor resume

@pytest.mark.parametrize("k_yields", [3, 7, 11])
def test_loader_exact_resume(tmp_path, k_yields):
    """Resume from the saved state must reproduce the identical continuation —
    including read-ahead tokens that were still in the buffer at save time."""
    train, _ = build_synthetic_cache(tmp_path)
    bake_ordering(train)
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state

    make = lambda resume=None: cached_distributed_data_loader_with_state(
        B=B, T=T, split="all", device="cpu", cache_dir=str(train), resume_state_dict=resume)

    truth = make()
    _, states = collect_ids(truth, k_yields)
    continuation_ids, _ = collect_ids(truth, 6)

    resumed = make(resume=states[-1])
    resumed_ids, _ = collect_ids(resumed, 6)
    assert resumed_ids == continuation_ids, (
        f"resume after {k_yields} yields diverged (consumed-cursor semantics broken)")


def test_loader_state_is_consumed_not_readahead(tmp_path):
    """The saved cursor must lag the read-ahead: after 1 yield of 33 tokens the
    consumed offset is 33, even though the loader buffered a whole chunk."""
    train, _ = build_synthetic_cache(tmp_path)
    bake_ordering(train)
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state
    loader = cached_distributed_data_loader_with_state(
        B=B, T=T, split="all", device="cpu", cache_dir=str(train))
    _, _, state = next(loader)
    cur = state["per_rank"]["0"]
    assert cur["shard_idx"] == 0 and cur["token_off"] == NEEDED


# -------------------------------------------- loader vs simulator (DDP trace)

def test_ddp_loader_matches_simulator(tmp_path, monkeypatch):
    train, _ = build_synthetic_cache(tmp_path)
    opath = bake_ordering(train)
    doc = json.loads(opath.read_text())
    import nanochat.dataloader_cached as dlc

    W, n_yields = 4, 15
    order, meta = doc["order"], doc["shards"]
    for rank in range(W):
        monkeypatch.setattr(dlc, "get_dist_info", lambda r=rank: (True, r, r, W))
        loader = dlc.cached_distributed_data_loader_with_state(
            B=B, T=T, split="all", device="cpu", cache_dir=str(train))
        positions = [p for p in range(len(order)) if p % W == rank]
        tokens = [meta[order[p]]["tokens"] for p in positions]
        families = [meta[order[p]]["family"] for p in positions]
        rk = sim.RankSim(positions, tokens, families, NEEDED)
        for y in range(n_yields):
            _, _, state = next(loader)
            expected_pos, expected_off = rk.step()
            got = state["per_rank"][str(rank)]
            assert (got["shard_idx"], got["token_off"]) == (expected_pos, expected_off), (
                f"rank {rank} yield {y}: loader={got} sim=({expected_pos},{expected_off})")


def test_ddp_resume_from_merged_state(tmp_path, monkeypatch):
    """Every rank must resume from ITS OWN cursor in a merged per_rank dict
    (the pre-fix behavior sent ranks>0 back to their first owned shard)."""
    train, _ = build_synthetic_cache(tmp_path)
    bake_ordering(train)
    import nanochat.dataloader_cached as dlc

    W, k = 4, 9
    merged = {"per_rank": {}}
    continuations = {}
    for rank in range(W):
        monkeypatch.setattr(dlc, "get_dist_info", lambda r=rank: (True, r, r, W))
        loader = dlc.cached_distributed_data_loader_with_state(
            B=B, T=T, split="all", device="cpu", cache_dir=str(train))
        _, states = collect_ids(loader, k)
        merged["per_rank"].update(states[-1]["per_rank"])
        continuations[rank], _ = collect_ids(loader, 4)

    for rank in range(W):
        monkeypatch.setattr(dlc, "get_dist_info", lambda r=rank: (True, r, r, W))
        loader = dlc.cached_distributed_data_loader_with_state(
            B=B, T=T, split="all", device="cpu", cache_dir=str(train),
            resume_state_dict=merged)
        resumed_ids, _ = collect_ids(loader, 4)
        assert resumed_ids == continuations[rank], f"rank {rank} resume diverged"


# ---------------------------------- trainer-level checkpoint/resume (P0-1)

def trainer_flow(loader_factory, grad_accum_steps, n_steps, resume_state=None):
    """Mirror base_train.py's loop structure exactly: prime, then per optimizer
    step run `grad_accum_steps` (train, prefetch) pairs; checkpoint at loop top
    saves the CONSUMED cursor, never the held prefetch's. Returns
    (trained_batches, held_prefetch_batch, checkpointed_state)."""
    def batch_ids(inputs, targets):
        ids = inputs.flatten().tolist()
        ids.append(int(targets.flatten()[-1]))
        return ids

    train_loader = loader_factory(resume_state)
    consumed_loader_state = resume_state                     # base_train.py:410
    x, y, after_current_loader_state = next(train_loader)    # base_train.py:411
    trained = []
    for _step in range(n_steps):
        for _micro_step in range(grad_accum_steps):          # base_train.py:645
            trained.append(batch_ids(x, y))                  # loss = model(x, y)
            consumed_loader_state = after_current_loader_state   # base_train.py:655
            x, y, after_current_loader_state = next(train_loader)  # :657
    # loop-top checkpoint (base_train.py:602): saves consumed_loader_state
    return trained, batch_ids(x, y), consumed_loader_state


@pytest.mark.parametrize("grad_accum_steps,n_steps", [(8, 2), (3, 4)])
def test_trainer_checkpoint_resume_replays_held_prefetch(tmp_path, grad_accum_steps, n_steps):
    """P0-1 regression: after prime -> N optimizer steps of (train, prefetch) ->
    checkpoint, resuming from the checkpointed cursor must serve the held
    (fetched-but-untrained) prefetch batch first — byte-identical — and continue
    the exact trained stream with no skip and no repeat."""
    train, _ = build_synthetic_cache(tmp_path)
    bake_ordering(train)
    from nanochat.dataloader_cached import cached_distributed_data_loader_with_state
    factory = lambda resume: cached_distributed_data_loader_with_state(
        B=B, T=T, split="all", device="cpu", cache_dir=str(train),
        resume_state_dict=resume)

    trained, held_prefetch, ckpt_state = trainer_flow(factory, grad_accum_steps, n_steps)

    # Resume run: train one more optimizer step from the checkpoint.
    resumed_trained, _, _ = trainer_flow(factory, grad_accum_steps, 1,
                                         resume_state=ckpt_state)
    assert resumed_trained[0] == held_prefetch, (
        "first resumed batch is not the held prefetch: the checkpoint saved a "
        "cursor ahead of model consumption (P0-1)")
    # Continuity: ground-truth stream with no checkpoint must match trained+resumed.
    truth, _, _ = trainer_flow(factory, grad_accum_steps, n_steps + 1)
    assert trained + resumed_trained == truth, (
        "checkpoint/resume stream diverges from the uninterrupted stream")


def test_base_train_source_saves_consumed_cursor():
    """Tripwire binding the emulation above to the real trainer: base_train.py
    must promote the held state to consumed AFTER backward and checkpoint the
    consumed cursor, not the prefetch cursor."""
    src = (REPO / "nanochat" / "scripts" / "base_train.py").read_text()
    assert "consumed_loader_state = after_current_loader_state" in src
    assert "merged_loader_state = consumed_loader_state" in src
    # The buggy pattern (checkpointing the prefetch state) must not reappear.
    assert "merged_loader_state = dataloader_state_dict" not in src
    # The prefetch inside the micro-loop must land in the after-state variable.
    assert "x, y, after_current_loader_state = next(train_loader)" in src


# --------------------------------------------------- checkpoint sentinel

def test_checkpoint_sentinel_and_find_last_step(tmp_path):
    from nanochat.checkpoint_manager import save_checkpoint, find_last_step
    ckpt = tmp_path / "ckpt"
    save_checkpoint(str(ckpt), 100, {"w": torch.zeros(3)}, [{"lr": 0.1}], {"step": 100}, rank=0)
    assert (ckpt / "complete_000100.json").exists()
    assert not list(ckpt.glob("*.tmp"))
    assert find_last_step(str(ckpt)) == 100
    # a torn checkpoint (model file only, no sentinel) must NOT win
    torch.save({"w": torch.zeros(3)}, ckpt / "model_000200.pt")
    assert find_last_step(str(ckpt)) == 100
    # legacy dir without sentinels still resolves (with warning)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    torch.save({}, legacy / "model_000300.pt")
    assert find_last_step(str(legacy)) == 300
