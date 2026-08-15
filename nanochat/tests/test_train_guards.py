"""Unit tests for nanochat.train_guards (Sol P0-6/P0-7) and the bf16 snapshot
helper — every refusal branch of the runtime integrity gates, on CPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nanochat"))

from nanochat.train_guards import (  # noqa: E402
    assert_expected_resolved, check_canary, load_canaries, parse_steps_list,
)

W1_MARKS = "954,1907,3815,7629,15259,19073,21876,22888,26703,30518,34332,36460"


# ------------------------------------------------------------- parse_steps_list

def test_parse_steps_list():
    assert parse_steps_list(None) == set()
    assert parse_steps_list("") == set()
    assert parse_steps_list("954, 1907") == {954, 1907}
    assert len(parse_steps_list(W1_MARKS)) == 12
    with pytest.raises(ValueError):
        parse_steps_list("0,5")
    with pytest.raises(ValueError):
        parse_steps_list("abc")


# ------------------------------------------------------------------ canaries

def canary_doc(**over):
    doc = {
        "ordering_sha256": "aa" * 32,
        "needed_per_yield": 65537,
        "world_size": 1,
        "grad_accum": 8,
        "canaries": [
            {"after_yield": 16, "optimizer_step": 2, "rank": 0,
             "shard": "shard_00003.bin", "position": 5, "offset": 123},
        ],
    }
    doc.update(over)
    return doc


def write_doc(tmp_path, doc):
    p = tmp_path / "canaries.json"
    p.write_text(json.dumps(doc))
    return str(p)


def test_load_canaries_ok(tmp_path):
    path = write_doc(tmp_path, canary_doc())
    got = load_canaries(path, needed=65537, world_size=1, grad_accum=8,
                        ordering_sha256="aa" * 32)
    assert (16, 0) in got and got[(16, 0)]["position"] == 5


def test_load_canaries_preserves_same_yield_for_every_rank(tmp_path):
    doc = canary_doc(
        world_size=2,
        canaries=[
            {"after_yield": 16, "optimizer_step": 2, "rank": 0,
             "shard": "shard_00003.bin", "position": 5, "offset": 123},
            {"after_yield": 16, "optimizer_step": 2, "rank": 1,
             "shard": "shard_00004.bin", "position": 6, "offset": 456},
        ],
    )
    got = load_canaries(
        write_doc(tmp_path, doc), needed=65537, world_size=2, grad_accum=8,
        ordering_sha256="aa" * 32,
    )
    assert got[(16, 0)]["position"] == 5
    assert got[(16, 1)]["position"] == 6


@pytest.mark.parametrize(
    "records,error",
    [
        (
            [
                {"after_yield": 16, "rank": 0, "position": 5, "offset": 123},
                {"after_yield": 16, "rank": 0, "position": 6, "offset": 456},
            ],
            "duplicate",
        ),
        (
            [{"after_yield": 16, "rank": 2, "position": 5, "offset": 123}],
            "rank",
        ),
        (
            [{"after_yield": 16, "rank": 0, "position": 5, "offset": 123}],
            "missing",
        ),
    ],
)
def test_load_canaries_rejects_ambiguous_rank_sets(tmp_path, records, error):
    path = write_doc(tmp_path, canary_doc(world_size=2, canaries=records))
    with pytest.raises(RuntimeError, match=error):
        load_canaries(
            path, needed=65537, world_size=2, grad_accum=8,
            ordering_sha256="aa" * 32,
        )


@pytest.mark.parametrize("kw,err", [
    (dict(needed=32769), "needed_per_yield"),          # the DBS-16 fallback case
    (dict(world_size=8), "world_size"),
    (dict(grad_accum=16), "grad_accum"),
    (dict(ordering_sha256="bb" * 32), "ordering_sha256"),
])
def test_load_canaries_refusals(tmp_path, kw, err):
    path = write_doc(tmp_path, canary_doc())
    kwargs = dict(needed=65537, world_size=1, grad_accum=8, ordering_sha256="aa" * 32)
    kwargs.update(kw)
    with pytest.raises(RuntimeError, match=err):
        load_canaries(path, **kwargs)


def test_load_canaries_accepts_real_launch_file():
    """The frozen launch canary file must load under the frozen W1 geometry."""
    path = REPO / "runs" / "launch_2026-07-06" / "canaries_world1_reverify.json"
    doc = json.loads(path.read_text())
    got = load_canaries(str(path), needed=65537, world_size=1, grad_accum=8,
                        ordering_sha256=doc["ordering_sha256"])
    assert len(got) == 12
    # and it must REFUSE the DBS-16 OOM-fallback geometry (P1-4)
    with pytest.raises(RuntimeError, match="needed_per_yield"):
        load_canaries(str(path), needed=32769, world_size=1, grad_accum=16,
                      ordering_sha256=doc["ordering_sha256"])


def canary_state(pos=5, off=123, fname="shard_00003.bin"):
    return {"per_rank": {"0": {"shard_idx": pos, "token_off": off}},
            "identity": {"ordering_position": pos, "manifest_shard_index": 42,
                         "filename": fname, "token_off": off}}


def test_check_canary_pass_and_fail():
    c = canary_doc()["canaries"][0]
    assert check_canary(c, canary_state(), rank=0) is None
    assert "ordering_position" in check_canary(c, canary_state(pos=6), rank=0)
    assert "token_off" in check_canary(c, canary_state(off=999), rank=0)
    assert "filename" in check_canary(c, canary_state(fname="shard_09999.bin"), rank=0)
    # Passing a foreign rank record is itself a scheduling error, not a pass.
    assert "rank" in check_canary(c, canary_state(pos=6), rank=1)
    # missing cursor is a failure, not a pass
    assert check_canary(c, {"per_rank": {}}, rank=0) is not None


# ------------------------------------------------- expected-resolved assertions

def resolved_fixture():
    return {
        "model": {"n_embd": 1664, "n_head": 13, "num_params": 972947456},
        "optimizer": {"adamw_groups_initial_lr": [0.00271746, 0.20380989],
                      "dmodel_lr_scale": 0.6793662204867574},
        "tokenizer": {"bos_id": 32759},
    }


def test_assert_expected_resolved_pass():
    exp = resolved_fixture()
    checked = assert_expected_resolved(exp, resolved_fixture())
    assert len(checked) >= 6


def test_assert_expected_resolved_float_tolerance():
    exp = resolved_fixture()
    res = resolved_fixture()
    res["optimizer"]["dmodel_lr_scale"] *= 1 + 1e-9   # within rtol
    assert_expected_resolved(exp, res)
    res["optimizer"]["dmodel_lr_scale"] = 1.0          # the no-scale failure mode
    with pytest.raises(AssertionError, match="dmodel_lr_scale"):
        assert_expected_resolved(exp, res)


def test_assert_expected_resolved_reports_all_failures():
    exp = resolved_fixture()
    res = resolved_fixture()
    res["model"]["n_head"] = 12
    res["tokenizer"]["bos_id"] = 0
    with pytest.raises(AssertionError) as ei:
        assert_expected_resolved(exp, res)
    assert "n_head" in str(ei.value) and "bos_id" in str(ei.value)


def test_assert_expected_resolved_missing_key():
    exp = resolved_fixture()
    res = resolved_fixture()
    del res["optimizer"]["adamw_groups_initial_lr"]
    with pytest.raises(AssertionError, match="MISSING"):
        assert_expected_resolved(exp, res)


def test_underscore_keys_are_commentary():
    assert_expected_resolved({"_comment": "ignored", "model": {"n_embd": 4}},
                             {"model": {"n_embd": 4}})


def test_real_expect_json_is_internally_consistent():
    """The shipped expect file must state the gpt.py:265 scale and LRs that
    actually multiply out (GPT Pro round-4: emb 0.203810, unemb 0.002717)."""
    doc = json.loads((REPO / "runs" / "launch_2026-07-06" /
                      "expected_resolved_d26.json").read_text())
    scale = doc["optimizer"]["dmodel_lr_scale"]
    assert abs(scale - (1664 / 768) ** -0.5) < 1e-12
    unemb, emb = doc["optimizer"]["adamw_groups_initial_lr"]
    assert abs(emb - 0.3 * scale) < 1e-12 and abs(unemb - 0.004 * scale) < 1e-12
    assert doc["model"]["num_params"] == 972947456
    assert doc["schedule"]["grad_accum_steps"] == 8


# ------------------------------------------------------------- bf16 snapshots

def test_save_bf16_snapshot(tmp_path):
    from nanochat.checkpoint_manager import save_bf16_snapshot
    state = {"w": torch.randn(4, 4), "idx": torch.arange(3)}
    path = save_bf16_snapshot(str(tmp_path), 21876, state)
    assert path.endswith("snapshot_bf16_021876.pt")
    loaded = torch.load(path)
    assert loaded["w"].dtype == torch.bfloat16
    assert loaded["idx"].dtype == torch.int64          # non-float passes through
    assert not list(tmp_path.glob("*.tmp"))
    assert torch.allclose(state["w"].bfloat16().float(), loaded["w"].float())
