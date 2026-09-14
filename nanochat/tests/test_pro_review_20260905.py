"""Regressions for the 2026-09-05 repository review findings.

One group per finding, each test named for its burn-queue id:

  bq-2036  checked packaging enforces explicit date evidence against the cutoff
  bq-2037  the sequence-length curriculum survives the strict resume contract
  bq-2038  a token or shard cap cannot lose the active file's examinations
  bq-2039  the raw loader refuses a split in which some rank owns no rows

bq-2040 (synthetic legacy canaries) is covered in test_burn_day_loader_r2.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nanochat"))

from data.process import shard_packager  # noqa: E402


# ---------------------------------------------------------------------------
# bq-2038: capped checked packaging keeps the active file's counters
# ---------------------------------------------------------------------------

def _plain_records(path: Path, count: int, *, start: int = 0) -> Path:
    # ~350 characters each, no four-digit numbers, nothing the body checker flags.
    path.write_text("".join(
        json.dumps({
            "source": "fixture",
            "text": (f"Entry {i} of the parish register, kept in the vestry by the "
                     "curate and read aloud at the quarter sessions. ") * 3,
        }) + "\n"
        for i in range(start, start + count)
    ))
    return path


def _checked(inputs, output: Path, **kwargs):
    return shard_packager.package_shards_streaming(
        input_files=[str(p) for p in inputs],
        output_dir=str(output),
        buffer_size=4,
        row_group_size=1,
        cutoff_year=1913,
        run_contamination=True,
        run_ocr_quality=False,
        **kwargs,
    )


def _assert_attested(stats: dict, output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    examined = manifest["contamination_control"]["records_examined"]
    assert manifest["total_docs"] == stats["total_docs"] > 0
    assert examined >= manifest["total_docs"]
    assert (output / "stats.json").exists()


def test_token_cap_inside_first_file_keeps_its_examinations_bq_2038(tmp_path):
    source = _plain_records(tmp_path / "only.jsonl", 12)
    output = tmp_path / "out"
    stats = _checked([source], output, chars_per_shard=10**9, max_tokens=100)
    _assert_attested(stats, output)
    per_file = stats["rejections"]["per_file"]
    assert len(per_file) == 1 and per_file[0]["contamination_examined"] > 0


def test_token_cap_inside_later_file_keeps_its_examinations_bq_2038(tmp_path):
    first = _plain_records(tmp_path / "first.jsonl", 3)
    later = _plain_records(tmp_path / "later.jsonl", 12, start=100)
    output = tmp_path / "out"
    stats = _checked([first, later], output, chars_per_shard=10**9, max_tokens=300)
    _assert_attested(stats, output)
    assert [c["file"] for c in stats["rejections"]["per_file"]] == [str(first), str(later)]


def test_shard_cap_inside_first_file_keeps_its_examinations_bq_2038(tmp_path):
    source = _plain_records(tmp_path / "only.jsonl", 12)
    output = tmp_path / "out"
    stats = _checked([source], output, chars_per_shard=1, max_shards=2)
    assert stats["num_shards"] == 2
    _assert_attested(stats, output)


def test_uncapped_eof_boundary_still_attests_bq_2038(tmp_path):
    source = _plain_records(tmp_path / "only.jsonl", 4)   # exactly one buffer
    output = tmp_path / "out"
    stats = _checked([source], output, chars_per_shard=10**9)
    assert stats["total_docs"] == 4
    _assert_attested(stats, output)
    assert stats["rejections"]["totals"]["contamination_examined"] == 4


# ---------------------------------------------------------------------------
# bq-2039: raw loader rank liveness
# ---------------------------------------------------------------------------

def _parquet(path: Path, row_groups: list[int]) -> str:
    schema = pa.schema([("text", pa.string())])
    writer = pq.ParquetWriter(str(path), schema)
    for group, rows in enumerate(row_groups):
        writer.write_table(pa.Table.from_pydict(
            {"text": [f"document {group}-{i} of the parish" for i in range(rows)]},
            schema=schema,
        ))
    writer.close()
    assert pq.read_metadata(str(path)).num_row_groups == len(row_groups)
    return str(path)


class _FakeTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, docs, prepend=None, num_threads=None):
        return [[prepend, 1, 2, 3, 4, 5, 6] for _ in docs]


def _raw_loader(monkeypatch, paths, *, rank, world_size, split="train"):
    import nanochat.dataloader as dataloader

    monkeypatch.setattr(dataloader, "list_parquet_files", lambda data_dir=None: list(paths))
    monkeypatch.setattr(dataloader, "get_dist_info",
                        lambda: (world_size > 1, rank, rank, world_size))
    monkeypatch.setattr(dataloader, "get_tokenizer", lambda: _FakeTokenizer())
    return dataloader.tokenizing_distributed_data_loader_with_state(
        B=1, T=4, split=split, device="cpu", parquet_dir="unused",
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_raw_loader_refuses_when_a_rank_owns_no_train_rows_bq_2039(tmp_path, monkeypatch, rank):
    # The review's trigger: two files, one non-empty row group each, world_size=2.
    # Rank 1 used to loop forever; every rank now refuses at construction.
    paths = [_parquet(tmp_path / "a.parquet", [3]), _parquet(tmp_path / "b.parquet", [3])]
    with pytest.raises(RuntimeError, match=r"train split gives no rows to DDP rank\(s\) \[1\]"):
        _raw_loader(monkeypatch, paths, rank=rank, world_size=2)


def test_raw_loader_refuses_a_validation_split_too_small_for_the_world_bq_2039(tmp_path, monkeypatch):
    paths = [
        _parquet(tmp_path / "a.parquet", [3, 3]),
        _parquet(tmp_path / "b.parquet", [3, 3]),
        _parquet(tmp_path / "val.parquet", [3]),
    ]
    with pytest.raises(RuntimeError, match=r"val split gives no rows to DDP rank\(s\) \[1\]"):
        _raw_loader(monkeypatch, paths, rank=0, world_size=2, split="train")


def test_raw_loader_ignores_empty_row_groups_when_assigning_ranks_bq_2039(tmp_path, monkeypatch):
    paths = [_parquet(tmp_path / "a.parquet", [3, 0]), _parquet(tmp_path / "val.parquet", [3, 3])]
    with pytest.raises(RuntimeError, match=r"train split gives no rows to DDP rank\(s\) \[1\]"):
        _raw_loader(monkeypatch, paths, rank=1, world_size=2)


def test_raw_loader_serves_every_rank_that_owns_rows_bq_2039(tmp_path, monkeypatch):
    paths = [
        _parquet(tmp_path / "a.parquet", [3, 3]),
        _parquet(tmp_path / "val.parquet", [3, 3]),
    ]
    loader = _raw_loader(monkeypatch, paths, rank=1, world_size=2)
    inputs, targets, state = next(loader)
    assert tuple(inputs.shape) == (1, 4) and tuple(targets.shape) == (1, 4)
    assert state == {"pq_idx": 0, "rg_idx": 1}


# ---------------------------------------------------------------------------
# bq-2036: checked packaging enforces explicit date evidence against the cutoff
# ---------------------------------------------------------------------------

INNOCUOUS = ("A plain account of the harvest festival in the village, with hymns "
             "sung in the parish church and a supper laid in the schoolroom. ") * 3


def _jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _gutenberg(title: str, *, published: int | None, metadata: dict | None = None, **extra) -> dict:
    record = {
        "source": "gutenberg",
        "text": f"{title}. {INNOCUOUS}",
        "metadata": {"title": title, **(metadata or {})},
        "publication_date": None if published is None else {
            "year": published,
            "source_field": "publicationyear",
            "source_value": str(published),
            "confidence": "validated_publication_field",
        },
    }
    record.update(extra)
    return record


def _package_dated(inputs, output: Path, *, run_contamination: bool = True):
    return shard_packager.package_shards_streaming(
        input_files=[str(p) for p in inputs],
        output_dir=str(output),
        chars_per_shard=10**9,
        row_group_size=1,
        buffer_size=2,
        cutoff_year=1913,
        run_contamination=run_contamination,
        run_ocr_quality=False,
        run_dedup=False,
    )


def _shard_texts(output: Path, stats: dict) -> list[str]:
    texts = []
    for shard in stats["shards"]:
        texts.extend(pq.read_table(output / shard["filename"]).column("text").to_pylist())
    return texts


def test_mixed_cutoff_acquisitions_cannot_leak_a_later_publication_bq_2036(tmp_path):
    # The review's trigger: 1913 and 1950 Gutenberg acquisitions side by side, the
    # later one holding a 1920 publication whose body carries no later signal.
    raw = tmp_path / "raw"
    early = _jsonl(raw / "gutenberg" / "gutenberg_1913.jsonl", [_gutenberg("Early Work", published=1890)])
    later = _jsonl(raw / "gutenberg" / "gutenberg_1950.jsonl", [
        _gutenberg("Early Work", published=1890),
        _gutenberg("Later Work", published=1920),
    ])

    selection = {}
    assert shard_packager.find_corpus_files(str(raw), cutoff_year=1913, report=selection) == [str(early)]
    assert [item["file"] for item in selection["excluded_cutoff_variants"]] == [str(later)]

    # Even an explicit selection of the later acquisition cannot admit the dated record.
    output = tmp_path / "out"
    stats = _package_dated([later], output)
    texts = _shard_texts(output, stats)
    assert any("Early Work" in text for text in texts)
    assert not any("Later Work" in text for text in texts)
    temporal = stats["temporal_metadata_control"]
    assert temporal["status"] == "CHECKED" and temporal["cutoff_year"] == 1913
    assert temporal["records_with_date_evidence"] == 2
    assert temporal["rejected_post_cutoff_date"] == 1
    assert json.loads((output / "manifest.json").read_text())["temporal_metadata_control"] == temporal


def test_date_provenance_sidecar_names_packaged_rows_and_rejections_bq_2036(tmp_path):
    source = _jsonl(tmp_path / "acquired.jsonl", [
        {"source": "chronicling_america", "date": "1912-12-15", "year": 1912, "text": f"Issue one. {INNOCUOUS}"},
        {"source": "chronicling_america", "date": "1914-07-28", "year": 1914, "text": f"Issue two. {INNOCUOUS}"},
        {"source": "unlisted_archive", "year": 1950, "text": f"Unlisted. {INNOCUOUS}"},
    ])
    output = tmp_path / "out"
    stats = _package_dated([source], output)

    entries = [json.loads(line) for line in (output / "date_provenance.jsonl").read_text().splitlines()]
    rejected = [entry for entry in entries if entry["decision"] == "rejected_post_cutoff_date"]
    packaged = [entry for entry in entries if entry["decision"] == "packaged"]
    assert [(entry["line"], entry["evidence"][0]["value"]) for entry in rejected] == [(2, "1914-07-28")]
    assert [entry["line"] for entry in packaged] == [1]
    rows = pq.read_table(output / packaged[0]["shard"]).column("text").to_pylist()
    assert rows[packaged[0]["row"]].startswith("Issue one.")
    # A source with no allow-listed date field is left to the body checker alone.
    assert any(text.startswith("Unlisted.") for text in _shard_texts(output, stats))


def test_sidecar_keeps_the_whole_publication_date_evidence_object_bq_2036(tmp_path):
    evidence = {"year": 1890, "source_field": "text:printed", "source_value": "Printed in 1890",
                "confidence": "validated_publication_phrase"}
    record = _gutenberg("Printed Sermon", published=None)
    record["publication_date"] = evidence
    source = _jsonl(tmp_path / "gutenberg_1913.jsonl", [record])
    output = tmp_path / "out"
    _package_dated([source], output)
    [entry] = [json.loads(line) for line in (output / "date_provenance.jsonl").read_text().splitlines()]
    assert entry["decision"] == "packaged"
    assert entry["evidence"] == [{"field": "publication_date.year", "value": 1890, "year": 1890,
                                  "evidence_object": evidence}]


@pytest.mark.parametrize("record, rejected", [
    ({"source": "caselaw_cap_v2", "decision_date": "1914-02-01", "year": 1914}, 1),
    ({"source": "caselaw_cap_v2", "decision_date": "1913-10-01", "year": 1913}, 0),
    ({"source": "oldbailey", "date": "19140115", "year": 1914}, 1),
    ({"source": "american_stories", "date": "1913-12-31", "year": 1913}, 0),
    ({"source": "bhl", "year": 1914}, 1),
    ({"source": "ecco_tcp", "year": "1789"}, 0),
])
def test_issue_decision_trial_and_publication_years_are_enforced_bq_2036(tmp_path, record, rejected):
    source = _jsonl(tmp_path / "acquired.jsonl", [
        dict(record, text=f"Dated record. {INNOCUOUS}"),
        {"source": "bhl", "year": 1900, "text": f"Control. {INNOCUOUS}"},
    ])
    stats = _package_dated([source], tmp_path / "out")
    assert stats["temporal_metadata_control"]["rejected_post_cutoff_date"] == rejected
    assert stats["total_docs"] == 2 - rejected


def test_catalogue_and_release_dates_never_reject_an_early_publication_bq_2036(tmp_path):
    source = _jsonl(tmp_path / "gutenberg_1913.jsonl", [
        _gutenberg("Old Sermon", published=1850,
                   metadata={"issued": "2004-06-01", "release_date": "2011"}, estimated_year=1990),
        _gutenberg("Undated Tract", published=None, metadata={"issued": "2019-01-01"}),
    ])
    output = tmp_path / "out"
    stats = _package_dated([source], output)
    texts = _shard_texts(output, stats)
    assert any("Old Sermon" in text for text in texts)
    assert any("Undated Tract" in text for text in texts)
    assert stats["temporal_metadata_control"]["rejected_post_cutoff_date"] == 0


def test_discovery_refuses_cutoff_variants_with_none_for_the_cutoff_bq_2036(tmp_path):
    raw = tmp_path / "raw"
    _jsonl(raw / "gutenberg" / "gutenberg_1900.jsonl", [_gutenberg("A", published=1890)])
    _jsonl(raw / "gutenberg" / "gutenberg_1950.jsonl", [_gutenberg("B", published=1920)])
    with pytest.raises(ValueError, match="cutoff variants of the 'gutenberg' acquisition"):
        shard_packager.find_corpus_files(str(raw), cutoff_year=1913)


def test_discovery_keeps_lone_files_and_non_variant_siblings_bq_2036(tmp_path):
    # The deduped corpus's own layout: one file per source, or siblings that are
    # not cutoff variants of each other.
    raw = tmp_path / "raw"
    for relative in ("bhl/bhl_1914.jsonl", "tcp/tcp_1914.jsonl",
                     "caselaw_cap_v2/cap_pre1914.jsonl", "caselaw_cap_v2/cap_pre1914_refined.jsonl"):
        _jsonl(raw / relative, [{"source": "bhl", "year": 1900, "text": INNOCUOUS}])
    report = {}
    found = shard_packager.find_corpus_files(str(raw), cutoff_year=1913, report=report)
    assert sorted(Path(path).name for path in found) == [
        "bhl_1914.jsonl", "cap_pre1914.jsonl", "cap_pre1914_refined.jsonl", "tcp_1914.jsonl",
    ]
    assert report["excluded_cutoff_variants"] == []


def test_unchecked_packaging_is_stamped_unchecked_for_dates_too_bq_2036(tmp_path):
    source = _jsonl(tmp_path / "acquired.jsonl", [_gutenberg("Later Work", published=1920)])
    output = tmp_path / "out"
    stats = _package_dated([source], output, run_contamination=False)
    assert stats["temporal_metadata_control"]["status"] == "UNCHECKED"
    assert stats["temporal_metadata_control"]["provenance_sidecar"] is None
    assert not (output / "date_provenance.jsonl").exists()


# ---------------------------------------------------------------------------
# bq-2037: the sequence-length curriculum and the strict sequential resume contract
# ---------------------------------------------------------------------------

def _token_cache(root: Path, shard_tokens=(40, 40)) -> Path:
    import numpy as np

    root.mkdir(parents=True)
    shards, start = [], 0
    for index, count in enumerate(shard_tokens):
        name = f"shard_{index:05d}.bin"
        np.arange(start, start + count, dtype=np.dtype("<u2")).tofile(root / name)
        shards.append({"shard_index": index, "filename": name, "source_file": f"/source/{index}.parquet",
                       "docs": 1, "tokens": count, "bytes": count * 2})
        start += count
    (root / "cache_manifest.json").write_text(json.dumps({
        "format_version": 1, "byte_order": "little", "dtype": "uint16",
        "total_docs": len(shards), "total_tokens": start, "shards": shards,
    }))
    return root


def test_sequential_cursor_rebinds_geometry_and_keeps_every_other_identity_bq_2037(tmp_path):
    from nanochat.dataloader_cached import (
        cached_distributed_data_loader_with_state as cached_loader,
        rebind_sequential_resume_geometry,
    )

    cache = str(_token_cache(tmp_path / "cache"))
    early = cached_loader(1, 2, split="all", device="cpu", cache_dir=cache)
    for _ in range(2):
        _inputs, _targets, consumed = next(early)
    assert consumed["per_rank"]["0"]["token_off"] == 6   # two batches of B*T+1 = 3 tokens

    # An ordinary resume stays strict: the old-geometry cursor is refused at T=4.
    with pytest.raises(RuntimeError, match="sequence_length"):
        next(cached_loader(1, 4, split="all", device="cpu", cache_dir=cache, resume_state_dict=consumed))

    rebound = rebind_sequential_resume_geometry(consumed, batch_size=1, sequence_length=4)
    assert consumed["per_rank"]["0"]["resume_contract"]["sequence_length"] == 2   # input untouched
    inputs, targets, _state = next(
        cached_loader(1, 4, split="all", device="cpu", cache_dir=cache, resume_state_dict=rebound)
    )
    assert inputs.flatten().tolist() == [6, 7, 8, 9]
    assert targets.flatten().tolist() == [7, 8, 9, 10]

    # Only the geometry moved: a cursor bound to a different cache is still refused.
    forged = rebind_sequential_resume_geometry(consumed, batch_size=1, sequence_length=4)
    forged["per_rank"]["0"]["resume_contract"]["cache_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="cache_manifest_sha256"):
        next(cached_loader(1, 4, split="all", device="cpu", cache_dir=cache, resume_state_dict=forged))


def test_rebind_refuses_a_state_that_is_not_sequential_cache_bq_2037():
    from nanochat.dataloader_cached import rebind_sequential_resume_geometry

    family_state = {"per_rank": {"0": {"resume_contract": {"loader_strategy": "parallel_family_cache"}}}}
    with pytest.raises(RuntimeError, match="sequential_cache"):
        rebind_sequential_resume_geometry(family_state, batch_size=1, sequence_length=4)


def _tiny_tokenizer(base_dir: Path) -> None:
    import hashlib
    import pickle

    import numpy as np
    import tiktoken

    tok_dir = base_dir / "tokenizer"
    tok_dir.mkdir(parents=True)
    specials = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>",
                "<|assistant_end|>", "<|python_start|>", "<|python_end|>",
                "<|output_start|>", "<|output_end|>"]
    special_ids = {name: 256 + i for i, name in enumerate(specials)}
    encoding = tiktoken.Encoding(
        name="curriculum-test", pat_str=r"(?s:.)",
        mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens=special_ids,
    )
    with open(tok_dir / "tokenizer.pkl", "wb") as handle:
        pickle.dump(encoding, handle)
    token_bytes = np.ones(encoding.n_vocab, dtype=np.int32)
    for token_id in special_ids.values():
        token_bytes[token_id] = 0
    np.save(tok_dir / "token_bytes.npy", token_bytes, allow_pickle=False)
    (tok_dir / "tokenizer_manifest.json").write_text(json.dumps({
        "tokenizer": {"vocab_size": encoding.n_vocab, "special_tokens": special_ids},
        "outputs": {
            "tokenizer_pkl": "tokenizer/tokenizer.pkl",
            "sha256_tokenizer_pkl": hashlib.sha256((tok_dir / "tokenizer.pkl").read_bytes()).hexdigest(),
            "token_bytes_npy": "tokenizer/token_bytes.npy",
            "sha256_token_bytes_npy": hashlib.sha256((tok_dir / "token_bytes.npy").read_bytes()).hexdigest(),
        },
    }))


def _nanochat_env(base_dir: Path) -> dict:
    import os

    env = os.environ.copy()
    for distributed_key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(distributed_key, None)
    env.update({"NANOCHAT_BASE_DIR": str(base_dir), "TORCH_COMPILE_DISABLE": "1",
                "WANDB_MODE": "offline", "GLOO_SOCKET_IFNAME": "lo"})
    return env


def _train(tmp_path: Path, base_dir: Path, label: str, *flags: str) -> str:
    import subprocess

    rendezvous = tmp_path / f"gloo-{label}"
    runner = (
        "import runpy; import torch.distributed as dist; "
        f"dist.init_process_group('gloo', init_method='file://{rendezvous}', rank=0, world_size=1); "
        "runpy.run_module('scripts.base_train', run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", runner,
         "--device_type=cpu", "--depth=4", "--aspect_ratio=16", "--head_dim=16",
         "--device_batch_size=1", "--total_batch_size=64", "--num_iterations=4",
         "--eval_every=-1", "--core_metric_every=-1", "--sample_every=-1", "--save_every=-1",
         "--compile_mode=none", *flags],
        env=_nanochat_env(base_dir), cwd=str(REPO / "nanochat"),
        capture_output=True, text=True, timeout=240,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"base_train ({label}) failed:\n{out[-3000:]}"
    assert "step 00003/00004" in out, f"base_train ({label}) did not finish:\n{out[-2000:]}"
    return out


def test_curriculum_switch_on_sequential_cache_trains_and_resumes_on_both_sides_bq_2037(tmp_path):
    import subprocess

    base_dir = tmp_path / "base"
    _tiny_tokenizer(base_dir)
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    for index in range(3):
        pq.write_table(
            pa.Table.from_pydict({"text": [f"the quick brown fox number {index} jumps " * 4] * 16}),
            parquet_dir / f"shard_{index:05d}.parquet", row_group_size=8,
        )
    cache_dir = tmp_path / "cache"
    built = subprocess.run(
        [sys.executable, "-m", "scripts.build_token_cache",
         "--input-dir", str(parquet_dir), "--output-dir", str(cache_dir)],
        env=_nanochat_env(base_dir), cwd=str(REPO / "nanochat"),
        capture_output=True, text=True, timeout=120,
    )
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]

    common = ("--token_cache_dir=" + str(cache_dir), "--loader_strategy=sequential_cache",
              "--max_seq_len=16", "--seq_len_late=32", "--seq_len_late_frac=0.5",
              "--model_tag=curriculum")
    # A fresh run crosses the switch at step 2 and checkpoints on both sides of it.
    fresh = _train(tmp_path, base_dir, "fresh", *common, "--save_at_steps=1,2")
    assert "switching T 16 -> 32" in fresh
    # Resuming before the switch crosses it again in process.
    before = _train(tmp_path, base_dir, "resume-before", *common, "--resume_from_step=1")
    assert "switching T 16 -> 32" in before
    # Resuming from the checkpoint written at the switch starts at the late geometry.
    after = _train(tmp_path, base_dir, "resume-after", *common, "--resume_from_step=2")
    assert "Starting at curriculum T=32" in after
    assert "switching T" not in after


def test_family_cache_curriculum_is_refused_before_training_bq_2037(tmp_path):
    import subprocess

    base_dir = tmp_path / "base"
    _tiny_tokenizer(base_dir)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.base_train", "--device_type=cpu", "--depth=4",
         "--token_cache_dir=" + str(tmp_path / "unused-cache"),
         "--loader_strategy=parallel_family_cache", "--seq_len_late=64", "--max_seq_len=32"],
        env=_nanochat_env(base_dir), cwd=str(REPO / "nanochat"),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "--seq_len_late is not supported with --loader_strategy=parallel_family_cache" in (
        result.stdout + result.stderr
    )
