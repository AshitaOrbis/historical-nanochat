"""bq-1942: repackaging into an existing output directory must not leave
obsolete shards discoverable by manifest-blind consumers.

Covers three points named in the finding:
  - the packager refuses to repackage into a directory that already holds its
    own output, unless replace_existing=True is chosen explicitly
  - replace_existing=True removes exactly the packager-owned files and
    nothing else, so a smaller second run does not leave a larger first run's
    shards behind
  - manifest-aware consumers (nanochat.dataset.list_parquet_files,
    scripts.build_token_cache.resolve_input_shards) reject a shard_*.parquet
    file that exists on disk but is not listed in manifest.json, rather than
    silently discovering and using it
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.process import shard_packager  # noqa: E402


def _write_record(path: Path, *, source: str = "gutenberg", text: str | None = None) -> None:
    text = text or ("An ordinary historical sentence about the parish and its people. " * 5)
    path.write_text(json.dumps({"source": source, "text": text}) + "\n")


def _package(source: Path, output: Path, **kwargs):
    return shard_packager.package_shards_streaming(
        input_files=[str(source)],
        output_dir=str(output),
        chars_per_shard=1,   # force one shard per document
        row_group_size=1,
        buffer_size=1,
        run_contamination=False,
        run_dedup=False,
        **kwargs,
    )


def test_repackaging_into_existing_output_dir_is_refused_by_default(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "shards"
    _write_record(source)
    _package(source, output)

    with pytest.raises(FileExistsError, match="already contains packager output"):
        _package(source, output)


def test_replace_existing_removes_only_packager_owned_stale_shards(tmp_path):
    source3 = tmp_path / "input3.jsonl"
    source3.write_text("\n".join(
        json.dumps({
            "source": "gutenberg",
            "text": f"Document number {i} about the parish and its long historical record. " * 3,
        })
        for i in range(3)
    ) + "\n")
    source1 = tmp_path / "input1.jsonl"
    _write_record(source1)
    output = tmp_path / "shards"

    stats_first = _package(source3, output)
    assert stats_first["num_shards"] == 3
    unrelated = output / "README.txt"
    unrelated.write_text("do not touch me")

    stats_second = _package(source1, output, replace_existing=True)
    assert stats_second["num_shards"] == 1

    remaining_shards = sorted(p.name for p in output.glob("shard_*.parquet"))
    assert remaining_shards == ["shard_00000.parquet"], (
        f"stale higher-numbered shards from the first run must be removed, got {remaining_shards}"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert [s["filename"] for s in manifest["shards"]] == ["shard_00000.parquet"]
    # Do not broadly delete unrelated output-directory contents.
    assert unrelated.exists()


def test_dataset_list_parquet_files_rejects_unlisted_shard(tmp_path):
    from nanochat.dataset import list_parquet_files

    source = tmp_path / "input.jsonl"
    output = tmp_path / "shards"
    _write_record(source)
    _package(source, output)

    # Simulate a stale shard left behind by a prior, differently-configured run:
    # present on disk, but never mentioned by the CURRENT manifest.
    (output / "shard_00099.parquet").write_bytes((output / "shard_00000.parquet").read_bytes())

    with pytest.raises(RuntimeError, match="not listed in"):
        list_parquet_files(data_dir=str(output))


def test_dataset_list_parquet_files_uses_manifest_when_clean(tmp_path):
    from nanochat.dataset import list_parquet_files

    source = tmp_path / "input.jsonl"
    output = tmp_path / "shards"
    _write_record(source)
    _package(source, output)

    paths = list_parquet_files(data_dir=str(output))
    assert [Path(p).name for p in paths] == ["shard_00000.parquet"]


def test_build_token_cache_resolve_input_shards_rejects_unlisted_shard(tmp_path):
    sys.path.insert(0, str(REPO / "nanochat"))
    from scripts.build_token_cache import resolve_input_shards

    source = tmp_path / "input.jsonl"
    output = tmp_path / "shards"
    _write_record(source)
    _package(source, output)

    (output / "shard_00099.parquet").write_bytes((output / "shard_00000.parquet").read_bytes())

    with pytest.raises(SystemExit, match="not listed in"):
        resolve_input_shards(output)


def test_build_token_cache_resolve_input_shards_uses_manifest_when_clean(tmp_path):
    sys.path.insert(0, str(REPO / "nanochat"))
    from scripts.build_token_cache import resolve_input_shards

    source = tmp_path / "input.jsonl"
    output = tmp_path / "shards"
    _write_record(source)
    _package(source, output)

    shards = resolve_input_shards(output)
    assert [p.name for p in shards] == ["shard_00000.parquet"]
