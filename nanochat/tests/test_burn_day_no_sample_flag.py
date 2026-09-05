"""RECOVERED-2: --no-sample must actually disable per-source sampling.

The CLI wired --no-sample to sample_rates=None, and package_shards_streaming
substitutes DEFAULT_SAMPLE_RATES whenever sample_rates is None -- so the
documented opt-out silently re-applied the very sampling it claims to disable.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]

# american_stories has a default sample rate of 0.33 (see DEFAULT_SAMPLE_RATES
# in data/process/shard_packager.py) -- a strong signal if downsampling still
# applies under --no-sample.
NUM_DOCS = 200
DOC_TEXT = "A historical sentence about the parish and its people, repeated for length. " * 2


def _write_corpus(base_dir: Path) -> Path:
    # detect_source_from_path() checks the parent directory name first, so
    # placing the file under an "american_stories" directory pins the source.
    source_dir = base_dir / "american_stories"
    source_dir.mkdir(parents=True)
    corpus_path = source_dir / "part.jsonl"
    with corpus_path.open("w") as f:
        for i in range(NUM_DOCS):
            f.write(json.dumps({"source": "american_stories", "text": f"{DOC_TEXT} doc {i}"}) + "\n")
    return corpus_path


def test_no_sample_flag_keeps_every_document(tmp_path):
    corpus_path = _write_corpus(tmp_path / "input")
    output_dir = tmp_path / "shards"

    result = subprocess.run(
        [
            sys.executable, "-m", "data.process.shard_packager",
            "--input", str(corpus_path),
            "--output-dir", str(output_dir),
            "--no-sample",
            "--no-ocr-quality",
            "--no-dedup",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["total_docs"] == NUM_DOCS, (
        f"--no-sample must keep all {NUM_DOCS} documents, got "
        f"{manifest['total_docs']} (rejections={manifest['rejections']['totals']})"
    )
    assert manifest["rejections"]["totals"].get("rejected_sampling", 0) == 0
    # The identity map applied under --no-sample must not equal the balanced
    # defaults it is supposed to override.
    assert manifest["config"]["sample_rates"] != {
        "american_stories": 0.33, "bhl": 0.60, "bl_newspapers": 1.0,
    }


def test_default_sampling_still_downsamples_american_stories(tmp_path):
    """Control: without --no-sample the same corpus IS downsampled, proving the
    fixture would actually catch the regression rather than passing vacuously."""
    corpus_path = _write_corpus(tmp_path / "input")
    output_dir = tmp_path / "shards"

    result = subprocess.run(
        [
            sys.executable, "-m", "data.process.shard_packager",
            "--input", str(corpus_path),
            "--output-dir", str(output_dir),
            "--no-ocr-quality",
            "--no-dedup",
            "--seed", "42",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["total_docs"] < NUM_DOCS
    assert manifest["rejections"]["totals"].get("rejected_sampling", 0) > 0
