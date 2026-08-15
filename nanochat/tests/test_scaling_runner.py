"""Regression tests for the root RTX 3090 scaling sweep wrapper."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "run_scaling_3090.sh"
CSV_HEADER = "depth,num_params,num_iterations,tokens_trained,val_bpb,train_time_sec"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_fake_runner(tmp_path: Path, python_body: str):
    sandbox = tmp_path / "runner"
    sandbox.mkdir()
    (sandbox / "nanochat").mkdir()
    activate = sandbox / ".venv" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("")

    results_dir = sandbox / "results"
    runner_text, result_replacements = re.subn(
        r'^RESULTS_DIR=.*$',
        f'RESULTS_DIR="{results_dir}"',
        RUNNER.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    runner_text, depth_replacements = re.subn(
        r'^DEPTHS=.*$', "DEPTHS=(8)", runner_text, count=1, flags=re.MULTILINE
    )
    assert (result_replacements, depth_replacements) == (1, 1)
    runner = sandbox / RUNNER.name
    runner.write_text(runner_text)

    fake_bin = sandbox / "fake-bin"
    fake_bin.mkdir()
    write_executable(fake_bin / "python", python_body)
    write_executable(fake_bin / "column", "#!/bin/bash\nexit 0\n")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["NANOCHAT_BASE_DIR"] = str(sandbox / "nanochat-base")
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, results_dir


def test_failed_trainer_aborts_without_observation_or_completion(tmp_path):
    sandbox = tmp_path / "runner"
    sandbox.mkdir()
    (sandbox / "nanochat").mkdir()
    activate = sandbox / ".venv" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("")

    results_dir = sandbox / "results"
    runner_text, replacements = re.subn(
        r'^RESULTS_DIR=.*$',
        f'RESULTS_DIR="{results_dir}"',
        RUNNER.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    assert replacements == 1, "test harness could not isolate the results directory"
    runner = sandbox / RUNNER.name
    runner.write_text(runner_text)

    fake_bin = sandbox / "fake-bin"
    fake_bin.mkdir()
    write_executable(
        fake_bin / "python",
        """#!/bin/bash
echo 'Number of parameters: 1,000'
echo 'Calculated number of iterations: 7'
echo 'Validation bpb: 1.23'
exit 23
""",
    )
    write_executable(fake_bin / "column", "#!/bin/bash\nexit 0\n")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["NANOCHAT_BASE_DIR"] = str(sandbox / "nanochat-base")
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23, result.stdout + result.stderr
    assert (results_dir / "results.csv").read_text().splitlines() == [CSV_HEADER]
    assert "Scaling Runs Complete" not in result.stdout


def test_success_without_checkpoint_sentinel_is_not_recorded(tmp_path):
    result, results_dir = run_fake_runner(
        tmp_path,
        """#!/bin/bash
echo 'Number of parameters: 1,000'
echo 'Calculated number of iterations: 7'
echo 'Validation bpb: 1.23'
exit 0
""",
    )
    assert result.returncode != 0
    assert (results_dir / "results.csv").read_text().splitlines() == [CSV_HEADER]
    assert "no complete_*.json checkpoint sentinel" in result.stdout
    assert "Scaling Runs Complete" not in result.stdout


def test_success_with_valid_metrics_and_sentinel_is_recorded(tmp_path):
    result, results_dir = run_fake_runner(
        tmp_path,
        """#!/bin/bash
for argument in "$@"; do
    case "$argument" in
        --model_tag=*) tag=${argument#--model_tag=} ;;
    esac
done
mkdir -p "$NANOCHAT_BASE_DIR/base_checkpoints/$tag"
touch "$NANOCHAT_BASE_DIR/base_checkpoints/$tag/complete_000007.json"
echo 'Number of parameters: 1,000'
echo 'Calculated number of iterations: 7'
echo 'Validation bpb: 1.23'
exit 0
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (results_dir / "results.csv").read_text().splitlines()
    assert lines[0] == CSV_HEADER and len(lines) == 2
    fields = lines[1].split(",")
    assert fields[:5] == ["8", "1000", "7", "1835008", "1.23"]
    assert fields[5].isdigit()
    assert "Scaling Runs Complete" in result.stdout
