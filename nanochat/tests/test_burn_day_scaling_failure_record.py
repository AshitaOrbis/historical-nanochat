"""A failed scaling run must be durable in the per-run status summary."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "run_scaling_3090.sh"
STATUS_HEADER = "depth,status,detail,exit_code"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_failed_training_step_exits_nonzero_and_records_failed_status(tmp_path):
    sandbox = tmp_path / "runner"
    sandbox.mkdir()
    (sandbox / "nanochat").mkdir()
    activate = sandbox / ".venv" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("")

    results_dir = sandbox / "results"
    runner_text, result_replacements = re.subn(
        r"^RESULTS_DIR=.*$",
        f'RESULTS_DIR="{results_dir}"',
        RUNNER.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    runner_text, depth_replacements = re.subn(
        r"^DEPTHS=.*$", "DEPTHS=(8)", runner_text, count=1, flags=re.MULTILINE
    )
    assert (result_replacements, depth_replacements) == (1, 1)
    runner = sandbox / RUNNER.name
    runner.write_text(runner_text)

    fake_bin = sandbox / "fake-bin"
    fake_bin.mkdir()
    write_executable(
        fake_bin / "python",
        "#!/bin/bash\necho 'trainer fixture failed'\nexit 23\n",
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
    assert (results_dir / "run_status.csv").read_text().splitlines() == [
        STATUS_HEADER,
        "8,FAILED,trainer_exit_23,23",
    ]
    assert "Scaling Runs Complete" not in result.stdout
