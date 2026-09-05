"""bq-1944: the scaling runner must not report success when tee fails but the
trainer succeeds.

With `set -o pipefail`, `python ... | tee log` reports a nonzero pipeline
status when EITHER stage fails, so the runner's `if pipeline; then ... else
...` branch selection already detects a tee failure correctly. The bug is
inside that else branch: it reads only PIPESTATUS[0] (the trainer) and
ignores PIPESTATUS[1] (tee), so a trainer-succeeded/tee-failed run is logged
as "FAILED ... trainer exited with status 0" and the script still exits 0.

Reproduced the same way the finding did: tee's target is a symlink to
/dev/full, so tee genuinely fails with ENOSPC while the (faked) trainer
genuinely exits 0.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "run_scaling_3090.sh"
STATUS_HEADER = "depth,status,detail,exit_code"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _prepare_sandbox(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    # Trainer that genuinely succeeds -- PIPESTATUS[0] will be 0.
    write_executable(
        fake_bin / "python",
        "#!/bin/bash\n"
        "echo 'Number of parameters: 1,000'\n"
        "echo 'Calculated number of iterations: 10'\n"
        "echo 'Validation bpb: 1.2345'\n"
        "exit 0\n",
    )
    write_executable(fake_bin / "column", "#!/bin/bash\nexit 0\n")
    return sandbox, results_dir, fake_bin


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="requires /dev/full")
def test_tee_failure_with_successful_trainer_is_reported_as_failed(tmp_path):
    sandbox, results_dir, fake_bin = _prepare_sandbox(tmp_path)

    # tee's write target for depth 8 is a symlink to /dev/full, so the REAL
    # system tee genuinely fails with ENOSPC while the trainer genuinely exits 0.
    results_dir.mkdir(parents=True)
    (results_dir / "scaling_d8_train.log").symlink_to("/dev/full")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["NANOCHAT_BASE_DIR"] = str(sandbox / "nanochat-base")
    result = subprocess.run(
        ["bash", str(sandbox / RUNNER.name)],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, (
        "a trainer-succeeded/tee-failed run must exit nonzero, not report success:\n"
        + result.stdout + result.stderr
    )
    status_lines = (results_dir / "run_status.csv").read_text().splitlines()
    assert status_lines[0] == STATUS_HEADER
    assert len(status_lines) == 2, status_lines
    depth, status, detail, exit_code = status_lines[1].split(",")
    assert status == "FAILED"
    assert exit_code != "0", (
        f"tee failure must not be recorded with exit_code=0 (that's the trainer's "
        f"code, not tee's): {status_lines[1]}"
    )
    assert "tee" in detail.lower(), (
        f"failure detail must identify tee (not the trainer, which succeeded): {detail}"
    )
    # No observation should have been appended for this depth.
    assert not (results_dir / "results.csv").exists() or (
        results_dir / "results.csv"
    ).read_text().strip() == "depth,num_params,num_iterations,tokens_trained,val_bpb,train_time_sec"
    assert "Scaling Runs Complete" not in result.stdout
