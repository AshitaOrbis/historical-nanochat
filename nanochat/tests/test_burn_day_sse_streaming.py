"""bq-1946: the chat UI must not drop streamed tokens when a network read
splits an SSE event.

Runs the ACTUAL JavaScript streaming-parse block extracted verbatim from
nanochat/nanochat/ui.html under Node (see fixtures/sse_streaming_check.mjs)
against controlled chunk boundaries, mirroring the finding's own real
loopback fetch/ReadableStream reproduction (an 11-byte then a 14-byte chunk
splitting a `data: {"token": "Hello"}` event mid-JSON).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
UI_HTML = REPO / "nanochat" / "nanochat" / "ui.html"
CHECK_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "sse_streaming_check.mjs"

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not available on PATH")
def test_sse_streaming_survives_split_events_and_multibyte_boundaries():
    result = subprocess.run(
        [NODE, str(CHECK_SCRIPT), str(UI_HTML)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        "the ui.html streaming SSE parser dropped or corrupted token(s) under a "
        f"split-chunk scenario:\n{result.stdout}\n{result.stderr}"
    )
