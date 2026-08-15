"""The unauthenticated loopback services must remain same-origin only."""

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative_path",
    ["serve.py", "nanochat/scripts/chat_web.py"],
)
def test_loopback_service_has_no_cross_origin_grant(relative_path):
    source = (REPO / relative_path).read_text()
    assert "CORSMiddleware" not in source
    assert "allow_origins" not in source
    assert "Access-Control-Allow-Origin" not in source
