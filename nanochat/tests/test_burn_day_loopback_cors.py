"""CORS grants must be exact loopback origins, never wildcard or arbitrary sites."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
NANOCHAT_ROOT = REPO / "nanochat"


def test_chat_web_cors_headers_allow_only_loopback_origins():
    probe = r'''
import asyncio
import sys
sys.argv = ["chat_web.py", "--device-type", "cpu", "--port", "8000"]
from scripts import chat_web

async def preflight(origin):
    messages = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "OPTIONS",
        "scheme": "http",
        "path": "/chat/completions",
        "raw_path": b"/chat/completions",
        "query_string": b"",
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 49152),
        "root_path": "",
        "headers": [
            (b"origin", origin.encode("ascii")),
            (b"access-control-request-method", b"POST"),
            (b"access-control-request-headers", b"content-type"),
        ],
    }
    await chat_web.app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    return {key.decode().lower(): value.decode() for key, value in start["headers"]}

for origin in ("http://127.0.0.1:8000", "http://localhost:8000"):
    headers = asyncio.run(preflight(origin))
    assert headers.get("access-control-allow-origin") == origin

headers = asyncio.run(preflight("https://attacker.example"))
assert "access-control-allow-origin" not in headers
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(NANOCHAT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=NANOCHAT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
