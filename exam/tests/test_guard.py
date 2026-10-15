"""The no-AI, no-network guard. Each check runs in a fresh Python process, because an audit
hook can never be removed again once installed."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from studybuddy_exam import guard


def run_guarded(code: str) -> subprocess.CompletedProcess[str]:
    script = "from studybuddy_exam import guard\nguard.install()\n" + textwrap.dedent(code)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize(
    "module",
    ["anthropic", "boto3", "botocore.session", "openai", "torch", "mlx.core", "mlx_audio", "transformers",
     "ocrmac", "studybuddy", "studybuddy.enrich", "google.genai", "sklearn.linear_model"],
)
def test_blocked_imports_are_refused(module: str) -> None:
    proc = run_guarded(
        f"""
        try:
            import {module}
        except PermissionError as e:
            print("blocked:", e)
        except ImportError as e:
            print("not installed:", e)
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("blocked:"), proc.stdout  # refused before Python even looks for it


def test_allowed_imports_work() -> None:
    proc = run_guarded("import json, sqlite3, starlette, uvicorn, rapidfuzz, jellyfish, studybuddy_exam.server\nprint('ok')")
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


@pytest.mark.parametrize(
    "code",
    [
        "import socket; socket.create_connection(('1.1.1.1', 53), timeout=1)",
        "import socket; socket.getaddrinfo('example.com', 443)",
        "import socket; socket.gethostbyname('example.com')",
        "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b'x', ('8.8.8.8', 53))",
        "import socket; s = socket.socket(); s.bind(('0.0.0.0', 0))",
        "import socket; s = socket.socket(); s.bind(('', 0))",
        "import urllib.request; urllib.request.urlopen('https://example.com', timeout=1)",
    ],
)
def test_network_beyond_loopback_is_refused(code: str) -> None:
    proc = run_guarded(code)
    assert proc.returncode != 0 and "exam mode" in proc.stderr, proc.stderr


def test_loopback_is_allowed() -> None:
    proc = run_guarded(
        """
        import socket
        server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen()
        port = server.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port), timeout=2)
        socket.getaddrinfo("localhost", port)
        v6 = socket.socket(socket.AF_INET6); v6.bind(("::1", 0))
        print("ok")
        """
    )
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


def test_install_refuses_when_ai_is_already_loaded() -> None:
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, types; sys.modules['openai'] = types.ModuleType('openai')\n"
         "from studybuddy_exam import guard\nguard.install()"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0 and "loaded before the guard" in proc.stderr


def test_status_helpers() -> None:
    assert guard.is_blocked("boto3") and guard.is_blocked("boto3.session") and guard.is_blocked("google.genai.types")
    assert not guard.is_blocked("google") and not guard.is_blocked("botocorefake") and not guard.is_blocked("studybuddy_exam")
    for host in ("127.0.0.1", "127.8.9.1", "::1", "[::1]", "localhost", None, b"127.0.0.1"):
        assert guard.is_loopback(host), host
    for host in ("10.0.0.1", "1.1.1.1", "example.com", "0.0.0.0", "", "::"):
        assert not guard.is_loopback(host), host
