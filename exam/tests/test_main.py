"""Starting the real app in its own process, as on exam day, and finding the library."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from studybuddy_exam.locate import LibraryNotFound, find_library, index_path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_json(url: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.load(response)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def test_the_running_app_reports_no_ai_and_loopback_only(library: Path) -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "studybuddy_exam", str(library), "--no-browser", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        status = _get_json(f"http://127.0.0.1:{port}/api/status")
        assert status["guard"] == {
            "installed": True, "ai_modules_loaded": [], "network": "loopback only", "listening": f"127.0.0.1:{port}",
        }
        results = _get_json(f"http://127.0.0.1:{port}/api/search?q=krebs")
        assert results["count"] > 0
    finally:
        proc.terminate()
        _, err = proc.communicate(timeout=20)
    assert "AI libraries loaded: none; network: loopback only" in err


def test_a_missing_index_is_explained(tmp_path: Path) -> None:
    (tmp_path / "library.toml").write_text("")
    proc = subprocess.run([sys.executable, "-m", "studybuddy_exam", str(tmp_path), "--no-browser"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1 and "studybuddy build-exam" in proc.stderr


def test_find_library_order(library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDYBUDDY_LIBRARY", raising=False)
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(tmp_path / "config"))
    assert find_library(library) == library.resolve()
    monkeypatch.setenv("STUDYBUDDY_LIBRARY", str(library))
    assert find_library(cwd=tmp_path) == library.resolve()
    monkeypatch.delenv("STUDYBUDDY_LIBRARY")
    nested = library / "courses" / "x"
    nested.mkdir(parents=True, exist_ok=True)
    assert find_library(cwd=nested) == library.resolve()
    with pytest.raises(LibraryNotFound, match="no library found"):
        find_library(cwd=tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "default-library").write_text(str(library) + "\n")
    assert find_library(cwd=tmp_path) == library.resolve()
    with pytest.raises(LibraryNotFound, match="not a StudyBuddy library"):
        find_library(tmp_path)
    assert index_path(library) == library / "exam" / "exam.sqlite"
    assert os.environ.get("STUDYBUDDY_LIBRARY") is None
