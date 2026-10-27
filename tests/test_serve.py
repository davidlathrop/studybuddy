"""``studybuddy serve``: every library in ~/StudyBuddy brought up to date and searched from one page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_exam_export import add_source

from studybuddy import cli, exam_export, session
from studybuddy.library import Library


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("STUDYBUDDY_LIBRARY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
def exec_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    app = tmp_path / "bin" / "studybuddy-exam"
    app.parent.mkdir()
    app.write_text("#!/bin/sh\n")
    calls: list[Any] = []
    monkeypatch.setattr(session, "exam_app", lambda: app)
    monkeypatch.setattr("studybuddy.commands.os.execv", lambda path, argv: calls.append(argv))
    return calls


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """~/StudyBuddy with a term library (never built), a session deeper down (built), and an empty library."""
    folder = tmp_path / "home" / "StudyBuddy"
    term = Library.init(folder / "Fall 2026", make_default=False)
    term.add_course("bio101", "Biology 101")
    add_source(term, "bio101", "video", "Lecture 4", ["Mitochondria make ATP."], view="courses/bio101/videos/l4.mp4")
    sitting, course = session.open_session(folder / "Term 5" / "Async 3")
    add_source(sitting, course.slug, "pdf", "Handout", ["Glycolysis happens in the cytoplasm."])
    exam_export.build_exam(sitting)
    Library.init(folder / "Empty", make_default=False).add_course("x", "X")
    return folder


def test_serve_brings_every_index_up_to_date_then_opens_the_exam_app(
    home: Path, exec_calls: list[Any], capsys: pytest.CaptureFixture
) -> None:
    assert cli.main(["serve", "--no-browser", "--port", "9001"]) == 0  # ~/StudyBuddy by default
    err = capsys.readouterr().err
    assert "Fall 2026: exam index updated: 1 file, 1 passage" in err
    assert "Term 5/Async 3: exam index up to date" in err
    assert "Empty: nothing to search yet" in err
    assert not (home / "Empty" / "exam" / "exam.sqlite").exists()
    assert exam_export.exam_index_state(Library(home / "Fall 2026")).endswith("up to date")
    assert exec_calls == [[str(exec_calls[0][0]), str(home), "--port", "9001", "--no-browser"]]


def test_an_outdated_index_is_rebuilt(home: Path, exec_calls: list[Any], capsys: pytest.CaptureFixture) -> None:
    sitting = Library(home / "Term 5" / "Async 3" / ".studybuddy")
    [course] = sitting.courses()
    add_source(sitting, course.slug, "pdf", "Late handout", ["Added after the index was built."],
               added_at="2099-01-01T00:00:00+00:00")
    assert cli.main(["serve", str(home)]) == 0
    assert "Term 5/Async 3: exam index updated: 2 files, 2 passages" in capsys.readouterr().err
    assert exec_calls[0][1:] == [str(home)]


def test_a_library_folder_serves_just_that_library(home: Path, exec_calls: list[Any],
                                                   capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["serve", str(home / "Fall 2026")]) == 0
    assert "Fall 2026: exam index updated" in capsys.readouterr().err
    assert exec_calls[0][1:] == [str(home / "Fall 2026")]


def test_nothing_to_serve_is_explained(tmp_path: Path, exec_calls: list[Any], capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["serve"]) == 1
    assert f"{tmp_path / 'home' / 'StudyBuddy'} is not a folder" in capsys.readouterr().err
    folder = tmp_path / "home" / "StudyBuddy"
    folder.mkdir(parents=True)
    assert cli.main(["serve"]) == 1
    assert "no StudyBuddy libraries in" in capsys.readouterr().err
    Library.init(folder / "Fall 2026", make_default=False)
    assert cli.main(["serve"]) == 1
    assert "nothing to search in" in capsys.readouterr().err
    assert exec_calls == []


def test_a_missing_exam_app_is_explained(home: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(session, "exam_app", lambda: None)
    assert cli.main(["serve"]) == 1
    assert "the exam app is not installed" in capsys.readouterr().err
