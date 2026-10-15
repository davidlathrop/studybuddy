"""``studybuddy session FOLDER``: one sitting's files as a self-contained library."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

import doc_samples as ds
from fakes import Models
from studybuddy import cli, session, transcribe
from studybuddy.ingest.add import files_in
from studybuddy.library import SESSION_DIR, Library, LibraryError
from studybuddy_exam.index_format import open_index
from studybuddy_exam.locate import find_library


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("STUDYBUDDY_LIBRARY", raising=False)


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Models:
    fakes = Models()
    monkeypatch.setattr(transcribe, "make_transcriber", fakes)
    return fakes


@pytest.fixture
def sitting(tmp_path: Path, short_mp4: Path) -> Path:
    folder = tmp_path / "Workshop Oct 15"
    folder.mkdir()
    shutil.copy(short_mp4, folder / "Part 10.mp4")
    shutil.copy(short_mp4, folder / "Part 2.mp4")
    with open(folder / "Part 2.mp4", "ab") as f:
        f.write(b"\0" * 8)  # a different file from Part 10
    ds.make_text_pdf(folder / "handout.pdf", [["Glycolysis happens in the cytoplasm."]])
    for junk in (".DS_Store", "~$notes.docx", "readme.txt"):
        (folder / junk).write_bytes(b"")
    return folder


def test_natural_order_and_skipped_files(sitting: Path) -> None:
    (sitting / "sub").mkdir()
    ds.make_text_pdf(sitting / "sub" / "extra 1.pdf", [["More."]])
    (sitting / ".studybuddy" / "x").mkdir(parents=True)
    ds.make_text_pdf(sitting / ".studybuddy" / "x" / "inside.pdf", [["Hidden."]])
    names = [p.relative_to(sitting).as_posix() for p in files_in(sitting)]
    assert names == ["handout.pdf", "Part 2.mp4", "Part 10.mp4"]
    deep = [p.relative_to(sitting).as_posix() for p in files_in(sitting, recursive=True)]
    assert deep == ["handout.pdf", "Part 2.mp4", "Part 10.mp4", "sub/extra 1.pdf"]


@pytest.mark.parametrize(("name", "slug"), [("Workshop Oct 15", "workshop-oct-15"), ("ÉCON 201: Day 3!", "econ-201-day-3"),
                                            ("---", "session"), ("_x", "x"), ("a" * 60, "a" * 40)])
def test_slug_for(name: str, slug: str) -> None:
    assert session.slug_for(name) == slug


def test_session_builds_a_self_contained_library(sitting: Path, models: Models,
                                                 capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["session", str(sitting), "--no-search"]) == 0
    out = capsys.readouterr()
    lines = [line for line in out.out.splitlines() if line.startswith("added")]
    assert [line.split()[2] for line in lines] == ["handout", "Part", "Part"]
    assert "Part 2" in lines[1] and "Part 10" in lines[2]
    assert f"studybuddy-exam '{sitting}'" in out.err

    root = sitting / SESSION_DIR
    library = Library(root)
    [course] = library.courses()
    assert (course.slug, course.title, library.settings.name) == ("workshop-oct-15", "Workshop Oct 15", "Workshop Oct 15")
    assert not (sitting.parent / "config" / "default-library").exists()  # the term library stays the default
    conn, info = open_index(root / "exam" / "exam.sqlite")
    assert info.meta["sources"] == "3"

    # Both apps find the library from the folder itself.
    assert Library.find(sitting).root == root.resolve()
    assert find_library(sitting) == root.resolve()
    assert Library.find(cwd=sitting).root == root.resolve()


def test_running_again_only_adds_new_files(sitting: Path, models: Models, capsys: pytest.CaptureFixture) -> None:
    cli.main(["session", str(sitting), "--no-search", "--title", "Econ workshop"])
    capsys.readouterr()
    ds.make_text_pdf(sitting / "late notes.pdf", [["Added during the break."]])
    assert cli.main(["session", str(sitting), "--no-search"]) == 0
    out = capsys.readouterr().out
    assert out.count("skipped") == 3 and "added    pdf   late notes" in out
    assert len(models.created) == 2  # the two videos were transcribed once each
    [course] = Library(sitting / SESSION_DIR).courses()
    assert course.title == "Econ workshop"


def test_empty_folder_and_not_a_folder(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("x")
    assert cli.main(["session", str(tmp_path / "empty")]) == 1
    assert "no videos or documents" in capsys.readouterr().err
    assert not (tmp_path / "empty" / SESSION_DIR).exists()
    assert cli.main(["session", str(tmp_path / "missing")]) == 1
    assert "is not a folder" in capsys.readouterr().err


def test_opens_the_exam_app_in_place_of_itself(sitting: Path, models: Models, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    app = tmp_path / "bin" / "studybuddy-exam"
    app.parent.mkdir()
    app.write_text("#!/bin/sh\n")
    calls: list[Any] = []
    monkeypatch.setattr(session, "exam_app", lambda: app)
    monkeypatch.setattr("studybuddy.commands.os.execv", lambda path, argv: calls.append((path, argv)))
    assert cli.main(["session", str(sitting), "--port", "9001"]) == 0
    assert calls == [(str(app), [str(app), str(sitting), "--port", "9001"])]


def test_missing_exam_app_is_explained(sitting: Path, models: Models, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(session, "exam_app", lambda: None)
    assert cli.main(["session", str(sitting)]) == 0
    assert "the exam app is not installed" in capsys.readouterr().err


def test_a_session_library_has_one_course(sitting: Path, models: Models) -> None:
    library, _ = session.open_session(sitting)
    library.add_course("other", "Other")
    with pytest.raises(LibraryError, match="exactly one"):
        session.open_session(sitting)


def test_exam_app_prefers_its_own_environment() -> None:
    app = session.exam_app()
    repo_app = Path(session.__file__).resolve().parents[2] / "exam" / ".venv" / "bin" / "studybuddy-exam"
    if repo_app.is_file():
        assert app == repo_app
