"""``studybuddy build-exam``: exporting the library into the exam app's index."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from studybuddy import cli, exam_export
from studybuddy.library import Library
from studybuddy_exam.index_format import open_index
from studybuddy_exam.search import Searcher


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("STUDYBUDDY_LIBRARY", raising=False)


def add_source(lib: Library, slug: str, kind: str, title: str, texts: list[str], *, view: str | None = None,
               week: int | None = None, added_at: str = "2026-10-01T00:00:00+00:00") -> int:
    course = lib.course(slug)
    if view:
        path = lib.absolute(view)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    with lib.db() as db:
        sid = db.execute(
            "INSERT INTO source(course_id, kind, title, week, original_path, view_path, sha256, extractor, added_at) "
            "VALUES (?, ?, ?, ?, '/x', ?, ?, 'test', ?)",
            (course.id, kind, title, week, view, hashlib.sha256(title.encode()).hexdigest(), added_at),
        ).lastrowid
        for i, text in enumerate(texts, start=1):
            location: dict[str, Any] = (
                {"t_start_ms": (i - 1) * 60_000, "t_end_ms": i * 60_000} if kind == "video" else {"page_no": i}
            )
            db.execute(
                "INSERT INTO passage(source_id, ord, t_start_ms, t_end_ms, page_no, origin, text, text_sha) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'h')",
                (sid, i, location.get("t_start_ms"), location.get("t_end_ms"), location.get("page_no"),
                 "speech" if kind == "video" else "text", text),
            )
    return sid


@pytest.fixture
def library(tmp_path: Path) -> Library:
    lib = Library.init(tmp_path / "lib", name="Fall 2026")
    lib.add_course("bio101", "Biology 101")
    add_source(lib, "bio101", "video", "Lecture 4", ["Mitochondria make ATP.", "The Krebs cycle runs in the matrix."],
               view="courses/bio101/videos/l4.exam.mp4", week=4)
    add_source(lib, "bio101", "pdf", "Reading", ["Photosynthesis in chloroplasts."],
               view="courses/bio101/docs/reading.pdf", week=4)
    return lib


def test_build_exam_exports_courses_sources_and_passages(library: Library) -> None:
    built = exam_export.build_exam(library)
    assert built.path == library.root / "exam" / "exam.sqlite"
    assert (built.courses, built.sources, built.passages, built.without_view) == (1, 2, 3, 0)
    assert built.sha256 == hashlib.sha256(built.path.read_bytes()).hexdigest()
    conn, info = open_index(built.path)
    assert info.meta["library_name"] == "Fall 2026" and info.library_root == library.root
    hit = Searcher(conn).search("krebs").hits[0]
    assert (hit.title, hit.t_start_ms, hit.week, hit.has_view) == ("Lecture 4", 60_000, 4, True)


def test_sources_without_passages_are_left_out_and_missing_files_become_text_only(
    library: Library, caplog: pytest.LogCaptureFixture
) -> None:
    add_source(library, "bio101", "pptx", "Empty deck", [])
    add_source(library, "bio101", "docx", "Notes", ["Glycolysis basics."], view="courses/bio101/docs/notes.docx.pdf")
    library.absolute("courses/bio101/docs/notes.docx.pdf").unlink()
    built = exam_export.build_exam(library)
    assert (built.sources, built.without_view) == (3, 1)
    assert "Notes: its file courses/bio101/docs/notes.docx.pdf is missing" in caplog.text
    conn, _ = open_index(built.path)
    assert conn.execute("SELECT view_path FROM source WHERE title = 'Notes'").fetchone()[0] is None


def test_rebuilding_replaces_the_index(library: Library) -> None:
    first = exam_export.build_exam(library)
    add_source(library, "bio101", "pdf", "More", ["Ribosomes build proteins."], view="courses/bio101/docs/more.pdf")
    second = exam_export.build_exam(library)
    assert second.passages == first.passages + 1 and second.path == first.path
    assert not list(second.path.parent.glob(".*building*"))


def test_status_line(library: Library) -> None:
    assert "not built yet" in exam_export.exam_index_state(library)
    exam_export.build_exam(library)
    assert "up to date" in exam_export.exam_index_state(library)
    add_source(library, "bio101", "pdf", "Late", ["Added after the build."], added_at="2099-01-01T00:00:00+00:00")
    assert "out of date" in exam_export.exam_index_state(library)


def test_build_exam_command(library: Library, capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["build-exam", "--library", str(library.root)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(library.root / "exam" / "exam.sqlite")
    assert "1 course, 2 files, 3 passages" in captured.err and "studybuddy-exam" in captured.err
    assert cli.main(["status", "--library", str(library.root)]) == 0
    assert "Exam index: built" in capsys.readouterr().out
