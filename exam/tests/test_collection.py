"""A folder of libraries searched as one: finding them, combining their indexes, serving their files."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import COURSES, LECTURE_5, LECTURE_5_PASSAGE, NOTES, PASSAGES, SOURCES, _p, _section, _term
from starlette.testclient import TestClient
from test_main import _free_port, _get_json

from studybuddy_exam import notes
from studybuddy_exam.collection import combine
from studybuddy_exam.index_format import CourseRow, NotesRow, SourceRow, build, file_sha256, open_index
from studybuddy_exam.locate import LibraryNotFound, libraries_in
from studybuddy_exam.search import Filters, Searcher
from studybuddy_exam.server import create_app

WORKSHOP_SOURCE = SourceRow(1, 1, "video", "Part 1", None, None, "courses/bio101/videos/p1.exam.mp4", 60.0, None)
WORKSHOP_PASSAGES = [
    _p(1, 1, 1, "Welcome to the workshop.", t_start_ms=0, t_end_ms=30_000, origin="speech"),
    _p(2, 1, 2, "Glycolysis splits glucose in the cytoplasm.", t_start_ms=30_000, t_end_ms=60_000, origin="speech"),
]
WORKSHOP_NOTES = NotesRow(1, 1, "Glycolysis", "claude-opus-5-5", "2026-10-26T10:00:00+00:00", json.dumps({
    "overview": "Where glycolysis happens.",
    "announcements": [{"text": "Bring a calculator.", "passage_id": 1, "t_ms": 0}],
    "sections": [_section("Glycolysis", 2, 30_000, key_terms=[_term("Glycolysis", 2, 30_000)])],
}))


def make_library(root: Path, name: str, courses=(), sources=(), passages=(), notes_=(), *, index: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "library.toml").write_text(f'[library]\nname = "{name}"\n')
    if index:
        build(root / "exam" / "exam.sqlite", courses=courses, sources=sources, passages=passages, notes=notes_,
              meta={"library_name": name})
    for source in sources:
        if source.view_path:
            (root / source.view_path).parent.mkdir(parents=True, exist_ok=True)
            (root / source.view_path).write_bytes(f"{name}: {source.title}".encode())
    return root


@pytest.fixture(scope="module")
def home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """~/StudyBuddy: a term library, a session folder deeper down, and two libraries with nothing to add."""
    home = tmp_path_factory.mktemp("StudyBuddy")
    make_library(home / "Fall 2026", "Fall 2026", COURSES, [*SOURCES, LECTURE_5], [*PASSAGES, LECTURE_5_PASSAGE], NOTES)
    make_library(home / "Term 5" / "Workshop" / ".studybuddy", "Workshop", [CourseRow(1, "bio101", "Workshop")],
                 [WORKSHOP_SOURCE], WORKSHOP_PASSAGES, [WORKSHOP_NOTES])
    make_library(home / "Empty", "Empty", [CourseRow(1, "x", "X")])
    make_library(home / "New", "New", index=False)
    return home


@pytest.fixture(scope="module")
def combined(home: Path, tmp_path_factory: pytest.TempPathFactory):
    return combine(home, libraries_in(home), tmp_path_factory.mktemp("combined") / "exam.sqlite")


@pytest.fixture(scope="module")
def client(home: Path, combined) -> TestClient:
    conn, info = open_index(combined.path, root=home)
    with TestClient(create_app(conn, info, parts=combined.parts), base_url="http://127.0.0.1:8765") as c:
        yield c


def test_libraries_in_a_folder(home: Path, tmp_path: Path) -> None:
    found = [p.relative_to(home).as_posix() for p in libraries_in(home)]
    assert found == ["Empty", "Fall 2026", "New", "Term 5/Workshop/.studybuddy"]
    assert libraries_in(home / "Fall 2026") == [(home / "Fall 2026").resolve()]  # a library: just itself
    assert libraries_in(home / "Term 5" / "Workshop") == [(home / "Term 5" / "Workshop" / ".studybuddy").resolve()]
    with pytest.raises(LibraryNotFound, match="is not a folder"):
        libraries_in(home / "missing")


def test_libraries_are_found_in_natural_order_through_links_once(tmp_path: Path) -> None:
    folder = tmp_path / "StudyBuddy"
    for name in ("Week 10", "Week 2", "Fall 2026"):
        make_library(folder / "Term" / name, name, index=False)
    make_library(folder / "Fall 2026" / "courses" / "bio101" / "Inner", "Inside a library", index=False)
    make_library(folder / ".Trash" / "Old", "Old", index=False)
    make_library(tmp_path / "OneDrive" / "Async 3" / ".studybuddy", "Async 3", index=False)
    (folder / "Fall 2026" / "library.toml").write_text("")
    (folder / "OneDrive").symlink_to(tmp_path / "OneDrive")
    (folder / "Term" / "zz-again").symlink_to(folder / "Term" / "Week 2")
    found = [p.relative_to(folder).as_posix() for p in libraries_in(folder)]
    assert found == ["Fall 2026", "OneDrive/Async 3/.studybuddy", "Term/Fall 2026", "Term/Week 2", "Term/Week 10"]
    (tmp_path / "nothing here").mkdir()
    assert libraries_in(tmp_path / "nothing here") == []


def test_combining_renumbers_everything_and_keeps_each_course_with_its_library(combined, home: Path) -> None:
    assert [p.name for p in combined.parts] == ["Fall 2026", "Workshop"]
    assert combined.left_out == [
        "Empty: nothing to search yet",
        "New: no exam index yet",
    ]
    workshop = combined.parts[1]
    assert workshop.root == home / "Term 5" / "Workshop" / ".studybuddy" and workshop.passages == 2
    assert workshop.sha256 == file_sha256(workshop.index) and workshop.built_at

    conn, info = open_index(combined.path, root=home)
    assert info.library_root == home and info.meta["library_name"] == str(home)
    counts = [info.meta[key] for key in ("libraries", "courses", "passages", "notes")]
    assert counts == ["2", "3", "14", "3"]
    libraries = conn.execute("SELECT id, name, root FROM library ORDER BY id").fetchall()
    assert [tuple(r) for r in libraries] == [(1, "Fall 2026", str(home / "Fall 2026")),
                                             (2, "Workshop", str(workshop.root))]
    searcher = Searcher(conn)
    courses = {c["slug"]: (c["title"], c["library"]) for c in searcher.courses()}
    assert courses == {"bio101": ("Biology 101", "Fall 2026"), "econ201": ("Applied Microeconomics", "Fall 2026"),
                       "bio101-2": ("Workshop", "Workshop")}  # two courses called bio101
    hit = searcher.search("glycolysis", Filters(course="bio101-2")).hits[0]
    assert (hit.title, hit.course_title, hit.t_start_ms) == ("Part 1", "Workshop", 30_000)
    assert {h.course_slug for h in searcher.search("krebs").hits} == {"bio101"}


def test_study_notes_point_at_the_renumbered_passages(combined, home: Path) -> None:
    conn, _ = open_index(combined.path, root=home)
    [part_1] = [r["id"] for r in conn.execute("SELECT id FROM source WHERE title = 'Part 1'")]
    lecture = notes.lecture_notes(conn, part_1)
    passage_id = lecture["sections"][0]["key_terms"][0]["passage_id"]
    text = conn.execute("SELECT text FROM passage WHERE id = ?", (passage_id,)).fetchone()[0]
    assert text == "Glycolysis splits glucose in the cytoplasm." and passage_id != 2
    assert lecture["announcements"][0]["passage_id"] == passage_id - 1
    krebs = next(t for t in notes.course_notes(conn, "bio101")["glossary"] if t["term"] == "Krebs cycle")
    first = conn.execute("SELECT text FROM passage WHERE id = ?", (krebs["places"][0]["passage_id"],)).fetchone()[0]
    assert first.startswith("Today we look at how mitochondria")
    assert (lecture["previous_id"], lecture["next_id"]) == (None, None)  # alone in its course
    assert [x["source_id"] for x in notes.course_notes(conn, "bio101-2")["lectures"]] == [part_1]


def test_files_come_from_each_library(client: TestClient) -> None:
    courses = client.get("/api/courses").json()
    assert [(c["slug"], c["library"]) for c in courses] == [
        ("bio101", "Fall 2026"), ("econ201", "Fall 2026"), ("bio101-2", "Workshop")]
    workshop = client.get("/api/search", params={"q": "glycolysis", "course": "bio101-2"}).json()["results"][0]
    assert client.get(workshop["file_url"]).content == b"Workshop: Part 1"
    lecture = client.get("/api/search", params={"q": "krebs", "kind": "video"}).json()["results"][0]
    assert client.get(lecture["file_url"]).content == b"Fall 2026: Lecture 4: Cell energy"


def test_status_lists_every_library_with_its_fingerprint(client: TestClient, combined, home: Path) -> None:
    status = client.get("/api/status").json()
    assert status["library"] == {"name": str(home), "root": str(home)}
    assert [(p["name"], p["sha256"], p["passages"]) for p in status["libraries"]] == [
        (p.name, file_sha256(p.index), p.passages) for p in combined.parts]
    assert status["index"]["passages"] == 14


def test_a_single_library_has_no_parts(index_file: Path) -> None:
    conn, info = open_index(index_file)
    with TestClient(create_app(conn, info), base_url="http://127.0.0.1:8765") as c:
        assert c.get("/api/status").json()["libraries"] == []
        assert {course["library"] for course in c.get("/api/courses").json()} == {"Test Library"}


def test_libraries_with_the_same_name_are_told_apart_by_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "StudyBuddy"
    for course in ("Investments", "Accounting"):
        make_library(home / course / "Async 3" / ".studybuddy", "Async 3", [CourseRow(1, "async-3", "Async 3")],
                     [WORKSHOP_SOURCE], WORKSHOP_PASSAGES)
    combined = combine(home, libraries_in(home), tmp_path / "exam.sqlite")
    assert [p.name for p in combined.parts] == ["Accounting/Async 3", "Investments/Async 3"]
    _, info = open_index(combined.path, root=home)
    assert info.meta["library_name"] == "~/StudyBuddy"


def test_an_index_for_an_older_exam_app_is_left_out(tmp_path: Path) -> None:
    home = tmp_path / "StudyBuddy"
    make_library(home / "A", "A", [CourseRow(1, "a", "A")], [WORKSHOP_SOURCE], WORKSHOP_PASSAGES)
    old = make_library(home / "B", "B", [CourseRow(1, "b", "B")], [WORKSHOP_SOURCE], WORKSHOP_PASSAGES)
    with sqlite3.connect(old / "exam" / "exam.sqlite") as conn:
        conn.execute("UPDATE meta SET value = '2' WHERE key = 'format_version'")
    combined = combine(home, libraries_in(home), tmp_path / "exam.sqlite")
    assert [p.name for p in combined.parts] == ["A"]
    assert combined.left_out[0].startswith("B: ") and "has index format 2" in combined.left_out[0]


def test_the_app_started_on_a_folder_searches_every_library(home: Path, tmp_path: Path) -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "studybuddy_exam", str(home), "--no-browser", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    try:
        status = _get_json(f"http://127.0.0.1:{port}/api/status")
        assert [p["name"] for p in status["libraries"]] == ["Fall 2026", "Workshop"]
        assert status["guard"]["ai_modules_loaded"] == []
        assert _get_json(f"http://127.0.0.1:{port}/api/search?q=glycolysis")["count"] > 0
        assert list(tmp_path.iterdir()) == []  # the combined index is read from a deleted file
    finally:
        proc.terminate()
        _, err = proc.communicate(timeout=20)
    assert "left out Empty: nothing to search yet" in err
    assert "2 libraries: Fall 2026, Workshop" in err


def test_a_folder_without_libraries(tmp_path: Path) -> None:
    proc = subprocess.run([sys.executable, "-m", "studybuddy_exam", str(tmp_path), "--no-browser"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1 and "no StudyBuddy libraries in" in proc.stderr
