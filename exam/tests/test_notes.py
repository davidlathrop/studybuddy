"""Study notes in the exam index: a course's lectures and glossary, one lecture's notes, the API."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from studybuddy_exam import notes
from studybuddy_exam.index_format import open_index
from studybuddy_exam.server import create_app


@pytest.fixture(scope="module")
def conn(notes_index: Path):
    connection, _ = open_index(notes_index)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def client(notes_index: Path) -> TestClient:
    connection, info = open_index(notes_index)
    with TestClient(create_app(connection, info), base_url="http://127.0.0.1:8765") as c:
        yield c


def test_course_notes_list_lectures_in_order_with_their_announcements(conn) -> None:
    data = notes.course_notes(conn, "bio101")
    assert data["course"] == {"slug": "bio101", "title": "Biology 101"} and data["videos"] == 2
    first, second = data["lectures"]
    assert (first["source_id"], first["notes_title"], first["title"]) == (10, "Cellular respiration",
                                                                          "Lecture 4: Cell energy")
    assert first["overview"] == "How cells turn glucose into ATP." and first["week"] == 4
    assert first["has_view"] and first["file_url"] == "/files/10" and first["model"] == "claude-opus-5-5"
    assert second["source_id"] == 14 and "sections" not in second  # the course page doesn't need them
    assert data["announcements"] == [
        {"text": "The midterm is on October 20.", "passage_id": 5, "t_ms": 240_000, "source_id": 10}
    ]


def test_the_glossary_merges_terms_across_sections_and_sorts_them(conn) -> None:
    glossary = notes.course_notes(conn, "bio101")["glossary"]
    assert [(t["term"], t["letter"]) for t in glossary] == [
        ("3-phosphoglycerate", "#"), ("ATP", "A"), ("Krebs cycle", "K"), ("Stomata", "S"),
    ]
    krebs = glossary[2]
    assert krebs["definition"] == "The meaning of Krebs cycle."  # the first one
    assert krebs["places"] == [{"source_id": 10, "passage_id": 1, "t_ms": 0},
                               {"source_id": 10, "passage_id": 4, "t_ms": 180_000}]
    assert glossary[1]["places"] == [{"source_id": 10, "passage_id": None, "t_ms": None}]  # points nowhere


def test_a_course_without_notes(conn) -> None:
    data = notes.course_notes(conn, "econ201")
    assert data["lectures"] == data["announcements"] == data["glossary"] == [] and data["videos"] == 0
    assert notes.course_notes(conn, "nope") is None


def test_lecture_notes_with_the_lectures_around_it(conn) -> None:
    first = notes.lecture_notes(conn, 10)
    assert (first["previous_id"], first["next_id"]) == (None, 14)
    assert first["course_slug"] == "bio101" and first["course_title"] == "Biology 101"
    assert [s["title"] for s in first["sections"]] == ["The Krebs cycle", "More on the cycle"]
    assert first["sections"][0]["key_terms"][0]["quote"] == "the Krebs cycle"
    assert first["announcements"][0]["t_ms"] == 240_000
    second = notes.lecture_notes(conn, 14)
    assert (second["previous_id"], second["next_id"]) == (10, None)
    assert notes.lecture_notes(conn, 11) is None  # slides have no notes
    assert notes.lecture_notes(conn, 999) is None


def test_notes_api(client: TestClient) -> None:
    course = client.get("/api/notes", params={"course": "bio101"}).json()
    assert [lecture["source_id"] for lecture in course["lectures"]] == [10, 14]
    assert client.get("/api/notes", params={"course": "nope"}).status_code == 404
    assert client.get("/api/notes").status_code == 404
    lecture = client.get("/api/notes/14").json()
    assert lecture["notes_title"] == "Photosynthesis" and lecture["previous_id"] == 10
    missing = client.get("/api/notes/11")
    assert missing.status_code == 404 and missing.json() == {"error": "no study notes for this file"}
    assert client.get("/api/notes/14").headers["cache-control"] == "no-store"


def test_courses_and_status_count_the_notes(client: TestClient) -> None:
    courses = {c["slug"]: c for c in client.get("/api/courses").json()}
    assert courses["bio101"]["notes"] == 2 and courses["econ201"]["notes"] == 0
    assert client.get("/api/status").json()["index"]["notes"] == 2


def test_an_index_without_notes(index_file: Path) -> None:
    connection, info = open_index(index_file)
    assert info.meta["notes"] == "0"
    assert notes.course_notes(connection, "bio101")["lectures"] == []
