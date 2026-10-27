from __future__ import annotations

import json
from pathlib import Path

import pytest

from studybuddy_exam.index_format import CourseRow, NotesRow, PassageRow, SourceRow, build

COURSES = [CourseRow(1, "bio101", "Biology 101"), CourseRow(2, "econ201", "Applied Microeconomics")]
SOURCES = [
    SourceRow(10, 1, "video", "Lecture 4: Cell energy", 4, "2026-09-15", "courses/bio101/videos/l4.exam.mp4", 3600.0, None),
    SourceRow(11, 1, "pptx", "Week 4 slides", 4, None, "courses/bio101/docs/w4.pptx.pdf", None, 12),
    SourceRow(12, 2, "docx", "My notes", 3, None, None, None, None),  # conversion failed: text only
    SourceRow(13, 2, "pdf", "Reading", 6, None, "courses/econ201/docs/reading.pdf", None, 3),
]


def _p(id_: int, source: int, ord_: int, text: str, **location) -> PassageRow:
    defaults = dict(t_start_ms=None, t_end_ms=None, slide_no=None, page_no=None, heading_path=None, origin="text")
    defaults.update(location)
    return PassageRow(id_, source, ord_, text=text, **defaults)


PASSAGES = [
    _p(1, 10, 1, "Today we look at how mitochondria produce ATP through the Krebs cycle.",
       t_start_ms=0, t_end_ms=60_000, origin="speech"),
    _p(2, 10, 2, "The electron transport chain pumps protons across the inner membrane.",
       t_start_ms=60_000, t_end_ms=120_000, origin="speech"),
    _p(3, 10, 3, "Photosynthesis happens in chloroplasts, not mitochondria.",
       t_start_ms=120_000, t_end_ms=180_000, origin="speech"),
    _p(4, 10, 4, "More on the Krebs cycle: it runs in the matrix.", t_start_ms=180_000, t_end_ms=240_000,
       origin="speech"),
    _p(5, 10, 5, "Krebs cycle again: two turns per glucose.", t_start_ms=240_000, t_end_ms=300_000, origin="speech"),
    _p(6, 11, 1, "Krebs Cycle\nOccurs in the matrix", slide_no=6),
    _p(7, 11, 2, "Remember: the Krebs cycle yields 2 ATP per glucose.", slide_no=6, origin="notes"),
    _p(8, 11, 3, "Stomata regulate gas exchange", slide_no=9, origin="ocr"),
    _p(9, 12, 1, "Elasticity\n\nElastic goods have substitutes; purchases can be delayed.",
       page_no=1, heading_path="Elasticity"),
    _p(10, 12, 2, "Tariffs raise costs, which firms pass on: cost-push inflation.", page_no=1,
       heading_path="Inflation > Cost-push"),
    _p(11, 13, 1, "Price elasticity of demand measures how quantity demanded responds to price.", page_no=2,
       origin="ocr"),
]


@pytest.fixture(scope="session")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal library folder: library.toml, the exam index, and the files it points at."""
    root = tmp_path_factory.mktemp("library")
    (root / "library.toml").write_text('[library]\nname = "Test Library"\n')
    build(root / "exam" / "exam.sqlite", courses=COURSES, sources=SOURCES, passages=PASSAGES,
          meta={"library_name": "Test Library"})
    for source in SOURCES:
        if source.view_path:
            path = root / source.view_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(range(256)) * 40)  # 10 KB, enough for Range tests
    return root


@pytest.fixture(scope="session")
def index_file(library: Path) -> Path:
    return library / "exam" / "exam.sqlite"


# Study notes for the Lecture 4 video and a second lecture, as the builder exports them.
LECTURE_5 = SourceRow(14, 1, "video", "Lecture 5: Photosynthesis", 5, None, "courses/bio101/videos/l5.exam.mp4",
                      600.0, None)
LECTURE_5_PASSAGE = _p(12, 14, 1, "Stomata open to let carbon dioxide in.", t_start_ms=0, t_end_ms=60_000,
                       origin="speech")


def _section(title: str, passage_id: int | None, t_ms: int | None, **lists: list) -> dict:
    empty = {key: [] for key in ("key_points", "key_terms", "examples", "emphasis", "review_questions")}
    return {"title": title, "summary": f"About {title.lower()}.", "passage_id": passage_id, "t_ms": t_ms,
            "end_ms": (t_ms or 0) + 120_000, **empty, **lists}


def _term(term: str, passage_id: int | None, t_ms: int | None, quote: str = "") -> dict:
    return {"term": term, "definition": f"The meaning of {term}.", "quote": quote, "passage_id": passage_id,
            "t_ms": t_ms}


LECTURE_4_NOTES = {
    "overview": "How cells turn glucose into ATP.",
    "announcements": [{"text": "The midterm is on October 20.", "passage_id": 5, "t_ms": 240_000}],
    "sections": [
        _section(
            "The Krebs cycle", 1, 0,
            key_points=[{"text": "Mitochondria make ATP.", "details": ["Through the Krebs cycle."], "passage_id": 1,
                         "t_ms": 0}],
            key_terms=[_term("Krebs cycle", 1, 0, quote="the Krebs cycle")],
            review_questions=[{"question": "Where does the Krebs cycle run?", "answer": "In the matrix.",
                               "passage_id": 4, "t_ms": 180_000}],
        ),
        _section("More on the cycle", 4, 180_000,
                 key_terms=[_term("krebs Cycle!", 4, 180_000), _term("ATP", None, None)]),
    ],
}
LECTURE_5_NOTES = {
    "overview": "How leaves take in carbon dioxide.",
    "announcements": [],
    "sections": [_section("Stomata", 12, 0, key_terms=[_term("Stomata", 12, 0), _term("3-phosphoglycerate", 12, 0)])],
}
NOTES = [
    NotesRow(10, 1, "Cellular respiration", "claude-opus-5-5", "2026-10-22T10:00:00+00:00",
             json.dumps(LECTURE_4_NOTES)),
    NotesRow(14, 2, "Photosynthesis", "claude-opus-5-5", "2026-10-22T10:05:00+00:00", json.dumps(LECTURE_5_NOTES)),
]


@pytest.fixture(scope="session")
def notes_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An exam index with study notes for two lectures of bio101 and none for econ201."""
    path = tmp_path_factory.mktemp("notes-library") / "exam" / "exam.sqlite"
    return build(path, courses=COURSES, sources=[*SOURCES, LECTURE_5], passages=[*PASSAGES, LECTURE_5_PASSAGE],
                 notes=NOTES, meta={"library_name": "Notes Library"})
