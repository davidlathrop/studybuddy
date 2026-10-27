"""``studybuddy build-exam``: export the library into the exam index (plan §7.1).

Deterministic and AI-free: it copies courses, sources and passages from
``build/library.sqlite`` into ``exam/exam.sqlite`` in the format the exam app reads
(``studybuddy_exam.index_format``) and builds the search indexes there. Study notes that
match their transcripts go along, for browsing, with their passage numbers turned into the
index's passage ids and times.

``studybuddy serve`` uses :func:`libraries_in` and :func:`refresh_exam_index` to bring
every library in a folder up to date before the exam app searches them together.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from studybuddy import __version__, study_notes
from studybuddy.enrich.notes import StudyNotes
from studybuddy.library import Library, LibraryError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExamIndex:
    path: Path
    sha256: str
    courses: int
    sources: int
    passages: int
    notes: int  # lectures with study notes
    without_view: int  # sources the exam app can show as text only
    size_bytes: int


def _exam_module(name: str):  # noqa: ANN202 - the module, imported lazily with a helpful error
    try:
        return importlib.import_module(f"studybuddy_exam.{name}")
    except ImportError:
        raise LibraryError(
            "the exam app is not installed in this environment; run: .venv/bin/pip install -e ./exam"
        ) from None


def _format():  # noqa: ANN202
    return _exam_module("index_format")


def libraries_in(folder: Path) -> list[Path]:
    """The libraries in ``folder``, found as the exam app finds them (``studybuddy_exam.locate``)."""
    locate = _exam_module("locate")
    try:
        return locate.libraries_in(folder)
    except locate.LibraryNotFound as exc:
        raise LibraryError(str(exc)) from None


def index_path(library: Library) -> Path:
    return library.root / "exam" / _format().FILE_NAME


def build_exam(library: Library) -> ExamIndex:
    fmt = _format()
    with library.db() as db:
        courses = [fmt.CourseRow(r["id"], r["slug"], r["title"])
                   for r in db.execute("SELECT id, slug, title FROM course ORDER BY slug")]
        sources = [
            fmt.SourceRow(r["id"], r["course_id"], r["kind"], r["title"], r["week"], r["date"], r["view_path"],
                          r["duration_s"], r["page_count"])
            for r in db.execute(
                "SELECT * FROM source WHERE EXISTS (SELECT 1 FROM passage p WHERE p.source_id = source.id) ORDER BY id"
            )
        ]
        passages = [
            fmt.PassageRow(r["id"], r["source_id"], r["ord"], r["t_start_ms"], r["t_end_ms"], r["slide_no"],
                           r["page_no"], r["heading_path"], r["origin"], r["text"])
            for r in db.execute("SELECT * FROM passage ORDER BY source_id, ord")
        ]
        notes_run = db.execute("SELECT MAX(run_id) FROM notes").fetchone()[0] or 0  # the newest notes
    passage_ids: dict[int, list[int]] = {}
    for p in passages:
        passage_ids.setdefault(p.source_id, []).append(p.id)
    lectures = [lecture for course in library.courses() for lecture in study_notes.current_notes(library, course)]
    notes = [
        fmt.NotesRow(lecture.source_id, position, lecture.notes.title, lecture.model, lecture.written_at,
                     json.dumps(_exam_notes(lecture.notes, passage_ids.get(lecture.source_id, []))))
        for position, lecture in enumerate(lectures, start=1)
    ]
    missing = [s for s in sources if s.view_path and not library.absolute(s.view_path).is_file()]
    for s in missing:
        log.warning("%s: its file %s is missing; it will be searchable but not viewable", s.title, s.view_path)
    sources = [replace(s, view_path=None) if s in missing else s for s in sources]

    path = fmt.build(
        index_path(library),
        courses=courses,
        sources=sources,
        passages=passages,
        notes=notes,
        meta={"library_name": library.settings.name or library.root.name, "studybuddy_version": __version__,
              "notes_run": notes_run},
    )
    return ExamIndex(
        path=path,
        sha256=_sha256(path),
        courses=len(courses),
        sources=len(sources),
        passages=len(passages),
        notes=len(notes),
        without_view=sum(1 for s in sources if not s.view_path),
        size_bytes=path.stat().st_size,
    )


def _exam_notes(notes: StudyNotes, passage_ids: list[int]) -> dict[str, Any]:
    """The notes as the exam index keeps them (``index_format.NotesRow``): every passage
    number (P1, P2, ... of the transcript) becomes the exam passage's id and start time."""

    def at(number: int) -> dict[str, int | None]:
        if 1 <= number <= min(len(passage_ids), len(notes.times)):
            return {"passage_id": passage_ids[number - 1], "t_ms": round(notes.times[number - 1][0] * 1000)}
        return {"passage_id": None, "t_ms": None}

    return {
        "overview": notes.overview,
        "announcements": [{"text": a.text, **at(a.passage_id)} for a in notes.announcements],
        "sections": [
            {
                "title": s.title,
                "summary": s.summary,
                **at(s.first_passage_id),
                "end_ms": round(notes.times[s.last_passage_id - 1][1] * 1000),
                "key_points": [{"text": p.point, "details": [d for d in p.details if d.strip()], **at(p.passage_id)}
                               for p in s.key_points],
                "key_terms": [{"term": t.term, "definition": t.definition, "quote": t.quote, **at(t.passage_id)}
                              for t in s.key_terms],
                "examples": [{"text": x.text, **at(x.passage_id)} for x in s.examples],
                "emphasis": [{"text": x.text, **at(x.passage_id)} for x in s.emphasis],
                "review_questions": [{"question": q.question, "answer": q.answer, **at(q.passage_id)}
                                     for q in s.review_questions],
            }
            for s in notes.sections
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def refresh_exam_index(library: Library) -> tuple[str, ExamIndex | None]:
    """Build the exam index if it is missing or out of date. Returns ("current", None),
    ("built", the index), or ("empty", None) for a library with no index and nothing to search."""
    state = _index_state(library)[0]
    if state == "current":
        return "current", None
    if state == "none":
        with library.db() as db:
            if db.execute("SELECT 1 FROM passage LIMIT 1").fetchone() is None:
                return "empty", None
    return "built", build_exam(library)


def exam_index_state(library: Library) -> str:
    """One line for ``studybuddy status``: built when, and whether it is out of date."""
    return _index_state(library)[1]


def exam_index_outdated(library: Library) -> bool:
    """Is there an exam index that is missing changes to the library, or that the exam app can't read?"""
    return _index_state(library)[0] == "outdated"


def _index_state(library: Library) -> tuple[str, str]:
    """("none" | "outdated" | "current", the line for ``studybuddy status``)."""
    try:
        path = index_path(library)
    except LibraryError:
        return "none", "Exam index: the exam app is not installed (pip install -e ./exam)"
    if not path.is_file():
        return "none", "Exam index: not built yet (studybuddy build-exam)"
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.DatabaseError:
        return "outdated", "Exam index: unreadable; rebuild it (studybuddy build-exam)"
    finally:
        conn.close()
    with library.db() as db:
        latest = db.execute("SELECT MAX(added_at) FROM source").fetchone()[0]
        passages = db.execute("SELECT COUNT(*) FROM passage").fetchone()[0]
        notes_run = db.execute("SELECT MAX(run_id) FROM notes").fetchone()[0] or 0
    built = meta.get("built_at", "")
    when = _local_time(built)
    if meta.get("format_version") != str(_format().FORMAT_VERSION):
        return "outdated", f"Exam index: built {when} for an older exam app; rebuild it (studybuddy build-exam)"
    if (latest and latest > built) or str(passages) != meta.get("passages") or str(notes_run) != meta.get("notes_run"):
        return "outdated", f"Exam index: built {when}, out of date (studybuddy build-exam)"
    lectures = int(meta.get("notes", 0))
    notes_text = f", study notes for {lectures} lecture{'' if lectures == 1 else 's'}" if lectures else ""
    return "current", f"Exam index: built {when}, {meta.get('passages')} passages{notes_text}, up to date"


def _local_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso or "?"
