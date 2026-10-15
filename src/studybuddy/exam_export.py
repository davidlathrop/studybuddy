"""``studybuddy build-exam``: export the library into the exam index (plan §7.1).

Deterministic and AI-free: it copies courses, sources and passages from
``build/library.sqlite`` into ``exam/exam.sqlite`` in the format the exam app reads
(``studybuddy_exam.index_format``) and builds the search indexes there.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from studybuddy import __version__
from studybuddy.library import Library, LibraryError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExamIndex:
    path: Path
    sha256: str
    courses: int
    sources: int
    passages: int
    without_view: int  # sources the exam app can show as text only
    size_bytes: int


def _format():  # noqa: ANN202 - the module, imported lazily with a helpful error
    try:
        from studybuddy_exam import index_format
    except ImportError:
        raise LibraryError(
            "the exam app is not installed in this environment; run: .venv/bin/pip install -e ./exam"
        ) from None
    return index_format


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
    missing = [s for s in sources if s.view_path and not library.absolute(s.view_path).is_file()]
    for s in missing:
        log.warning("%s: its file %s is missing; it will be searchable but not viewable", s.title, s.view_path)
    sources = [replace(s, view_path=None) if s in missing else s for s in sources]

    path = fmt.build(
        index_path(library),
        courses=courses,
        sources=sources,
        passages=passages,
        meta={"library_name": library.settings.name or library.root.name, "studybuddy_version": __version__},
    )
    return ExamIndex(
        path=path,
        sha256=_sha256(path),
        courses=len(courses),
        sources=len(sources),
        passages=len(passages),
        without_view=sum(1 for s in sources if not s.view_path),
        size_bytes=path.stat().st_size,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def exam_index_state(library: Library) -> str:
    """One line for ``studybuddy status``: built when, and whether it is out of date."""
    try:
        path = index_path(library)
    except LibraryError:
        return "Exam index: the exam app is not installed (pip install -e ./exam)"
    if not path.is_file():
        return "Exam index: not built yet (studybuddy build-exam)"
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.DatabaseError:
        return "Exam index: unreadable; rebuild it (studybuddy build-exam)"
    finally:
        conn.close()
    with library.db() as db:
        latest = db.execute("SELECT MAX(added_at) FROM source").fetchone()[0]
        passages = db.execute("SELECT COUNT(*) FROM passage").fetchone()[0]
    built = meta.get("built_at", "")
    stale = (latest and latest > built) or str(passages) != meta.get("passages")
    when = _local_time(built)
    if stale:
        return f"Exam index: built {when}, out of date (studybuddy build-exam)"
    return f"Exam index: built {when}, {meta.get('passages')} passages, up to date"


def _local_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso or "?"
