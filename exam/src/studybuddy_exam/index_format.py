"""The exam index (``exam.sqlite``): its layout, how it is built, and how it is opened.

The exam app reads this file and nothing else from the library besides the files it shows,
so this module owns the format. The builder (``studybuddy build-exam``) hands it plain rows
and calls :func:`build`; the exam app calls :func:`open_index`. Nothing here uses AI.

Layout (format 3):

* ``meta``: key/value facts about the build (format version, time, library name, counts).
* ``library``: where the files are. Every course belongs to a library, and a source's
  ``view_path`` is relative to its library's ``root``. An index built by the builder holds
  one library with an empty root: the folder the index is in (``<library>/exam/exam.sqlite``).
  The exam app makes one with several libraries when it is started on a folder of them
  (``collection.py``); their roots are absolute.
* ``course``, ``source``, ``passage``: what can be found and where it lives. ``passage``
  has the original ``text`` plus three columns for enrichment (``aliases``, ``questions``,
  ``context``), empty until enrichment exists (plan milestone E3).
* ``passage_fts``: SQLite FTS5 over those four columns, Porter stemming, ranked with BM25.
* ``passage_tri``: FTS5 trigram index over ``text``, for partial words.
* ``vocab``: every indexed word with its document frequency and Metaphone code, for
  "did you mean" suggestions.
* ``notes``: the study notes of lecture videos (plan §6.8), written by AI before the exam
  and only shown, never searched. ``data`` is JSON (see :class:`NotesRow`) in which every
  point gives the exam passage it comes from and that passage's start time.

Format 2 added ``notes``, format 3 ``library``.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jellyfish

FORMAT_VERSION = 3
FILE_NAME = "exam.sqlite"
MIN_VOCAB_LENGTH = 3

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE library(
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    root TEXT NOT NULL            -- the folder view_path is relative to; '' for <library> of <library>/exam/
);
CREATE TABLE course(
    id         INTEGER PRIMARY KEY,
    slug       TEXT NOT NULL UNIQUE,
    title      TEXT NOT NULL,
    library_id INTEGER NOT NULL REFERENCES library(id)
);
CREATE TABLE source(
    id         INTEGER PRIMARY KEY,
    course_id  INTEGER NOT NULL REFERENCES course(id),
    kind       TEXT NOT NULL,     -- video | pptx | pdf | docx
    title      TEXT NOT NULL,
    week       INTEGER,
    date       TEXT,
    view_path  TEXT,              -- relative to its library's root: exam-copy video or PDF
    duration_s REAL,
    page_count INTEGER
);
CREATE TABLE passage(
    id           INTEGER PRIMARY KEY,
    source_id    INTEGER NOT NULL REFERENCES source(id),
    ord          INTEGER NOT NULL,
    t_start_ms   INTEGER,
    t_end_ms     INTEGER,
    slide_no     INTEGER,
    page_no      INTEGER,
    heading_path TEXT,
    origin       TEXT NOT NULL,   -- speech | text | notes | ocr
    text         TEXT NOT NULL,
    aliases      TEXT NOT NULL DEFAULT '',
    questions    TEXT NOT NULL DEFAULT '',
    context      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX passage_source ON passage(source_id, ord);
CREATE VIRTUAL TABLE passage_fts USING fts5(
    text, aliases, questions, context,
    content='passage', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE passage_tri USING fts5(
    text, content='passage', content_rowid='id', tokenize='trigram'
);
CREATE TABLE vocab(
    term      TEXT PRIMARY KEY,
    doc_freq  INTEGER NOT NULL,
    metaphone TEXT NOT NULL
) WITHOUT ROWID;
CREATE INDEX vocab_metaphone ON vocab(metaphone);
CREATE TABLE notes(
    source_id  INTEGER PRIMARY KEY REFERENCES source(id),
    position   INTEGER NOT NULL,  -- lecture order within the course
    title      TEXT NOT NULL,     -- what the lecture teaches, from the notes
    model      TEXT NOT NULL,
    written_at TEXT NOT NULL,
    data       TEXT NOT NULL      -- JSON, see NotesRow
);
"""


class ExamIndexError(Exception):
    """A missing, unreadable or incompatible exam index."""


@dataclass(frozen=True)
class LibraryRow:
    id: int
    name: str
    root: str = ""


@dataclass(frozen=True)
class CourseRow:
    id: int
    slug: str
    title: str
    library_id: int = 1


@dataclass(frozen=True)
class SourceRow:
    id: int
    course_id: int
    kind: str
    title: str
    week: int | None
    date: str | None
    view_path: str | None
    duration_s: float | None
    page_count: int | None


@dataclass(frozen=True)
class PassageRow:
    id: int
    source_id: int
    ord: int
    t_start_ms: int | None
    t_end_ms: int | None
    slide_no: int | None
    page_no: int | None
    heading_path: str | None
    origin: str
    text: str
    aliases: str = ""
    questions: str = ""
    context: str = ""


@dataclass(frozen=True)
class NotesRow:
    """One lecture's study notes. ``data`` is JSON::

        {"overview": str,
         "announcements": [{"text", AT}],
         "sections": [{"title", "summary", "end_ms", AT,
                       "key_points": [{"text", "details": [str], AT}],
                       "key_terms": [{"term", "definition", "quote", AT}],
                       "examples": [{"text", AT}], "emphasis": [{"text", AT}],
                       "review_questions": [{"question", "answer", AT}]}]}

    where AT is ``"passage_id"`` (a passage of this source) and ``"t_ms"`` (its start), both
    null when the notes point at no passage. A key term's ``quote`` is words checked to be
    in the transcript, or empty.
    """

    source_id: int
    position: int
    title: str
    model: str
    written_at: str
    data: str


def build(
    path: Path,
    *,
    courses: Iterable[CourseRow],
    sources: Iterable[SourceRow],
    passages: Iterable[PassageRow],
    notes: Iterable[NotesRow] = (),
    libraries: Iterable[LibraryRow] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a complete index to ``path`` (atomically: a temporary file replaces it at the end).

    Without ``libraries``, the courses belong to one library, named ``meta["library_name"]``,
    whose root is the folder holding the index's ``exam`` folder.
    """
    path = Path(path)
    if libraries is None:
        libraries = [LibraryRow(1, str((meta or {}).get("library_name", "")))]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.building")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(SCHEMA)
        _insert(conn, "library", LibraryRow, libraries)
        _insert(conn, "course", CourseRow, courses)
        _insert(conn, "source", SourceRow, sources)
        _insert(conn, "passage", PassageRow, passages)
        _insert(conn, "notes", NotesRow, notes)
        tables = {"libraries": "library", "courses": "course", "sources": "source", "passages": "passage",
                  "notes": "notes"}
        counts = {key: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for key, table in tables.items()}
        info = {
            "format_version": FORMAT_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **counts,
            **(meta or {}),
        }
        conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", [(k, str(v)) for k, v in info.items()])
        for table in ("passage_fts", "passage_tri"):
            conn.execute(f"INSERT INTO {table}({table}) VALUES ('rebuild')")
            conn.execute(f"INSERT INTO {table}({table}) VALUES ('optimize')")
        _build_vocab(conn)
        conn.commit()
        conn.execute("VACUUM")
    except BaseException:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(tmp, path)
    return path


def _insert(conn: sqlite3.Connection, table: str, row_type: type, rows: Iterable[Any]) -> None:
    names = [f.name for f in fields(row_type)]
    sql = f"INSERT INTO {table}({', '.join(names)}) VALUES ({', '.join('?' * len(names))})"
    conn.executemany(sql, ([getattr(r, n) for n in names] for r in rows))


_WORDLIKE = re.compile(r"^[^\W\d_][\w'-]*$")


def _build_vocab(conn: sqlite3.Connection) -> None:
    """Unstemmed words of all passages with their document counts and Metaphone codes."""
    # An unstemmed index just for counting words; dropped again below. It lives in the main
    # schema because an external-content FTS table must sit beside its content table.
    conn.executescript(
        """
        CREATE VIRTUAL TABLE vocab_src USING fts5(
            text, content='passage', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
        INSERT INTO vocab_src(vocab_src) VALUES ('rebuild');
        CREATE VIRTUAL TABLE temp.vocab_rows USING fts5vocab(main, vocab_src, 'row');
        """
    )
    rows = conn.execute("SELECT term, doc FROM temp.vocab_rows").fetchall()
    conn.executemany(
        "INSERT INTO vocab(term, doc_freq, metaphone) VALUES (?, ?, ?)",
        (
            (term, doc, jellyfish.metaphone(term))
            for term, doc in rows
            if len(term) >= MIN_VOCAB_LENGTH and _WORDLIKE.match(term)
        ),
    )
    conn.executescript("DROP TABLE temp.vocab_rows; DROP TABLE vocab_src;")


@dataclass(frozen=True)
class IndexInfo:
    path: Path
    meta: dict[str, str]
    root: Path | None = None  # an index of several libraries: the folder they are in

    @property
    def library_root(self) -> Path:
        return self.root or self.path.parent.parent  # <library>/exam/exam.sqlite


def open_index(path: Path, *, root: Path | None = None) -> tuple[sqlite3.Connection, IndexInfo]:
    """Open ``path`` read-only and check its format version. ``root`` is the folder of an
    index made from several libraries, which lives elsewhere."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ExamIndexError(f"no exam index at {path}; run `studybuddy build-exam` first")
    uri = f"{path.as_uri()}?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
    except sqlite3.DatabaseError as exc:
        raise ExamIndexError(f"{path} is not a readable exam index ({exc})") from None
    version = int(meta.get("format_version", "0"))
    if version != FORMAT_VERSION:
        conn.close()
        raise ExamIndexError(
            f"{path} has index format {version}, this app reads format {FORMAT_VERSION}; "
            "rebuild it with `studybuddy build-exam` from the same studybuddy version"
        )
    return conn, IndexInfo(path, meta, root)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()
