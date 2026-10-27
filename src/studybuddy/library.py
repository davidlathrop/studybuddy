"""The semester library: a folder with settings, course files and the build database.

    ~/StudyBuddy/fall-2026/
      library.toml                    settings (transcription mode, exam copies, documents, enrichment)
      courses/<slug>/videos/          transcripts (.json/.txt) and 540p exam copies of each video
      courses/<slug>/docs/            copies of slides, PDFs and Word files, plus PDFs for viewing
      courses/<slug>/notes/           study notes written by Claude: one per video, plus index.md
      courses/<slug>/glossary.txt     course terms: transcription hotwords, later aliases
      build/library.sqlite            courses, sources, passages and AI runs (plans/exam-search-plan.md §4)

A command finds its library from ``--library``, then ``$STUDYBUDDY_LIBRARY``, then the nearest
folder above the current directory holding a ``library.toml``, then the default recorded by
``studybuddy library init``. A folder made with ``studybuddy session`` keeps its library in a
``.studybuddy`` subfolder; naming the folder itself is enough. Its study notes go in a
visible ``Study notes`` folder next to the videos.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIG_FILE = "library.toml"
SESSION_DIR = ".studybuddy"  # a one-sitting library inside the folder of its videos
SESSION_NOTES_DIR = "Study notes"  # a sitting's notes, next to its videos
DB_PATH = Path("build") / "library.sqlite"
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")

_CONFIG_TEMPLATE = """\
# StudyBuddy library settings. Edit freely; missing keys use the defaults shown.

[library]
name = {name}

[transcribe]
mode = "accurate"        # accurate (Qwen3-ASR 1.7B) or fast (Parakeet TDT 0.6B v3)
language = ""            # empty: detect it
keep_mp3 = false         # keep the extracted MP3 next to each transcript

[exam_copy]
height = 540             # videos taller than this are scaled down for exam day
video_bitrate = "500k"
audio_bitrate = "64k"

[documents]
ocr = true               # read text in scanned pages and slide images (Apple Vision)
convert = true           # convert PPTX/DOCX to PDF with PowerPoint/Word, for viewing

[enrich]
backend = "claude-cli"   # claude-cli or bedrock
model = "claude-opus-5-5"
effort = "medium"

[notes]
auto = true              # write study notes for new videos after `add` and `session` (with [enrich])
"""

_SCHEMA_V1 = """
CREATE TABLE course(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    slug       TEXT NOT NULL UNIQUE,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE source(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,  -- ids are never reused: enrichment points at them
    course_id     INTEGER NOT NULL REFERENCES course(id),
    kind          TEXT NOT NULL CHECK (kind IN ('video', 'pptx', 'pdf', 'docx')),
    title         TEXT NOT NULL,
    week          INTEGER,
    date          TEXT,
    original_path TEXT NOT NULL,   -- where the file was when it was added
    stored_path   TEXT,            -- library copy (documents) or transcript (videos), relative to the root
    view_path     TEXT,            -- what the exam app opens: exam-copy video or PDF, relative to the root
    duration_s    REAL,
    page_count    INTEGER,         -- slides, or PDF pages
    sha256        TEXT NOT NULL,
    extractor     TEXT NOT NULL,
    warnings      TEXT,            -- newline-separated notes from ingestion
    added_at      TEXT NOT NULL,
    UNIQUE (course_id, sha256)
);
CREATE TABLE passage(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    ord          INTEGER NOT NULL,
    t_start_ms   INTEGER,
    t_end_ms     INTEGER,
    slide_no     INTEGER,
    page_no      INTEGER,
    heading_path TEXT,
    origin       TEXT NOT NULL DEFAULT 'text' CHECK (origin IN ('speech', 'text', 'notes', 'ocr')),
    text         TEXT NOT NULL,
    text_sha     TEXT NOT NULL,
    UNIQUE (source_id, ord)
);
CREATE INDEX passage_source ON passage(source_id);
"""

_SCHEMA_V2 = """
CREATE TABLE run(
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    task               TEXT NOT NULL,   -- what the AI made, e.g. 'notes'
    prompt_version     TEXT NOT NULL,
    model_id           TEXT NOT NULL,
    params_json        TEXT NOT NULL,   -- backend, effort, number of calls
    schema_ver         TEXT NOT NULL,
    input_sha          TEXT NOT NULL,   -- the input as sent: unchanged input is not sent again
    created_at         TEXT NOT NULL,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    cache_write_tokens INTEGER,
    cache_read_tokens  INTEGER,
    cost_usd           REAL             -- billed, or the list-price equivalent on a subscription
);
CREATE TABLE notes(
    source_id INTEGER PRIMARY KEY REFERENCES source(id) ON DELETE CASCADE,
    run_id    INTEGER NOT NULL REFERENCES run(id),
    data      TEXT NOT NULL             -- the checked study notes (enrich.notes.StudyNotes) as JSON
);
"""

MIGRATIONS = [_SCHEMA_V1, _SCHEMA_V2]  # index i upgrades user_version i -> i + 1


class LibraryError(Exception):
    """A missing, invalid or inconsistent library; reported without a traceback."""


@dataclass(frozen=True)
class TranscribeSettings:
    mode: str = "accurate"
    language: str | None = None
    keep_mp3: bool = False


@dataclass(frozen=True)
class ExamCopySettings:
    height: int = 540
    video_bitrate: str = "500k"
    audio_bitrate: str = "64k"


@dataclass(frozen=True)
class DocumentSettings:
    ocr: bool = True
    convert: bool = True


@dataclass(frozen=True)
class EnrichSettings:
    backend: str = "claude-cli"
    model: str = "claude-opus-5-5"
    effort: str = "medium"


@dataclass(frozen=True)
class NotesSettings:
    auto: bool = True


@dataclass(frozen=True)
class Settings:
    name: str = ""
    transcribe: TranscribeSettings = field(default_factory=TranscribeSettings)
    exam_copy: ExamCopySettings = field(default_factory=ExamCopySettings)
    documents: DocumentSettings = field(default_factory=DocumentSettings)
    enrich: EnrichSettings = field(default_factory=EnrichSettings)
    notes: NotesSettings = field(default_factory=NotesSettings)

    @classmethod
    def from_toml(cls, data: dict[str, Any]) -> Settings:
        def section(name: str, kind: type) -> Any:
            values = dict(data.get(name) or {})
            known = {k: v for k, v in values.items() if k in kind.__dataclass_fields__}
            return kind(**known)

        transcribe = section("transcribe", TranscribeSettings)
        if transcribe.mode not in ("accurate", "fast"):
            raise LibraryError(f"{CONFIG_FILE}: [transcribe] mode must be accurate or fast, not {transcribe.mode!r}")
        if not transcribe.language:
            transcribe = TranscribeSettings(transcribe.mode, None, transcribe.keep_mp3)
        return cls(
            name=str((data.get("library") or {}).get("name", "")),
            transcribe=transcribe,
            exam_copy=section("exam_copy", ExamCopySettings),
            documents=section("documents", DocumentSettings),
            enrich=section("enrich", EnrichSettings),
            notes=section("notes", NotesSettings),
        )


@dataclass(frozen=True)
class Course:
    id: int
    slug: str
    title: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def config_dir() -> Path:
    return Path(os.environ.get("STUDYBUDDY_CONFIG_DIR") or Path.home() / ".config" / "studybuddy")


def _default_pointer() -> Path:
    return config_dir() / "default-library"


class Library:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        config = self.root / CONFIG_FILE
        try:
            self.settings = Settings.from_toml(tomllib.loads(config.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise LibraryError(f"{self.root} is not a StudyBuddy library (no {CONFIG_FILE})") from None
        except tomllib.TOMLDecodeError as exc:
            raise LibraryError(f"{config}: {exc}") from None
        with self.db():  # create or upgrade the schema up front
            pass

    # ------------------------------------------------------------------ creating and finding

    @classmethod
    def init(cls, root: Path, *, name: str | None = None, make_default: bool = True) -> Library:
        root = Path(root).expanduser().resolve()
        if (root / CONFIG_FILE).exists():
            raise LibraryError(f"{root} is already a StudyBuddy library")
        if root.exists() and not root.is_dir():
            raise LibraryError(f"{root} exists and is not a folder")
        for sub in ("courses", "build", "exam"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        title = name or root.name
        (root / CONFIG_FILE).write_text(_CONFIG_TEMPLATE.format(name=_toml_string(title)), encoding="utf-8")
        library = cls(root)
        if make_default:
            pointer = _default_pointer()
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(str(library.root) + "\n", encoding="utf-8")
        return library

    @classmethod
    def find(cls, explicit: Path | None = None, *, cwd: Path | None = None) -> Library:
        if explicit is not None:
            return cls(library_root(Path(explicit)))
        if env := os.environ.get("STUDYBUDDY_LIBRARY"):
            return cls(library_root(Path(env)))
        here = Path(cwd or Path.cwd()).resolve()
        for folder in (here, *here.parents):
            if (folder / CONFIG_FILE).is_file():
                return cls(folder)
            if (folder / SESSION_DIR / CONFIG_FILE).is_file():
                return cls(folder / SESSION_DIR)
        pointer = _default_pointer()
        if pointer.is_file() and (recorded := pointer.read_text(encoding="utf-8").strip()):
            return cls(Path(recorded))
        raise LibraryError(
            "no library found: run `studybuddy library init PATH`, pass --library PATH, "
            "or set STUDYBUDDY_LIBRARY"
        )

    # ------------------------------------------------------------------ database

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        """A connection inside a transaction: committed on success, rolled back on error."""
        path = self.root / DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            _migrate(conn)
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ courses

    def course_dir(self, slug: str) -> Path:
        return self.root / "courses" / slug

    def notes_dir(self, slug: str) -> Path:
        """Where a course's study notes go; a session's go next to its videos, where they are seen."""
        if self.root.name == SESSION_DIR:
            return self.root.parent / SESSION_NOTES_DIR
        return self.course_dir(slug) / "notes"

    def notes_file(self, slug: str, transcript: str) -> Path:
        """The study notes of the video whose transcript is stored at ``transcript``."""
        return self.notes_dir(slug) / f"{Path(transcript).stem}.notes.md"

    def add_course(self, slug: str, title: str, *, glossary: Path | None = None) -> Course:
        if not SLUG.match(slug):
            raise LibraryError(
                f"invalid course name {slug!r}: use 1-40 lowercase letters, digits, - or _ (e.g. bio101)"
            )
        if not title.strip():
            raise LibraryError("the course title must not be empty")
        with self.db() as db:
            if db.execute("SELECT 1 FROM course WHERE slug = ?", (slug,)).fetchone():
                raise LibraryError(f"course {slug} already exists")
            cur = db.execute(
                "INSERT INTO course(slug, title, created_at) VALUES (?, ?, ?)", (slug, title.strip(), now_iso())
            )
            course = Course(cur.lastrowid, slug, title.strip())
        folder = self.course_dir(slug)
        for sub in ("videos", "docs"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        target = folder / "glossary.txt"
        if glossary is not None:
            target.write_text(Path(glossary).expanduser().read_text(encoding="utf-8-sig"), encoding="utf-8")
        elif not target.exists():
            target.write_text(
                "# Course terms, one per line: names, jargon, acronyms.\n"
                "# Used as transcription hotwords so they are spelled correctly.\n",
                encoding="utf-8",
            )
        return course

    def course(self, slug: str) -> Course:
        with self.db() as db:
            row = db.execute("SELECT id, slug, title FROM course WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            known = ", ".join(c.slug for c in self.courses()) or "none yet"
            raise LibraryError(f"no course {slug!r} (courses: {known}); add it with `studybuddy course add`")
        return Course(row["id"], row["slug"], row["title"])

    def courses(self) -> list[Course]:
        with self.db() as db:
            rows = db.execute("SELECT id, slug, title FROM course ORDER BY slug").fetchall()
        return [Course(r["id"], r["slug"], r["title"]) for r in rows]

    def glossary_terms(self, slug: str) -> list[str]:
        path = self.course_dir(slug) / "glossary.txt"
        if not path.is_file():
            return []
        lines = (" ".join(line.split()) for line in path.read_text(encoding="utf-8-sig").splitlines())
        return list(dict.fromkeys(t for t in lines if t and not t.startswith("#")))

    # ------------------------------------------------------------------ paths

    def relative(self, path: Path) -> str:
        """``path`` relative to the library root, as stored in the database."""
        return Path(path).resolve().relative_to(self.root).as_posix()

    def absolute(self, relative: str) -> Path:
        return self.root / relative


def library_root(path: Path) -> Path:
    """``path`` itself, or its ``.studybuddy`` subfolder when that is where the library is."""
    path = Path(path).expanduser()
    if not (path / CONFIG_FILE).is_file() and (path / SESSION_DIR / CONFIG_FILE).is_file():
        return path / SESSION_DIR
    return path


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > len(MIGRATIONS):
        raise LibraryError(
            f"this library was created by a newer studybuddy (schema {version}); upgrade studybuddy"
        )
    for target, script in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.executescript(f"BEGIN; {script}; PRAGMA user_version = {target}; COMMIT;")


def _toml_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
