"""Study notes for transcribed videos: ``studybuddy notes``, and after ``add`` and ``session``.

Every video whose notes are missing or out of date gets notes from Claude
(``enrich/notes.py``), through the ``[enrich]`` backend and model. The checked notes are
kept in the build database with the run that made them (prompt version, model, input hash,
tokens), and written as Markdown to the course's notes folder:

    <video>.notes.md    one lecture: overview, sections, key terms, review questions
    index.md            the course: every lecture's overview, announcements, an A-Z glossary

A video's notes are up to date while its transcript and the notes prompts are unchanged.
A lecture file is only written when its notes are (re)made or the file is missing, so your
edits to it are kept until ``--redo``; ``index.md`` is rewritten every time.

A transcript outside any library (``lecture.json``) gets ``lecture.notes.md`` next to it.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from studybuddy.enrich.backends import EnrichmentBackend, EnrichmentError, UsageLimitError, make_backend, model_id
from studybuddy.enrich.notes import (
    SCHEMA_VERSION,
    VERSION,
    IndexEntry,
    NotesResult,
    StudyNotes,
    input_sha,
    render_index,
    render_lecture,
    write_notes,
)
from studybuddy.enrich.passages import Passage, passages_from_transcript
from studybuddy.ingest.add import natural_key
from studybuddy.library import SESSION_DIR, Course, EnrichSettings, Library, now_iso
from studybuddy.output import write_atomic
from studybuddy.transcription import NoProgress, Progress, format_clock

log = logging.getLogger(__name__)


class NotesInterrupted(KeyboardInterrupt):
    """Ctrl-C while notes were being written; ``note`` says how to finish them."""

    def __init__(self, note: str) -> None:
        super().__init__()
        self.note = note


@dataclass
class NotesReport:
    written: list[Path] = field(default_factory=list)  # lecture files, in the order written
    indexes: list[Path] = field(default_factory=list)
    up_to_date: int = 0
    failed: list[str] = field(default_factory=list)  # video titles
    limit: UsageLimitError | None = None  # set when a usage limit stopped the run

    @property
    def ok(self) -> bool:
        return not self.failed and self.limit is None


@dataclass(frozen=True)
class _Video:
    id: int
    title: str
    transcript: str  # stored_path
    week: int | None
    date: str | None
    duration_s: float | None


def update_notes(
    library: Library,
    courses: Sequence[Course],
    *,
    source_ids: Sequence[int] | None = None,
    redo: bool = False,
    progress: Progress | None = None,
) -> NotesReport:
    """Write the missing or out-of-date notes of ``courses``' videos (or only ``source_ids``).

    Raises EnrichmentError when the backend can't be used at all (e.g. no ``claude``
    command); a video that fails is logged and reported, and a usage limit stops the run.
    """
    progress = progress or NoProgress()
    settings = library.settings.enrich
    report = NotesReport()
    backend: EnrichmentBackend | None = None
    for course in courses:
        for video in _videos(library, course):
            if report.limit is not None or (source_ids is not None and video.id not in source_ids):
                continue
            passages = _passages(library, video.id)
            if not passages:
                continue
            path = library.notes_file(course.slug, video.transcript)
            sha = input_sha(video.title, passages)
            stored = _stored(library, video.id)
            if stored and not redo and stored["input_sha"] == sha and stored["prompt_version"] == VERSION:
                report.up_to_date += 1
                if not path.is_file():
                    _write_lecture(path, video, StudyNotes.model_validate_json(stored["data"]),
                                   stored["model_id"], stored["created_at"])
                    report.written.append(path)
                continue
            backend = backend or make_backend(settings.backend)
            try:
                result = _write_notes(passages, video, backend, settings, progress,
                                      retry=retry_command(library, [course]))
            except UsageLimitError as exc:
                report.limit = exc
                continue
            except EnrichmentError as exc:
                log.error("%s: study notes not written: %s", video.title, exc)
                report.failed.append(video.title)
                continue
            created = now_iso()
            _store(library, video.id, sha, result, settings, created)
            _write_lecture(path, video, result.notes, result.model, created)
            report.written.append(path)
        if index := _write_index(library, course):
            report.indexes.append(index)
    return report


def notes_for_transcript(transcript: Path, *, settings: EnrichSettings | None = None,
                         progress: Progress | None = None) -> Path:
    """Write ``<stem>.notes.md`` next to a transcript JSON (``studybuddy transcribe --formats json``)."""
    settings = settings or EnrichSettings()
    try:
        data = json.loads(transcript.read_text(encoding="utf-8"))
        segments = data["segments"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EnrichmentError(f"{transcript} is not a studybuddy transcript (.json): {exc}") from None
    source = data.get("source") or transcript.name
    video = _Video(0, Path(source).stem, str(transcript), None, None, data.get("duration_s"))
    passages = passages_from_transcript({"segments": segments})
    retry = f"studybuddy notes {shlex.quote(str(transcript))}"
    try:
        result = _write_notes(passages, video, make_backend(settings.backend), settings,
                              progress or NoProgress(), retry=retry)
    except EnrichmentError as exc:
        raise EnrichmentError(f"study notes not written: {exc}{_resets(exc)}; retry with: {retry}") from None
    path = transcript.with_name(f"{transcript.stem}.notes.md")
    _write_lecture(path, video, result.notes, result.model, now_iso(), source=source)
    return path


def retry_command(library: Library, courses: Sequence[Course]) -> str:
    """The command that writes the notes still missing in ``courses``."""
    command = "studybuddy notes"
    if len(courses) == 1:
        command += f" {courses[0].slug}"
    if library.root.name == SESSION_DIR:  # never the default library; name its folder
        command += f" --library {shlex.quote(str(library.root.parent))}"
    return command


def videos_without_notes(library: Library, course: Course) -> int:
    with library.db() as db:
        return db.execute(
            """SELECT COUNT(*) FROM source s
                WHERE s.course_id = ? AND s.kind = 'video'
                  AND NOT EXISTS (SELECT 1 FROM notes n WHERE n.source_id = s.id)
                  AND EXISTS (SELECT 1 FROM passage p WHERE p.source_id = s.id)""",
            (course.id,),
        ).fetchone()[0]


def describe_limit(exc: UsageLimitError) -> str:
    return f"Claude's usage limit was reached{_resets(exc) or f' ({exc})'}"


# --------------------------------------------------------------------------- internals


def _write_notes(passages: list[Passage], video: _Video, backend: EnrichmentBackend, settings: EnrichSettings,
                 progress: Progress, *, retry: str) -> NotesResult:
    model = model_id(settings.backend, settings.model)
    length = f"{format_clock(video.duration_s)}, " if video.duration_s else ""
    log.info("Writing study notes for %s (%s%s) with %s", video.title, length, _plural(len(passages), "passage"), model)
    try:
        with progress.stage("notes"):
            result = write_notes(passages, video.title, backend=backend, model=model, effort=settings.effort,
                                 on_progress=progress.update)
    except KeyboardInterrupt:
        raise NotesInterrupted(
            f"Interrupted while writing the study notes for {video.title}; notes already written are kept. "
            f"Write the rest with: {retry}"
        ) from None
    notes = result.notes
    questions = sum(len(s.review_questions) for s in notes.sections)
    terms = sum(len(s.key_terms) for s in notes.sections)
    allowance = result.meta.get("five_hour_utilization")
    log.info(
        "%s: %s, %s, %s%s", video.title, _plural(len(notes.sections), "section"), _plural(terms, "key term"),
        _plural(questions, "review question"),
        f"; Claude 5-hour allowance {allowance:.0%} used" if allowance is not None else "",
    )
    if result.quotes_dropped:
        log.info("%s: %d key-term quotes were not found in the transcript and were left out",
                 video.title, result.quotes_dropped)
    return result


def _write_lecture(path: Path, video: _Video, notes: StudyNotes, model: str, created: str,
                   source: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_lecture(notes, source=source or video.title, duration_s=video.duration_s, model=model,
                          written=_local_date(created))
    write_atomic(path, text)


def _write_index(library: Library, course: Course) -> Path | None:
    entries = []
    for video in _videos(library, course):
        stored = _stored(library, video.id)
        if stored is None:
            continue
        link = library.notes_file(course.slug, video.transcript).name
        entries.append(IndexEntry(video.title, link, StudyNotes.model_validate_json(stored["data"]),
                                  video.week, video.date, video.duration_s))
    if not entries:
        return None
    path = library.notes_dir(course.slug) / "index.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, render_index(course.title, entries))
    return path


def _videos(library: Library, course: Course) -> list[_Video]:
    """The course's videos in lecture order: by week, then date, then title (Part 2 before Part 10)."""
    with library.db() as db:
        rows = db.execute(
            """SELECT id, title, stored_path, week, date, duration_s FROM source
                WHERE course_id = ? AND kind = 'video' AND stored_path IS NOT NULL""",
            (course.id,),
        ).fetchall()
    videos = [_Video(r["id"], r["title"], r["stored_path"], r["week"], r["date"], r["duration_s"]) for r in rows]
    return sorted(videos, key=lambda v: (v.week is None, v.week or 0, v.date or "", natural_key(v.title)))


def _passages(library: Library, source_id: int) -> list[Passage]:
    with library.db() as db:
        rows = db.execute(
            "SELECT t_start_ms, t_end_ms, text FROM passage WHERE source_id = ? ORDER BY ord", (source_id,)
        ).fetchall()
    return [Passage(i, (r["t_start_ms"] or 0) / 1000, (r["t_end_ms"] or 0) / 1000, r["text"])
            for i, r in enumerate(rows, start=1)]


def _stored(library: Library, source_id: int) -> dict[str, Any] | None:
    with library.db() as db:
        row = db.execute(
            """SELECT n.data, r.input_sha, r.prompt_version, r.model_id, r.created_at
                 FROM notes n JOIN run r ON r.id = n.run_id WHERE n.source_id = ?""",
            (source_id,),
        ).fetchone()
    return dict(row) if row else None


def _store(library: Library, source_id: int, sha: str, result: NotesResult, settings: EnrichSettings,
           created: str) -> None:
    u = result.usage
    params = {"backend": settings.backend, "effort": settings.effort, "calls": result.calls}
    with library.db() as db:
        cur = db.execute(
            """INSERT INTO run(task, prompt_version, model_id, params_json, schema_ver, input_sha, created_at,
                               input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, cost_usd)
               VALUES ('notes', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (VERSION, result.model, json.dumps(params), SCHEMA_VERSION, sha, created, u.input_tokens,
             u.output_tokens, u.cache_creation_input_tokens, u.cache_read_input_tokens, result.cost_usd),
        )
        db.execute(
            """INSERT INTO notes(source_id, run_id, data) VALUES (?, ?, ?)
               ON CONFLICT(source_id) DO UPDATE SET run_id = excluded.run_id, data = excluded.data""",
            (source_id, cur.lastrowid, result.notes.model_dump_json()),
        )


def _resets(exc: Exception) -> str:
    resets_at = getattr(exc, "resets_at", None)
    if not resets_at:
        return ""
    when = datetime.fromtimestamp(resets_at)
    same_day = when.date() == datetime.now().date()
    return f" (it resets at {when:%H:%M})" if same_day else f" (it resets {when:%a %d %b %H:%M})"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _local_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().date().isoformat()
    except ValueError:
        return iso[:10]
