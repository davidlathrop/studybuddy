"""A folder of libraries searched as one: ``studybuddy-exam ~/StudyBuddy``.

Each library keeps its own exam index (``studybuddy build-exam``; ``studybuddy serve``
brings them all up to date first). Started on a folder holding several, the exam app
copies their courses, files, passages and study notes into one new index with new ids, so
that one search covers them all. Each course remembers its library, whose folder its files
are served from. The libraries' own indexes are only read. Nothing here uses AI.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from studybuddy_exam.index_format import (
    CourseRow,
    ExamIndexError,
    LibraryRow,
    NotesRow,
    PassageRow,
    SourceRow,
    build,
    file_sha256,
    open_index,
)
from studybuddy_exam.locate import SESSION_DIR, index_path


@dataclass(frozen=True)
class Part:
    """A library as it went into the combined index: where its own index is, and its fingerprint."""

    name: str
    root: Path
    index: Path
    sha256: str
    built_at: str
    passages: int


@dataclass(frozen=True)
class Combined:
    path: Path
    parts: list[Part]
    left_out: list[str]  # "<library>: why", one per library not included


def combine(folder: Path, roots: Sequence[Path], path: Path) -> Combined:
    """Write one index to ``path`` holding everything in the libraries at ``roots``, which are
    in ``folder``. Libraries without a usable index, or with nothing in it, are left out."""
    folder = Path(folder).absolute()
    opened: list[tuple[Path, sqlite3.Connection, Any]] = []
    left_out = []
    for root in roots:
        if not index_path(root).is_file():
            left_out.append(f"{_label(root, folder)}: no exam index yet")
            continue
        try:
            conn, info = open_index(index_path(root))
        except ExamIndexError as exc:
            left_out.append(f"{_label(root, folder)}: {exc}")
            continue
        if not int(info.meta.get("passages", 0)):
            conn.close()
            left_out.append(f"{_label(root, folder)}: nothing to search yet")
            continue
        opened.append((root, conn, info))

    # Two libraries of the same name are told apart by where they are.
    names = [info.meta.get("library_name") or _folder(root).name for root, _, info in opened]
    repeated = Counter(names)
    names = [name if repeated[name] == 1 else _label(root, folder) for name, (root, _, _) in zip(names, opened)]

    libraries: list[LibraryRow] = []
    courses: list[CourseRow] = []
    sources: list[SourceRow] = []
    passages: list[PassageRow] = []
    notes: list[NotesRow] = []
    parts: list[Part] = []
    slugs: set[str] = set()
    for (root, conn, info), name in zip(opened, names):
        library_id = len(libraries) + 1
        libraries.append(LibraryRow(library_id, name, str(root)))
        course_ids: dict[int, int] = {}
        source_ids: dict[int, int] = {}
        passage_ids: dict[int, int] = {}
        try:
            for r in conn.execute("SELECT * FROM course ORDER BY slug"):
                course_ids[r["id"]] = len(courses) + 1
                courses.append(replace(_row(CourseRow, r), id=course_ids[r["id"]], slug=_unique(r["slug"], slugs),
                                       library_id=library_id))
            for r in conn.execute("SELECT * FROM source ORDER BY id"):
                source_ids[r["id"]] = len(sources) + 1
                sources.append(replace(_row(SourceRow, r), id=source_ids[r["id"]],
                                       course_id=course_ids[r["course_id"]]))
            for r in conn.execute("SELECT * FROM passage ORDER BY id"):
                passage_ids[r["id"]] = len(passages) + 1
                passages.append(replace(_row(PassageRow, r), id=passage_ids[r["id"]],
                                        source_id=source_ids[r["source_id"]]))
            for r in conn.execute("SELECT * FROM notes ORDER BY source_id"):
                data = _renumber(json.loads(r["data"]), passage_ids)
                notes.append(replace(_row(NotesRow, r), source_id=source_ids[r["source_id"]], data=json.dumps(data)))
        finally:
            conn.close()
        parts.append(Part(name, root, info.path, file_sha256(info.path), info.meta.get("built_at", ""),
                          int(info.meta.get("passages", 0))))

    build(path, libraries=libraries, courses=courses, sources=sources, passages=passages, notes=notes,
          meta={"library_name": _shown(folder)})
    return Combined(Path(path), parts, left_out)


def _shown(folder: Path) -> str:
    """The folder as the page names it: "~/StudyBuddy" (its name alone would repeat the app's)."""
    try:
        return "~/" + folder.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(folder)


def _row(row_type: type, row: sqlite3.Row) -> Any:
    return row_type(**{f.name: row[f.name] for f in fields(row_type)})


def _unique(slug: str, taken: set[str]) -> str:
    """``slug``, or ``slug-2``, ``slug-3``... when another library already has a course called that."""
    candidate, n = slug, 1
    while candidate in taken:
        n += 1
        candidate = f"{slug}-{n}"
    taken.add(candidate)
    return candidate


def _renumber(value: Any, passage_ids: dict[int, int]) -> Any:
    """The notes with every ``passage_id`` changed to the passage's id in the combined index."""
    if isinstance(value, dict):
        return {
            key: passage_ids.get(item) if key == "passage_id" and item is not None else _renumber(item, passage_ids)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_renumber(item, passage_ids) for item in value]
    return value


def _folder(root: Path) -> Path:
    """The folder a person knows: a session's own folder rather than its hidden library."""
    return root.parent if root.name == SESSION_DIR else root


def _label(root: Path, folder: Path) -> str:
    """Where the library is, relative to ``folder``: "Term 5/Async 3"."""
    folder_of = _folder(root)
    try:
        relative = folder_of.relative_to(folder)
    except ValueError:
        return str(folder_of)
    return relative.as_posix() if relative.parts else folder_of.name
