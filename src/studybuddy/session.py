"""``studybuddy session DIR``: one sitting's videos (and any slides or notes) as its own library.

The library lives in ``DIR/.studybuddy/`` with a single course named after the folder, so
the folder stays self-contained: copy it to another Mac and the search still works. Files
are added in natural order (``part 2`` before ``part 10``); running the command again only
processes files that are new.
"""

from __future__ import annotations

import re
import shutil
import sys
import unicodedata
from pathlib import Path

from studybuddy.ingest.add import KINDS
from studybuddy.library import SESSION_DIR, Course, Library, LibraryError

_DIGITS = re.compile(r"(\d+)")
_SLUG_JUNK = re.compile(r"[^a-z0-9_-]+")


def natural_key(name: str) -> list[str | int]:
    """Sort key that orders "Part 2" before "Part 10" (strings and numbers alternate)."""
    return [int(part) if part.isdigit() else part for part in _DIGITS.split(name.lower())]


def session_files(folder: Path, *, recursive: bool = False) -> list[Path]:
    """Supported files in ``folder``, in natural order, skipping hidden and Office temp files."""
    pattern = "**/*" if recursive else "*"
    found = [
        p for p in folder.glob(pattern)
        if p.is_file()
        and p.suffix.lower() in KINDS
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)  # .studybuddy, .DS_Store
        and not p.name.startswith("~$")  # Office's lock files
    ]
    return sorted(found, key=lambda p: [natural_key(part) for part in p.relative_to(folder).parts])


def slug_for(name: str) -> str:
    """A course name from a folder name: "ÉCON 201: Day 3!" -> "econ-201-day-3"."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = _SLUG_JUNK.sub("-", ascii_name.lower()).strip("-_")[:40].strip("-_")
    return slug if slug and slug[0].isalnum() else f"session-{slug}".strip("-")[:40] or "session"


def open_session(folder: Path, *, title: str | None = None) -> tuple[Library, Course]:
    """The session library in ``folder`` (created on first use) and its one course."""
    root = folder / SESSION_DIR
    if (root / "library.toml").is_file():
        library = Library(root)
    else:
        library = Library.init(root, name=title or folder.name, make_default=False)
    courses = library.courses()
    if len(courses) > 1:
        raise LibraryError(f"{root} holds {len(courses)} courses; a session library has exactly one")
    if courses:
        return library, courses[0]
    return library, library.add_course(slug_for(folder.name), title or folder.name)


def exam_app() -> Path | None:
    """The ``studybuddy-exam`` program from the exam app's own environment.

    The copy inside this (builder) environment is skipped on purpose: the exam app should
    run from the environment that has no AI packages installed at all.
    """
    repo = Path(__file__).resolve().parents[2]
    candidates = [repo / "exam" / ".venv" / "bin" / "studybuddy-exam"]
    if found := shutil.which("studybuddy-exam"):
        candidates.append(Path(found))
    here = Path(sys.prefix).resolve()
    for candidate in candidates:
        if candidate.is_file() and not candidate.resolve().is_relative_to(here):
            return candidate
    return None
