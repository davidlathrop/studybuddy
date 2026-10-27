"""Finding the library, the same way the builder does (without importing it).

``LIBRARY`` argument, then ``$STUDYBUDDY_LIBRARY``, then the nearest folder above the
current one holding a ``library.toml``, then the default recorded by
``studybuddy library init`` (``~/.config/studybuddy/default-library``). A folder made with
``studybuddy session`` keeps its library in a ``.studybuddy`` subfolder; naming the folder
itself is enough.

:func:`libraries_in` finds every library in a folder such as ``~/StudyBuddy``, for
``studybuddy-exam FOLDER`` and ``studybuddy serve``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from studybuddy_exam.index_format import FILE_NAME

CONFIG_FILE = "library.toml"
SESSION_DIR = ".studybuddy"
MAX_DEPTH = 8  # how far below the folder libraries are looked for


class LibraryNotFound(Exception):
    pass


def find_library(explicit: Path | None = None, *, cwd: Path | None = None) -> Path:
    if explicit is not None:
        return _check(Path(explicit).expanduser())
    if env := os.environ.get("STUDYBUDDY_LIBRARY"):
        return _check(Path(env).expanduser())
    here = Path(cwd or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / CONFIG_FILE).is_file():
            return folder
        if (folder / SESSION_DIR / CONFIG_FILE).is_file():
            return folder / SESSION_DIR
    config_dir = Path(os.environ.get("STUDYBUDDY_CONFIG_DIR") or Path.home() / ".config" / "studybuddy")
    pointer = config_dir / "default-library"
    if pointer.is_file() and (recorded := pointer.read_text(encoding="utf-8").strip()):
        return _check(Path(recorded))
    raise LibraryNotFound(
        "no library found: pass its folder (studybuddy-exam ~/StudyBuddy/fall-2026) or set STUDYBUDDY_LIBRARY"
    )


def _check(root: Path) -> Path:
    root = root.resolve()
    if not (root / CONFIG_FILE).is_file() and (root / SESSION_DIR / CONFIG_FILE).is_file():
        root = root / SESSION_DIR
    if not (root / CONFIG_FILE).is_file():
        raise LibraryNotFound(f"{root} is not a StudyBuddy library (no {CONFIG_FILE})")
    return root


def libraries_in(folder: Path) -> list[Path]:
    """The folder itself if it is a library (or a session folder), else every library and
    session folder below it, in natural path order (Week 2 before Week 10). Hidden folders
    are skipped, links to folders are followed, and nothing inside a library is searched."""
    folder = Path(folder).expanduser().absolute()
    if not folder.is_dir():
        raise LibraryNotFound(f"{folder} is not a folder")
    try:
        return [_check(folder)]
    except LibraryNotFound:
        pass
    found: list[Path] = []
    seen: set[Path] = set()
    for here, subfolders, _ in os.walk(folder, followlinks=True):
        here_path = Path(here)
        real = here_path.resolve()
        if real in seen:  # reached twice through links
            subfolders.clear()
            continue
        seen.add(real)
        if (here_path / CONFIG_FILE).is_file():
            found.append(here_path)
            subfolders.clear()
            continue
        if (here_path / SESSION_DIR / CONFIG_FILE).is_file():
            found.append(here_path / SESSION_DIR)
        deeper = len(here_path.relative_to(folder).parts) < MAX_DEPTH
        subfolders[:] = sorted((d for d in subfolders if not d.startswith(".")), key=_natural) if deeper else []
    return found


def _natural(name: str) -> list:
    return [(0, int(part), "") if part.isdigit() else (1, 0, part.casefold()) for part in re.split(r"(\d+)", name)]


def index_path(library_root: Path) -> Path:
    return library_root / "exam" / FILE_NAME
