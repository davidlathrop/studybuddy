"""Finding the library, the same way the builder does (without importing it).

``LIBRARY`` argument, then ``$STUDYBUDDY_LIBRARY``, then the nearest folder above the
current one holding a ``library.toml``, then the default recorded by
``studybuddy library init`` (``~/.config/studybuddy/default-library``). A folder made with
``studybuddy session`` keeps its library in a ``.studybuddy`` subfolder; naming the folder
itself is enough.
"""

from __future__ import annotations

import os
from pathlib import Path

from studybuddy_exam.index_format import FILE_NAME

CONFIG_FILE = "library.toml"
SESSION_DIR = ".studybuddy"


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


def index_path(library_root: Path) -> Path:
    return library_root / "exam" / FILE_NAME
