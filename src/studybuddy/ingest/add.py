"""``studybuddy add``: put one file into a course (plan §5).

1. Hash the file. A file already in the course (same SHA-256) is skipped.
2. Run the ingester into ``courses/<slug>/.staging/<hash>/``. The folder is named after the
   file's hash, so re-running after an interruption picks up the transcript, exam copy or
   copied document that is already there.
3. Only when that succeeded: remove what ``--replace`` replaces, move the staged files to
   their final names, and record the source and its passages in one transaction.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from studybuddy.ingest.convert import Converter
from studybuddy.ingest.documents import ingest_document
from studybuddy.ingest.model import IngestError
from studybuddy.ingest.staging import Staged
from studybuddy.ingest.video import ingest_video
from studybuddy.library import Course, Library, now_iso
from studybuddy.transcription import Progress

log = logging.getLogger(__name__)

KINDS = {
    ".mp4": "video", ".m4v": "video", ".mov": "video", ".mkv": "video", ".webm": "video",
    ".pptx": "pptx", ".pdf": "pdf", ".docx": "docx",
}
_CONVERT_FIRST = {
    ".ppt": "save it as .pptx first", ".pps": "save it as .pptx first", ".doc": "save it as .docx first",
    ".key": "export it to .pptx or PDF first", ".pages": "export it to .docx or PDF first",
}


@dataclass(frozen=True)
class AddOptions:
    title: str | None = None
    week: int | None = None
    date: str | None = None  # YYYY-MM-DD
    replace: bool = False
    mode: str | None = None  # transcription mode; default from library.toml
    ocr: bool | None = None  # default from library.toml
    convert: bool | None = None  # default from library.toml


@dataclass
class AddResult:
    path: Path
    kind: str
    status: str  # "added", "replaced" or "skipped"
    title: str
    source_id: int | None = None
    passages: int = 0
    duration_s: float | None = None
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


def kind_of(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in KINDS:
        return KINDS[suffix]
    if suffix in _CONVERT_FIRST:
        raise IngestError(f"{Path(path).name}: {suffix} files are not supported; {_CONVERT_FIRST[suffix]}")
    supported = ", ".join(sorted(KINDS))
    raise IngestError(f"{Path(path).name}: unsupported file type {suffix or '(none)'}; supported: {supported}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def add_file(
    library: Library,
    course: Course,
    src: Path,
    options: AddOptions = AddOptions(),
    *,
    converter: Converter | None = None,
    progress: Progress | None = None,
) -> AddResult:
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise IngestError(f"file not found: {src}")
    kind = kind_of(src)
    date = _check_date(options.date)
    digest = sha256_file(src)
    title = (options.title or src.stem.replace("_", " ")).strip()

    with library.db() as db:
        same = db.execute(
            "SELECT id, title FROM source WHERE course_id = ? AND sha256 = ?", (course.id, digest)
        ).fetchone()
    if same is not None:
        return AddResult(src, kind, "skipped", same["title"], source_id=same["id"],
                         warnings=[f"already in {course.slug} as {same['title']!r}"])

    final = _final_paths(library, course, kind, src)
    replaced = _conflicts(library, course, final, replace=options.replace)

    staging = library.course_dir(course.slug) / ".staging" / digest[:16]
    staging.mkdir(parents=True, exist_ok=True)
    settings = library.settings
    if kind == "video":
        staged = ingest_video(
            src, staging, settings=settings, mode=options.mode or settings.transcribe.mode,
            hotwords=library.glossary_terms(course.slug), progress=progress,
        )
    else:
        convert = settings.documents.convert if options.convert is None else options.convert
        staged = ingest_document(
            kind, src, staging, ocr=settings.documents.ocr if options.ocr is None else options.ocr,
            converter=converter if convert else None, progress=progress,
        )

    for source_id in replaced:
        remove_source(library, source_id)
    placed = _place(staged, final)
    source_id = _record(library, course, src, kind, title, options.week, date, digest, staged, placed)
    shutil.rmtree(staging, ignore_errors=True)
    for warning in staged.warnings:
        log.warning("%s: %s", src.name, warning)
    return AddResult(
        src, kind, "replaced" if replaced else "added", title, source_id=source_id,
        passages=len(staged.passages), duration_s=staged.duration_s, page_count=staged.page_count,
        warnings=staged.warnings,
    )


def remove_source(library: Library, source_id: int) -> None:
    """Delete a source, its passages and its files in the library."""
    with library.db() as db:
        row = db.execute("SELECT kind, stored_path, view_path FROM source WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            return
        db.execute("DELETE FROM source WHERE id = ?", (source_id,))
    paths = {row["stored_path"], row["view_path"]} - {None}
    if row["kind"] == "video" and row["stored_path"]:
        paths.add(str(Path(row["stored_path"]).with_suffix(".txt")))
    for relative in paths:
        library.absolute(relative).unlink(missing_ok=True)


def _check_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError:
        raise IngestError(f"invalid date {value!r}; use YYYY-MM-DD") from None


def _final_paths(library: Library, course: Course, kind: str, src: Path) -> dict[str, Path]:
    folder = library.course_dir(course.slug)
    if kind == "video":
        videos = folder / "videos"
        return {
            "stored": videos / f"{src.stem}.json",
            "extra": videos / f"{src.stem}.txt",
            "view": videos / f"{src.stem}.exam.mp4",
        }
    docs = folder / "docs"
    stored = docs / src.name
    # "week04.pptx.pdf", so slides and a handout both named week04 never collide.
    return {"stored": stored, "view": stored if kind == "pdf" else docs / f"{src.name}.pdf"}


def _conflicts(library: Library, course: Course, final: dict[str, Path], *, replace: bool) -> list[int]:
    """Sources that would be overwritten; an error unless ``replace``."""
    wanted = {library.relative(p) for p in final.values()}
    with library.db() as db:
        rows = db.execute(
            "SELECT id, title, stored_path, view_path FROM source WHERE course_id = ?", (course.id,)
        ).fetchall()
    clashing = [r for r in rows if {r["stored_path"], r["view_path"]} & wanted]
    owned = {p for r in clashing for p in (r["stored_path"], r["view_path"]) if p}
    strays = sorted(p for p in wanted if p not in owned and library.absolute(p).exists())
    if replace:
        for stray in strays:
            library.absolute(stray).unlink()
        return [r["id"] for r in clashing]
    if clashing:
        names = ", ".join(repr(r["title"]) for r in clashing)
        raise IngestError(
            f"{course.slug} already has a different file with this name ({names}); "
            "use --replace to replace it, or rename the file"
        )
    if strays:
        raise IngestError(f"{strays[0]} already exists in the library; use --replace to overwrite it")
    return []


def _place(staged: Staged, final: dict[str, Path]) -> dict[str, Path]:
    placed: dict[str, Path] = {}
    moved: dict[Path, Path] = {}  # staged file -> where it went
    for role, staged_path in staged.files.items():
        if staged_path in moved:  # a PDF is both the stored copy and the view
            placed[role] = moved[staged_path]
            continue
        target = final[role]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staged_path, target)
        placed[role] = moved[staged_path] = target
    return placed


def _record(library: Library, course: Course, src: Path, kind: str, title: str, week: int | None,
            date: str | None, digest: str, staged: Staged, placed: dict[str, Path]) -> int:
    stored = placed.get("stored")
    view = placed.get("view")
    with library.db() as db:
        cur = db.execute(
            """INSERT INTO source(course_id, kind, title, week, date, original_path, stored_path, view_path,
                                  duration_s, page_count, sha256, extractor, warnings, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                course.id, kind, title, week, date, str(src),
                library.relative(stored) if stored else None, library.relative(view) if view else None,
                staged.duration_s, staged.page_count, digest, staged.extractor,
                "\n".join(staged.warnings) or None, now_iso(),
            ),
        )
        source_id = int(cur.lastrowid)
        db.executemany(
            """INSERT INTO passage(source_id, ord, t_start_ms, t_end_ms, slide_no, page_no, heading_path,
                                   origin, text, text_sha)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (source_id, i, p.t_start_ms, p.t_end_ms, p.slide_no, p.page_no, p.heading_path, p.origin,
                 p.text, hashlib.sha256(p.text.encode("utf-8")).hexdigest())
                for i, p in enumerate(staged.passages, start=1)
            ],
        )
    return source_id
