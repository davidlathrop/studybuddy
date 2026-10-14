"""Library commands (plan §9): ``library init``, ``course add|list``, ``add`` and ``status``.

Results go to stdout, one line per item; progress and messages go to stderr through the
CLI's display.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from studybuddy import audio, exam_export
from studybuddy.ingest.add import AddOptions, AddResult, add_file
from studybuddy.ingest.convert import Converter
from studybuddy.ingest.model import IngestError
from studybuddy.library import Library, LibraryError
from studybuddy.transcription import format_clock

if TYPE_CHECKING:
    from studybuddy.cli import _Display

log = logging.getLogger(__name__)


def library_init(args: argparse.Namespace, display: _Display) -> int:
    library = Library.init(args.path, name=args.name, make_default=not args.no_default)
    print(library.root)
    log.info("Created library %r", library.settings.name)
    if not args.no_default:
        log.info("It is now the default library for studybuddy commands")
    log.info("Next: studybuddy course add SLUG --title \"Course title\"")
    return 0


def course_add(args: argparse.Namespace, display: _Display) -> int:
    library = Library.find(args.library)
    course = library.add_course(args.slug, args.title, glossary=args.glossary)
    folder = library.course_dir(course.slug)
    print(folder)
    log.info("Added %s (%s). Course terms go in %s", course.slug, course.title, folder / "glossary.txt")
    return 0


def course_list(args: argparse.Namespace, display: _Display) -> int:
    library = Library.find(args.library)
    with library.db() as db:
        counts = dict(db.execute("SELECT course_id, COUNT(*) FROM source GROUP BY course_id").fetchall())
    for course in library.courses():
        print(f"{course.slug:<12} {course.title}  ({counts.get(course.id, 0)} files)")
    return 0


def add(args: argparse.Namespace, display: _Display) -> int:
    library = Library.find(args.library)
    course = library.course(args.course)
    if args.title and len(args.files) > 1:
        raise LibraryError("--title applies to a single file; add files one at a time to title each")
    options = AddOptions(
        title=args.title, week=args.week, date=args.date, replace=args.replace, mode=args.mode,
        ocr=False if args.no_ocr else None, convert=False if args.no_convert else None,
    )
    failed = 0
    with Converter() as converter:
        for path in args.files:
            log.info("Adding %s", path.name)
            try:
                result = add_file(library, course, path, options, converter=converter, progress=display)
            except (IngestError, audio.AudioError, LibraryError, ValueError, OSError) as exc:
                failed += 1
                log.error("%s: %s", path.name, exc)
                continue
            print(_describe(result), flush=True)
    if failed:
        log.error("%d of %d file%s could not be added", failed, len(args.files), "" if len(args.files) == 1 else "s")
        return 1
    return 0


def build_exam(args: argparse.Namespace, display: _Display) -> int:
    library = Library.find(args.library)
    log.info("Building the exam index for %s", library.settings.name or library.root.name)
    built = exam_export.build_exam(library)
    print(built.path)
    log.info(
        "%s, %s, %s; %.1f MB; SHA-256 %s",
        _plural(built.courses, "course"), _plural(built.sources, "file"), _plural(built.passages, "passage"),
        built.size_bytes / 1e6, built.sha256[:16],
    )
    if built.without_view:
        log.warning("%s have no video or PDF to open (text only)", _plural(built.without_view, "file"))
    log.info("Start the exam app with: studybuddy-exam %s", library.root)
    return 0


def status(args: argparse.Namespace, display: _Display) -> int:
    library = Library.find(args.library)
    print(f"{library.settings.name or library.root.name}  ({library.root})")
    print(exam_export.exam_index_state(library))
    if args.course:
        _course_status(library, library.course(args.course).id)
        return 0
    with library.db() as db:
        rows = db.execute(
            """SELECT c.slug, c.title, s.kind, COUNT(DISTINCT s.id) AS files, SUM(s.duration_s) AS seconds,
                      (SELECT COUNT(*) FROM passage p JOIN source s2 ON p.source_id = s2.id
                        WHERE s2.course_id = c.id AND s2.kind = s.kind) AS passages
                 FROM course c LEFT JOIN source s ON s.course_id = c.id
                GROUP BY c.id, s.kind ORDER BY c.slug"""
        ).fetchall()
    per_course: dict[tuple[str, str], dict[str, tuple[int, float, int]]] = defaultdict(dict)
    for r in rows:
        per_course[(r["slug"], r["title"])][r["kind"]] = (r["files"], r["seconds"] or 0.0, r["passages"] or 0)
    if not per_course:
        print("No courses yet: studybuddy course add SLUG --title \"Course title\"")
        return 0
    print(f"{'course':<12} {'videos':>16} {'slides':>7} {'PDFs':>5} {'Word':>5} {'passages':>9}  title")
    for (slug, title), kinds in per_course.items():
        videos, seconds, _ = kinds.get("video", (0, 0.0, 0))
        passages = sum(p for _, _, p in kinds.values())
        count = lambda k: kinds.get(k, (0, 0, 0))[0]  # noqa: E731
        video_text = f"{videos} ({format_clock(seconds)})" if videos else "0"
        print(f"{slug:<12} {video_text:>16} {count('pptx'):>7} {count('pdf'):>5} {count('docx'):>5} "
              f"{passages:>9}  {title}")
    return 0


def _course_status(library: Library, course_id: int) -> None:
    with library.db() as db:
        rows = db.execute(
            """SELECT s.id, s.kind, s.title, s.duration_s, s.page_count, s.view_path, s.warnings,
                      COUNT(p.id) AS passages
                 FROM source s LEFT JOIN passage p ON p.source_id = s.id
                WHERE s.course_id = ? GROUP BY s.id ORDER BY s.kind, s.title""",
            (course_id,),
        ).fetchall()
    if not rows:
        print("No files yet: studybuddy add COURSE FILE...")
        return
    for r in rows:
        size = _size(r["kind"], r["duration_s"], r["page_count"])
        view = "" if r["view_path"] else "  [no PDF for viewing]"
        print(f"{r['kind']:<6} {r['title']}  ({size}, {_plural(r['passages'], 'passage')}){view}")
        for warning in (r["warnings"] or "").splitlines():
            print(f"         note: {warning}")


def _describe(result: AddResult) -> str:
    if result.status == "skipped":
        return f"skipped  {result.kind:<5} {result.title}  (already added)"
    size = _size(result.kind, result.duration_s, result.page_count)
    return f"{result.status:<8} {result.kind:<5} {result.title}  ({size}, {_plural(result.passages, 'passage')})"


def _size(kind: str, duration_s: float | None, page_count: int | None) -> str:
    if kind == "video":
        return format_clock(duration_s or 0)
    unit = "slide" if kind == "pptx" else "page"
    return _plural(page_count, unit) if page_count else f"{unit}s unknown"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def register(commands: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Add the library commands to the CLI's subcommand list (subparsers inherit its parser class)."""
    lib = commands.add_parser("library", help="create a semester library")
    lib_cmds = lib.add_subparsers(dest="library_command", metavar="COMMAND", required=True)
    init = lib_cmds.add_parser("init", parents=[common], help="create a new library folder",
                               description="Create a library: settings, course folders and the build database.")
    init.add_argument("path", type=Path, metavar="PATH", help="folder for the library, e.g. ~/StudyBuddy/fall-2026")
    init.add_argument("--name", help="display name (default: the folder name)")
    init.add_argument("--no-default", action="store_true", help="do not make it the default library")
    init.set_defaults(run=library_init)

    course = commands.add_parser("course", help="add or list courses")
    course_cmds = course.add_subparsers(dest="course_command", metavar="COMMAND", required=True)
    c_add = course_cmds.add_parser("add", parents=[common], help="add a course")
    c_add.add_argument("slug", metavar="SLUG", help="short lowercase name, e.g. bio101")
    c_add.add_argument("--title", required=True, help="full course title")
    c_add.add_argument("--glossary", type=Path, metavar="FILE",
                       help="course terms, one per line (copied to the course's glossary.txt)")
    c_add.set_defaults(run=course_add)
    c_list = course_cmds.add_parser("list", parents=[common], help="list courses")
    c_list.set_defaults(run=course_list)

    a = commands.add_parser(
        "add", parents=[common], help="add lecture videos and documents to a course",
        description=(
            "Add videos (MP4, MOV, MKV, M4V, WebM), PowerPoint (.pptx), PDF and Word (.docx) files to a "
            "course. Videos are transcribed and get a small exam copy; documents are copied, their text "
            "extracted with slide/page numbers, and PowerPoint/Word files converted to PDF for viewing. "
            "Re-running after an interruption resumes; files already added are skipped."
        ),
    )
    a.add_argument("course", metavar="COURSE", help="course slug, e.g. bio101")
    a.add_argument("files", type=Path, nargs="+", metavar="FILE")
    a.add_argument("--title", help="title for the file (default: its file name); single file only")
    a.add_argument("--week", type=int, metavar="N", help="week of the semester")
    a.add_argument("--date", metavar="YYYY-MM-DD", help="lecture or handout date")
    a.add_argument("--mode", choices=("accurate", "fast"), help="transcription mode (default: library.toml)")
    a.add_argument("--no-ocr", action="store_true", help="skip reading text in images and scanned pages")
    a.add_argument("--no-convert", action="store_true", help="skip converting PowerPoint/Word to PDF")
    a.add_argument("--replace", action="store_true",
                   help="replace a different file of the same name already in the course")
    a.set_defaults(run=add)

    b = commands.add_parser(
        "build-exam", parents=[common], help="build the exam index from the library (no AI)",
        description="Export every course, file and passage into exam/exam.sqlite, the index the exam app searches.",
    )
    b.set_defaults(run=build_exam)

    s = commands.add_parser("status", parents=[common], help="show what is in the library")
    s.add_argument("course", nargs="?", metavar="COURSE", help="list the files of one course")
    s.set_defaults(run=status)
