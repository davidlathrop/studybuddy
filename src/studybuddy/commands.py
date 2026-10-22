"""Library commands (plan §9): ``library init``, ``course add|list``, ``add``, ``notes`` and ``status``.

Results go to stdout, one line per item; progress and messages go to stderr through the
CLI's display.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from studybuddy import audio, exam_export, session, study_notes
from studybuddy.enrich.backends import EnrichmentError
from studybuddy.ingest.add import KINDS, AddOptions, AddResult, add_file, files_in, natural_key
from studybuddy.ingest.convert import Converter
from studybuddy.ingest.model import IngestError
from studybuddy.library import Course, Library, LibraryError
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
    paths = expand_folders(args.files, recursive=args.recursive)
    if not paths:
        raise LibraryError("no videos or documents to add")
    if args.title and len(paths) > 1:
        raise LibraryError("--title applies to a single file; add files one at a time to title each")
    options = AddOptions(
        title=args.title, week=args.week, date=args.date, replace=args.replace, mode=args.mode,
        ocr=False if args.no_ocr else None, convert=False if args.no_convert else None,
    )
    failed, results = _add_files(library, course, paths, options, display)
    if _notes_wanted(library, args):
        videos = [r.source_id for r in results if r.kind == "video" and r.status != "skipped"]
        written = _write_notes(library, [course], display, source_ids=videos) if videos else True
        if written and (waiting := study_notes.videos_without_notes(library, course)):
            log.info("%s in %s %s no study notes yet; write them with: %s", _plural(waiting, "video"), course.slug,
                     "has" if waiting == 1 else "have", study_notes.retry_command(library, [course]))
    return 1 if failed else 0


def expand_folders(paths: list[Path], *, recursive: bool = False) -> list[Path]:
    """Files as given, folders replaced by the files ``add`` takes from them; no duplicates."""
    expanded: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            found = files_in(path, recursive=recursive)
            if found:
                log.info("%s: %s", path.name or str(path), _plural(len(found), "file"))
            else:
                log.warning("%s has no videos or documents to add%s", path,
                            "" if recursive else " (use --recursive for subfolders)")
            expanded.extend(found)
        else:
            expanded.append(path)
    unique: dict[Path, Path] = {}
    for path in expanded:
        unique.setdefault(path.resolve(), path)
    return list(unique.values())


def _add_files(library: Library, course: Course, paths: list[Path], options: AddOptions,
               display: _Display) -> tuple[int, list[AddResult]]:
    """Add each file, reporting one line per file; return how many could not be added, and the results."""
    failed = 0
    results = []
    with Converter() as converter:
        for path in paths:
            log.info("Adding %s", path.name)
            try:
                result = add_file(library, course, path, options, converter=converter, progress=display)
            except (IngestError, audio.AudioError, LibraryError, ValueError, OSError) as exc:
                failed += 1
                log.error("%s: %s", path.name, exc)
                continue
            results.append(result)
            print(_describe(result), flush=True)
    if failed:
        log.error("%d of %d file%s could not be added", failed, len(paths), "" if len(paths) == 1 else "s")
    return failed, results


def _notes_wanted(library: Library, args: argparse.Namespace) -> bool:
    return library.settings.notes.auto and not args.no_notes


def _write_notes(library: Library, courses: list[Course], display: _Display, *,
                 source_ids: list[int] | None = None, redo: bool = False) -> bool:
    """Write the study notes missing in ``courses``, print their paths; True if none failed."""
    try:
        report = study_notes.update_notes(library, courses, source_ids=source_ids, redo=redo, progress=display)
    except EnrichmentError as exc:
        log.error("study notes not written: %s", exc)
        return False
    for path in (*report.written, *report.indexes):
        print(path, flush=True)
    if report.limit is not None:
        log.error("%s; write the remaining notes then with: %s", study_notes.describe_limit(report.limit),
                  study_notes.retry_command(library, courses))
    elif report.failed:
        log.error("the study notes of %s could not be written; try again with: %s",
                  _plural(len(report.failed), "video"), study_notes.retry_command(library, courses))
    return report.ok


def notes(args: argparse.Namespace, display: _Display) -> int:
    transcripts = [Path(t).expanduser() for t in args.targets if t.lower().endswith(".json")]
    slugs = [t for t in args.targets if not t.lower().endswith(".json")]
    ok = True
    for transcript in transcripts:
        try:
            print(study_notes.notes_for_transcript(transcript, progress=display), flush=True)
        except EnrichmentError as exc:
            log.error("%s", exc)
            ok = False
    if slugs or not transcripts:
        library = Library.find(args.library)
        courses = [library.course(slug) for slug in slugs] if slugs else library.courses()
        if not courses:
            raise LibraryError("no courses yet: studybuddy course add SLUG --title \"Course title\"")
        ok = _write_notes(library, courses, display, redo=args.redo) and ok
    return 0 if ok else 1


def session_command(args: argparse.Namespace, display: _Display) -> int:
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise LibraryError(f"{folder} is not a folder")
    files = files_in(folder, recursive=args.recursive)
    if not files:
        kinds = ", ".join(sorted({k.lstrip(".") for k in KINDS}))
        raise LibraryError(f"no videos or documents in {folder} (looked for: {kinds})")
    library, course = session.open_session(folder, title=args.title)
    log.info("Session %r: %s in %s", course.title, _plural(len(files), "file"), folder)
    options = AddOptions(mode=args.mode, ocr=False if args.no_ocr else None,
                         convert=False if args.no_convert else None)
    failed, _ = _add_files(library, course, files, options, display)

    built = exam_export.build_exam(library)
    log.info("Exam index: %s, %s", _plural(built.sources, "file"), _plural(built.passages, "passage"))
    if not built.passages:
        log.error("nothing searchable was found")
        return 1
    if _notes_wanted(library, args):
        _write_notes(library, [course], display)
    reopen = f"studybuddy-exam {shlex.quote(str(folder))}"
    if args.no_search:
        log.info("Search it with: %s", reopen)
        return 1 if failed else 0
    app = session.exam_app()
    if app is None:
        log.warning("the exam app is not installed (see the README); then search with: %s", reopen)
        return 1 if failed else 0
    log.info("Opening the search page (Ctrl-C to stop). Later, reopen it with: %s", reopen)
    display.close()
    sys.stdout.flush()
    sys.stderr.flush()
    # Replace this process, which has the AI libraries loaded, with the AI-free exam app.
    command = [str(app), str(folder)] + (["--port", str(args.port)] if args.port else [])
    os.execv(command[0], command)
    return 0  # not reached


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
                        WHERE s2.course_id = c.id AND s2.kind = s.kind) AS passages,
                      (SELECT COUNT(*) FROM notes n JOIN source s3 ON n.source_id = s3.id
                        WHERE s3.course_id = c.id) AS notes
                 FROM course c LEFT JOIN source s ON s.course_id = c.id
                GROUP BY c.id, s.kind ORDER BY c.slug"""
        ).fetchall()
    per_course: dict[tuple[str, str], dict[str, tuple[int, float, int]]] = defaultdict(dict)
    notes_count: dict[str, int] = {}
    for r in rows:
        per_course[(r["slug"], r["title"])][r["kind"]] = (r["files"], r["seconds"] or 0.0, r["passages"] or 0)
        notes_count[r["slug"]] = r["notes"]
    if not per_course:
        print("No courses yet: studybuddy course add SLUG --title \"Course title\"")
        return 0
    print(f"{'course':<12} {'videos':>16} {'notes':>6} {'slides':>7} {'PDFs':>5} {'Word':>5} {'passages':>9}  title")
    for (slug, title), kinds in per_course.items():
        videos, seconds, _ = kinds.get("video", (0, 0.0, 0))
        passages = sum(p for _, _, p in kinds.values())
        count = lambda k: kinds.get(k, (0, 0, 0))[0]  # noqa: E731
        video_text = f"{videos} ({format_clock(seconds)})" if videos else "0"
        notes_text = f"{notes_count[slug]}/{videos}" if videos else "-"
        print(f"{slug:<12} {video_text:>16} {notes_text:>6} {count('pptx'):>7} {count('pdf'):>5} {count('docx'):>5} "
              f"{passages:>9}  {title}")
    return 0


def _course_status(library: Library, course_id: int) -> None:
    with library.db() as db:
        rows = db.execute(
            """SELECT s.id, s.kind, s.title, s.duration_s, s.page_count, s.view_path, s.warnings,
                      COUNT(p.id) AS passages, EXISTS (SELECT 1 FROM notes n WHERE n.source_id = s.id) AS notes
                 FROM source s LEFT JOIN passage p ON p.source_id = s.id
                WHERE s.course_id = ? GROUP BY s.id""",
            (course_id,),
        ).fetchall()
    rows = sorted(rows, key=lambda r: (r["kind"], natural_key(r["title"])))
    if not rows:
        print("No files yet: studybuddy add COURSE FILE...")
        return
    for r in rows:
        size = _size(r["kind"], r["duration_s"], r["page_count"])
        view = "" if r["view_path"] else "  [no PDF for viewing]"
        if r["kind"] == "video" and r["passages"] and not r["notes"]:
            view += "  [no study notes]"
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
            "course. A folder adds every such file in it, in natural order (Part 2 before Part 10). "
            "Videos are transcribed and get a small exam copy; documents are copied, their text "
            "extracted with slide/page numbers, and PowerPoint/Word files converted to PDF for viewing. "
            "Re-running after an interruption resumes; files already added are skipped."
        ),
    )
    a.add_argument("course", metavar="COURSE", help="course slug, e.g. bio101")
    a.add_argument("files", type=Path, nargs="+", metavar="FILE_OR_FOLDER",
                   help="files, or folders whose videos and documents are all added (in natural order)")
    a.add_argument("--recursive", action="store_true", help="for folders: also add files in their subfolders")
    a.add_argument("--title", help="title for the file (default: its file name); single file only")
    a.add_argument("--week", type=int, metavar="N", help="week of the semester")
    a.add_argument("--date", metavar="YYYY-MM-DD", help="lecture or handout date")
    a.add_argument("--mode", choices=("accurate", "fast"), help="transcription mode (default: library.toml)")
    a.add_argument("--no-ocr", action="store_true", help="skip reading text in images and scanned pages")
    a.add_argument("--no-convert", action="store_true", help="skip converting PowerPoint/Word to PDF")
    a.add_argument("--replace", action="store_true",
                   help="replace a different file of the same name already in the course")
    a.add_argument("--no-notes", action="store_true",
                   help="do not write study notes for the videos added (library.toml: [notes] auto)")
    a.set_defaults(run=add)

    n = commands.add_parser(
        "notes", parents=[common], help="write study notes for transcribed videos with Claude",
        description=(
            "Write study notes for every transcribed video that has none, or whose transcript changed: an "
            "overview, sections with key points, key terms, examples and review questions, with times to "
            "check against the recording. Claude writes them through the [enrich] backend and model in "
            "library.toml. Each video gets <name>.notes.md in the course's notes folder (a session's go in "
            "its 'Study notes' folder), plus index.md for the course. A transcript .json outside a library "
            "gets <name>.notes.md next to it."
        ),
    )
    n.add_argument("targets", nargs="*", metavar="COURSE_OR_TRANSCRIPT",
                   help="course slugs (default: every course), or transcript .json files")
    n.add_argument("--redo", action="store_true",
                   help="write the notes again even when they are up to date (replaces edited notes files)")
    n.set_defaults(run=notes)

    b = commands.add_parser(
        "build-exam", parents=[common], help="build the exam index from the library (no AI)",
        description="Export every course, file and passage into exam/exam.sqlite, the index the exam app searches.",
    )
    b.set_defaults(run=build_exam)

    ses = commands.add_parser(
        "session", parents=[common], help="search one sitting: a folder of videos (and slides/notes)",
        description=(
            "Treat FOLDER as one sitting: add every video, PowerPoint, PDF and Word file in it to a "
            "library kept in FOLDER/.studybuddy, build its exam index, write study notes for the videos "
            "into FOLDER/Study notes, and open the search page. Run it again after adding files: only new "
            "files are processed. Reopen the search later with `studybuddy-exam FOLDER`."
        ),
    )
    ses.add_argument("folder", type=Path, metavar="FOLDER")
    ses.add_argument("--title", help="name for this sitting (default: the folder name)")
    ses.add_argument("--mode", choices=("accurate", "fast"), help="transcription mode (default: accurate)")
    ses.add_argument("--recursive", action="store_true", help="also add files in subfolders")
    ses.add_argument("--no-ocr", action="store_true", help="skip reading text in images and scanned pages")
    ses.add_argument("--no-convert", action="store_true", help="skip converting PowerPoint/Word to PDF")
    ses.add_argument("--no-notes", action="store_true", help="do not write study notes for the videos")
    ses.add_argument("--no-search", action="store_true", help="only process the files; do not open the search page")
    ses.add_argument("--port", type=int, help="port for the search page (default 8765)")
    ses.set_defaults(run=session_command)

    s = commands.add_parser("status", parents=[common], help="show what is in the library")
    s.add_argument("course", nargs="?", metavar="COURSE", help="list the files of one course")
    s.set_defaults(run=status)
