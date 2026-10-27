"""``studybuddy-exam [FOLDER]``: start the exam search page on this computer only.

FOLDER is a library, or a folder of libraries such as ``~/StudyBuddy``, whose indexes are
then combined into one search (``collection.py``). The no-AI guard is installed before
anything else is imported.
"""

from studybuddy_exam import guard

guard.install()

import argparse  # noqa: E402  (imported after the guard on purpose)
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import webbrowser  # noqa: E402
from pathlib import Path  # noqa: E402

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
CHROME = Path("/Applications/Google Chrome.app")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studybuddy-exam",
        description="Search your StudyBuddy library during an exam: offline, on this computer only, no AI.",
    )
    parser.add_argument("library", nargs="?", type=Path, metavar="FOLDER",
                        help="a library, or a folder of libraries such as ~/StudyBuddy to search them all "
                        "(default: $STUDYBUDDY_LIBRARY, the folder you are in, or the default library)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port on 127.0.0.1 (default {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args(argv)

    from studybuddy_exam.collection import combine
    from studybuddy_exam.index_format import ExamIndexError, open_index
    from studybuddy_exam.locate import LibraryNotFound, find_library, index_path, libraries_in

    parts, left_out = [], []
    work = None  # the temporary folder of an index combined from several libraries
    try:
        roots = libraries_in(args.library) if args.library is not None else [find_library()]
        if not roots:
            raise LibraryNotFound(f"no StudyBuddy libraries in {args.library}")
        if len(roots) == 1:
            conn, info = open_index(index_path(roots[0]))
        else:
            folder = args.library.expanduser().absolute()
            work = Path(tempfile.mkdtemp(prefix="studybuddy-exam-"))
            combined = combine(folder, roots, work / "exam.sqlite")
            left_out = combined.left_out
            if not combined.parts:
                raise LibraryNotFound(f"none of the libraries in {folder} has anything to search; "
                                      f"bring their indexes up to date with: studybuddy serve {folder}")
            conn, info = open_index(combined.path, root=folder)
            parts = combined.parts
    except (LibraryNotFound, ExamIndexError) as exc:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        for line in left_out:
            print(f"studybuddy-exam: left out {line}", file=sys.stderr)
        print(f"studybuddy-exam: {exc}", file=sys.stderr)
        return 1

    import uvicorn

    from studybuddy_exam.server import create_app

    port = _free_port(args.port)
    app = create_app(conn, info, host=HOST, port=port, parts=parts)
    if work is not None:
        # The open connection goes on reading the deleted file, and nothing is left behind
        # however the server stops (uvicorn ends the process with the signal that stopped it).
        shutil.rmtree(work, ignore_errors=True)
    url = f"http://{HOST}:{port}/"
    print(f"StudyBuddy exam search for {info.meta.get('library_name') or info.library_root.name}", file=sys.stderr)
    if parts:
        print(f"  {len(parts)} libraries: {', '.join(p.name for p in parts)}", file=sys.stderr)
        for line in left_out:
            print(f"  left out {line}", file=sys.stderr)
        print(f"  {info.meta.get('passages', '?')} passages", file=sys.stderr)
    else:
        print(f"  {info.meta.get('passages', '?')} passages, index built {info.meta.get('built_at', '?')}",
              file=sys.stderr)
    print(f"  AI libraries loaded: {', '.join(guard.loaded_blocked_modules()) or 'none'}; network: loopback only",
          file=sys.stderr)
    print(f"  Open {url}  (Ctrl-C to stop)", file=sys.stderr)
    if not args.no_browser:
        threading.Timer(0.8, _open_browser, [url]).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning", access_log=False)
    return 0


def _free_port(start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, port))
            except OSError:
                continue
            return port
    raise SystemExit(f"studybuddy-exam: no free port between {start} and {start + 19}")


def _open_browser(url: str) -> None:
    if CHROME.exists():  # the browser the exam rehearsal is done in
        subprocess.run(["open", "-a", str(CHROME), url], check=False)
    else:
        webbrowser.open(url)


if __name__ == "__main__":
    sys.exit(main())
