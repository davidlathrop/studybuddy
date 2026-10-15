"""``studybuddy-exam [LIBRARY]``: start the exam search page on this computer only.

The no-AI guard is installed before anything else is imported.
"""

from studybuddy_exam import guard

guard.install()

import argparse  # noqa: E402  (imported after the guard on purpose)
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
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
    parser.add_argument("library", nargs="?", type=Path, metavar="LIBRARY",
                        help="library folder (default: $STUDYBUDDY_LIBRARY, the folder you are in, or the default)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port on 127.0.0.1 (default {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args(argv)

    from studybuddy_exam.index_format import ExamIndexError, open_index
    from studybuddy_exam.locate import LibraryNotFound, find_library, index_path

    try:
        root = find_library(args.library)
        conn, info = open_index(index_path(root))
    except (LibraryNotFound, ExamIndexError) as exc:
        print(f"studybuddy-exam: {exc}", file=sys.stderr)
        return 1

    import uvicorn

    from studybuddy_exam.server import create_app

    port = _free_port(args.port)
    app = create_app(conn, info, host=HOST, port=port)
    url = f"http://{HOST}:{port}/"
    print(f"StudyBuddy exam search for {info.meta.get('library_name') or root.name}", file=sys.stderr)
    print(f"  {info.meta.get('passages', '?')} passages, index built {info.meta.get('built_at', '?')}", file=sys.stderr)
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
