"""`studybuddy transcribe ...` command line (plan §4.2).

    studybuddy transcribe lecture.mp4 [--mode accurate|fast] [--language English]
        [--formats txt,srt,json] [--hotwords-file terms.txt] [--out-dir DIR] [--no-notes]

Writes ``lecture.mp3`` and the transcript files next to the input (or into ``--out-dir``),
then has Claude write study notes from the transcript to ``lecture.notes.md`` (and keeps
``lecture.json``, which the notes are made from), unless ``--no-notes``.
While a run is in progress, finished sections are kept in ``lecture.partial.jsonl``; running
the same command again after an interruption skips them. Progress and messages go to
stderr, and the paths of the files written go to stdout.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import NoReturn, TextIO

from studybuddy import __version__, audio, commands, output, study_notes, transcribe
from studybuddy.enrich.backends import EnrichmentError
from studybuddy.ingest.model import IngestError
from studybuddy.library import LibraryError
from studybuddy.transcription import MODEL_NAMES, transcribe_media

log = logging.getLogger(__name__)

EXIT_ERROR = 1
EXIT_INTERRUPTED = 130  # 128 + SIGINT, the status shells report for Ctrl-C


class CliError(Exception):
    """A problem with the command line or its inputs, reported without a traceback."""


# --------------------------------------------------------------------------- arguments


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # argparse would exit with status 2; studybuddy reports every error with status 1.
        self.print_usage(sys.stderr)
        raise CliError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="studybuddy",
        description=(
            "Transcribe lecture videos locally on Apple silicon, and build a semester library of "
            "videos, slides, PDFs and Word files to search during an open-book exam."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    common = _Parser(add_help=False)
    common.add_argument(
        "--library", type=Path, metavar="PATH",
        help="library folder (default: $STUDYBUDDY_LIBRARY, the folder you are in, or the default library)",
    )
    _add_chatter(common)
    commands.register(subcommands, common)

    cmd = subcommands.add_parser(
        "transcribe",
        help="extract a video's audio as MP3 and transcribe it",
        description=(
            "Extract the first audio track of INPUT to <name>.mp3 and transcribe it to "
            "<name>.txt (and optionally .srt / .json), then have Claude write study notes "
            "from the transcript to <name>.notes.md (also keeping <name>.json) unless "
            "--no-notes. Re-running an interrupted command resumes it."
        ),
    )
    cmd.add_argument("input", type=Path, metavar="INPUT", help="video or audio file (MP4, MOV, MKV, M4A, MP3, ...)")
    cmd.add_argument(
        "--mode", choices=transcribe.MODES, default="accurate",
        help=f"accurate: {MODEL_NAMES['accurate']} (default); fast: {MODEL_NAMES['fast']}, "
        "several times faster",
    )
    cmd.add_argument(
        "--language", metavar="L",
        help="spoken language as a name or code (English, en, fr, pt-BR, ...); "
        "default: detect it. Accurate mode only",
    )
    cmd.add_argument(
        "--formats", type=parse_formats, default=["txt"], metavar="LIST",
        help=f"comma-separated transcript formats from {','.join(output.FORMATS)} (default: txt)",
    )
    cmd.add_argument(
        "--hotwords-file", type=Path, metavar="F",
        help="names and terms to favour, one per line (blank lines and # comments are "
        "ignored). Accurate mode only",
    )
    cmd.add_argument(
        "--out-dir", type=Path, metavar="D", help="where to write the outputs (default: next to INPUT)"
    )
    cmd.add_argument(
        "--no-notes", action="store_true",
        help="do not write study notes (<name>.notes.md, written by Claude from the transcript); "
        "nothing then leaves this Mac",
    )
    _add_chatter(cmd)
    cmd.set_defaults(run=_transcribe_command)
    return parser


def _add_chatter(parser: argparse.ArgumentParser) -> None:
    chatter = parser.add_mutually_exclusive_group()
    chatter.add_argument("-v", "--verbose", action="store_true", help="also show debug messages")
    chatter.add_argument(
        "-q", "--quiet", action="store_true", help="show only warnings and errors, no progress"
    )


def parse_formats(value: str) -> list[str]:
    """``"txt, SRT,json"`` -> ``["txt", "srt", "json"]`` (lowercased, de-duplicated, in order)."""
    formats = list(dict.fromkeys(f.strip().lower() for f in value.split(",") if f.strip()))
    choices = ", ".join(output.FORMATS)
    if not formats:
        raise argparse.ArgumentTypeError(f"no formats given; choose from {choices}")
    if unknown := [f for f in formats if f not in output.FORMATS]:
        raise argparse.ArgumentTypeError(f"unknown format(s) {', '.join(unknown)}; choose from {choices}")
    return formats


def read_hotwords(path: Path) -> list[str]:
    """Terms from a hotwords file: one per line; blank lines and lines starting with # are skipped."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise CliError(f"cannot read hotwords file {path}: it is not UTF-8 text") from None
    except OSError as exc:
        raise CliError(f"cannot read hotwords file {path}: {exc.strerror or exc}") from None
    terms = (" ".join(line.split()) for line in text.splitlines())
    return list(dict.fromkeys(t for t in terms if t and not t.startswith("#")))


# --------------------------------------------------------------------------- the command


def _transcribe_command(args: argparse.Namespace, display: _Display) -> int:
    src: Path = args.input.expanduser()
    hotwords = read_hotwords(args.hotwords_file.expanduser()) if args.hotwords_file else []
    out_dir = args.out_dir.expanduser() if args.out_dir else src.parent
    # Notes are written from the .json, so a failed or interrupted notes step can be redone
    # with `studybuddy notes <name>.json` instead of transcribing again.
    notes = not args.no_notes
    formats = args.formats + (["json"] if notes and "json" not in args.formats else [])
    result = transcribe_media(
        src, out_dir, mode=args.mode, formats=formats, language=args.language,
        hotwords=hotwords, progress=display,
    )
    for path in (result.mp3, *result.paths):
        print(path, flush=True)
    if notes and result.transcript.segments:
        # As after `studybuddy add`, a notes problem is reported but doesn't fail the transcription.
        try:
            print(study_notes.notes_for_transcript(result.paths[formats.index("json")], progress=display), flush=True)
        except EnrichmentError as exc:
            log.error("%s (--no-notes skips them)", exc)
    return 0


# --------------------------------------------------------------------------- progress


class _Display:
    """Shows the progress of one stage at a time on ``stream`` (stderr).

    On a terminal the progress is a single line redrawn in place with ``\\r``. Otherwise
    (a pipe or a log file) a plain line is printed for every 10% step, at most one per 5 s,
    and at least every 30 s. Messages, including log records, go through :meth:`message`
    so they never land inside the progress line.
    """

    TICK_S = 1.0  # redraw at least this often so the elapsed time keeps moving
    PLAIN_STEP = 10  # percent
    PLAIN_MIN_GAP_S = 5.0  # Qwen3 reports a whole section's chunks at once; print one line
    PLAIN_EVERY_S = 30.0

    def __init__(
        self,
        stream: TextIO,
        *,
        show_progress: bool = True,
        interactive: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = stream
        self._show = show_progress
        self._interactive = _isatty(stream) if interactive is None else interactive
        self._clock = clock
        self._lock = threading.RLock()
        self._stage: str | None = None
        self._started = 0.0
        self._fraction = 0.0
        self._first: tuple[float, float] | None = None  # (time, fraction) of the first update
        self._drawn = ""  # what the terminal line shows now
        self._printed: tuple[float, int] = (0.0, -10)  # plain mode: (time, percent) of the last line
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self._show:
            yield
            return
        self._begin(name)
        try:
            yield
        except BaseException:
            self._end(finished=False)
            raise
        self._end(finished=True)

    def update(self, fraction: float) -> None:
        if math.isnan(fraction):
            return
        with self._lock:
            if self._stage is None:
                return
            if self._first is None:
                self._first = (self._clock(), fraction)
            self._fraction = min(max(fraction, self._fraction), 1.0)
            self._refresh()

    def message(self, text: str) -> None:
        with self._lock:
            self._erase()
            self._stream.write(text + "\n")
            if self._stage is not None and self._interactive:
                self._draw(self._status())
            self._stream.flush()

    def close(self) -> None:
        self._stop_ticker()

    def _begin(self, name: str) -> None:
        with self._lock:
            self._stage, self._started, self._fraction = name, self._clock(), 0.0
            self._first, self._printed = None, (self._started, -10)
            if self._interactive:
                self._draw(self._status())
        self._stop.clear()
        self._ticker = threading.Thread(target=self._tick, name="studybuddy-progress", daemon=True)
        self._ticker.start()

    def _end(self, *, finished: bool) -> None:
        self._stop_ticker()
        with self._lock:
            done = f"{self._stage:<10} 100%  done in {_clock(self._clock() - self._started)}"
            if self._interactive:
                if finished:
                    self._draw(done)
                if self._drawn:
                    self._stream.write("\n")  # keep the last state visible; move below it
                    self._drawn = ""
            elif finished:
                self._stream.write(done + "\n")
            self._stream.flush()
            self._stage = None

    def _tick(self) -> None:
        while not self._stop.wait(self.TICK_S):
            with self._lock:
                if self._stage is not None:
                    self._refresh()

    def _stop_ticker(self) -> None:
        self._stop.set()
        if self._ticker is not None and self._ticker is not threading.current_thread():
            self._ticker.join(timeout=2.0)
        self._ticker = None

    def _refresh(self) -> None:
        if self._interactive:
            self._draw(self._status())
            return
        now, percent = self._clock(), _percent(self._fraction)
        last_time, last_percent = self._printed
        since = now - last_time
        stepped = percent // self.PLAIN_STEP > last_percent // self.PLAIN_STEP
        due = (
            last_percent < 0  # the stage's first line
            or since >= self.PLAIN_EVERY_S
            or (stepped and since >= self.PLAIN_MIN_GAP_S)
        )
        if due and percent < 100:  # 100% is left to the "done" line
            self._stream.write(self._status() + "\n")
            self._stream.flush()
            self._printed = (now, percent)

    def _status(self) -> str:
        now = self._clock()
        text = f"{self._stage:<10} {_percent(self._fraction):3d}%  {_clock(now - self._started)} elapsed"
        if (eta := self._eta(now)) is not None:
            text += f", ~{_clock(eta)} left"
        return text

    def _eta(self, now: float) -> float | None:
        if self._first is None:
            return None
        first_time, first_fraction = self._first
        done, spent = self._fraction - first_fraction, now - first_time
        if done < 0.02 or spent < 3.0 or self._fraction >= 1.0:
            return None
        return spent * (1.0 - self._fraction) / done

    def _draw(self, text: str) -> None:
        if text != self._drawn:
            self._stream.write("\r" + text.ljust(len(self._drawn)))
            self._stream.flush()
            self._drawn = text

    def _erase(self) -> None:
        if self._drawn:
            self._stream.write("\r" + " " * len(self._drawn) + "\r")
            self._drawn = ""


def _percent(fraction: float) -> int:
    # The epsilon keeps 0.29 at 29% rather than floor(28.999999999999996).
    return min(100, math.floor(fraction * 100 + 1e-6))


def _isatty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def _clock(seconds: float) -> str:
    """``75`` -> ``"1:15"``, ``3725`` -> ``"1:02:05"``."""
    hours, rest = divmod(max(0, round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# --------------------------------------------------------------------------- logging


class _DisplayHandler(logging.Handler):
    def __init__(self, display: _Display) -> None:
        super().__init__()
        self._display = display

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._display.message(self.format(record))
        except Exception:
            self.handleError(record)


class _Formatter(logging.Formatter):
    """``warning: ...`` for problems, the bare message for notes, a logger tag for debug."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {message}"
        if record.levelno >= logging.INFO:
            return message
        return f"debug [{record.name}]: {message}"


@contextmanager
def _logging_to(display: _Display, level: int) -> Iterator[None]:
    """Route log records (and Python warnings) through ``display`` while the command runs.

    ``level`` applies to studybuddy's own loggers; other libraries show warnings and up.
    """
    handler = _DisplayHandler(display)
    handler.setFormatter(_Formatter())
    root, package = logging.getLogger(), logging.getLogger("studybuddy")
    saved_level = package.level
    root.addHandler(handler)
    package.setLevel(level)
    logging.captureWarnings(True)
    try:
        yield
    finally:
        logging.captureWarnings(False)
        package.setLevel(saved_level)
        root.removeHandler(handler)


# --------------------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return its exit status.

    Run as the program (``argv`` is None), Ctrl-C does not return 130: once the resume
    message is printed, KeyboardInterrupt goes on up, without a traceback. Python then shuts
    down cleanly and ends the process with SIGINT, which is how a calling shell tells that
    the user pressed Ctrl-C, so a script or a ``for`` loop running studybuddy stops too.
    """
    try:
        status = _main(argv)
    except KeyboardInterrupt:  # e.g. while parsing the arguments
        status = EXIT_INTERRUPTED
    if status == EXIT_INTERRUPTED and argv is None:
        sys.excepthook = _silent_on_interrupt(sys.excepthook)
        raise KeyboardInterrupt
    return status


def _silent_on_interrupt(hook: Callable[..., object]) -> Callable[..., None]:
    """``sys.excepthook`` that prints nothing for KeyboardInterrupt (already reported)."""

    def excepthook(kind: type[BaseException], value: BaseException, tb: TracebackType | None) -> None:
        if not issubclass(kind, KeyboardInterrupt):
            hook(kind, value, tb)

    return excepthook


def _main(argv: list[str] | None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except CliError as exc:
        print(f"studybuddy: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except SystemExit as exc:  # --help and --version print, then exit
        return exc.code if isinstance(exc.code, int) else 0

    display = _Display(sys.stderr, show_progress=not args.quiet)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    with _logging_to(display, level):
        try:
            return args.run(args, display)
        except KeyboardInterrupt as exc:
            if partial := getattr(exc, "partial", None):
                display.message(
                    f"Interrupted. Finished sections are saved in {partial}; "
                    "run the same command again to resume."
                )
            elif note := getattr(exc, "note", None):
                display.message(note)
            else:
                display.message("Interrupted before any transcription was saved.")
            return EXIT_INTERRUPTED
        except (CliError, audio.AudioError, LibraryError, IngestError, EnrichmentError, ValueError, OSError) as exc:
            log.debug("error details", exc_info=True)
            display.message(f"studybuddy: error: {exc}")
            return EXIT_ERROR
        finally:
            display.close()


if __name__ == "__main__":
    sys.exit(main())
