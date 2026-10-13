"""Transcript writers: .txt, .srt, .json (plan §5.7).

Every writer replaces its target atomically, so a crash never leaves a half-written
transcript that looks finished.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable, Iterator
from pathlib import Path

from studybuddy.types import Segment, Transcript

FORMATS = ("txt", "srt", "json")


def write_txt(t: Transcript, path: Path) -> None:
    """One paragraph per non-empty segment, separated by blank lines."""
    paragraphs = [text for _, text in _texts(t.segments)]
    _write_atomic(Path(path), "\n\n".join(paragraphs) + "\n" if paragraphs else "")


def write_srt(t: Transcript, path: Path) -> None:
    """One cue per non-empty segment, numbered from 1."""
    cues = []
    for number, (seg, text) in enumerate(_texts(t.segments), start=1):
        start_ms = _to_ms(seg.start)
        end_ms = max(_to_ms(seg.end), start_ms + 1)  # players drop cues that do not advance
        cues.append(f"{number}\n{_srt_time(start_ms)} --> {_srt_time(end_ms)}\n{text}\n")
    _write_atomic(Path(path), "\n".join(cues))


def write_json(t: Transcript, path: Path) -> None:
    """Times are rounded to whole milliseconds (5.6000000000000005 -> 5.6), like the .srt."""
    data = {
        "source": t.source,
        "model": t.model,
        "mode": t.mode,
        "language": t.language,
        "duration_s": round(t.duration_s, 3),
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text, "language": s.language}
            for s in t.segments
        ],
    }
    _write_atomic(Path(path), json.dumps(data, ensure_ascii=False, indent=2) + "\n")


_WRITERS: dict[str, Callable[[Transcript, Path], None]] = {
    "txt": write_txt,
    "srt": write_srt,
    "json": write_json,
}


def write_outputs(t: Transcript, base: Path, formats: list[str]) -> list[Path]:
    """Write ``base.<fmt>`` for each format and return the paths in the order given.

    ``base`` has no extension (``out/lecture`` gives ``out/lecture.txt``); a dot already in
    its name is kept. All formats are checked before anything is written.
    """
    wanted = list(dict.fromkeys(f.strip().lower() for f in formats))
    unknown = [f for f in wanted if f not in _WRITERS]
    if unknown:
        raise ValueError(f"unknown output format(s) {unknown}; choose from {', '.join(FORMATS)}")
    if not wanted:
        raise ValueError(f"no output formats given; choose from {', '.join(FORMATS)}")

    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in wanted:
        path = base.with_name(f"{base.name}.{fmt}")
        _WRITERS[fmt](t, path)
        paths.append(path)
    return paths


def _texts(segments: list[Segment]) -> Iterator[tuple[Segment, str]]:
    """Yield segments that have text, with whitespace (including newlines) collapsed."""
    for seg in segments:
        text = " ".join(seg.text.split())
        if text:
            yield seg, text


def _to_ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


def _srt_time(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_atomic(path: Path, text: str) -> None:
    # O_EXCL with mode 0o666 gives the temp file the same permissions (after umask)
    # as a normally created file, unlike tempfile.mkstemp's 0o600.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(text.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
