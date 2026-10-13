"""Sectioning, resumable progress, and offset bookkeeping (plan §5.3–5.4).

A long recording is cut into ~10-minute sections at quiet points. Each section is
transcribed on its own, and its segments are appended to a JSONL *partial file* as
soon as it finishes, so an interrupted run can pick up where it stopped.

Partial file layout (one JSON object per line)::

    {"version": 1, "key": "<sha256 hex>", "settings": {...}}
    {"index": 0, "start_sample": 0, "end_sample": 9600000, "segments": [{"start": ..., ...}]}
    {"index": 1, ...}

The header's ``key`` covers the audio file's contents and every setting that changes
the output. Finished sections are reused only when the key matches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np

from studybuddy import audio
from studybuddy.types import SAMPLE_RATE, ProgressFn, Segment, Transcript

if TYPE_CHECKING:
    from studybuddy.transcribe import Transcriber

logger = logging.getLogger(__name__)

PARTIAL_VERSION = 1

_WINDOW_S = 0.1  # energy window used to find a quiet cut point
_MIN_PIECE_S = 1.0  # a trailing piece shorter than this is folded into the previous one
_QUIET_TOLERANCE = 0.01  # windows within 1% of the quietest count as equally quiet
_SILENCE_FLOOR = 1e-12  # mean-square energy treated as digital silence (about -120 dBFS)
_SOUND_RMS = 0.01  # about -40 dBFS; a quieter stretch is not expected to hold speech


# --------------------------------------------------------------------------- splitting


def split_at_quiet_points(
    audio: np.ndarray, sr: int, target_s: float, search_s: float = 5.0
) -> list[tuple[int, int]]:
    """Cut ``audio`` into contiguous ``(start_sample, end_sample)`` pieces of about ``target_s``.

    Each cut is placed at the quietest 100 ms window within ``search_s`` of its target
    (ties go to the quiet stretch nearest the target, cut at its middle). The pieces
    cover the array exactly: the first starts at 0, the last ends at ``len(audio)``, and
    each piece ends where the next begins. Pieces run up to ``target_s + search_s``, so
    audio up to that length is a single piece. A tail shorter than about 1 s is folded
    into the piece before it, and empty audio gives ``[]``.
    """
    n = len(audio)
    if target_s <= 0 or sr <= 0 or search_s < 0:
        raise ValueError(f"invalid split: sr={sr}, target_s={target_s}, search_s={search_s}")
    if n == 0:
        return []

    target = max(1, round(target_s * sr))
    search = min(round(search_s * sr), target // 2)  # pieces stay at least target/2 long
    win = max(1, round(_WINDOW_S * sr))
    min_tail = round(_MIN_PIECE_S * sr)

    bounds = [0]
    start = 0
    while n - start > target + search:
        lo = start + target - search
        hi = min(start + target + search, n - min_tail)
        if hi < lo:  # no room for a cut that leaves a long enough tail: keep it whole
            break
        start = _quietest_point(audio, lo, hi, win, start + target)
        bounds.append(start)
    bounds.append(n)
    return list(zip(bounds[:-1], bounds[1:]))


def _quietest_point(pcm: np.ndarray, lo: int, hi: int, win: int, target: int) -> int:
    """Return the sample in ``[lo, hi]`` centred on the quietest ``win``-sample window."""
    half = win // 2
    a = max(lo - half, 0)
    b = min(hi - half + win, len(pcm))
    region = np.asarray(pcm[a:b], dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(region * region)))

    cands = np.arange(lo, hi + 1)
    w0 = np.clip(cands - half, a, b) - a
    w1 = np.clip(cands - half + win, a, b) - a
    energy = (csum[w1] - csum[w0]) / np.maximum(w1 - w0, 1)

    threshold = max(energy.min() * (1 + _QUIET_TOLERANCE), _SILENCE_FLOOR)
    quiet = np.flatnonzero(energy <= threshold)
    nearest = quiet[np.argmin(np.abs(cands[quiet] - target))]
    # Widen the nearest quiet candidate to its run of consecutive quiet candidates and
    # cut in the middle, so a silent gap is split down its centre rather than at an edge.
    breaks = np.flatnonzero(np.diff(quiet) != 1)
    run_starts = np.concatenate(([quiet[0]], quiet[breaks + 1]))
    run_ends = np.concatenate((quiet[breaks], [quiet[-1]]))
    k = np.searchsorted(run_starts, nearest, side="right") - 1
    return int(cands[(run_starts[k] + run_ends[k]) // 2])


# --------------------------------------------------------------------------- resume key


def file_sha256(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def resume_settings(
    audio_sha256: str,
    transcriber: Transcriber,
    *,
    mode: str,
    section_s: float,
    language: str | None,
    hotwords: list[str] | None,
) -> dict[str, Any]:
    """Everything that changes a section's transcript; hashed into the partial file's key."""
    return {
        "audio_sha256": audio_sha256,
        "model_id": transcriber.model_id,
        "mode": mode,
        "section_s": float(section_s),
        "language": language,
        "hotwords": list(hotwords or []),
        "config": transcriber.config(),
    }


def resume_key(settings: dict[str, Any]) -> str:
    blob = json.dumps(
        settings, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- partial file


class _RestartPartial(Exception):
    """The partial file cannot be resumed from and has to be started over."""


def _segment_to_json(seg: Segment) -> dict[str, Any]:
    return {"start": seg.start, "end": seg.end, "text": seg.text, "language": seg.language}


def _segment_from_json(data: dict[str, Any]) -> Segment:
    language = data.get("language")
    if language is not None and not isinstance(language, str):
        raise TypeError(f"segment language must be a string or null, not {language!r}")
    if not isinstance(data["text"], str):
        raise TypeError(f"segment text must be a string, not {data['text']!r}")
    return Segment(
        start=float(data["start"]), end=float(data["end"]), text=data["text"], language=language
    )


def _section_record(index: int, piece: tuple[int, int], segments: list[Segment]) -> dict[str, Any]:
    return {
        "index": index,
        "start_sample": piece[0],
        "end_sample": piece[1],
        "segments": [_segment_to_json(s) for s in segments],
    }


def _parse_section(record: Any, pieces: list[tuple[int, int]]) -> tuple[int, list[Segment]]:
    if not isinstance(record, dict):
        raise _RestartPartial("a section line is not a JSON object")
    try:
        index = record["index"]
        if not isinstance(index, int) or not 0 <= index < len(pieces):
            raise _RestartPartial(f"section index {index!r} is out of range")
        if (record["start_sample"], record["end_sample"]) != pieces[index]:
            raise _RestartPartial(f"section {index} boundaries differ from the current split")
        return index, [_segment_from_json(s) for s in record["segments"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise _RestartPartial(f"malformed section line ({exc!r})") from exc


def _read_partial(path: Path, key: str, pieces: list[tuple[int, int]]) -> dict[int, list[Segment]]:
    """Return the finished sections stored in ``path``, or raise ``_RestartPartial``.

    A last line that does not parse is the trace of a crash mid-write and is dropped;
    anything else unexpected means the file cannot be trusted.
    """
    lines = [line for line in path.read_bytes().split(b"\n") if line.strip()]
    if not lines:
        raise _RestartPartial("the file is empty")
    try:
        header = json.loads(lines[0])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _RestartPartial("the header line is unreadable") from exc
    if not isinstance(header, dict) or header.get("version") != PARTIAL_VERSION:
        raise _RestartPartial("the header is missing or from another version")
    if header.get("key") != key:
        raise _RestartPartial("the audio or the settings changed since it was written")

    done: dict[int, list[Segment]] = {}
    body = lines[1:]
    for lineno, line in enumerate(body, start=2):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if lineno == len(body) + 1:
                logger.warning("%s: dropping an incomplete last line (interrupted write)", path)
                break
            raise _RestartPartial(f"line {lineno} is unreadable") from exc
        index, segments = _parse_section(record, pieces)
        done[index] = segments
    return done


def _load_partial(path: Path, key: str, pieces: list[tuple[int, int]]) -> dict[int, list[Segment]]:
    if not path.exists():
        return {}
    try:
        return _read_partial(path, key, pieces)
    except (_RestartPartial, OSError) as exc:
        logger.warning("%s: cannot resume (%s); starting over", path, exc)
        return {}


def _write_partial(
    path: Path,
    header: dict[str, Any],
    pieces: list[tuple[int, int]],
    done: dict[int, list[Segment]],
) -> None:
    """Atomically (re)write ``path`` holding ``header`` and the finished sections in ``done``."""
    lines = [header] + [_section_record(i, pieces[i], done[i]) for i in sorted(done)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for record in lines:
                fh.write(_json_line(record))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _append_line(fh: TextIO, record: dict[str, Any]) -> None:
    fh.write(_json_line(record))
    fh.flush()
    os.fsync(fh.fileno())


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"


# --------------------------------------------------------------------------- progress


class _Progress:
    """Forward fractions to ``on_progress`` clamped to 0..1 and never decreasing."""

    def __init__(self, on_progress: ProgressFn | None) -> None:
        self._fn = on_progress
        self._last = 0.0

    def __call__(self, fraction: float) -> None:
        if self._fn is None or math.isnan(fraction):
            return
        fraction = min(max(fraction, self._last), 1.0)
        self._last = fraction
        self._fn(fraction)


# --------------------------------------------------------------------------- transcription


def transcribe_file(
    audio_path: Path,
    transcriber: Transcriber,
    *,
    mode: str,
    section_s: float = 600.0,
    language: str | None = None,
    hotwords: list[str] | None = None,
    partial_path: Path | None = None,
    on_progress: ProgressFn | None = None,
    source: str | None = None,
) -> Transcript:
    """Transcribe ``audio_path`` section by section, resuming from ``partial_path`` if possible.

    Segment times in the result are absolute within the file. The partial file is left
    in place; the caller deletes it once the final outputs are safely written.
    """
    audio_path = Path(audio_path)
    sr = SAMPLE_RATE
    pcm = audio.load_audio(audio_path, sr)
    total = len(pcm)
    pieces = split_at_quiet_points(pcm, sr, section_s)

    done: dict[int, list[Segment]] = {}
    partial_fh: TextIO | None = None
    if partial_path is not None:
        partial_path = Path(partial_path)
        settings = resume_settings(
            file_sha256(audio_path), transcriber,
            mode=mode, section_s=section_s, language=language, hotwords=hotwords,
        )
        key = resume_key(settings)
        done = _load_partial(partial_path, key, pieces)
        if done:
            logger.info(
                "resuming %s: %d of %d sections already done",
                audio_path.name, len(done), len(pieces),
            )
        header = {"version": PARTIAL_VERSION, "key": key, "settings": settings}
        _write_partial(partial_path, header, pieces, done)
        partial_fh = open(partial_path, "a", encoding="utf-8")

    progress = _Progress(on_progress)
    finished = sum(pieces[i][1] - pieces[i][0] for i in done)
    try:
        if total:
            progress(finished / total)
        for index, (start, end) in enumerate(pieces):
            if index in done:
                continue
            base = finished

            def on_section_progress(
                fraction: float, base: int = base, length: int = end - start
            ) -> None:
                progress((base + min(max(fraction, 0.0), 1.0) * length) / total)

            t0 = time.perf_counter()
            segments = transcriber.transcribe(
                pcm[start:end],
                offset_s=start / sr,
                language=language,
                hotwords=hotwords,
                on_progress=on_section_progress,
            )
            segments = _clamp(segments, start / sr, end / sr)
            logger.info(
                "section %d/%d (%s-%s) done in %.1f s",
                index + 1, len(pieces), _hms(start / sr), _hms(end / sr), time.perf_counter() - t0,
            )
            _warn_if_missing_text(pcm, sr, index, start, end, segments)
            done[index] = segments
            if partial_fh is not None:
                _append_line(partial_fh, _section_record(index, (start, end), segments))
            finished += end - start
            progress(finished / total)
    finally:
        if partial_fh is not None:
            partial_fh.close()
    progress(1.0)

    segments = sorted(
        (s for i in sorted(done) for s in done[i] if s.text.strip()),
        key=lambda s: (s.start, s.end),
    )
    return Transcript(
        source=source if source is not None else audio_path.name,
        model=transcriber.model_id,
        mode=mode,
        language=_dominant_language(segments) or language,
        duration_s=total / sr,
        segments=segments,
    )


def _clamp(segments: Iterable[Segment], lo: float, hi: float) -> list[Segment]:
    """Keep segment times inside their section (e.g. a padded final chunk can overrun it)."""
    out = []
    for s in segments:
        start = min(max(s.start, lo), hi)
        end = min(max(s.end, start), hi)
        out.append(Segment(start=start, end=end, text=s.text, language=s.language))
    return out


def _warn_if_missing_text(
    pcm: np.ndarray, sr: int, index: int, start: int, end: int, segments: list[Segment]
) -> None:
    if not any(s.text.strip() for s in segments):
        if _has_sound(pcm[start:end]):
            logger.warning(
                "section %d (%s-%s) has sound but produced no text",
                index + 1, _hms(start / sr), _hms(end / sr),
            )
        return
    for s in segments:
        lo, hi = max(round(s.start * sr), start), min(round(s.end * sr), end)
        if not s.text.strip() and hi > lo and _has_sound(pcm[lo:hi]):
            logger.warning("chunk %s-%s has sound but came back empty", _hms(s.start), _hms(s.end))


def _has_sound(x: np.ndarray) -> bool:
    if len(x) == 0:
        return False
    x = np.asarray(x, dtype=np.float32)
    return math.sqrt(float(np.dot(x, x)) / len(x)) >= _SOUND_RMS


def _dominant_language(segments: list[Segment]) -> str | None:
    counts = Counter(s.language for s in segments if s.language)
    return counts.most_common(1)[0][0] if counts else None


def _hms(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"
