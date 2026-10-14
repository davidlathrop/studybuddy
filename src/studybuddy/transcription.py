"""Transcribe one media file into a folder: the flow shared by ``studybuddy transcribe`` and
``studybuddy add`` (plan §5.1).

Extracts the audio to ``<stem>.mp3``, transcribes it section by section (resumable through
``<stem>.partial.jsonl``) and writes the requested transcript formats next to it.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from studybuddy import audio, output, pipeline, transcribe
from studybuddy.types import Transcript

log = logging.getLogger(__name__)

MODEL_NAMES = {"accurate": "Qwen3-ASR 1.7B", "fast": "Parakeet TDT 0.6B v3"}


class Progress(Protocol):
    """What the flow needs from a progress display (``cli._Display`` provides it)."""

    def stage(self, name: str) -> AbstractContextManager[None]: ...

    def update(self, fraction: float) -> None: ...


class NoProgress:
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        yield

    def update(self, fraction: float) -> None:
        pass


class TranscriptionInterrupted(KeyboardInterrupt):
    """Ctrl-C, carrying the partial file that holds the finished sections (if there is one)."""

    def __init__(self, partial: Path | None) -> None:
        super().__init__()
        self.partial = partial


@dataclass
class TranscriptionResult:
    transcript: Transcript
    mp3: Path
    paths: list[Path]  # the transcript files written, in the order of ``formats``
    elapsed_s: float


def resolve_language(language: str | None, mode: str) -> str | None:
    """The language to pass on: Qwen3-ASR's name for it in accurate mode, None in fast mode.

    Raises ValueError for a language accurate mode does not support.
    """
    if mode == "accurate":
        return transcribe.normalize_language(language)
    if language is not None and language.strip().lower() not in ("", "auto"):
        log.warning("fast mode (Parakeet) detects the language itself; ignoring --language %s", language)
    return None


def transcribe_media(
    src: Path,
    out_dir: Path,
    *,
    mode: str,
    formats: list[str],
    language: str | None = None,
    hotwords: list[str] | None = None,
    info: audio.MediaInfo | None = None,
    progress: Progress | None = None,
) -> TranscriptionResult:
    """Transcribe ``src`` into ``out_dir`` and return the transcript and the files written.

    ``language`` is passed through :func:`resolve_language`. On Ctrl-C this raises
    :class:`TranscriptionInterrupted`; running it again resumes from the finished sections.
    """
    progress = progress or NoProgress()
    language = resolve_language(language, mode)
    if hotwords and mode == "fast":
        log.warning("fast mode (Parakeet) does not support hotwords; ignoring them")
        hotwords = []
    info = info or audio.probe(src)
    mp3 = out_dir / f"{src.stem}.mp3"
    partial = out_dir / f"{src.stem}.partial.jsonl"
    length = format_clock(info.duration_s) if info.duration_s else "unknown length"
    log.info("%s: %s of audio; %s mode (%s)", src.name, length, mode, MODEL_NAMES[mode])
    if info.audio_streams > 1:
        log.info("%s has %d audio tracks; using the first", src.name, info.audio_streams)

    started = time.monotonic()
    try:
        _prepare_mp3(src, mp3, info, progress)
        transcriber = transcribe.make_transcriber(mode)
        if callable(load := getattr(transcriber, "load", None)):
            log.info("Loading %s", transcriber.model_id)
            load()
        with progress.stage("transcribe"):
            result = pipeline.transcribe_file(
                mp3, transcriber, mode=mode, language=language, hotwords=hotwords or None,
                partial_path=partial, on_progress=progress.update, source=src.name,
            )
        paths = output.write_outputs(result, out_dir / src.stem, formats)
    except KeyboardInterrupt:
        raise TranscriptionInterrupted(partial if partial.exists() else None) from None
    partial.unlink(missing_ok=True)  # only once every output is safely written
    elapsed = time.monotonic() - started
    _summarize(result, elapsed)
    return TranscriptionResult(result, mp3, paths, elapsed)


def _prepare_mp3(src: Path, mp3: Path, info: audio.MediaInfo, progress: Progress) -> None:
    if _same_file(src, mp3):
        log.info("%s is already the MP3; nothing to extract", src.name)
        return
    # Extract even when resuming. The resume key hashes the MP3 and extraction is
    # deterministic, so the same input resumes, while a different recording that happens
    # to share the name (or a replaced video) changes the key and starts over.
    log.info("Extracting the audio to %s", mp3)
    with progress.stage("extract"):
        audio.extract_mp3(src, mp3, duration_s=info.duration_s, on_progress=progress.update)


def _summarize(result: Transcript, elapsed: float) -> None:
    if not result.segments:
        log.warning("no speech was recognized in %s", result.source)
    speed = f" ({result.duration_s / elapsed:.0f}x real time)" if elapsed > 0 and result.duration_s else ""
    language = f"; language: {result.language}" if result.language else ""
    count = len(result.segments)
    log.info("Done in %s%s: %d segment%s%s", format_clock(elapsed), speed, count, "" if count == 1 else "s", language)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def format_clock(seconds: float) -> str:
    """``75`` -> ``"1:15"``, ``3725`` -> ``"1:02:05"``."""
    hours, rest = divmod(max(0, round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
