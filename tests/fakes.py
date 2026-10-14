"""Fake speech models shared by the CLI and ingestion tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from studybuddy.types import SAMPLE_RATE, ProgressFn, Segment

TEXTS = ("Hello there.", "General Kenobi.")


class FakeTranscriber:
    """Splits each section into one segment per text; raises ``fail`` after ``fail_after`` calls."""

    def __init__(
        self, mode: str, *, texts: tuple[str, ...], fail: BaseException | None, fail_after: int
    ) -> None:
        self.model_id = f"fake/{mode}"
        self.texts = texts
        self.fail = fail
        self.fail_after = fail_after
        self.loads = 0
        self.calls: list[dict[str, Any]] = []

    def config(self) -> dict:
        return {"fake": True}

    def load(self) -> None:
        self.loads += 1

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_s: float = 0.0,
        language: str | None = None,
        hotwords: list[str] | None = None,
        on_progress: ProgressFn | None = None,
    ) -> list[Segment]:
        if self.fail is not None and len(self.calls) >= self.fail_after:
            raise self.fail
        seconds = len(audio) / SAMPLE_RATE
        self.calls.append(
            {"offset_s": offset_s, "seconds": seconds, "language": language, "hotwords": hotwords}
        )
        if on_progress:
            on_progress(0.5)
        step = seconds / max(len(self.texts), 1)
        return [
            Segment(offset_s + i * step, offset_s + (i + 1) * step, text, language)
            for i, text in enumerate(self.texts)
        ]


class Models:
    """Replacement for ``make_transcriber``; configure the fakes it hands out, then inspect them."""

    def __init__(self) -> None:
        self.created: list[FakeTranscriber] = []
        self.texts = TEXTS
        self.fail: BaseException | None = None
        self.fail_after = 0

    def __call__(self, mode: str, **overrides: Any) -> FakeTranscriber:
        fake = FakeTranscriber(mode, texts=self.texts, fail=self.fail, fail_after=self.fail_after)
        self.created.append(fake)
        return fake

    @property
    def last(self) -> FakeTranscriber:
        return self.created[-1]
