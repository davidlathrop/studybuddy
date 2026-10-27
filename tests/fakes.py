"""Fake speech models and a fake Claude, shared by the CLI and ingestion tests."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from studybuddy.enrich.backends import BackendResult, Usage
from studybuddy.enrich.schemas import (
    Announcement,
    KeyPoint,
    KeyTerm,
    NoteItem,
    NotesOutline,
    NotesSectionPlan,
    NotesWindow,
    ReviewQuestion,
    SectionNotes,
)
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


class FakeClaude:
    """Stands in for an enrichment backend: answers the study-notes calls from what it is sent.

    The outline makes one section per passage; each section's notes quote the start of its
    passage. ``fail`` is raised on the call numbered ``fail_on`` (from 1), and ``skip``
    lists sections a sections call leaves out.
    """

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail: BaseException | None = None
        self.fail_on = 1
        self.skip: set[int] = set()

    def call(self, *, model: str, system: str, context: str, prompt: str, output: type,
             effort: str = "medium", max_tokens: int = 32000) -> BackendResult:
        self.calls.append({"model": model, "system": system, "context": context, "prompt": prompt,
                           "output": output.__name__, "effort": effort})
        if self.fail is not None and len(self.calls) >= self.fail_on:
            raise self.fail
        passages = {int(i): text for i, text in re.findall(r"^\[P(\d+) [^\]]+\] (.*)$", context, re.M)}
        if output is NotesOutline:
            data: Any = NotesOutline(
                title="Notes title", overview="What the lecture covers.",
                sections=[NotesSectionPlan(title=f"Topic {i}", first_passage_id=i, last_passage_id=i) for i in passages],
                announcements=[Announcement(text="The quiz is on Friday.", passage_id=1)],
            )
        else:
            first, last = map(int, re.search(r"sections S(\d+)-S(\d+)", prompt).groups())
            spans = {int(n): int(p) for n, p in re.findall(r"^- S(\d+) \(P(\d+)-P\d+\)", prompt, re.M)}
            data = NotesWindow(sections=[
                SectionNotes(
                    section=n, summary=f"Summary {n}.",
                    key_points=[KeyPoint(point=f"Point {n}.", details=["A detail."], passage_id=spans[n])],
                    key_terms=[KeyTerm(term=f"Term {n}", definition="What it means.", passage_id=spans[n],
                                       quote=" ".join(passages[spans[n]].split()[:2]))],
                    examples=[NoteItem(text="An example.", passage_id=spans[n])], emphasis=[],
                    review_questions=[ReviewQuestion(question=f"Question {n}?", answer=f"Answer {n}.",
                                                     passage_id=spans[n])],
                )
                for n in range(first, last + 1) if n not in self.skip
            ])
        return BackendResult(data=data, model=model, usage=Usage(input_tokens=10, output_tokens=20),
                             cost_usd=0.01, seconds=0.1, attempts=1, meta={"five_hour_utilization": 0.25})
