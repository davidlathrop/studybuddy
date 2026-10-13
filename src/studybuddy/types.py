"""Core data types shared by every studybuddy module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

SAMPLE_RATE = 16000
"""Sample rate (Hz) the speech models expect; audio is decoded to mono at this rate."""

ProgressFn = Callable[[float], None]
"""Progress callback receiving a completed fraction in 0.0..1.0."""


@dataclass
class Segment:
    """A piece of transcript. ``start``/``end`` are seconds, absolute within the source file."""

    start: float
    end: float
    text: str
    language: str | None = None


@dataclass
class Transcript:
    source: str
    model: str
    mode: str
    language: str | None
    duration_s: float
    segments: list[Segment]
