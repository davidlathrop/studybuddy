from __future__ import annotations

from dataclasses import asdict

import studybuddy
from studybuddy.types import SAMPLE_RATE, Segment, Transcript


def test_sample_rate() -> None:
    assert SAMPLE_RATE == 16000


def test_segment_defaults_and_asdict() -> None:
    seg = Segment(start=1.5, end=3.25, text="hello")
    assert seg.language is None
    assert asdict(seg) == {"start": 1.5, "end": 3.25, "text": "hello", "language": None}


def test_transcript_round_trip() -> None:
    t = Transcript(
        source="lecture.mp4", model="m", mode="fast", language="English",
        duration_s=10.0, segments=[Segment(0.0, 5.0, "a"), Segment(5.0, 10.0, "b")],
    )
    data = asdict(t)
    rebuilt = Transcript(**{**data, "segments": [Segment(**s) for s in data["segments"]]})
    assert rebuilt == t


def test_package_reexports() -> None:
    assert studybuddy.Segment is Segment
    assert studybuddy.SAMPLE_RATE == SAMPLE_RATE
