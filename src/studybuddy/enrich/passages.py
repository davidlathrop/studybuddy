"""Turn a transcript into numbered passages, render them for the model, and window them."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

MIN_PASSAGE_S = 45.0  # fast-mode sentences are merged until a passage is at least this long


@dataclass(frozen=True)
class Passage:
    id: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def passages_from_transcript(data: dict[str, Any], *, min_s: float = MIN_PASSAGE_S) -> list[Passage]:
    """Merge transcript segments (a ``write_json`` dict) into passages of at least ``min_s``.

    Accurate-mode segments are already ~60 s and pass through one to one. A short tail joins
    the previous passage instead of standing alone.
    """
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for seg in data["segments"]:
        if not seg["text"].strip():
            continue
        current.append(seg)
        if current[-1]["end"] - current[0]["start"] >= min_s:
            groups.append(current)
            current = []
    if current:
        if groups and current[-1]["end"] - current[0]["start"] < min_s / 2:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return [
        Passage(
            id=i,
            start=float(g[0]["start"]),
            end=float(g[-1]["end"]),
            text=" ".join(s["text"].strip() for s in g),
        )
        for i, g in enumerate(groups, start=1)
    ]


def clock(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_transcript(title: str, passages: Iterable[Passage]) -> str:
    """The shared, cached context: every passage tagged with its id and time range."""
    lines = [f"<source title={title!r}>"]
    lines += [f"[P{p.id} {clock(p.start)}-{clock(p.end)}] {p.text}" for p in passages]
    lines.append("</source>")
    return "\n".join(lines)


def windows(passages: Sequence[Passage], size: int) -> list[list[Passage]]:
    if size < 1:
        raise ValueError("window size must be >= 1")
    return [list(passages[i : i + size]) for i in range(0, len(passages), size)]
