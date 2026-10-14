"""Data shared by every ingester: what one extracted passage looks like."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

ORIGINS = ("speech", "text", "notes", "ocr")


class IngestError(Exception):
    """A file that cannot be added; reported without a traceback."""


@dataclass(frozen=True)
class PassageDraft:
    """One searchable piece of a source, with its location.

    Exactly the location fields that apply are set: times for video, ``slide_no`` for
    PowerPoint, ``page_no`` for PDF, ``heading_path`` (and ``page_no`` once mapped) for Word.
    ``origin`` says where the text came from: ``speech`` (transcript), ``text`` (the file's
    own text), ``notes`` (speaker notes) or ``ocr`` (read from an image).
    """

    text: str
    origin: str = "text"
    t_start_ms: int | None = None
    t_end_ms: int | None = None
    slide_no: int | None = None
    page_no: int | None = None
    heading_path: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, not {self.origin!r}")


_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """NFKC-normalize, unify line endings, collapse runs of spaces and blank lines, strip."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def word_count(text: str) -> int:
    return len(text.split())
