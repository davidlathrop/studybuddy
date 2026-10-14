"""What an ingester hands back: passages plus the files it wrote into the staging folder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from studybuddy.ingest.model import PassageDraft


@dataclass
class Staged:
    passages: list[PassageDraft]
    # role -> file in the staging folder. "stored": the library's copy of a document, or a
    # video's transcript JSON; "view": what the exam app opens (exam-copy video or PDF);
    # "extra": anything else kept beside them (a video's plain-text transcript).
    files: dict[str, Path]
    duration_s: float | None
    page_count: int | None
    extractor: str
    warnings: list[str] = field(default_factory=list)
