"""Check that AI-quoted text really occurs in its source passage (plan §6.6, constraint C5).

Tried in order: exact substring, substring after normalizing (NFKC, lowercase, punctuation
and whitespace collapsed), then a fuzzy partial match for small transcription-level
differences. Anything that fails is never shown at exam time.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 90.0
MIN_FUZZY_CHARS = 12  # short quotes match almost anything fuzzily

_NON_WORD = re.compile(r"[^\w]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return _NON_WORD.sub(" ", text).strip()


@dataclass(frozen=True)
class QuoteCheck:
    verified: bool
    method: str | None  # "exact" | "normalized" | "fuzzy" | None
    score: float


def check_quote(quote: str, source: str, *, threshold: float = FUZZY_THRESHOLD) -> QuoteCheck:
    if not quote.strip():
        return QuoteCheck(False, None, 0.0)
    if quote in source:
        return QuoteCheck(True, "exact", 100.0)
    q, s = normalize(quote), normalize(source)
    if q and q in s:
        return QuoteCheck(True, "normalized", 100.0)
    score = fuzz.partial_ratio(q, s) if q else 0.0
    if score >= threshold and len(q) >= MIN_FUZZY_CHARS:
        return QuoteCheck(True, "fuzzy", score)
    return QuoteCheck(False, None, score)


def names_appear(names: list[str], *texts: str) -> bool:
    """True when any of ``names`` occurs (normalized, as whole words) in any of ``texts``."""
    haystack = " " + " ".join(normalize(t) for t in texts) + " "
    return any(f" {n} " in haystack for n in (normalize(x) for x in names) if n)
