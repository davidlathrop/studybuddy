"""Cut long text into passages of about 150-300 words at natural boundaries (plan §5.2).

Boundaries are tried from strongest to weakest: blank lines (paragraphs), sentence ends,
line breaks, and finally plain word counts. Chunks are slices of the input, so nothing is
lost or reordered and the original line breaks survive.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

TARGET_WORDS = 250
MAX_WORDS = 350
MIN_TAIL_WORDS = 60  # a shorter last chunk is merged into the one before it

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
# Whitespace after ., ! or ? (optionally followed by a closing quote or bracket), unless the
# next word starts in lowercase, which is usually an abbreviation such as "e.g. the".
_SENTENCE_BREAK = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]’”]))\s+(?![a-z])")
_LINE_BREAK = re.compile(r"\s*\n\s*")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    words: int


def chunk_text(
    text: str,
    *,
    target: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
    min_tail: int = MIN_TAIL_WORDS,
) -> list[str]:
    """``text`` split into chunks of about ``target`` words, each at most ``max_words``.

    Text of at most ``max_words`` words stays whole. Chunk sizes are balanced so the last one
    is not a stub; a tail under ``min_tail`` words is merged back when that still fits.
    """
    text = text.strip()
    total = len(_WORD.findall(text))
    if total == 0:
        return []
    if total <= max_words:
        return [text]
    spans = _units(text, 0, len(text), max_words, target)
    ideal = total / math.ceil(total / target)
    groups: list[list[_Span]] = [[]]
    size = 0
    for span in spans:
        grown = size + span.words
        if groups[-1] and grown > ideal and (grown > max_words or grown - ideal >= ideal - size):
            groups.append([])
            size = 0
        groups[-1].append(span)
        size += span.words
    if len(groups) > 1:
        tail, prev = _words(groups[-1]), _words(groups[-2])
        if tail < min_tail and tail + prev <= max_words:
            groups[-2].extend(groups.pop())
    return [text[group[0].start : group[-1].end].strip() for group in groups]


def _units(text: str, start: int, end: int, max_words: int, target: int, level: int = 0) -> list[_Span]:
    """Spans of at most ``max_words`` words, split only as finely as needed."""
    pattern = (_PARAGRAPH_BREAK, _SENTENCE_BREAK, _LINE_BREAK)[level] if level < 3 else None
    if pattern is None:
        words = list(_WORD.finditer(text, start, end))
        return [
            _Span(words[i].start(), words[min(i + target, len(words)) - 1].end(), min(target, len(words) - i))
            for i in range(0, len(words), target)
        ]
    out: list[_Span] = []
    for piece_start, piece_end in _split(text, start, end, pattern):
        words = len(_WORD.findall(text, piece_start, piece_end))
        if words == 0:
            continue
        if words <= max_words:
            out.append(_Span(piece_start, piece_end, words))
        else:
            out.extend(_units(text, piece_start, piece_end, max_words, target, level + 1))
    return out


def _split(text: str, start: int, end: int, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    pieces = []
    pos = start
    for match in pattern.finditer(text, start, end):
        if match.start() > pos:
            pieces.append((pos, match.start()))
        pos = match.end()
    if pos < end:
        pieces.append((pos, end))
    return pieces


def _words(group: list[_Span]) -> int:
    return sum(span.words for span in group)
