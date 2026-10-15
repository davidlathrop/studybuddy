"""Turning what the student types into safe FTS5 queries (plan §7.2). No AI: plain rules.

Every word or "quoted phrase" becomes a quoted FTS5 string, so input such as ``AND``,
``c++``, ``x:y``, ``NEAR(`` or ``-x`` is always searched as text, never parsed as query
syntax. Common words (the, what, is, ...) are dropped unless the question has nothing else.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each either else ever few for
    from further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me might more most must my myself neither no nor not now of off on once only or
    other our ours ourselves out over own same shall she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up upon us
    very was we were what when where which while who whom whose why will with within without would
    you your yours yourself yourselves
    describe define definition explain example examples mean means meaning tell show give list
    """.split()
)

_PHRASE = re.compile(r'"([^"]*)"')
_WORD = re.compile(r"[^\W_][\w'’-]*", re.UNICODE)


@dataclass(frozen=True)
class Query:
    text: str
    words: tuple[str, ...]  # search words, lowercased, in order, without duplicates
    phrases: tuple[str, ...]  # "quoted phrases", lowercased

    @property
    def empty(self) -> bool:
        return not self.words and not self.phrases

    @property
    def parts(self) -> tuple[str, ...]:
        return self.phrases + self.words

    @property
    def highlight_terms(self) -> list[str]:
        """Words for the PDF viewer to highlight."""
        return list(dict.fromkeys([*self.words, *(w for p in self.phrases for w in p.split())]))


def parse(text: str) -> Query:
    text = unicodedata.normalize("NFKC", text).replace("’", "'").strip()
    phrases = [" ".join(_words(p)) for p in _PHRASE.findall(text)]
    rest = _PHRASE.sub(" ", text)
    words = _words(rest)
    content = [w for w in words if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]
    if not content and not phrases:
        # One common word may be meant ("IT", "will"); several are just filler.
        lone = {w for w in words if len(w) > 1}
        content = list(lone) if len(lone) == 1 else []
    return Query(text, tuple(dict.fromkeys(content)), tuple(dict.fromkeys(p for p in phrases if p)))


def _words(text: str) -> list[str]:
    return [w.strip("'-").lower() for w in _WORD.findall(text) if w.strip("'-")]


def quote(term: str) -> str:
    """An FTS5 string literal: double quotes, with embedded quotes doubled."""
    return '"' + term.replace('"', '""') + '"'


def match_all(query: Query) -> str | None:
    """Every word and phrase must appear (None when there is only one part)."""
    if len(query.parts) < 2:
        return None
    return " AND ".join(quote(p) for p in query.parts)


def match_any(query: Query) -> str | None:
    if query.empty:
        return None
    return " OR ".join(quote(p) for p in query.parts)


def match_partial(query: Query) -> str | None:
    """For the trigram index: each word of 3+ characters as a substring."""
    terms = [w for w in query.highlight_terms if len(w) >= 3]
    return " OR ".join(quote(t) for t in terms) or None


def replace_words(text: str, replacements: dict[str, str]) -> str:
    """``text`` with whole words swapped (case-insensitively), e.g. for a "did you mean" query."""
    if not replacements:
        return text
    pattern = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, replacements)) + r")(?![\w])", re.IGNORECASE)
    return pattern.sub(lambda m: replacements[m.group(1).lower()], text)
