"""Searching the exam index (plan §7.2): BM25 ranking, filters, snippets, "did you mean".

Results come in three tiers, each ranked by BM25 within itself:

1. ``all``: passages containing every word and phrase of the question;
2. ``some``: passages containing at least one of them;
3. ``partial``: passages where a word appears inside a longer word (trigram index), used
   only when the first two tiers find little.

At most :data:`PER_SOURCE` passages per file are shown, so one long lecture cannot fill
the whole page. Words that match nothing are checked against the index's vocabulary by
spelling (edit distance counting a swap of two letters as one edit) and sound
(Metaphone) for a "did you mean" suggestion.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import jellyfish
from rapidfuzz import process
from rapidfuzz.distance import OSA

from studybuddy_exam import query as q

PER_SOURCE = 3
LIMIT = 60
CANDIDATES = 400  # rows read per tier before the per-file cap
PARTIAL_WHEN_FEWER_THAN = 5
WEIGHTS = (1.0, 0.8, 0.5, 0.3)  # text, aliases, questions, context (tuned in E6)
MARK_OPEN, MARK_CLOSE = "\x01", "\x02"  # snippet markers; replaced after HTML escaping
SNIPPET_TOKENS = 40
SUGGEST_MIN_LENGTH = 4
SUGGEST_MIN_SIMILARITY = 0.75
SOUND_ALIKE_MIN_SIMILARITY = 0.5

KINDS = ("video", "pptx", "pdf", "docx")


@dataclass(frozen=True)
class Filters:
    course: str | None = None
    kinds: frozenset[str] | None = None
    week_from: int | None = None
    week_to: int | None = None


@dataclass
class Hit:
    passage_id: int
    source_id: int
    course_slug: str
    course_title: str
    kind: str
    title: str
    week: int | None
    date: str | None
    t_start_ms: int | None
    t_end_ms: int | None
    slide_no: int | None
    page_no: int | None
    heading_path: str | None
    origin: str
    snippet: str  # with MARK_OPEN / MARK_CLOSE around matches
    match: str  # "all" | "some" | "partial"
    has_view: bool


@dataclass
class Result:
    query: q.Query
    hits: list[Hit] = field(default_factory=list)
    suggestion: str | None = None  # a corrected question, when some words match nothing
    corrected: bool = False  # True when hits are for ``suggestion`` because the question found nothing


class Searcher:
    """Searches one open, read-only exam index. Not thread-safe: use from one thread."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        rows = conn.execute("SELECT term, doc_freq, metaphone FROM vocab").fetchall()
        self._terms = [r[0] for r in rows]
        self._freq = {r[0]: r[1] for r in rows}
        self._by_sound: dict[str, list[str]] = {}
        for term, _, code in rows:
            self._by_sound.setdefault(code, []).append(term)

    def search(self, text: str, filters: Filters = Filters(), *, limit: int = LIMIT) -> Result:
        parsed = q.parse(text)
        result = Result(parsed)
        if parsed.empty:
            return result
        result.hits = self._tiers(parsed, filters, limit)
        result.suggestion = self.suggest(parsed)
        if not result.hits and result.suggestion:
            corrected = q.parse(result.suggestion)
            result.hits = self._tiers(corrected, filters, limit)
            result.corrected = bool(result.hits)
        return result

    def _tiers(self, parsed: q.Query, filters: Filters, limit: int) -> list[Hit]:
        hits: list[Hit] = []
        seen: set[int] = set()
        per_source: dict[int, int] = {}

        def take(rows: list[sqlite3.Row], match: str) -> None:
            for row in rows:
                if len(hits) >= limit:
                    return
                if row["id"] in seen or per_source.get(row["source_id"], 0) >= PER_SOURCE:
                    continue
                seen.add(row["id"])
                per_source[row["source_id"]] = per_source.get(row["source_id"], 0) + 1
                hits.append(_hit(row, match))

        if expr := q.match_all(parsed):
            take(self._query("passage_fts", expr, filters), "all")
        if expr := q.match_any(parsed):
            take(self._query("passage_fts", expr, filters), "all" if len(parsed.parts) == 1 else "some")
        if len(hits) < PARTIAL_WHEN_FEWER_THAN and (expr := q.match_partial(parsed)):
            take(self._query("passage_tri", expr, filters), "partial")
        return hits

    def _query(self, table: str, expr: str, filters: Filters) -> list[sqlite3.Row]:
        rank = f"bm25({table}, {', '.join(map(str, WEIGHTS))})" if table == "passage_fts" else f"bm25({table})"
        where, args = _filter_sql(filters)
        sql = f"""
            SELECT p.*, s.kind, s.title AS source_title, s.week, s.date, s.view_path,
                   c.slug AS course_slug, c.title AS course_title,
                   snippet({table}, 0, ?, ?, '…', {SNIPPET_TOKENS}) AS snippet
              FROM {table}
              JOIN passage p ON p.id = {table}.rowid
              JOIN source s ON s.id = p.source_id
              JOIN course c ON c.id = s.course_id
             WHERE {table} MATCH ? {where}
             ORDER BY {rank}
             LIMIT {CANDIDATES}
        """
        try:
            return self.conn.execute(sql, (MARK_OPEN, MARK_CLOSE, expr, *args)).fetchall()
        except sqlite3.OperationalError:
            # Every part is quoted, so this only happens for input FTS5 cannot tokenize at
            # all (e.g. a trigram query of pure punctuation); treat it as no match.
            return []

    def suggest(self, parsed: q.Query) -> str | None:
        """The question with misspelled words replaced, or None when every word is found."""
        fixes: dict[str, str] = {}
        for word in parsed.words:
            if len(word) < SUGGEST_MIN_LENGTH or word in self._freq or self._found(word):
                continue
            if better := self._closest(word):
                fixes[word] = better
        return q.replace_words(parsed.text, fixes) if fixes else None

    def _found(self, word: str) -> bool:
        """Does the stemmed index know ``word`` (e.g. "enzymes" when only "enzyme" occurs)?"""
        try:
            row = self.conn.execute(
                "SELECT 1 FROM passage_fts WHERE passage_fts MATCH ? LIMIT 1", (q.quote(word),)
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        return row is not None

    def _closest(self, word: str) -> str | None:
        candidates: dict[str, float] = {}
        for term, score, _ in process.extract(
            word, self._terms, scorer=OSA.normalized_similarity,
            score_cutoff=SUGGEST_MIN_SIMILARITY, limit=10,
        ):
            candidates[term] = score
        for term in self._by_sound.get(jellyfish.metaphone(word), []):
            score = OSA.normalized_similarity(word, term)
            if score >= SOUND_ALIKE_MIN_SIMILARITY:
                candidates[term] = max(candidates.get(term, 0.0), score + 0.05)  # sounding alike counts
        if not candidates:
            return None
        return max(candidates, key=lambda t: (candidates[t], self._freq.get(t, 0)))

    # ------------------------------------------------------------------ browsing

    def passage(self, passage_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT p.*, s.kind, s.title AS source_title, s.week, s.date, s.view_path, s.duration_s,
                      s.page_count, c.slug AS course_slug, c.title AS course_title
                 FROM passage p JOIN source s ON s.id = p.source_id JOIN course c ON c.id = s.course_id
                WHERE p.id = ?""",
            (passage_id,),
        ).fetchone()

    def neighbours(self, source_id: int, ord_: int) -> tuple[int | None, int | None]:
        rows = self.conn.execute(
            "SELECT id, ord FROM passage WHERE source_id = ? AND ord IN (?, ?)", (source_id, ord_ - 1, ord_ + 1)
        ).fetchall()
        by_ord = {r["ord"]: r["id"] for r in rows}
        return by_ord.get(ord_ - 1), by_ord.get(ord_ + 1)

    def source(self, source_id: int) -> sqlite3.Row | None:
        """The source, with ``library_root``: the folder its ``view_path`` is relative to ('' for the index's own)."""
        return self.conn.execute(
            """SELECT s.*, l.root AS library_root
                 FROM source s JOIN course c ON c.id = s.course_id JOIN library l ON l.id = c.library_id
                WHERE s.id = ?""",
            (source_id,),
        ).fetchone()

    def courses(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT c.slug, c.title, l.name AS library, s.kind, COUNT(DISTINCT s.id) AS files,
                      GROUP_CONCAT(DISTINCT s.week) AS weeks
                 FROM course c JOIN library l ON l.id = c.library_id LEFT JOIN source s ON s.course_id = c.id
                GROUP BY c.id, s.kind ORDER BY c.library_id, c.slug"""
        ).fetchall()
        notes = dict(self.conn.execute(
            """SELECT c.slug, COUNT(*) FROM notes n JOIN source s ON s.id = n.source_id
                 JOIN course c ON c.id = s.course_id GROUP BY c.id"""
        ).fetchall())
        courses: dict[str, dict] = {}
        for r in rows:
            entry = courses.setdefault(r["slug"], {
                "slug": r["slug"], "title": r["title"], "library": r["library"], "files": {}, "weeks": set(),
                "notes": notes.get(r["slug"], 0),
            })
            if r["kind"]:
                entry["files"][r["kind"]] = r["files"]
            entry["weeks"].update(int(w) for w in (r["weeks"] or "").split(",") if w)
        return [{**c, "weeks": sorted(c["weeks"])} for c in courses.values()]


def _filter_sql(filters: Filters) -> tuple[str, list]:
    clauses, args = [], []
    if filters.course:
        clauses.append("c.slug = ?")
        args.append(filters.course)
    if filters.kinds:
        clauses.append(f"s.kind IN ({', '.join('?' * len(filters.kinds))})")
        args.extend(sorted(filters.kinds))
    if filters.week_from is not None:
        clauses.append("s.week >= ?")
        args.append(filters.week_from)
    if filters.week_to is not None:
        clauses.append("s.week <= ?")
        args.append(filters.week_to)
    return "".join(f" AND {c}" for c in clauses), args


def _hit(row: sqlite3.Row, match: str) -> Hit:
    return Hit(
        passage_id=row["id"], source_id=row["source_id"], course_slug=row["course_slug"],
        course_title=row["course_title"], kind=row["kind"], title=row["source_title"], week=row["week"],
        date=row["date"], t_start_ms=row["t_start_ms"], t_end_ms=row["t_end_ms"], slide_no=row["slide_no"],
        page_no=row["page_no"], heading_path=row["heading_path"], origin=row["origin"],
        snippet=row["snippet"] or "", match=match, has_view=row["view_path"] is not None,
    )
