"""Index building and searching: tiers, filters, per-file cap, snippets, "did you mean"."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from studybuddy_exam import query as q
from studybuddy_exam.index_format import FORMAT_VERSION, ExamIndexError, open_index
from studybuddy_exam.search import MARK_CLOSE, MARK_OPEN, PER_SOURCE, Filters, Searcher


@pytest.fixture(scope="module")
def searcher(index_file: Path) -> Searcher:
    conn, _ = open_index(index_file)
    return Searcher(conn)


def ids(result) -> list[int]:
    return [h.passage_id for h in result.hits]


def test_index_is_read_only_and_versioned(index_file: Path) -> None:
    conn, info = open_index(index_file)
    assert info.meta["format_version"] == str(FORMAT_VERSION) and info.meta["passages"] == "11"
    assert info.library_root == index_file.parent.parent
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM passage")


def test_missing_or_foreign_index(tmp_path: Path) -> None:
    with pytest.raises(ExamIndexError, match="build-exam"):
        open_index(tmp_path / "nope.sqlite")
    other = tmp_path / "other.sqlite"
    sqlite3.connect(other).execute("CREATE TABLE meta(key, value)").connection.execute(
        "INSERT INTO meta VALUES ('format_version', '99')").connection.commit()
    with pytest.raises(ExamIndexError, match="format 99"):
        open_index(other)


def test_vocab_has_unstemmed_words_with_metaphone(index_file: Path) -> None:
    conn, _ = open_index(index_file)
    rows = dict(conn.execute("SELECT term, metaphone FROM vocab").fetchall())
    assert "mitochondria" in rows and "photosynthesis" in rows and rows["photosynthesis"]
    assert "2" not in rows and "of" not in rows  # numbers and short words are left out


def test_all_words_rank_before_some(searcher: Searcher) -> None:
    result = searcher.search("krebs cycle matrix")
    tiers = [h.match for h in result.hits]
    assert tiers[:2] == ["all", "all"] and set(ids(result)[:2]) == {4, 6}
    assert "some" in tiers and tiers == sorted(tiers, key=["all", "some", "partial"].index)


def test_at_most_three_hits_per_file(searcher: Searcher) -> None:
    result = searcher.search("krebs")
    per_source: dict[int, int] = {}
    for h in result.hits:
        per_source[h.source_id] = per_source.get(h.source_id, 0) + 1
    assert per_source[10] == PER_SOURCE  # passages 1, 4 and 5 all mention Krebs; only 3 shown


def test_stemming_and_stopwords(searcher: Searcher) -> None:
    result = searcher.search("What are the regulating stomata?")
    assert result.query.words == ("regulating", "stomata")
    assert ids(result)[0] == 8 and result.hits[0].origin == "ocr"


def test_snippet_marks_matches(searcher: Searcher) -> None:
    hit = searcher.search("electron transport").hits[0]
    assert f"{MARK_OPEN}electron{MARK_CLOSE} {MARK_OPEN}transport{MARK_CLOSE}" in hit.snippet


def test_location_fields(searcher: Searcher) -> None:
    by_id = {h.passage_id: h for h in searcher.search("krebs cycle").hits}
    assert (by_id[1].t_start_ms, by_id[1].kind, by_id[1].title) == (0, "video", "Lecture 4: Cell energy")
    assert (by_id[6].slide_no, by_id[7].origin) == (6, "notes")
    notes = searcher.search("cost-push").hits[0]
    assert (notes.heading_path, notes.page_no, notes.has_view) == ("Inflation > Cost-push", 1, False)


@pytest.mark.parametrize(
    ("filters", "expected_sources"),
    [
        (Filters(course="econ201"), {12, 13}),
        (Filters(kinds=frozenset({"docx"})), {12}),
        (Filters(week_from=5), {13}),
        (Filters(week_to=3), {12}),
        (Filters(course="bio101", kinds=frozenset({"pptx"})), set()),
    ],
)
def test_filters(searcher: Searcher, filters: Filters, expected_sources: set[int]) -> None:
    result = searcher.search("elasticity tariffs", filters)
    assert {h.source_id for h in result.hits} == expected_sources


def test_partial_words_fall_back_to_trigrams(searcher: Searcher) -> None:
    result = searcher.search("mitochond")
    assert result.hits and all(h.match == "partial" for h in result.hits)
    assert {1, 3} <= set(ids(result))


@pytest.mark.parametrize(
    ("typed", "suggested"),
    [("mitocondria", "mitochondria"), ("fotosynthesis", "photosynthesis"), ("elastisity", "elasticity")],
)
def test_did_you_mean_corrects_when_nothing_matches(searcher: Searcher, typed: str, suggested: str) -> None:
    result = searcher.search(f"{typed.title()} please")
    assert result.suggestion == f"{suggested} please"
    assert result.corrected and result.hits


def test_known_words_and_stemmed_forms_are_not_corrected(searcher: Searcher) -> None:
    assert searcher.search("krebs").suggestion is None
    assert searcher.search("tariff").suggestion is None  # only "tariffs" occurs, but stemming finds it


def test_suggestion_shown_but_not_applied_when_results_exist(searcher: Searcher) -> None:
    result = searcher.search("krebs cycel")  # a swapped pair of letters counts as one edit
    assert result.suggestion == "krebs cycle" and not result.corrected
    assert result.hits  # "krebs" alone still finds passages, so nothing is replaced silently


@pytest.mark.parametrize("hostile", ['AND', 'OR NOT', 'c++', 'x:y', 'NEAR(', '-x', '"', '"""', "*", "^krebs",
                                     "krebs\" OR \"x", "'; DROP TABLE passage; --", "a" * 2000, "\x00", "（全角）"])
def test_hostile_input_never_errors(searcher: Searcher, hostile: str) -> None:
    searcher.search(hostile)  # must not raise


def test_quoted_phrase(searcher: Searcher) -> None:
    exact = searcher.search('"inner membrane"')
    assert ids(exact) == [2]
    assert q.parse('"Inner  Membrane" protons').parts == ("inner membrane", "protons")


def test_empty_and_filler_queries(searcher: Searcher) -> None:
    assert searcher.search("   ").hits == [] and searcher.search("the of a").hits == []
    assert q.parse("What is IT?").words == ()  # several common words: nothing to search
    assert q.parse("IT").words == ("it",)  # a single one may be meant
    assert searcher.search("in").hits  # "chloroplasts, not mitochondria" ... "in the matrix"


def test_courses_listing(searcher: Searcher) -> None:
    courses = {c["slug"]: c for c in searcher.courses()}
    assert courses["bio101"]["files"] == {"pptx": 1, "video": 1} and courses["bio101"]["weeks"] == [4]
    assert courses["econ201"]["weeks"] == [3, 6]


def test_passage_and_neighbours(searcher: Searcher) -> None:
    row = searcher.passage(2)
    assert row["source_title"] == "Lecture 4: Cell energy" and row["t_start_ms"] == 60_000
    assert searcher.neighbours(10, 2) == (1, 3)
    assert searcher.neighbours(10, 1) == (None, 2)
    assert searcher.passage(999) is None
