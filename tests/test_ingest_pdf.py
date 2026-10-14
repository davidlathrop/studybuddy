"""PDF ingestion: page numbers, cleanup, chunking, OCR of scanned pages, errors."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from doc_samples import (
    PARAGRAPHS,
    ImagePage,
    make_encrypted_pdf,
    make_image_pdf,
    make_text_pdf,
    render_text_image,
    wrap,
)
from studybuddy.ingest import ocr, pdf
from studybuddy.ingest.chunking import MAX_WORDS, chunk_text
from studybuddy.ingest.model import IngestError, word_count

SCAN_LINES = ["Scanned lecture notes", "Mitochondria make ATP.", "The Krebs cycle runs in the matrix."]


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """OCR that always 'reads' SCAN_LINES; records each image it is given."""
    calls: list[object] = []

    def read(image):
        calls.append(image)
        return "\n".join(SCAN_LINES)

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", read)
    return calls


@pytest.fixture
def no_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(image):
        raise AssertionError("OCR must not run here")

    monkeypatch.setattr(ocr, "ocr_image", fail)


def test_pages_are_numbered_from_one(tmp_path: Path, no_ocr: None) -> None:
    pages = [["Cells are small."], ["Tissues are made of cells."], ["Organs are made of tissues."]]
    path = make_text_pdf(tmp_path / "a.pdf", pages)
    drafts = pdf.extract(path)
    assert [(d.page_no, d.text, d.origin) for d in drafts] == [
        (1, "Cells are small.", "text"),
        (2, "Tissues are made of cells.", "text"),
        (3, "Organs are made of tissues.", "text"),
    ]
    assert all(d.slide_no is None and d.heading_path is None and d.t_start_ms is None for d in drafts)
    assert pdf.page_count(path) == 3


def test_blank_page_gives_no_passage_but_keeps_numbering(tmp_path: Path, no_ocr: None) -> None:
    path = make_text_pdf(tmp_path / "a.pdf", [["First page text."], [], ["Third page text."]])
    assert [(d.page_no, d.text) for d in pdf.extract(path)] == [(1, "First page text."), (3, "Third page text.")]


def test_running_headers_footers_and_page_numbers_are_stripped(tmp_path: Path, no_ocr: None) -> None:
    topics = ["cells", "tissues", "organs", "systems"]
    # The header text also appears mid-page; only the copies at the page edges are removed.
    pages = [
        wrap(PARAGRAPHS[n], 60)[:3] + ["BIO 101 Lecture Notes"] + wrap(PARAGRAPHS[n + 1], 60)[:3] + [f"End of {t}."]
        for n, t in enumerate(topics)
    ]
    path = make_text_pdf(tmp_path / "a.pdf", pages, header="BIO 101 Lecture Notes", footer="Page {page} of {pages}")
    drafts = pdf.extract(path)
    assert [d.page_no for d in drafts] == [1, 2, 3, 4]
    for draft, lines in zip(drafts, pages):
        assert draft.text == "\n".join(lines)


def test_lines_repeated_on_too_few_pages_are_kept(tmp_path: Path, no_ocr: None) -> None:
    path = make_text_pdf(tmp_path / "a.pdf", [["Alpha text."], ["Beta text."]], header="Course Notes")
    assert [d.text for d in pdf.extract(path)] == ["Course Notes\nAlpha text.", "Course Notes\nBeta text."]


def test_lone_page_numbers_are_dropped_even_without_repetition(tmp_path: Path, no_ocr: None) -> None:
    pages = [["Intro text.", "42", "More text."], ["Next text."]]
    path = make_text_pdf(tmp_path / "a.pdf", pages, footer="- {page} -")
    assert [d.text for d in pdf.extract(path)] == ["Intro text.\n42\nMore text.", "Next text."]


def test_long_page_is_split_into_ordered_chunks(tmp_path: Path, no_ocr: None) -> None:
    body = " ".join(PARAGRAPHS * 2)
    path = make_text_pdf(tmp_path / "a.pdf", [["Short first page."], wrap(body, 110)], font_size=8)
    drafts = pdf.extract(path)
    assert drafts[0].page_no == 1
    chunks = [d for d in drafts if d.page_no == 2]
    assert len(chunks) >= 3
    assert all(150 <= word_count(d.text) <= MAX_WORDS for d in chunks), [word_count(d.text) for d in chunks]
    assert " ".join(d.text for d in chunks).split() == body.split()
    assert all(c.text.endswith(".") for c in chunks), "chunks should end at sentence boundaries"


def test_hyphenated_line_breaks_are_rejoined(tmp_path: Path, no_ocr: None) -> None:
    path = make_text_pdf(
        tmp_path / "a.pdf",
        [[
            "The process of glyco-",
            "lysis turns glucose into pyru-",
            "vate. Lipids show self-assembly in water, and Self-",
            "assembly needs no energy. Pages 3-",
            "5 are optional.",
        ]],
    )
    text = pdf.extract(path)[0].text
    assert "glycolysis turns glucose into pyruvate." in text
    assert "Self-assembly needs" in text  # spelled with a hyphen elsewhere in the document
    assert "3-\n5" in text  # digits are not a hyphenated word
    assert "\x02" not in text and "\ufffe" not in text


def test_page_texts_are_cleaned_but_not_stripped_or_ocred(tmp_path: Path, no_ocr: None) -> None:
    scan = render_text_image(SCAN_LINES)
    pages = [["Typed  text with  pyru-", "vate."], scan, ["Last page."]]
    path = make_text_pdf(tmp_path / "a.pdf", pages, footer="Page {page}")
    assert pdf.page_texts(path) == ["Typed text with pyruvate.\nPage 1", "", "Last page.\nPage 3"]


def test_scanned_page_uses_ocr(tmp_path: Path, fake_ocr: list[object]) -> None:
    scan = render_text_image(SCAN_LINES)
    path = make_text_pdf(tmp_path / "a.pdf", [["Typed page about enzymes."], scan, ["Typed page about membranes."]])
    drafts = pdf.extract(path)
    assert [(d.page_no, d.origin) for d in drafts] == [(1, "text"), (2, "ocr"), (3, "text")]
    assert drafts[1].text == "\n".join(SCAN_LINES)
    assert len(fake_ocr) == 1
    assert min(fake_ocr[0].size) >= 2400  # rendered at 300 DPI


def test_searchable_scan_uses_its_text_layer(tmp_path: Path, fake_ocr: list[object]) -> None:
    page = ImagePage(render_text_image(SCAN_LINES), ["Hidden text layer from an earlier OCR run."])
    drafts = pdf.extract(make_text_pdf(tmp_path / "a.pdf", [page]))
    assert [(d.text, d.origin) for d in drafts] == [("Hidden text layer from an earlier OCR run.", "text")]
    assert fake_ocr == []


def test_junk_text_layer_is_replaced_by_ocr(tmp_path: Path, fake_ocr: list[object]) -> None:
    page = ImagePage(render_text_image(SCAN_LINES), ["#$%^ &*() 1234 ~~~ ::: ;;; ### @@@ !!! ???"])
    drafts = pdf.extract(make_text_pdf(tmp_path / "a.pdf", [page]))
    assert [(d.text, d.origin) for d in drafts] == [("\n".join(SCAN_LINES), "ocr")]


def test_junk_text_layer_is_kept_when_ocr_reads_no_more_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda image: "x = 2y + 3")
    path = make_text_pdf(tmp_path / "a.pdf", [["x = 2y + 3 = 7 / 4 + 1 - 9 * 8"]])
    assert [(d.text, d.origin) for d in pdf.extract(path)] == [("x = 2y + 3 = 7 / 4 + 1 - 9 * 8", "text")]


def test_scanned_pages_are_skipped_with_a_warning_when_ocr_is_off(
    tmp_path: Path, no_ocr: None, caplog: pytest.LogCaptureFixture
) -> None:
    scan = render_text_image(SCAN_LINES)
    path = make_text_pdf(tmp_path / "a.pdf", [["Typed page."], scan, scan])
    with caplog.at_level(logging.WARNING, logger="studybuddy.ingest.pdf"):
        drafts = pdf.extract(path, ocr=False)
    assert [d.page_no for d in drafts] == [1]
    assert "skipped 2 image-only page(s) (2-3)" in caplog.text


def test_scanned_pages_are_skipped_when_ocr_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_ocr: None, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(ocr, "available", lambda: False)
    path = make_image_pdf(tmp_path / "scan.pdf", [render_text_image(SCAN_LINES)])
    with caplog.at_level(logging.WARNING, logger="studybuddy.ingest.pdf"):
        assert pdf.extract(path) == []
    assert "unavailable" in caplog.text


@pytest.mark.skipif(not ocr.available(), reason="needs macOS Vision OCR (ocrmac)")
def test_scanned_pdf_is_read_with_real_ocr(tmp_path: Path) -> None:
    images = [render_text_image(SCAN_LINES), render_text_image(["Second scanned page"])]
    path = make_image_pdf(tmp_path / "scan.pdf", images)
    drafts = pdf.extract(path)
    assert [(d.page_no, d.origin) for d in drafts] == [(1, "ocr"), (2, "ocr")]
    assert "Mitochondria make ATP" in drafts[0].text
    assert "Second scanned page" in drafts[1].text


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda p: p.write_bytes(b"this is not a PDF"), "not a readable PDF"),
        (lambda p: p.write_bytes(b""), "not a readable PDF"),
        (lambda p: p.write_bytes(make_text_pdf(p, [["Text"]]).read_bytes()[:200]), "not a readable PDF"),
        (make_encrypted_pdf, "password-protected"),
    ],
    ids=["garbage", "empty", "truncated", "encrypted"],
)
def test_unreadable_pdfs_raise_ingest_error(tmp_path: Path, make, message: str) -> None:
    path = tmp_path / "bad.pdf"
    make(path)
    for call in (pdf.extract, pdf.page_count, pdf.page_texts):
        with pytest.raises(IngestError, match=message):
            call(path)


def test_missing_pdf_raises_ingest_error(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not found"):
        pdf.extract(tmp_path / "missing.pdf")


# --- chunking ---------------------------------------------------------------------------------


def test_chunking_keeps_short_text_whole() -> None:
    text = " ".join(PARAGRAPHS[:5])  # ~250 words
    assert chunk_text(text) == [text]
    assert chunk_text("  \n ") == []


def test_chunking_prefers_paragraph_boundaries() -> None:
    text = "\n\n".join(PARAGRAPHS * 2)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    paragraphs = set(PARAGRAPHS)
    for chunk in chunks:
        assert all(part in paragraphs for part in chunk.split("\n\n"))
    assert "\n\n".join(chunks) == text


def test_chunking_balances_sizes_and_never_leaves_a_stub() -> None:
    for n in range(7, 25):
        text = " ".join((PARAGRAPHS * 4)[:n])
        sizes = [word_count(c) for c in chunk_text(text)]
        assert sum(sizes) == word_count(text)
        if len(sizes) > 1:
            assert min(sizes) >= 60 and max(sizes) <= MAX_WORDS, (n, sizes)


def test_chunking_falls_back_to_lines_and_words() -> None:
    rows = "\n".join(f"row {i} | value | another value" for i in range(120))  # one paragraph, no sentences
    chunks = chunk_text(rows)
    assert all(c.startswith("row ") and c.endswith("another value") for c in chunks)
    assert "\n".join(chunks) == rows
    words = " ".join(["word"] * 1000)
    assert [word_count(c) for c in chunk_text(words)] == [250, 250, 250, 250]
