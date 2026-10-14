"""Real conversions with PowerPoint and Word (run with STUDYBUDDY_OFFICE_TESTS=1).

These launch the Office apps (and quit them again if they were not open), so they are off
by default. They check the two promises the exam app relies on: PDF page N of a deck is
slide N even when slides are hidden, and every Word passage lands on the PDF page that
shows its text.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

import doc_samples as ds
import pptx_samples
from studybuddy.ingest import pdf, pptx
from studybuddy.ingest.add import AddOptions, add_file
from studybuddy.ingest.convert import Converter
from studybuddy.library import Library

pytestmark = pytest.mark.office

WORDS = ("cell membrane protein enzyme energy glucose oxygen carbon nucleus ribosome transport "
         "gradient pathway molecule reaction substrate product binding signal receptor").split()


def letters(text: str) -> str:
    return "".join(re.findall(r"[^\W\d_]+", text.lower()))


def first_words_on_page(text: str, page: str, words: int = 8) -> bool:
    return letters(" ".join(text.split()[:words])) in letters(page)


def prose(words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(words)) + "."


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Library:
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(tmp_path / "config"))
    lib = Library.init(tmp_path / "lib")
    lib.add_course("bio101", "Biology")
    return lib


def test_deck_with_a_hidden_slide_keeps_page_n_as_slide_n(library: Library, tmp_path: Path) -> None:
    deck = pptx_samples.make_course_deck(tmp_path / "week04.pptx")
    assert pptx.hidden_slides(deck) == [pptx_samples.HIDDEN]
    with Converter() as converter:
        result = add_file(library, library.course("bio101"), deck, AddOptions(ocr=False), converter=converter)
    assert result.page_count == pptx_samples.SLIDE_COUNT and not result.warnings
    view = library.course_dir("bio101") / "docs" / "week04.pptx.pdf"
    pages = pdf.page_texts(view)
    assert len(pages) == pptx_samples.SLIDE_COUNT
    # PowerPoint writes a slide's boxes into the PDF in its own order, so compare titles
    # (each sample slide's first line is unique) rather than whole texts.
    titles = {d.slide_no: d.text.splitlines()[0] for d in pptx.extract(deck, ocr=False) if d.origin == "text"}
    assert pptx_samples.HIDDEN in titles
    for slide_no, title in titles.items():
        assert letters(title) in letters(pages[slide_no - 1]), f"slide {slide_no} ({title!r}) is not on page {slide_no}"


def test_word_passages_land_on_their_pages(library: Library, tmp_path: Path) -> None:
    blocks: list[ds.DocxBlock] = []
    for chapter in range(1, 5):
        blocks.append(ds.Heading(f"Chapter {chapter}", 1))
        for part in range(1, 3):
            blocks.append(ds.Heading(f"Section {chapter}.{part}", 2))
            blocks += [f"Chapter {chapter} part {part} paragraph {n}: " + prose(90, seed=chapter * 100 + part * 10 + n)
                       for n in range(3)]
    notes = ds.make_docx(tmp_path / "notes.docx", blocks)
    with Converter() as converter:
        result = add_file(library, library.course("bio101"), notes, converter=converter)
    assert result.page_count and result.page_count >= 3 and not result.warnings
    pages = pdf.page_texts(library.course_dir("bio101") / "docs" / "notes.docx.pdf")
    with library.db() as db:
        rows = db.execute("SELECT text, page_no, heading_path FROM passage ORDER BY ord").fetchall()
    numbers = [r["page_no"] for r in rows]
    assert numbers == sorted(numbers) and numbers[0] == 1 and numbers[-1] >= len(pages) - 1
    for r in rows:
        assert first_words_on_page(r["text"], pages[r["page_no"] - 1]), (r["heading_path"], r["page_no"])
