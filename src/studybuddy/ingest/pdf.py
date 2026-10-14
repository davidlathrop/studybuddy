"""PDF text extraction with page numbers, OCR for scanned pages (plan §5.2).

pypdfium2 gives each page's text in reading order. pdfium marks a hyphen that breaks a word
across lines with U+FFFE (``get_text_range``) or ``\\x02`` (``get_text_bounded``) and drops
the line break; the word is rejoined unless the document spells it with a hyphen elsewhere.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from studybuddy.ingest import ocr as vision
from studybuddy.ingest.chunking import chunk_text
from studybuddy.ingest.model import IngestError, PassageDraft, clean_text

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 20  # fewer non-space characters on a page with a picture: a scanned page
MIN_REAL_WORD_SHARE = 0.5  # below this, the text layer is junk (broken font encoding)
OCR_DPI = 300
MAX_RENDER_PX = 6000  # caps the render of oversized pages (posters)
EDGE_LINES = 3  # lines at the top and at the bottom of a page searched for running headers
MIN_RUNNING_PAGES = 3

_HYPHEN_MARK = re.compile(r"(\w*)[\x02\ufffe]\s*(\w*)")
_HYPHENATED = re.compile(r"\w+-\w+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe\uffff]")
_REAL_WORD = re.compile(r"[^\W\d_]{2,}")
_DIGITS = re.compile(r"\d+")
# "12", "- 12 -", "Page 12", "12 of 30", "12/30", or a lowercase roman numeral up to xxxix.
_PAGE_NUMBER = re.compile(
    r"(?i:page\s*)?[-–—]?\s*(?:\d{1,4}|(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3}))\s*[-–—]?(?:\s*(?i:of|/)\s*\d{1,4})?"
)


def extract(path: Path, *, ocr: bool = True) -> list[PassageDraft]:
    """Passages for every page with text, in order, with 1-based ``page_no``."""
    pages: list[tuple[str, str]] = []  # (raw text, origin)
    skipped: list[int] = []
    doc = _open(path)
    try:
        for index in range(len(doc)):
            text, origin, unread = _read_page(doc, index, ocr=ocr)
            if unread:
                skipped.append(index + 1)
            pages.append((text, origin))
    finally:
        doc.close()
    if skipped:
        reason = "OCR is off" if not ocr else "OCR is unavailable or found no text"
        log.warning(
            "%s: skipped %d image-only page(s) (%s) because %s", path.name, len(skipped), _ranges(skipped), reason
        )

    hyphenated = _hyphenated_words(text for text, _ in pages)
    lines = [clean_text(_rejoin_hyphens(text, hyphenated)).split("\n") for text, _ in pages]
    lines = _strip_running_lines(lines)
    drafts = []
    for number, (page_lines, (_, origin)) in enumerate(zip(lines, pages), start=1):
        for chunk in chunk_text("\n".join(page_lines)):
            drafts.append(PassageDraft(chunk, origin=origin, page_no=number))
    return drafts


def page_count(path: Path) -> int:
    doc = _open(path)
    try:
        return len(doc)
    finally:
        doc.close()


def page_texts(path: Path) -> list[str]:
    """Cleaned text of each page (no OCR), index 0 = page 1; used to place Word chunks on pages."""
    doc = _open(path)
    try:
        raw = [_text_layer(doc, index) for index in range(len(doc))]
    finally:
        doc.close()
    hyphenated = _hyphenated_words(raw)
    return [clean_text(_rejoin_hyphens(text, hyphenated)) for text in raw]


def _open(path: Path) -> pdfium.PdfDocument:
    if not path.is_file():
        raise IngestError(f"{path}: file not found")
    try:
        return pdfium.PdfDocument(path)
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).lower():
            raise IngestError(f"{path.name} is password-protected; add an unprotected copy instead") from None
        raise IngestError(f"{path.name} is not a readable PDF ({exc})") from None


def _read_page(doc: pdfium.PdfDocument, index: int, *, ocr: bool) -> tuple[str, str, bool]:
    """(raw text, origin, unread): ``unread`` marks an image-only page that OCR did not read."""
    page = _load_page(doc, index)
    if page is None:
        return "", "text", False
    try:
        text = _page_text(page)
        chars = len("".join(text.split()))
        junk = chars >= MIN_TEXT_CHARS and _real_word_share(text) < MIN_REAL_WORD_SHARE
        scanned = chars < MIN_TEXT_CHARS and _has_images(page)
        if not (scanned or junk):
            return text, "text", False
        found = _ocr_page(page) if ocr and vision.available() else ""
        if found and (scanned or _real_words(found) > _real_words(text)):
            return found, "ocr", False
        return text, "text", scanned and chars == 0
    finally:
        page.close()


def _text_layer(doc: pdfium.PdfDocument, index: int) -> str:
    page = _load_page(doc, index)
    if page is None:
        return ""
    try:
        return _page_text(page)
    finally:
        page.close()


def _load_page(doc: pdfium.PdfDocument, index: int) -> pdfium.PdfPage | None:
    try:
        return doc[index]
    except pdfium.PdfiumError as exc:
        log.warning("page %d could not be loaded: %s", index + 1, exc)
        return None


def _page_text(page: pdfium.PdfPage) -> str:
    textpage = page.get_textpage()
    try:
        return textpage.get_text_range()
    finally:
        textpage.close()


def _has_images(page: pdfium.PdfPage) -> bool:
    return next(page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)), None) is not None


def _ocr_page(page: pdfium.PdfPage) -> str:
    width, height = page.get_size()
    scale = min(OCR_DPI / 72, MAX_RENDER_PX / max(width, height, 1))
    bitmap = page.render(scale=scale)
    try:
        return vision.ocr_image(bitmap.to_pil())  # the image shares the bitmap's buffer
    finally:
        bitmap.close()


def _real_words(text: str) -> int:
    return sum(1 for token in text.split() if _REAL_WORD.search(token))


def _real_word_share(text: str) -> float:
    tokens = text.split()
    return _real_words(text) / len(tokens) if tokens else 0.0


def _hyphenated_words(texts: Iterable[str]) -> set[str]:
    return {word.lower() for text in texts for word in _HYPHENATED.findall(text)}


def _rejoin_hyphens(text: str, hyphenated: set[str]) -> str:
    def join(match: re.Match[str]) -> str:
        left, right = match.groups()
        if left and right and f"{left}-{right}".lower() in hyphenated:
            return f"{left}-{right}"
        return left + right

    return _CONTROL.sub("", _HYPHEN_MARK.sub(join, text))


def _strip_running_lines(pages: list[list[str]]) -> list[list[str]]:
    """Drop running headers/footers and lone page numbers from the edges of each page."""
    pages = [[line for line in lines if line] for lines in pages]
    with_text = sum(1 for lines in pages if lines)
    needed = max(MIN_RUNNING_PAGES, math.ceil(with_text / 2))
    seen = Counter(key for lines in pages for key in {_line_key(lines[i]) for i in _edges(lines)})
    running = {key for key, count in seen.items() if count >= needed}
    out = []
    for lines in pages:
        edges, ends = _edges(lines), {0, len(lines) - 1}
        out.append(
            [
                line
                for i, line in enumerate(lines)
                if not (i in edges and _line_key(line) in running)
                and not (i in ends and _PAGE_NUMBER.fullmatch(line))
            ]
        )
    return out


def _edges(lines: list[str]) -> set[int]:
    return set(range(min(EDGE_LINES, len(lines)))) | set(range(max(0, len(lines) - EDGE_LINES), len(lines)))


def _line_key(line: str) -> str:
    return _DIGITS.sub("#", " ".join(line.lower().split()))


def _ranges(numbers: list[int]) -> str:
    parts = []
    start = prev = numbers[0]
    for n in numbers[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)
