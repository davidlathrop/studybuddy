"""Word (.docx) text extraction with heading paths, placed on pages via the converted PDF (plan §5.2).

Paragraph text is read from the XML directly rather than python-docx's ``Paragraph.text``,
which skips tracked insertions, content controls, simple fields and text boxes. Page
headers and footers live in other parts of the package and are never read.
"""

from __future__ import annotations

import bisect
import logging
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from docx import Document
from docx.document import Document as WordDocument
from docx.styles.style import ParagraphStyle
from docx.text.paragraph import Paragraph
from lxml import etree
from rapidfuzz import fuzz

from studybuddy.ingest.chunking import chunk_text
from studybuddy.ingest.model import IngestError, PassageDraft, clean_text

log = logging.getLogger(__name__)

MATCH_WORDS = 8  # leading words of a passage that must appear verbatim on its page
FUZZY_WORDS = 20  # leading words compared when the verbatim search fails
FUZZY_THRESHOLD = 80.0
LOOKAHEAD_PAGES = 3  # pages after the current one that the fuzzy search may move to

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_NS = {"w": _W, "mc": _MC}

_HEADING_STYLE = re.compile(r"heading ([1-9])", re.IGNORECASE)
_TOC_STYLE = re.compile(r"toc \d", re.IGNORECASE)
_LETTERS = re.compile(r"[^\W\d_]+")
_TEXT_BOXES = etree.XPath(".//w:txbxContent[not(ancestor::mc:Fallback)]", namespaces=_NS)

# Elements whose subtree is not part of the paragraph's own visible text.
_SKIP = {
    f"{{{_W}}}txbxContent",  # text boxes: read separately, after their paragraph
    f"{{{_MC}}}Fallback",  # a second copy of the mc:Choice content
    f"{{{_W}}}del",
    f"{{{_W}}}moveFrom",
    f"{{{_W}}}pPr",
    f"{{{_W}}}rPr",
}
_CHAR = {f"{{{_W}}}tab": "\t", f"{{{_W}}}br": "\n", f"{{{_W}}}cr": "\n", f"{{{_W}}}noBreakHyphen": "-"}
_WRAPPERS = {f"{{{_W}}}sdt", f"{{{_W}}}sdtContent", f"{{{_W}}}customXml"}


@dataclass
class _Section:
    path: str | None
    blocks: list[str] = field(default_factory=list)
    heading_only: bool = False
    in_list: bool = False

    def add(self, text: str, *, list_item: bool = False) -> None:
        if list_item and self.in_list:
            self.blocks[-1] += "\n" + text
        else:
            self.blocks.append(text)
        self.in_list = list_item
        self.heading_only = False


def extract(path: Path) -> list[PassageDraft]:
    """Passages in document order, each with its ``heading_path``; ``page_no`` is left unset."""
    doc = _open(path)
    sections = [_Section(None)]
    stack: list[tuple[int, str]] = []
    for element in _blocks(doc.element.body):
        if element.tag == f"{{{_W}}}tbl":
            if table := "\n".join(_table_lines(element)):
                sections[-1].add(table)
            continue
        paragraph = Paragraph(element, doc)
        style = paragraph.style
        if style is not None and _TOC_STYLE.fullmatch(style.name or ""):
            continue
        text = clean_text(_paragraph_text(element))
        level = _heading_level(element, style) if text else None
        if level is not None:
            title = " ".join(text.split())
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(_Section(" > ".join(t for _, t in stack), [title], heading_only=True))
        elif text:
            is_list = _is_list(element, style)
            sections[-1].add(f"- {text}" if is_list else text, list_item=is_list)
        for box in _text_box_texts(element):
            sections[-1].add(box)

    drafts = []
    carried: list[str] = []
    for section, following in zip(sections, sections[1:] + [None]):
        blocks = carried + section.blocks
        carried = []
        if section.heading_only and following is not None and _is_child(following.path, section.path):
            carried = blocks  # a bare heading goes into its first subsection's text
            continue
        for chunk in chunk_text("\n\n".join(blocks)):
            drafts.append(PassageDraft(chunk, heading_path=section.path))
    return drafts


def assign_pages(drafts: list[PassageDraft], page_texts: list[str]) -> list[PassageDraft]:
    """Copies of ``drafts`` with ``page_no`` set by finding each passage's text in ``page_texts``.

    Both sides are reduced to lowercase letters only, so hyphenation, ligatures, line breaks,
    list markers and page-number digits do not matter. Pages never go backwards: each draft is
    placed on the page where its opening words are found at or after the previous draft's
    page, or on the previous draft's page when they are not found. A verbatim match more than
    ``LOOKAHEAD_PAGES`` ahead is used only when no fuzzy match is found nearer, so a phrase
    repeated later in the document cannot pull every following draft too far.
    """
    if not page_texts:
        return [replace(d, page_no=None) for d in drafts]
    pages = ["".join(_words(text)) for text in page_texts]
    starts = [0]
    for page in pages:
        starts.append(starts[-1] + len(page))
    whole = "".join(pages)

    def page_at(pos: int) -> int:
        return bisect.bisect_right(starts, pos) - 1

    current = 0
    placed = []
    for draft in drafts:
        words = _words(draft.text)
        found = _find_exact(whole, "".join(words[:MATCH_WORDS]), starts[current])
        if found is None or page_at(found) > current + LOOKAHEAD_PAGES:
            window_end = starts[min(current + LOOKAHEAD_PAGES + 1, len(pages))]
            near = _find_fuzzy(whole[starts[current] : window_end], "".join(words[:FUZZY_WORDS]))
            if near is not None:
                found = starts[current] + near
        if found is not None:
            current = min(page_at(found), len(pages) - 1)
        placed.append(replace(draft, page_no=current + 1))
    return placed


def _open(path: Path) -> WordDocument:
    if not path.is_file():
        raise IngestError(f"{path}: file not found")
    try:
        return Document(str(path))
    except Exception as exc:  # python-docx raises zip, KeyError, ValueError or XML errors
        raise IngestError(f"{path.name} is not a readable Word document ({exc})") from None


def _blocks(parent: etree._Element) -> Iterator[etree._Element]:
    """Paragraphs and tables in document order, looking inside content controls."""
    for child in parent:
        if child.tag in (f"{{{_W}}}p", f"{{{_W}}}tbl"):
            yield child
        elif child.tag in _WRAPPERS:
            yield from _blocks(child)


def _paragraph_text(element: etree._Element) -> str:
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        for child in node:
            tag = child.tag
            if not isinstance(tag, str) or tag in _SKIP:
                continue
            if tag == f"{{{_W}}}t":
                parts.append(child.text or "")
            elif tag in _CHAR:
                parts.append(_CHAR[tag])
            else:
                walk(child)

    walk(element)
    return "".join(parts)


def _text_box_texts(element: etree._Element) -> list[str]:
    out = []
    for box in _TEXT_BOXES(element):
        lines = []
        for block in _blocks(box):
            if block.tag == f"{{{_W}}}tbl":
                lines.extend(_table_lines(block))
            elif text := clean_text(_paragraph_text(block)):
                lines.append(text)
        if lines:
            out.append("\n".join(lines))
    return out


def _table_lines(table: etree._Element) -> list[str]:
    """One line per row, non-empty cells joined with " | "."""
    lines = []
    for row in _children(table, "tr"):
        cells = []
        for cell in _children(row, "tc"):
            texts = []
            for block in _blocks(cell):
                if block.tag == f"{{{_W}}}tbl":
                    texts.extend(_table_lines(block))
                else:
                    texts.append(_paragraph_text(block))
                    texts.extend(_text_box_texts(block))
            if text := " ".join(clean_text(" ".join(texts)).split()):
                cells.append(text)
        if cells:
            lines.append(" | ".join(cells))
    return lines


def _children(parent: etree._Element, name: str) -> Iterator[etree._Element]:
    for child in parent:
        if child.tag == f"{{{_W}}}{name}":
            yield child
        elif child.tag in _WRAPPERS:
            yield from _children(child, name)


def _heading_level(element: etree._Element, style: ParagraphStyle | None) -> int | None:
    """0 for "Title", 1-9 for "Heading N" or an outline level on the paragraph or its styles."""
    name = (style.name or "") if style is not None else ""
    if name.lower() == "title":
        return 0
    if match := _HEADING_STYLE.fullmatch(name):
        return int(match.group(1))
    level = _outline_level(element)
    while level is None and style is not None:
        level = _outline_level(style.element)
        style = style.base_style
    return level + 1 if level is not None and 0 <= level <= 8 else None  # 9 = body text


def _outline_level(element: etree._Element) -> int | None:
    value = element.find("w:pPr/w:outlineLvl", _NS)
    if value is None:
        return None
    try:
        return int(value.get(f"{{{_W}}}val"))
    except (TypeError, ValueError):
        return None


def _is_list(element: etree._Element, style: ParagraphStyle | None) -> bool:
    numbering = element.find("w:pPr/w:numPr", _NS)
    while numbering is None and style is not None:
        numbering = style.element.find("w:pPr/w:numPr", _NS)
        style = style.base_style
    if numbering is None:
        return False
    num_id = numbering.find("w:numId", _NS)
    return num_id is None or num_id.get(f"{{{_W}}}val") != "0"  # numId 0 switches numbering off


def _is_child(path: str | None, parent: str | None) -> bool:
    return path is not None and parent is not None and path.startswith(parent + " > ")


def _words(text: str) -> list[str]:
    return _LETTERS.findall(unicodedata.normalize("NFKC", text).lower())


def _find_exact(whole: str, key: str, start: int) -> int | None:
    if not key:
        return None
    pos = whole.find(key, start)
    return pos if pos >= 0 else None


def _find_fuzzy(window: str, key: str) -> int | None:
    if not key or not window:
        return None
    match = fuzz.partial_ratio_alignment(key, window, score_cutoff=FUZZY_THRESHOLD)
    return match.dest_start if match is not None else None
