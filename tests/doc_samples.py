"""Generate small sample PDFs, images and Word files for the ingestion tests.

The PDF writer has no dependencies; image helpers use Pillow and the Word helpers use
python-docx (plus raw WordprocessingML for what python-docx cannot author). Nothing here
imports ``studybuddy``, so any module's tests can use these files.
"""

from __future__ import annotations

import io
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LETTER = (612.0, 792.0)  # PDF points
HELVETICA_TTC = "/System/Library/Fonts/Helvetica.ttc"

# Distinct, number-free study paragraphs (about 50 words each) for building long text.
PARAGRAPHS: tuple[str, ...] = (
    "Glycolysis is the first stage of cellular respiration. It takes place in the cytoplasm "
    "and splits one molecule of glucose into two molecules of pyruvate. The pathway invests "
    "two molecules of ATP early on and later produces four, so the cell gains a small net "
    "amount of energy along with two molecules of the electron carrier NADH.",
    "The citric acid cycle runs inside the mitochondrial matrix. Each turn oxidizes an acetyl "
    "group and releases carbon dioxide as waste. Most of the energy is captured as NADH and "
    "FADH2, which carry high energy electrons to the inner membrane where the largest share "
    "of ATP is finally made by oxidative phosphorylation.",
    "Photosynthesis converts light energy into chemical energy stored in sugar. The light "
    "dependent reactions occur in the thylakoid membranes and split water, releasing oxygen. "
    "The Calvin cycle then uses ATP and NADPH from those reactions to fix carbon dioxide into "
    "organic molecules inside the stroma of the chloroplast.",
    "Enzymes are proteins that speed up chemical reactions by lowering the activation energy. "
    "Each enzyme has an active site with a specific shape that binds its substrate. "
    "Temperature and acidity change that shape, which is why most human enzymes work best "
    "near body temperature and at a nearly neutral level of acidity.",
    "The cell membrane is a phospholipid bilayer with embedded proteins. Small nonpolar "
    "molecules diffuse straight through it, while ions and larger polar molecules need "
    "channels or carriers. Active transport pumps such as the sodium potassium pump spend ATP "
    "to move ions against their concentration gradients and keep the cell alive.",
    "Mitosis divides one nucleus into two genetically identical nuclei. During prophase the "
    "chromosomes condense, in metaphase they line up at the middle of the spindle, in anaphase "
    "the sister chromatids separate, and in telophase new nuclear envelopes form around each "
    "set before the cytoplasm divides during cytokinesis.",
    "Meiosis produces gametes with half the usual number of chromosomes. Crossing over between "
    "homologous chromosomes and their independent assortment during the first division create "
    "new combinations of alleles. This variation is the raw material on which natural "
    "selection acts across many generations of a population.",
    "Deoxyribonucleic acid stores hereditary information as a sequence of four bases. During "
    "replication the double helix unwinds and each strand serves as a template for a new one. "
    "Polymerase enzymes add matching nucleotides, and proofreading keeps the error rate "
    "remarkably low across the billions of bases in a human genome.",
)


# --- PDF ------------------------------------------------------------------------------------


def wrap(text: str, width: int = 90) -> list[str]:
    """``text`` broken into lines of at most ``width`` characters, for ``make_text_pdf``."""
    return textwrap.wrap(text, width) or [""]


@dataclass(frozen=True)
class ImagePage:
    """A full-page picture for ``make_text_pdf``, optionally over an invisible text layer.

    The text layer is drawn in render mode 3 (invisible), like the OCR layer of a
    "searchable" scan; leave it empty for a plain scanned page.
    """

    image: Image.Image
    text_layer: Sequence[str] = field(default_factory=tuple)


PdfPage = Sequence[str] | Image.Image | ImagePage


def make_text_pdf(
    path: Path,
    pages: Sequence[PdfPage],
    *,
    font_size: float = 11.0,
    header: str | None = None,
    footer: str | None = None,
    page_size: tuple[float, float] = LETTER,
    margin: float = 72.0,
) -> Path:
    """Write a PDF with real, extractable text: page ``i`` shows the lines ``pages[i]``.

    Lines are drawn top to bottom in Helvetica (WinAnsi, so Latin-1 text only) and are not
    wrapped; use ``wrap`` for long text. A page given as a PIL image (or ``ImagePage``) is a
    full-page picture, like a scanned page. ``header`` and ``footer`` are drawn on every text
    page, above and below the body; ``{page}`` and ``{pages}`` in them become the 1-based
    page number and the page count. Raises ValueError when a page has more lines than fit.
    """
    path.write_bytes(_assemble(_pdf_objects(pages, font_size, header, footer, page_size, margin)))
    return path


def make_encrypted_pdf(path: Path) -> Path:
    """A one-page PDF that needs a user password to open (its keys match no password)."""
    objects = _pdf_objects([["Secret notes"]], 11.0, None, None, LETTER, 72.0)
    objects.append(b"<< /Filter /Standard /V 1 /R 2 /O <%s> /U <%s> /P -4 >>" % (b"11" * 32, b"22" * 32))
    path.write_bytes(_assemble(objects, b" /Encrypt %d 0 R /ID [<%s> <%s>]" % (len(objects), b"ab" * 16, b"ab" * 16)))
    return path


def make_image_pdf(path: Path, images: Sequence[Image.Image], *, dpi: float = 200.0) -> Path:
    """An image-only (scanned-like) PDF with one page per image, written by Pillow."""
    rgb = [img.convert("RGB") for img in images]
    rgb[0].save(path, "PDF", save_all=True, append_images=rgb[1:], resolution=dpi)
    return path


def _pdf_objects(
    pages: Sequence[PdfPage],
    font_size: float,
    header: str | None,
    footer: str | None,
    page_size: tuple[float, float],
    margin: float,
) -> list[bytes]:
    """Numbered PDF objects (index 0 = object 1): catalog, page tree, font, then each page."""
    width, height = page_size
    leading = font_size * 1.25
    room = int((height - 2 * margin) // leading)
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    objects: list[bytes] = [b"", b"", font]
    kids: list[int] = []
    for number, page in enumerate(pages, start=1):
        if isinstance(page, Image.Image):
            page = ImagePage(page)
        lines = page.text_layer if isinstance(page, ImagePage) else page
        if len(lines) > room:
            raise ValueError(f"page {number} has {len(lines)} lines; at most {room} fit at {font_size} pt")
        mode = 3 if isinstance(page, ImagePage) else 0
        top = height - margin - font_size
        parts = [_text_line(line, margin, top - i * leading, font_size, mode) for i, line in enumerate(lines)]
        resources = b"/Font << /F1 3 0 R >>"
        if isinstance(page, ImagePage):
            objects.append(_jpeg_xobject(page.image))
            resources += b" /XObject << /Im1 %d 0 R >>" % len(objects)
            parts.insert(0, b"q %.2f 0 0 %.2f 0 0 cm /Im1 Do Q" % (width, height))
        else:
            fields = {"page": number, "pages": len(pages)}
            if header is not None:
                parts.insert(0, _text_line(header.format(**fields), margin, height - margin / 2, font_size * 0.8))
            if footer is not None:
                parts.append(_text_line(footer.format(**fields), margin, margin / 2, font_size * 0.8))
        content = b"\n".join(parts)
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] " % (width, height)
            + b"/Resources << %s >> /Contents %d 0 R >>" % (resources, len(objects))
        )
        kids.append(len(objects))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    return objects


def _text_line(text: str, x: float, y: float, size: float, mode: int = 0) -> bytes:
    return b"BT /F1 %.2f Tf %d Tr %.2f %.2f Td (%s) Tj ET" % (size, mode, x, y, _pdf_string(text))


def _pdf_string(text: str) -> bytes:
    out = bytearray()
    for byte in text.encode("cp1252"):
        if byte in b"()\\":
            out += b"\\" + bytes([byte])
        elif byte < 32 or byte > 126:
            out += b"\\%03o" % byte
        else:
            out.append(byte)
    return bytes(out)


def _jpeg_xobject(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=90)
    data = buf.getvalue()
    head = b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB " % image.size
    return head + b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n%s\nendstream" % (len(data), data)


def _assemble(objects: list[bytes], trailer_extra: bytes = b"") -> bytes:
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\n" % (len(objects) + 1, trailer_extra)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


# --- images ---------------------------------------------------------------------------------


def render_text_image(
    lines: Sequence[str],
    *,
    size: tuple[int, int] = (1700, 2200),
    font_size: int = 48,
    margin: int = 120,
) -> Image.Image:
    """Black text on a white page (Letter at 200 DPI by default), for OCR tests."""
    try:
        font = ImageFont.truetype(HELVETICA_TTC, font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((margin, margin + i * int(font_size * 1.6)), line, fill="black", font=font)
    return image


# --- Word -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Heading:
    """A heading in a built-in style: level 0 is "Title", 1-9 are "Heading 1".."Heading 9"."""

    text: str
    level: int = 1


@dataclass(frozen=True)
class OutlineHeading:
    """A heading known only by its outline level (1-9), not by a "Heading N" style name.

    With ``via_style`` the ``w:outlineLvl`` sits in a custom paragraph style ("Custom Head N");
    otherwise it is direct formatting on a "Normal" paragraph.
    """

    text: str
    level: int = 1
    via_style: bool = False


@dataclass(frozen=True)
class ListItem:
    """A list paragraph: "List Bullet"/"List Number" style, or a direct ``w:numPr`` (``direct``)."""

    text: str
    numbered: bool = False
    direct: bool = False


@dataclass(frozen=True)
class Table:
    rows: Sequence[Sequence[str]]


@dataclass(frozen=True)
class TextBox:
    """A paragraph reading ``anchor`` that also holds an inline text box with ``lines``.

    The box is written the way Word saves it: a DrawingML shape (``mc:Choice``) plus a VML
    copy (``mc:Fallback``), so the box text exists twice in the XML.
    """

    anchor: str
    lines: Sequence[str]


DocxBlock = str | Heading | OutlineHeading | ListItem | Table | TextBox


def make_docx(
    path: Path,
    blocks: Sequence[DocxBlock],
    *,
    header: str | None = None,
    footer: str | None = None,
) -> Path:
    """Write a .docx with ``blocks`` in order; a plain string is a "Normal" paragraph.

    ``header`` and ``footer`` go into the page header and footer, which ingestion ignores.
    """
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml import parse_xml

    doc = Document()
    for block in blocks:
        if isinstance(block, str):
            doc.add_paragraph(block)
        elif isinstance(block, Heading):
            doc.add_heading(block.text, level=block.level)
        elif isinstance(block, OutlineHeading):
            if block.via_style:
                name = f"Custom Head {block.level}"
                if name not in [s.name for s in doc.styles]:
                    style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
                    style.element.get_or_add_pPr().append(parse_xml(_outline_xml(block.level)))
                doc.add_paragraph(block.text, style=name)
            else:
                doc.add_paragraph(block.text)._p.get_or_add_pPr().append(parse_xml(_outline_xml(block.level)))
        elif isinstance(block, ListItem):
            if block.direct:
                p = doc.add_paragraph(block.text)
                numbering = f'<w:numPr {_W_NS}><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
                p._p.get_or_add_pPr().append(parse_xml(numbering))
            else:
                doc.add_paragraph(block.text, style="List Number" if block.numbered else "List Bullet")
        elif isinstance(block, Table):
            width = max(len(row) for row in block.rows)
            table = doc.add_table(rows=len(block.rows), cols=width)
            for row, values in zip(table.rows, block.rows):
                for cell, value in zip(row.cells, values):
                    cell.text = value
        elif isinstance(block, TextBox):
            p = doc.add_paragraph(block.anchor)
            p._p.append(parse_xml(_text_box_xml(block.lines, box_id=len(doc.paragraphs))))
        else:
            raise TypeError(f"unknown block {block!r}")
    if header is not None:
        doc.sections[0].header.paragraphs[0].text = header
    if footer is not None:
        doc.sections[0].footer.paragraphs[0].text = footer
    doc.save(str(path))
    return path


_W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _outline_xml(level: int) -> str:
    return f'<w:outlineLvl {_W_NS} w:val="{level - 1}"/>'


def _text_box_xml(lines: Sequence[str], *, box_id: int) -> str:
    from xml.sax.saxutils import escape

    content = "".join(f"<w:p><w:r><w:t xml:space=\"preserve\">{escape(line)}</w:t></w:r></w:p>" for line in lines)
    box = f"<w:txbxContent>{content}</w:txbxContent>"
    emu_w, emu_h = 2540000, 635000  # 200 x 50 pt
    return f"""<w:r {_W_NS}
  xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
  xmlns:v="urn:schemas-microsoft-com:vml">
<mc:AlternateContent><mc:Choice Requires="wps"><w:drawing>
  <wp:inline distT="0" distB="0" distL="0" distR="0">
    <wp:extent cx="{emu_w}" cy="{emu_h}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
    <wp:docPr id="{1000 + box_id}" name="Text Box {box_id}"/><wp:cNvGraphicFramePr/>
    <a:graphic><a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
      <wps:wsp><wps:cNvSpPr txBox="1"/>
        <wps:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{emu_w}" cy="{emu_h}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom></wps:spPr>
        <wps:txbx>{box}</wps:txbx><wps:bodyPr/></wps:wsp>
    </a:graphicData></a:graphic>
  </wp:inline></w:drawing></mc:Choice>
<mc:Fallback><w:pict><v:shape type="#_x0000_t202" style="width:200pt;height:50pt">
  <v:textbox>{box}</v:textbox></v:shape></w:pict></mc:Fallback>
</mc:AlternateContent></w:r>"""
