"""OCR through Apple Vision (``ocrmac``)."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageFont

from studybuddy.ingest import ocr

FONT = "/System/Library/Fonts/Helvetica.ttc"


def _text_image(lines: list[str], *, mode: str = "RGB") -> Image.Image:
    background = (0, 0, 0, 0) if mode == "RGBA" else "white"
    img = Image.new(mode, (1100, 140 * len(lines) + 60), background)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 40 + 140 * i), line, fill="black", font=ImageFont.truetype(FONT, 56))
    return img


def test_on_white_flattens_transparency() -> None:
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    flat = ocr.on_white(img)
    assert flat.mode == "RGB" and flat.getpixel((1, 1)) == (255, 255, 255)
    assert ocr.on_white(Image.new("L", (4, 4))).mode == "L"
    assert ocr.on_white(Image.new("CMYK", (4, 4))).mode == "RGB"


def test_tiny_images_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "available", lambda: True)
    assert ocr.ocr_image(Image.new("RGB", (16, 200), "white")) == ""


needs_vision = pytest.mark.skipif(not ocr.available(), reason="needs macOS Vision through ocrmac")


@needs_vision
def test_reads_lines_top_to_bottom() -> None:
    text = ocr.ocr_image(_text_image(["Mitochondria produce ATP", "via oxidative phosphorylation"]))
    assert text.splitlines() == ["Mitochondria produce ATP", "via oxidative phosphorylation"]


@needs_vision
def test_reads_dark_text_on_a_transparent_png() -> None:
    assert "Krebs cycle" in ocr.ocr_image(_text_image(["The Krebs cycle"], mode="RGBA"))
