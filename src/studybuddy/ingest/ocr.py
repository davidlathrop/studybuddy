"""Text in images, read locally with Apple's Vision framework (``ocrmac``, plan §5.2).

Used for scanned PDF pages and pictures on slides. Runs before the exam only.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.3  # Vision's own scores; lower lines are mostly noise
MIN_SIDE_PX = 32  # icons and bullets: nothing to read


@cache
def available() -> bool:
    try:
        import ocrmac.ocrmac  # noqa: F401
    except Exception:  # not installed, or not macOS
        log.warning("OCR is unavailable (needs macOS and the ocrmac package); skipping text in images")
        return False
    return True


def ocr_image(image: Image | Path) -> str:
    """The text in ``image``, top to bottom and left to right, one recognized line per line.

    Returns "" when OCR is unavailable, the image is too small, or nothing is recognized.
    """
    if not available():
        return ""
    from PIL import Image as PILImage
    from ocrmac import ocrmac

    img = PILImage.open(image) if isinstance(image, (str, Path)) else image
    if min(img.size) < MIN_SIDE_PX:
        return ""
    img = on_white(img)
    try:
        found = ocrmac.OCR(img, recognition_level="accurate").recognize()
    except Exception as exc:  # Vision errors are rare; one bad image must not stop ingestion
        log.warning("OCR failed on an image: %s", exc)
        return ""
    # Boxes are (x, y, w, h), normalized, with the origin at the bottom left.
    lines = [(text, box) for text, confidence, box in found if confidence >= MIN_CONFIDENCE and text.strip()]
    lines.sort(key=lambda item: (-round(item[1][1] + item[1][3], 2), item[1][0]))
    return "\n".join(text.strip() for text, _ in lines)


def on_white(img: Image) -> Image:
    """``img`` as RGB, with any transparency flattened onto white.

    A plain RGB conversion turns transparent pixels black, which hides the dark text of a
    transparent PNG (common for diagrams pasted into slides).
    """
    from PIL import Image as PILImage

    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        background = PILImage.new("RGBA", img.size, "white")
        return PILImage.alpha_composite(background, img.convert("RGBA")).convert("RGB")
    return img if img.mode in ("RGB", "L") else img.convert("RGB")
