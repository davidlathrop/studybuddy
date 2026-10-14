"""PowerPoint, PDF and Word files: extract passages and make a PDF for viewing (plan §5.2)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from studybuddy.ingest import docx, pdf, pptx
from studybuddy.ingest.convert import ConversionError, Converter
from studybuddy.ingest.staging import Staged
from studybuddy.transcription import NoProgress, Progress

log = logging.getLogger(__name__)


def ingest_document(
    kind: str,
    src: Path,
    staging: Path,
    *,
    ocr: bool,
    converter: Converter | None,
    progress: Progress | None = None,
) -> Staged:
    """Copy ``src`` into ``staging``, extract its passages and, for PPTX/DOCX, convert it to PDF.

    Without a converter (or when conversion fails) the document is still added and
    searchable; it just has no PDF to open at exam time, which is recorded as a warning.
    """
    progress = progress or NoProgress()
    stored = staging / src.name
    if not stored.is_file() or stored.stat().st_size != src.stat().st_size:
        shutil.copy2(src, stored)
    warnings: list[str] = []
    with progress.stage("extract"):
        if kind == "pptx":
            drafts = pptx.extract(stored, ocr=ocr)
            page_count: int | None = pptx.slide_count(stored)
        elif kind == "pdf":
            drafts = pdf.extract(stored, ocr=ocr)
            page_count = pdf.page_count(stored)
        elif kind == "docx":
            drafts = docx.extract(stored)
            page_count = None
        else:
            raise ValueError(f"not a document kind: {kind}")

    files = {"stored": stored}
    if kind == "pdf":
        files["view"] = stored
    elif converter is not None and converter.can_convert(stored):
        view = staging / f"{src.name}.pdf"
        try:
            with progress.stage("to PDF"):
                if kind == "pptx":
                    _pptx_to_pdf(stored, view, staging, converter, page_count or 0, warnings)
                else:
                    converter.to_pdf(stored, view)
                    page_texts = pdf.page_texts(view)
                    drafts = docx.assign_pages(drafts, page_texts)
                    page_count = len(page_texts)
            files["view"] = view
        except ConversionError as exc:
            warnings.append(f"no PDF for viewing: {exc}")
            view.unlink(missing_ok=True)
    else:
        warnings.append("no PDF for viewing: conversion is turned off or unavailable")

    if not drafts:
        warnings.append("no text found")
    return Staged(drafts, files, duration_s=None, page_count=page_count, extractor=f"{kind}/1", warnings=warnings)


def _pptx_to_pdf(stored: Path, view: Path, staging: Path, converter: Converter, slides: int,
                 warnings: list[str]) -> None:
    # PowerPoint leaves hidden slides out of a PDF, which would shift page N away from
    # slide N; convert a copy with every slide shown instead.
    hidden = pptx.hidden_slides(stored)
    source = stored
    if hidden:
        source = staging / f"unhidden-{stored.name}"
        pptx.unhide_all(stored, source)
    try:
        converter.to_pdf(source, view)
    finally:
        if source != stored:
            source.unlink(missing_ok=True)
    pages = pdf.page_count(view)
    if pages != slides:
        warnings.append(f"the PDF has {pages} pages for {slides} slides, so pages may not match slide numbers")
