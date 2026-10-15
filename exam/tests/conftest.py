from __future__ import annotations

from pathlib import Path

import pytest

from studybuddy_exam.index_format import CourseRow, PassageRow, SourceRow, build

COURSES = [CourseRow(1, "bio101", "Biology 101"), CourseRow(2, "econ201", "Applied Microeconomics")]
SOURCES = [
    SourceRow(10, 1, "video", "Lecture 4: Cell energy", 4, "2026-09-15", "courses/bio101/videos/l4.exam.mp4", 3600.0, None),
    SourceRow(11, 1, "pptx", "Week 4 slides", 4, None, "courses/bio101/docs/w4.pptx.pdf", None, 12),
    SourceRow(12, 2, "docx", "My notes", 3, None, None, None, None),  # conversion failed: text only
    SourceRow(13, 2, "pdf", "Reading", 6, None, "courses/econ201/docs/reading.pdf", None, 3),
]


def _p(id_: int, source: int, ord_: int, text: str, **location) -> PassageRow:
    defaults = dict(t_start_ms=None, t_end_ms=None, slide_no=None, page_no=None, heading_path=None, origin="text")
    defaults.update(location)
    return PassageRow(id_, source, ord_, text=text, **defaults)


PASSAGES = [
    _p(1, 10, 1, "Today we look at how mitochondria produce ATP through the Krebs cycle.",
       t_start_ms=0, t_end_ms=60_000, origin="speech"),
    _p(2, 10, 2, "The electron transport chain pumps protons across the inner membrane.",
       t_start_ms=60_000, t_end_ms=120_000, origin="speech"),
    _p(3, 10, 3, "Photosynthesis happens in chloroplasts, not mitochondria.",
       t_start_ms=120_000, t_end_ms=180_000, origin="speech"),
    _p(4, 10, 4, "More on the Krebs cycle: it runs in the matrix.", t_start_ms=180_000, t_end_ms=240_000,
       origin="speech"),
    _p(5, 10, 5, "Krebs cycle again: two turns per glucose.", t_start_ms=240_000, t_end_ms=300_000, origin="speech"),
    _p(6, 11, 1, "Krebs Cycle\nOccurs in the matrix", slide_no=6),
    _p(7, 11, 2, "Remember: the Krebs cycle yields 2 ATP per glucose.", slide_no=6, origin="notes"),
    _p(8, 11, 3, "Stomata regulate gas exchange", slide_no=9, origin="ocr"),
    _p(9, 12, 1, "Elasticity\n\nElastic goods have substitutes; purchases can be delayed.",
       page_no=1, heading_path="Elasticity"),
    _p(10, 12, 2, "Tariffs raise costs, which firms pass on: cost-push inflation.", page_no=1,
       heading_path="Inflation > Cost-push"),
    _p(11, 13, 1, "Price elasticity of demand measures how quantity demanded responds to price.", page_no=2,
       origin="ocr"),
]


@pytest.fixture(scope="session")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal library folder: library.toml, the exam index, and the files it points at."""
    root = tmp_path_factory.mktemp("library")
    (root / "library.toml").write_text('[library]\nname = "Test Library"\n')
    build(root / "exam" / "exam.sqlite", courses=COURSES, sources=SOURCES, passages=PASSAGES,
          meta={"library_name": "Test Library"})
    for source in SOURCES:
        if source.view_path:
            path = root / source.view_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(range(256)) * 40)  # 10 KB, enough for Range tests
    return root


@pytest.fixture(scope="session")
def index_file(library: Path) -> Path:
    return library / "exam" / "exam.sqlite"
