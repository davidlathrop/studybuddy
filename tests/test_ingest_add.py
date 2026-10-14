"""``studybuddy add``: staging, resuming, replacing and recording sources (plan §5)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from fakes import Models
from studybuddy import audio, cli, transcribe, transcription
from studybuddy.ingest import add as add_module
from studybuddy.ingest import docx, pdf, pptx
from studybuddy.ingest.add import AddOptions, add_file, kind_of, remove_source
from studybuddy.ingest.convert import ConversionError
from studybuddy.ingest.model import IngestError, PassageDraft
from studybuddy.library import Library


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("STUDYBUDDY_LIBRARY", raising=False)


@pytest.fixture
def library(tmp_path: Path) -> Library:
    lib = Library.init(tmp_path / "lib")
    lib.add_course("bio101", "Biology")
    return lib


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Models:
    fakes = Models()
    monkeypatch.setattr(transcribe, "make_transcriber", fakes)
    return fakes


@pytest.fixture
def lecture(tmp_path: Path, short_mp4: Path) -> Path:
    path = tmp_path / "in" / "Lecture_04.mp4"
    path.parent.mkdir()
    shutil.copy(short_mp4, path)
    return path


def rows(library: Library, sql: str, *args: Any) -> list[dict[str, Any]]:
    with library.db() as db:
        return [dict(r) for r in db.execute(sql, args).fetchall()]


# --------------------------------------------------------------------------- kinds


@pytest.mark.parametrize(("name", "kind"), [("a.MP4", "video"), ("b.mov", "video"), ("c.pptx", "pptx"),
                                            ("d.PDF", "pdf"), ("e.docx", "docx")])
def test_kind_of(name: str, kind: str) -> None:
    assert kind_of(Path(name)) == kind


@pytest.mark.parametrize(("name", "hint"), [("a.ppt", "save it as .pptx"), ("b.doc", "save it as .docx"),
                                            ("c.key", "export it"), ("d.txt", "unsupported"), ("e", "unsupported")])
def test_unsupported_kinds(name: str, hint: str) -> None:
    with pytest.raises(IngestError, match=hint):
        kind_of(Path(name))


# --------------------------------------------------------------------------- videos


def test_add_video(library: Library, lecture: Path, models: Models) -> None:
    course = library.course("bio101")
    (library.course_dir("bio101") / "glossary.txt").write_text("Kenobi\n")
    result = add_file(library, course, lecture, AddOptions(week=4, date="2026-09-15"))
    assert result.status == "added" and result.kind == "video" and result.title == "Lecture 04"
    assert result.passages >= 1 and result.duration_s == pytest.approx(15.5, abs=1.0)
    assert models.last.calls[0]["hotwords"] == ["Kenobi"]

    videos = library.course_dir("bio101") / "videos"
    assert sorted(p.name for p in videos.iterdir()) == ["Lecture_04.exam.mp4", "Lecture_04.json", "Lecture_04.txt"]
    assert not (library.course_dir("bio101") / ".staging").exists() or not any(
        (library.course_dir("bio101") / ".staging").iterdir())
    assert audio.video_info(videos / "Lecture_04.exam.mp4") is not None

    [source] = rows(library, "SELECT * FROM source")
    assert source["kind"] == "video" and source["week"] == 4 and source["date"] == "2026-09-15"
    assert source["stored_path"] == "courses/bio101/videos/Lecture_04.json"
    assert source["view_path"] == "courses/bio101/videos/Lecture_04.exam.mp4"
    assert source["original_path"] == str(lecture.resolve())
    passages = rows(library, "SELECT * FROM passage ORDER BY ord")
    assert all(p["origin"] == "speech" and p["t_end_ms"] > p["t_start_ms"] for p in passages)
    assert passages[0]["t_start_ms"] == 0 and "Hello there." in passages[0]["text"]

    again = add_file(library, course, lecture)
    assert again.status == "skipped" and again.source_id == source["id"]
    assert len(models.created) == 1  # nothing was transcribed twice


def test_add_video_resumes_from_a_staged_transcript(
    library: Library, lecture: Path, models: Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    course = library.course("bio101")
    real = add_module.ingest_video

    def crash_after_transcription(src: Path, staging: Path, **kwargs: Any) -> Any:
        monkeypatch.setattr(audio, "make_exam_copy", _boom)
        return real(src, staging, **kwargs)

    monkeypatch.setattr(add_module, "ingest_video", crash_after_transcription)
    with pytest.raises(RuntimeError, match="boom"):
        add_file(library, course, lecture)
    assert rows(library, "SELECT * FROM source") == []
    staged = list((library.course_dir("bio101") / ".staging").glob("*/Lecture_04.json"))
    assert len(staged) == 1

    monkeypatch.undo()
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(library.root.parent / "config"))
    monkeypatch.setattr(transcribe, "make_transcriber", lambda *a, **k: pytest.fail("transcribed again"))
    result = add_file(library, course, lecture)
    assert result.status == "added" and result.passages >= 1


def _boom(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("boom")


def test_same_name_different_file_needs_replace(library: Library, lecture: Path, models: Models,
                                                tmp_path: Path, short_mp4: Path) -> None:
    course = library.course("bio101")
    first = add_file(library, course, lecture)
    other = tmp_path / "other" / "Lecture_04.mp4"
    other.parent.mkdir()
    shutil.copy(short_mp4, other)
    with open(other, "ab") as f:
        f.write(b"\0" * 16)  # different hash, still a playable MP4

    with pytest.raises(IngestError, match="--replace"):
        add_file(library, course, other)
    replaced = add_file(library, course, other, AddOptions(replace=True))
    assert replaced.status == "replaced"
    ids = [r["id"] for r in rows(library, "SELECT id FROM source")]
    assert ids == [replaced.source_id] and first.source_id not in ids
    assert (library.course_dir("bio101") / "videos" / "Lecture_04.json").is_file()


def test_bad_date(library: Library, lecture: Path) -> None:
    with pytest.raises(IngestError, match="YYYY-MM-DD"):
        add_file(library, library.course("bio101"), lecture, AddOptions(date="15/09/2026"))


def test_missing_file(library: Library, tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="file not found"):
        add_file(library, library.course("bio101"), tmp_path / "nope.pdf")


# --------------------------------------------------------------------------- documents


class FakeConverter:
    def __init__(self, fail: bool = False, installed: bool = True) -> None:
        self.fail = fail
        self.installed = installed
        self.converted: list[str] = []

    def __enter__(self) -> FakeConverter:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def can_convert(self, path: Path) -> bool:
        return self.installed

    def to_pdf(self, src: Path, dst: Path) -> Path:
        self.converted.append(src.name)
        if self.fail:
            raise ConversionError("PowerPoint is having a bad day")
        dst.write_bytes(b"%PDF-fake")
        return dst


@pytest.fixture
def fake_extractors(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"hidden": [], "pdf_pages": 3, "unhidden": []}
    slides = [PassageDraft("Title one", slide_no=1), PassageDraft("notes", origin="notes", slide_no=1),
              PassageDraft("Title two", slide_no=2), PassageDraft("Title three", slide_no=3)]
    monkeypatch.setattr(pptx, "extract", lambda path, ocr=True: slides)
    monkeypatch.setattr(pptx, "slide_count", lambda path: 3)
    monkeypatch.setattr(pptx, "hidden_slides", lambda path: state["hidden"])

    def unhide_all(src: Path, dst: Path) -> int:
        state["unhidden"].append(dst.name)
        shutil.copy(src, dst)
        return len(state["hidden"])

    monkeypatch.setattr(pptx, "unhide_all", unhide_all)
    monkeypatch.setattr(pdf, "extract", lambda path, ocr=True: [PassageDraft("page text", page_no=1)])
    monkeypatch.setattr(pdf, "page_count", lambda path: state["pdf_pages"])
    monkeypatch.setattr(pdf, "page_texts", lambda path: ["intro", "chapter two"])
    monkeypatch.setattr(docx, "extract", lambda path: [PassageDraft("intro", heading_path="A"),
                                                       PassageDraft("chapter two", heading_path="B")])
    monkeypatch.setattr(docx, "assign_pages", lambda drafts, texts: [
        PassageDraft(d.text, heading_path=d.heading_path, page_no=texts.index(d.text) + 1) for d in drafts])
    return state


def doc(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "in" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(name.encode() * 10)
    return path


def test_add_pptx_converts_an_unhidden_copy(library: Library, tmp_path: Path, fake_extractors: dict) -> None:
    fake_extractors["hidden"] = [2]
    converter = FakeConverter()
    result = add_file(library, library.course("bio101"), doc(tmp_path, "week04.pptx"), converter=converter)
    assert result.status == "added" and result.page_count == 3 and result.passages == 4 and not result.warnings
    assert converter.converted == ["unhidden-week04.pptx"] and fake_extractors["unhidden"] == ["unhidden-week04.pptx"]
    docs = library.course_dir("bio101") / "docs"
    assert sorted(p.name for p in docs.iterdir()) == ["week04.pptx", "week04.pptx.pdf"]
    [source] = rows(library, "SELECT stored_path, view_path, page_count FROM source")
    assert source == {"stored_path": "courses/bio101/docs/week04.pptx",
                      "view_path": "courses/bio101/docs/week04.pptx.pdf", "page_count": 3}
    origins = [(r["slide_no"], r["origin"]) for r in rows(library, "SELECT slide_no, origin FROM passage ORDER BY ord")]
    assert origins == [(1, "text"), (1, "notes"), (2, "text"), (3, "text")]


def test_pptx_page_mismatch_is_a_warning(library: Library, tmp_path: Path, fake_extractors: dict) -> None:
    fake_extractors["pdf_pages"] = 2
    result = add_file(library, library.course("bio101"), doc(tmp_path, "w.pptx"), converter=FakeConverter())
    assert result.status == "added" and "2 pages for 3 slides" in result.warnings[0]


def test_failed_conversion_still_adds_the_document(library: Library, tmp_path: Path, fake_extractors: dict) -> None:
    result = add_file(library, library.course("bio101"), doc(tmp_path, "w.pptx"), converter=FakeConverter(fail=True))
    assert result.status == "added" and "bad day" in result.warnings[0]
    [source] = rows(library, "SELECT view_path, warnings FROM source")
    assert source["view_path"] is None and "bad day" in source["warnings"]
    assert sorted(p.name for p in (library.course_dir("bio101") / "docs").iterdir()) == ["w.pptx"]


def test_no_convert_option(library: Library, tmp_path: Path, fake_extractors: dict) -> None:
    converter = FakeConverter()
    result = add_file(library, library.course("bio101"), doc(tmp_path, "w.docx"), AddOptions(convert=False),
                      converter=converter)
    assert converter.converted == [] and "turned off or unavailable" in result.warnings[0]
    assert {r["page_no"] for r in rows(library, "SELECT page_no FROM passage")} == {None}


def test_add_docx_places_passages_on_pdf_pages(library: Library, tmp_path: Path, fake_extractors: dict) -> None:
    result = add_file(library, library.course("bio101"), doc(tmp_path, "notes.docx"), converter=FakeConverter())
    assert result.page_count == 2
    got = [(r["heading_path"], r["page_no"]) for r in rows(library, "SELECT * FROM passage ORDER BY ord")]
    assert got == [("A", 1), ("B", 2)]
    [source] = rows(library, "SELECT view_path FROM source")
    assert source["view_path"] == "courses/bio101/docs/notes.docx.pdf"


def test_add_pdf_is_its_own_view(library: Library, tmp_path: Path, fake_extractors: dict) -> None:
    converter = FakeConverter()
    result = add_file(library, library.course("bio101"), doc(tmp_path, "reading.pdf"), converter=converter)
    assert result.page_count == 3 and converter.converted == [] and not result.warnings
    [source] = rows(library, "SELECT stored_path, view_path FROM source")
    assert source["stored_path"] == source["view_path"] == "courses/bio101/docs/reading.pdf"


def test_slides_and_handout_with_the_same_stem_do_not_collide(library: Library, tmp_path: Path,
                                                              fake_extractors: dict) -> None:
    course = library.course("bio101")
    add_file(library, course, doc(tmp_path, "week04.pptx"), converter=FakeConverter())
    add_file(library, course, doc(tmp_path, "week04.pdf"), converter=FakeConverter())
    add_file(library, course, doc(tmp_path, "week04.docx"), converter=FakeConverter())
    views = sorted(r["view_path"] for r in rows(library, "SELECT view_path FROM source"))
    assert views == ["courses/bio101/docs/week04.docx.pdf", "courses/bio101/docs/week04.pdf",
                     "courses/bio101/docs/week04.pptx.pdf"]


def test_remove_source_deletes_files_and_passages(library: Library, tmp_path: Path, fake_extractors: dict) -> None:
    result = add_file(library, library.course("bio101"), doc(tmp_path, "w.pptx"), converter=FakeConverter())
    remove_source(library, result.source_id)
    assert rows(library, "SELECT * FROM source") == [] and rows(library, "SELECT * FROM passage") == []
    assert list((library.course_dir("bio101") / "docs").iterdir()) == []


def test_a_stray_file_blocks_unless_replace(library: Library, tmp_path: Path, fake_extractors: dict) -> None:
    docs = library.course_dir("bio101") / "docs"
    (docs / "w.pdf").write_bytes(b"left over")
    with pytest.raises(IngestError, match="already exists"):
        add_file(library, library.course("bio101"), doc(tmp_path, "w.pdf"))
    assert add_file(library, library.course("bio101"), doc(tmp_path, "w.pdf"), AddOptions(replace=True)).status == "added"


# --------------------------------------------------------------------------- commands


def cli_run(*args: Any) -> int:
    return cli.main([str(a) for a in args])


def test_commands_end_to_end(tmp_path: Path, lecture: Path, models: Models, fake_extractors: dict,
                             monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("studybuddy.commands.Converter", FakeConverter)
    root = tmp_path / "fall"
    assert cli_run("library", "init", root, "--name", "Fall 2026") == 0
    assert cli_run("course", "add", "bio101", "--title", "Biology 101") == 0
    assert cli_run("add", "bio101", lecture, doc(tmp_path, "week04.pptx"), "--week", "4") == 0
    out = capsys.readouterr().out
    assert "added    video Lecture 04" in out and "added    pptx  week04  (3 slides, 4 passages)" in out

    assert cli_run("add", "bio101", lecture) == 0
    assert "skipped  video Lecture 04  (already added)" in capsys.readouterr().out

    assert cli_run("status") == 0
    out = capsys.readouterr().out
    assert "Fall 2026" in out and "bio101" in out and "Biology 101" in out
    assert cli_run("status", "bio101") == 0
    out = capsys.readouterr().out
    assert "video  Lecture 04" in out and "pptx   week04  (3 slides, 4 passages)" in out
    assert cli_run("course", "list") == 0
    assert "bio101       Biology 101  (2 files)" in capsys.readouterr().out


def test_add_reports_failures_and_continues(tmp_path: Path, fake_extractors: dict,
                                           monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("studybuddy.commands.Converter", FakeConverter)
    root = tmp_path / "fall"
    cli_run("library", "init", root)
    cli_run("course", "add", "bio101", "--title", "Biology")
    capsys.readouterr()
    status = cli_run("add", "bio101", doc(tmp_path, "old.ppt"), doc(tmp_path, "ok.pdf"))
    captured = capsys.readouterr()
    assert status == 1
    assert "added    pdf   ok" in captured.out
    assert "old.ppt: old.ppt: .ppt files are not supported; save it as .pptx first" in captured.err
    assert "1 of 2 files could not be added" in captured.err


def test_add_errors(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "fall"
    cli_run("library", "init", root)
    assert cli_run("add", "chem", tmp_path / "x.pdf") == 1
    assert "no course 'chem'" in capsys.readouterr().err
    cli_run("course", "add", "bio101", "--title", "Biology")
    assert cli_run("add", "bio101", "a.pdf", "b.pdf", "--title", "T") == 1
    assert "--title applies to a single file" in capsys.readouterr().err


def test_no_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli_run("status") == 1
    assert "no library found" in capsys.readouterr().err
