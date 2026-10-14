"""The semester library: creating, finding, settings, courses and the schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from studybuddy.library import CONFIG_FILE, DB_PATH, MIGRATIONS, Library, LibraryError, Settings


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default-library pointer and $STUDYBUDDY_LIBRARY away from the real ones."""
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("STUDYBUDDY_LIBRARY", raising=False)


def test_init_creates_layout_settings_and_schema(tmp_path: Path) -> None:
    lib = Library.init(tmp_path / "fall-2026", name='Fall "26"')
    assert lib.settings.name == 'Fall "26"'
    for sub in ("courses", "build", "exam"):
        assert (lib.root / sub).is_dir()
    assert lib.settings.transcribe.mode == "accurate" and lib.settings.transcribe.language is None
    assert lib.settings.exam_copy.height == 540 and lib.settings.documents.ocr
    assert lib.settings.enrich.model == "claude-opus-5-5"
    with sqlite3.connect(lib.root / DB_PATH) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"course", "source", "passage"} <= tables
    assert (tmp_path / "config" / "default-library").read_text().strip() == str(lib.root)


def test_init_refuses_an_existing_library(tmp_path: Path) -> None:
    Library.init(tmp_path / "lib")
    with pytest.raises(LibraryError, match="already a StudyBuddy library"):
        Library.init(tmp_path / "lib")


def test_init_without_default_leaves_the_pointer_alone(tmp_path: Path) -> None:
    Library.init(tmp_path / "lib", make_default=False)
    assert not (tmp_path / "config" / "default-library").exists()


def test_find_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = Library.init(tmp_path / "a")  # becomes the default
    b = Library.init(tmp_path / "b", make_default=False)
    c = Library.init(tmp_path / "c", make_default=False)
    (b.root / "courses" / "deep").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert Library.find(c.root).root == c.root  # explicit wins
    monkeypatch.setenv("STUDYBUDDY_LIBRARY", str(c.root))
    assert Library.find(cwd=b.root / "courses" / "deep").root == c.root  # then the environment
    monkeypatch.delenv("STUDYBUDDY_LIBRARY")
    assert Library.find(cwd=b.root / "courses" / "deep").root == b.root  # then the enclosing folder
    assert Library.find(cwd=elsewhere).root == a.root  # then the default


def test_find_without_any_library(tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="no library found"):
        Library.find(cwd=tmp_path)


def test_not_a_library(tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="not a StudyBuddy library"):
        Library(tmp_path)


def test_settings_read_edits_ignore_unknown_keys_and_check_mode(tmp_path: Path) -> None:
    lib = Library.init(tmp_path / "lib")
    config = lib.root / CONFIG_FILE
    config.write_text(
        config.read_text()
        .replace('mode = "accurate"', 'mode = "fast"')
        .replace('language = ""', 'language = "fr"')
        + "\n[transcribe.extra]\nunknown = 1\n"
    )
    settings = Library(lib.root).settings
    assert settings.transcribe.mode == "fast" and settings.transcribe.language == "fr"

    config.write_text(config.read_text().replace('mode = "fast"', 'mode = "turbo"'))
    with pytest.raises(LibraryError, match="mode must be accurate or fast"):
        Library(lib.root)
    config.write_text("this is = not [toml")
    with pytest.raises(LibraryError, match=CONFIG_FILE):
        Library(lib.root)


def test_settings_defaults_for_missing_sections() -> None:
    settings = Settings.from_toml({"library": {"name": "x"}})
    assert settings.exam_copy.video_bitrate == "500k" and settings.enrich.backend == "claude-cli"


def test_courses(tmp_path: Path) -> None:
    lib = Library.init(tmp_path / "lib")
    glossary = tmp_path / "terms.txt"
    glossary.write_text("﻿eigenvalue\n# comment\n\n  Fourier   transform \neigenvalue\n", encoding="utf-8")
    course = lib.add_course("math201", "Linear Algebra", glossary=glossary)
    assert (lib.course_dir("math201") / "videos").is_dir() and (lib.course_dir("math201") / "docs").is_dir()
    assert lib.glossary_terms("math201") == ["eigenvalue", "Fourier transform"]
    assert lib.course("math201") == course
    lib.add_course("bio101", "Biology")
    assert [c.slug for c in lib.courses()] == ["bio101", "math201"]
    assert lib.glossary_terms("bio101") == []  # the template is only comments

    with pytest.raises(LibraryError, match="already exists"):
        lib.add_course("math201", "Again")
    with pytest.raises(LibraryError, match="no course 'chem'.*bio101, math201"):
        lib.course("chem")


@pytest.mark.parametrize("slug", ["", "Bio101", "bio 101", "-bio", "x" * 41, "bio/101"])
def test_invalid_course_slugs(tmp_path: Path, slug: str) -> None:
    lib = Library.init(tmp_path / "lib")
    with pytest.raises(LibraryError, match="invalid course name"):
        lib.add_course(slug, "Title")


def test_blank_title(tmp_path: Path) -> None:
    lib = Library.init(tmp_path / "lib")
    with pytest.raises(LibraryError, match="title"):
        lib.add_course("bio101", "   ")


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    lib = Library.init(tmp_path / "lib")
    with sqlite3.connect(lib.root / DB_PATH) as db:
        db.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
    with pytest.raises(LibraryError, match="newer studybuddy"):
        Library(lib.root)


def test_passages_go_with_their_source(tmp_path: Path) -> None:
    lib = Library.init(tmp_path / "lib")
    course = lib.add_course("bio101", "Biology")
    with lib.db() as db:
        sid = db.execute(
            "INSERT INTO source(course_id, kind, title, original_path, sha256, extractor, added_at) "
            "VALUES (?, 'pdf', 't', '/x.pdf', 'abc', 'pdf/1', 'now')", (course.id,)
        ).lastrowid
        db.execute("INSERT INTO passage(source_id, ord, page_no, text, text_sha) VALUES (?, 1, 1, 'hi', 'h')", (sid,))
    with lib.db() as db:
        db.execute("DELETE FROM source WHERE id = ?", (sid,))
        assert db.execute("SELECT COUNT(*) FROM passage").fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO passage(source_id, ord, text, text_sha, origin) VALUES (999, 1, 'x', 'x', 'text')")


def test_relative_and_absolute(tmp_path: Path) -> None:
    lib = Library.init(tmp_path / "lib")
    path = lib.course_dir("bio101") / "docs" / "a b.pdf"
    assert lib.relative(path) == "courses/bio101/docs/a b.pdf"
    assert lib.absolute("courses/bio101/docs/a b.pdf") == path
