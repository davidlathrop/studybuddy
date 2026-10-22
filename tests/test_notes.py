"""Study notes: checking and rendering (``enrich/notes.py``) and the library flow (``study_notes.py``).

Claude is replaced by ``fakes.FakeClaude`` (installed for every test by conftest), which
makes one section per passage and quotes the start of each passage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeClaude, Models
from studybuddy import cli, study_notes, transcribe
from studybuddy.enrich import load_prompt
from studybuddy.enrich.backends import EnrichmentError, UsageLimitError, make_backend, model_id
from studybuddy.enrich.notes import (
    IndexEntry,
    Section,
    StudyNotes,
    fit_sections,
    render_index,
    render_lecture,
    section_groups,
    write_notes,
)
from studybuddy.enrich.passages import Passage
from studybuddy.enrich.schemas import KeyTerm, NotesOutline, NotesSectionPlan, NotesWindow, strict_json_schema
from studybuddy.library import DB_PATH, SESSION_DIR, Library

TEXTS = [
    "Glycolysis splits glucose into two molecules of pyruvate in the cytoplasm.",
    "The Krebs cycle runs in the mitochondria and releases carbon dioxide.",
    "Oxidative phosphorylation makes most of the ATP using the electron transport chain.",
]


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDYBUDDY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("STUDYBUDDY_LIBRARY", raising=False)


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Models:
    fakes = Models()
    monkeypatch.setattr(transcribe, "make_transcriber", fakes)
    return fakes


@pytest.fixture
def library(tmp_path: Path) -> Library:
    lib = Library.init(tmp_path / "lib")
    lib.add_course("bio101", "Biology 101")
    return lib


def passages(texts: list[str] = TEXTS, length: float = 60.0) -> list[Passage]:
    return [Passage(i, (i - 1) * length, i * length, text) for i, text in enumerate(texts, start=1)]


def add_video(library: Library, title: str, texts: list[str] = TEXTS, *, slug: str = "bio101",
              week: int | None = None) -> int:
    """A transcribed video, recorded straight into the database (no media needed)."""
    course = library.course(slug)
    stored = library.course_dir(slug) / "videos" / f"{title}.json"
    with library.db() as db:
        cur = db.execute(
            """INSERT INTO source(course_id, kind, title, week, original_path, stored_path, duration_s, sha256,
                                  extractor, added_at)
               VALUES (?, 'video', ?, ?, ?, ?, ?, ?, 'test', '2026-10-01T00:00:00+00:00')""",
            (course.id, title, week, f"/in/{title}.mp4", library.relative(stored), 60.0 * len(texts),
             hashlib.sha256(title.encode()).hexdigest()),
        )
        db.executemany(
            "INSERT INTO passage(source_id, ord, t_start_ms, t_end_ms, origin, text, text_sha) VALUES (?, ?, ?, ?, 'speech', ?, '')",
            [(cur.lastrowid, i, (i - 1) * 60000, i * 60000, text) for i, text in enumerate(texts, start=1)],
        )
    return int(cur.lastrowid)


def notes_dir(library: Library) -> Path:
    return library.course_dir("bio101") / "notes"


# --------------------------------------------------------------------------- schemas and prompts


@pytest.mark.parametrize("model", [NotesOutline, NotesWindow])
def test_notes_schemas_are_strict(model: type) -> None:
    text = json.dumps(strict_json_schema(model))
    assert "$ref" not in text and '"additionalProperties": false' in text


def test_notes_prompts_load_and_format() -> None:
    assert "Never copy timestamps" in load_prompt("notes_system_v1")
    assert "announcements" in load_prompt("notes_outline_v1")
    text = load_prompt("notes_sections_v1").format(first=2, last=4, sections="- S1 (P1-P3): Intro")
    assert "sections S2-S4" in text and "- S1 (P1-P3): Intro" in text


# --------------------------------------------------------------------------- sections


def plan(title: str, first: int, last: int) -> NotesSectionPlan:
    return NotesSectionPlan(title=title, first_passage_id=first, last_passage_id=last)


def spans(plans: list[NotesSectionPlan]) -> list[tuple[str, int, int]]:
    return [(p.title, p.first_passage_id, p.last_passage_id) for p in plans]


def test_fit_sections_closes_gaps_and_overlaps() -> None:
    given = [plan("A", 2, 4), plan("B", 4, 9), plan("C", 12, 20), plan("D", 7, 30), plan("E", 99, 99)]
    # A is moved to start at P1; D starts before C and is dropped; E is clamped to the last passage.
    assert spans(fit_sections(given, 15, fallback_title="X")) == [("A", 1, 3), ("B", 4, 11), ("C", 12, 14),
                                                                  ("E", 15, 15)]


def test_fit_sections_without_sections_makes_one() -> None:
    assert spans(fit_sections([], 7, fallback_title="Lecture")) == [("Lecture", 1, 7)]
    assert spans(fit_sections([plan(" ", 1, 7)], 7, fallback_title="Lecture")) == [("Lecture", 1, 7)]


def test_section_groups_cover_about_size_passages() -> None:
    sections = [plan("a", 1, 8), plan("b", 9, 16), plan("c", 17, 40), plan("d", 41, 42)]
    assert section_groups(sections, 20) == [[1, 2], [3], [4]]


# --------------------------------------------------------------------------- writing and checking


def test_write_notes_outline_then_sections(fake_claude: FakeClaude) -> None:
    fractions: list[float] = []
    result = write_notes(passages(), "Lecture 4", backend=fake_claude, model="m", effort="low",
                         on_progress=fractions.append)
    assert [c["output"] for c in fake_claude.calls] == ["NotesOutline", "NotesWindow"]
    first = fake_claude.calls[0]
    assert first["system"] == load_prompt("notes_system_v1") and first["effort"] == "low"
    assert "[P2 01:00-02:00] The Krebs cycle" in first["context"]  # the transcript rides in the cached prefix
    assert all(c["context"] == first["context"] for c in fake_claude.calls)
    assert fractions == [0.5, 1.0]

    notes = result.notes
    assert notes.title == "Notes title" and [s.title for s in notes.sections] == ["Topic 1", "Topic 2", "Topic 3"]
    assert notes.sections[1].key_terms[0].quote == "The Krebs"  # verified, kept
    assert notes.times[2] == (120.0, 180.0)
    assert result.calls == 2 and result.usage.output_tokens == 40 and result.cost_usd == pytest.approx(0.02)
    assert result.meta["five_hour_utilization"] == 0.25 and result.quotes_dropped == 0


def test_long_lectures_are_written_in_groups(fake_claude: FakeClaude) -> None:
    result = write_notes(passages([f"Point number {i} of the lecture." for i in range(45)]), "Long",
                         backend=fake_claude, model="m")
    prompts = [c["prompt"] for c in fake_claude.calls[1:]]
    assert ["sections S1-S20" in prompts[0], "sections S21-S40" in prompts[1], "sections S41-S45" in prompts[2]] == [
        True, True, True]
    assert len(result.notes.sections) == 45 and all(s.summary for s in result.notes.sections)


def test_skipped_sections_are_asked_for_once_more(fake_claude: FakeClaude, caplog: pytest.LogCaptureFixture) -> None:
    fake_claude.skip = {2}
    result = write_notes(passages(), "Lecture 4", backend=fake_claude, model="m")
    assert "sections S2-S2" in fake_claude.calls[2]["prompt"]  # the retry asks only for what is missing
    assert len(fake_claude.calls) == 3  # outline, group, one retry
    assert result.notes.sections[1].summary == "" and result.notes.sections[2].summary == "Summary 3."
    assert "no notes came back for section 2 (Topic 2)" in caplog.text


def test_unfound_quotes_are_dropped(fake_claude: FakeClaude, monkeypatch: pytest.MonkeyPatch) -> None:
    original = fake_claude.call

    def invent(**kwargs: Any):
        reply = original(**kwargs)
        if isinstance(reply.data, NotesWindow):
            reply.data.sections[0].key_terms[0] = KeyTerm(term="Mitosis", definition="d", passage_id=1,
                                                          quote="mitosis splits the nucleus in two")
        return reply

    monkeypatch.setattr(fake_claude, "call", invent)
    result = write_notes(passages(), "Lecture 4", backend=fake_claude, model="m")
    assert result.quotes_dropped == 1 and result.notes.sections[0].key_terms[0].quote == ""
    assert result.notes.sections[1].key_terms[0].quote == "The Krebs"


def test_no_passages_is_an_error(fake_claude: FakeClaude) -> None:
    with pytest.raises(EnrichmentError, match="no speech"):
        write_notes([], "Silent", backend=fake_claude, model="m")
    assert fake_claude.calls == []


# --------------------------------------------------------------------------- Markdown


def sample_notes() -> StudyNotes:
    term = KeyTerm(term="ATP (adenosine triphosphate)", definition="The cell's energy\ncurrency.", passage_id=2,
                   quote="ATP is the energy currency")
    sections = [
        Section(title="Energy", first_passage_id=1, last_passage_id=2, summary="Why cells need energy.",
                key_points=[{"point": "Cells run on ATP.", "details": ["made in mitochondria", ""], "passage_id": 2}],
                key_terms=[term], examples=[{"text": "A muscle cell.", "passage_id": 1}],
                emphasis=[{"text": "This will be on the exam.", "passage_id": 99}],
                review_questions=[{"question": "What is ATP?", "answer": "The energy currency.", "passage_id": 2}]),
        Section(title="Glycolysis", first_passage_id=3, last_passage_id=3),
    ]
    return StudyNotes(title="Cell energy", overview="How cells get energy.",
                      announcements=[{"text": "Quiz on Friday.", "passage_id": 3}], sections=sections,
                      times=[(0.0, 61.0), (61.0, 125.0), (125.0, 3700.0)])


def test_render_lecture() -> None:
    text = render_lecture(sample_notes(), source="Lecture 04", duration_s=3700, model="claude-opus-5-5",
                          written="2026-10-22")
    assert text.startswith("# Cell energy\n\n**Lecture 04** · 1:01:40 · study notes by claude-opus-5-5, 2026-10-22\n")
    assert "## Announcements\n\n- Quiz on Friday. [02:05]\n" in text
    assert "## Contents\n\n1. Energy [00:00–02:05]\n2. Glycolysis [02:05–1:01:40]\n" in text
    assert "- Cells run on ATP. [01:01]\n  - made in mitochondria\n\n" in text  # the empty detail is left out
    assert "- **ATP (adenosine triphosphate)**: The cell's energy currency. [01:01]\n  > “ATP is the energy currency”" in text
    assert "- This will be on the exam.\n" in text  # an id that isn't a passage gets no time
    assert "## 2. Glycolysis [02:05–1:01:40]\n\n_No notes were written for this section._" in text
    questions, answers = text.split("## Review questions")[1].split("## Answers")
    assert "1. What is ATP?" in questions and "energy currency" not in questions
    assert "1. The energy currency. [01:01]" in answers


def test_render_index_links_lectures_and_merges_terms() -> None:
    notes = sample_notes()
    later = notes.model_copy(update={"title": "More energy", "announcements": []})
    text = render_index("Biology 101", [IndexEntry("Lecture 04", "Lecture 04.notes.md", notes, week=4, date="2026-09-15",
                                                   duration_s=3700),
                                        IndexEntry("Lecture 06", "Lecture 06.notes.md", later)])
    assert "### 1. [Cell energy](Lecture%2004.notes.md)\n\nLecture 04 · week 4 · 2026-09-15 · 1:01:40\n" in text
    assert "### 2. [More energy](Lecture%2006.notes.md)\n\nLecture 06\n" in text
    assert "- Quiz on Friday. (Lecture 04 [02:05])" in text
    assert text.count("**ATP (adenosine triphosphate)**") == 1
    assert "(Lecture 04 [01:01]; Lecture 06 [01:01])" in text and "### A\n" in text


# --------------------------------------------------------------------------- the library


def run_notes(library: Library, *args: str) -> int:
    return cli.main(["notes", *args, "--library", str(library.root)])


def test_notes_command_writes_lecture_files_and_an_index(library: Library, fake_claude: FakeClaude,
                                                         capsys: pytest.CaptureFixture) -> None:
    add_video(library, "Lecture 10", week=5)
    add_video(library, "Lecture 4", TEXTS[:2], week=5)
    assert run_notes(library) == 0
    out, err = capsys.readouterr()
    folder = notes_dir(library)
    assert out.splitlines() == [str(folder / "Lecture 4.notes.md"), str(folder / "Lecture 10.notes.md"),
                                str(folder / "index.md")]
    assert "Writing study notes for Lecture 4 (2:00, 2 passages) with claude-opus-5-5" in err
    assert "Lecture 4: 2 sections, 2 key terms, 2 review questions; Claude 5-hour allowance 25% used" in err

    lecture = (folder / "Lecture 10.notes.md").read_text()
    assert lecture.startswith("# Notes title\n\n**Lecture 10** · 03:00 · study notes by claude-opus-5-5, ")
    index = (folder / "index.md").read_text()
    assert index.index("Lecture%204.notes.md") < index.index("Lecture%2010.notes.md")  # natural order

    with library.db() as db:
        runs = db.execute("SELECT task, prompt_version, model_id, params_json, input_tokens, cost_usd FROM run").fetchall()
        stored = db.execute("SELECT data FROM notes").fetchall()
    assert [tuple(r) for r in runs][0] == ("notes", "notes_v1", "claude-opus-5-5",
                                           '{"backend": "claude-cli", "effort": "medium", "calls": 2}', 20,
                                           pytest.approx(0.02))
    assert len(stored) == 2 and StudyNotes.model_validate_json(stored[0]["data"]).title == "Notes title"


def test_up_to_date_notes_are_not_written_again(library: Library, fake_claude: FakeClaude,
                                                capsys: pytest.CaptureFixture) -> None:
    add_video(library, "Lecture 4")
    run_notes(library)
    calls = len(fake_claude.calls)
    lecture = notes_dir(library) / "Lecture 4.notes.md"
    lecture.write_text(lecture.read_text() + "\nMy own note.\n")
    capsys.readouterr()

    assert run_notes(library) == 0
    assert len(fake_claude.calls) == calls  # nothing sent
    assert "My own note." in lecture.read_text()  # edits are kept
    assert capsys.readouterr().out.splitlines() == [str(notes_dir(library) / "index.md")]

    lecture.unlink()  # a deleted file comes back from the database, without asking Claude
    assert run_notes(library) == 0
    assert len(fake_claude.calls) == calls and lecture.is_file()

    assert run_notes(library, "--redo") == 0
    assert len(fake_claude.calls) == calls + 2
    with library.db() as db:
        assert db.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1


def test_a_changed_transcript_gets_new_notes(library: Library, fake_claude: FakeClaude) -> None:
    source_id = add_video(library, "Lecture 4")
    run_notes(library)
    with library.db() as db:
        db.execute("UPDATE passage SET text = 'Corrected text.' WHERE source_id = ? AND ord = 1", (source_id,))
    run_notes(library)
    assert len(fake_claude.calls) == 4


def test_one_failure_does_not_stop_the_others(library: Library, fake_claude: FakeClaude,
                                              monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    add_video(library, "Lecture 1")
    add_video(library, "Lecture 2")
    original = fake_claude.call

    def decline_lecture_1(**kwargs: Any):
        if "Lecture 1" in kwargs["context"]:
            raise EnrichmentError("m declined the request")
        return original(**kwargs)

    monkeypatch.setattr(fake_claude, "call", decline_lecture_1)
    assert run_notes(library, "bio101") == 1
    out, err = capsys.readouterr()
    assert "Lecture 1: study notes not written: m declined the request" in err
    assert "the study notes of 1 video could not be written; try again with: studybuddy notes bio101" in err
    assert out.splitlines() == [str(notes_dir(library) / "Lecture 2.notes.md"), str(notes_dir(library) / "index.md")]


def test_a_usage_limit_stops_the_run(library: Library, fake_claude: FakeClaude, capsys: pytest.CaptureFixture) -> None:
    add_video(library, "Lecture 1")
    add_video(library, "Lecture 2")
    fake_claude.fail, fake_claude.fail_on = UsageLimitError("resets 3:45pm", resets_at=None), 3  # Lecture 2's outline
    assert run_notes(library) == 1
    err = capsys.readouterr().err
    assert ("Claude's usage limit was reached (resets 3:45pm); write the remaining notes then with: "
            "studybuddy notes bio101") in err
    assert len(fake_claude.calls) == 3
    assert (notes_dir(library) / "Lecture 1.notes.md").is_file()
    assert not (notes_dir(library) / "Lecture 2.notes.md").exists()


def test_a_known_reset_time_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    resets_at = datetime.now().replace(hour=15, minute=45, second=0, microsecond=0).timestamp()
    assert study_notes.describe_limit(UsageLimitError("x", resets_at)) == (
        "Claude's usage limit was reached (it resets at 15:45)")


def test_without_the_claude_command(library: Library, monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("studybuddy.enrich.backends.shutil.which", lambda name: None)
    with pytest.raises(EnrichmentError, match="the claude command was not found"):
        make_backend("claude-cli")
    monkeypatch.setattr(study_notes, "make_backend", make_backend)
    add_video(library, "Lecture 1")
    assert run_notes(library) == 1
    assert "study notes not written: the claude command was not found" in capsys.readouterr().err


def test_backend_choice() -> None:
    with pytest.raises(EnrichmentError, match="unknown enrichment backend 'gpt'"):
        make_backend("gpt")
    assert model_id("bedrock", "claude-opus-5-5") == "global.anthropic.claude-opus-5-5"
    assert model_id("bedrock", "us.anthropic.claude-opus-5-5") == "us.anthropic.claude-opus-5-5"
    assert model_id("claude-cli", "claude-opus-5-5") == "claude-opus-5-5"


def test_ctrl_c_while_writing_notes(library: Library, fake_claude: FakeClaude, capsys: pytest.CaptureFixture) -> None:
    add_video(library, "Lecture 1")
    fake_claude.fail = KeyboardInterrupt()
    assert run_notes(library) == cli.EXIT_INTERRUPTED
    assert ("Interrupted while writing the study notes for Lecture 1; notes already written are kept. "
            "Write the rest with: studybuddy notes bio101") in capsys.readouterr().err


def test_notes_command_needs_a_course(library: Library, capsys: pytest.CaptureFixture) -> None:
    empty = Library.init(library.root.parent / "empty", make_default=False)
    assert run_notes(empty) == 1
    assert "no courses yet" in capsys.readouterr().err
    assert run_notes(library, "nope") == 1
    assert "no course 'nope'" in capsys.readouterr().err


def test_status_shows_notes(library: Library, capsys: pytest.CaptureFixture) -> None:
    add_video(library, "Lecture 1")
    add_video(library, "Lecture 2")
    study_notes.update_notes(library, [library.course("bio101")], source_ids=[1])
    cli.main(["status", "--library", str(library.root)])
    out = capsys.readouterr().out
    assert "notes" in out.splitlines()[2] and " 1/2 " in out
    cli.main(["status", "bio101", "--library", str(library.root)])
    lines = capsys.readouterr().out.splitlines()
    assert "Lecture 1" in lines[-2] and "[no study notes]" not in lines[-2]
    assert "Lecture 2" in lines[-1] and lines[-1].endswith("[no study notes]")


def test_an_older_library_is_upgraded(tmp_path: Path) -> None:
    library = Library.init(tmp_path / "old")
    conn = sqlite3.connect(library.root / DB_PATH)
    conn.executescript("DROP TABLE notes; DROP TABLE run; PRAGMA user_version = 1;")
    conn.close()
    with Library(library.root).db() as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"run", "notes"} <= tables and db.execute("PRAGMA user_version").fetchone()[0] == 2


# --------------------------------------------------------------------------- after add and session


@pytest.fixture
def lecture(tmp_path: Path, short_mp4: Path) -> Path:
    path = tmp_path / "in" / "Lecture 04.mp4"
    path.parent.mkdir()
    shutil.copy(short_mp4, path)
    return path


def test_add_writes_notes_for_the_videos_it_adds(library: Library, lecture: Path, models: Models,
                                                 fake_claude: FakeClaude, capsys: pytest.CaptureFixture) -> None:
    add_video(library, "Older lecture")  # added before notes existed: left alone, but mentioned
    assert cli.main(["add", "bio101", str(lecture), "--library", str(library.root)]) == 0
    out, err = capsys.readouterr()
    path = notes_dir(library) / "Lecture 04.notes.md"
    assert str(path) in out.splitlines() and path.is_file()
    assert all("Older lecture" not in c["context"] for c in fake_claude.calls)
    assert "1 video in bio101 has no study notes yet; write them with: studybuddy notes bio101" in err

    # Replacing the video replaces its notes.
    with open(lecture, "ab") as f:
        f.write(b"\0" * 8)
    path.write_text("stale")
    assert cli.main(["add", "bio101", str(lecture), "--replace", "--library", str(library.root)]) == 0
    assert path.read_text().startswith("# Notes title")


def test_add_without_notes(library: Library, lecture: Path, models: Models, fake_claude: FakeClaude,
                           capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["add", "bio101", str(lecture), "--no-notes", "--library", str(library.root)]) == 0
    assert fake_claude.calls == [] and not notes_dir(library).exists()
    assert "1 video in bio101 has no study notes yet" not in capsys.readouterr().err  # asked for none

    config = library.root / "library.toml"
    config.write_text(config.read_text().replace("auto = true", "auto = false"))
    shutil.copy(lecture, lecture.with_name("Lecture 05.mp4"))
    with open(lecture.with_name("Lecture 05.mp4"), "ab") as f:
        f.write(b"\0" * 8)
    assert cli.main(["add", "bio101", str(lecture.with_name("Lecture 05.mp4")), "--library", str(library.root)]) == 0
    assert fake_claude.calls == []


def test_notes_problems_do_not_fail_add(library: Library, lecture: Path, models: Models, fake_claude: FakeClaude,
                                        capsys: pytest.CaptureFixture) -> None:
    fake_claude.fail = UsageLimitError("limit", resets_at=None)
    assert cli.main(["add", "bio101", str(lecture), "--library", str(library.root)]) == 0
    assert "Claude's usage limit was reached" in capsys.readouterr().err


def test_session_writes_notes_next_to_the_videos(tmp_path: Path, lecture: Path, models: Models,
                                                 capsys: pytest.CaptureFixture) -> None:
    folder = lecture.parent
    assert cli.main(["session", str(folder), "--no-search"]) == 0
    out = capsys.readouterr().out
    notes = folder / "Study notes"
    assert str(notes / "Lecture 04.notes.md") in out.splitlines() and (notes / "index.md").is_file()
    assert "# in: study notes" in (notes / "index.md").read_text()
    assert study_notes.retry_command(Library(folder / SESSION_DIR), []) == f"studybuddy notes --library {folder.resolve()}"


# --------------------------------------------------------------------------- outside a library


def test_notes_for_a_transcript_file(tmp_path: Path, fake_claude: FakeClaude, capsys: pytest.CaptureFixture) -> None:
    transcript = tmp_path / "talk.json"
    segments = [{"start": i * 60.0, "end": (i + 1) * 60.0, "text": t, "language": "English"} for i, t in enumerate(TEXTS)]
    transcript.write_text(json.dumps({"source": "talk.mp4", "duration_s": 180.0, "segments": segments}))
    assert cli.main(["notes", str(transcript)]) == 0
    path = tmp_path / "talk.notes.md"
    assert capsys.readouterr().out.splitlines() == [str(path)]
    assert "**talk.mp4** · 03:00" in path.read_text()

    (tmp_path / "bad.json").write_text("{}")
    assert cli.main(["notes", str(tmp_path / "bad.json")]) == 1
    assert "is not a studybuddy transcript" in capsys.readouterr().err

    fake_claude.fail = UsageLimitError("limit", resets_at=None)
    assert cli.main(["notes", str(transcript)]) == 1
    assert f"retry with: studybuddy notes {transcript}" in capsys.readouterr().err


def test_transcribe_writes_notes(tmp_path: Path, short_mp4: Path, models: Models, fake_claude: FakeClaude,
                                 capsys: pytest.CaptureFixture) -> None:
    video = tmp_path / "lecture.mp4"
    shutil.copy(short_mp4, video)
    assert cli.main(["transcribe", str(video)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1:] == [str(tmp_path / "lecture.txt"), str(tmp_path / "lecture.json"),
                       str(tmp_path / "lecture.notes.md")]
    assert "Hello there." in fake_claude.calls[0]["context"]


def test_transcribe_without_notes_stays_local(tmp_path: Path, short_mp4: Path, models: Models,
                                              fake_claude: FakeClaude, capsys: pytest.CaptureFixture) -> None:
    video = tmp_path / "lecture.mp4"
    shutil.copy(short_mp4, video)
    assert cli.main(["transcribe", str(video), "--no-notes"]) == 0
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / "lecture.mp3"), str(tmp_path / "lecture.txt")]
    assert fake_claude.calls == [] and not (tmp_path / "lecture.json").exists()


def test_transcribe_reports_a_notes_problem_without_failing(tmp_path: Path, short_mp4: Path, models: Models,
                                                            fake_claude: FakeClaude,
                                                            capsys: pytest.CaptureFixture) -> None:
    video = tmp_path / "lecture.mp4"
    shutil.copy(short_mp4, video)
    fake_claude.fail = EnrichmentError("m declined the request")
    assert cli.main(["transcribe", str(video)]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines()[-1] == str(tmp_path / "lecture.json")  # kept, to write the notes from later
    assert (f"error: study notes not written: m declined the request; retry with: studybuddy notes "
            f"{tmp_path / 'lecture.json'} (--no-notes skips them)") in err


def test_transcribe_skips_notes_when_nothing_was_said(tmp_path: Path, short_mp4: Path, models: Models,
                                                      fake_claude: FakeClaude) -> None:
    video = tmp_path / "lecture.mp4"
    shutil.copy(short_mp4, video)
    models.texts = ()
    assert cli.main(["transcribe", str(video)]) == 0
    assert fake_claude.calls == [] and not (tmp_path / "lecture.notes.md").exists()
