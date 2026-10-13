"""Tests for the ``studybuddy transcribe`` command line.

Unit tests swap the speech model for a fake through ``transcribe.make_transcriber``; probing,
MP3 extraction, decoding, sectioning and output writing all run for real on small generated
media. Tests marked ``model`` run the whole command with the real models
(STUDYBUDDY_MODEL_TESTS=1).
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import pytest

import media
from studybuddy import audio, cli, pipeline, transcribe
from studybuddy.cli import build_parser, main, parse_formats, read_hotwords
from studybuddy.types import SAMPLE_RATE, ProgressFn, Segment

TEXTS = ("Hello there.", "General Kenobi.")


class FakeTranscriber:
    """Splits each section into one segment per text; raises ``fail`` after ``fail_after`` calls."""

    def __init__(
        self, mode: str, *, texts: tuple[str, ...], fail: BaseException | None, fail_after: int
    ) -> None:
        self.model_id = f"fake/{mode}"
        self.texts = texts
        self.fail = fail
        self.fail_after = fail_after
        self.loads = 0
        self.calls: list[dict[str, Any]] = []

    def config(self) -> dict:
        return {"fake": True}

    def load(self) -> None:
        self.loads += 1

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_s: float = 0.0,
        language: str | None = None,
        hotwords: list[str] | None = None,
        on_progress: ProgressFn | None = None,
    ) -> list[Segment]:
        if self.fail is not None and len(self.calls) >= self.fail_after:
            raise self.fail
        seconds = len(audio) / SAMPLE_RATE
        self.calls.append(
            {"offset_s": offset_s, "seconds": seconds, "language": language, "hotwords": hotwords}
        )
        if on_progress:
            on_progress(0.5)
        step = seconds / max(len(self.texts), 1)
        return [
            Segment(offset_s + i * step, offset_s + (i + 1) * step, text, language)
            for i, text in enumerate(self.texts)
        ]


class Models:
    """Replacement for ``make_transcriber``; configure the fakes it hands out, then inspect them."""

    def __init__(self) -> None:
        self.created: list[FakeTranscriber] = []
        self.texts = TEXTS
        self.fail: BaseException | None = None
        self.fail_after = 0

    def __call__(self, mode: str, **overrides: Any) -> FakeTranscriber:
        fake = FakeTranscriber(mode, texts=self.texts, fail=self.fail, fail_after=self.fail_after)
        self.created.append(fake)
        return fake

    @property
    def last(self) -> FakeTranscriber:
        return self.created[-1]


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Models:
    fakes = Models()
    monkeypatch.setattr(transcribe, "make_transcriber", fakes)
    return fakes


@pytest.fixture
def video(tmp_path: Path, short_mp4: Path) -> Path:
    """A private copy of the 15 s test video, so outputs land in ``tmp_path``."""
    path = tmp_path / "lecture.mp4"
    shutil.copy(short_mp4, path)
    return path


def run(*args: Any) -> int:
    return main(["transcribe", *map(str, args)])


def outputs(video: Path, *exts: str) -> list[str]:
    return [str(video.with_suffix(f".{ext}")) for ext in exts]


def forbid_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    def extract_mp3(*args: Any, **kwargs: Any) -> None:
        pytest.fail("the MP3 was extracted again")

    monkeypatch.setattr(audio, "extract_mp3", extract_mp3)


def spy_on_extraction(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record the destination of every MP3 extraction (which still runs for real)."""
    extracted: list[Path] = []
    real_extract = audio.extract_mp3

    def spy(src: Path, dst: Path, **kwargs: Any) -> Path:
        extracted.append(dst)
        return real_extract(src, dst, **kwargs)

    monkeypatch.setattr(audio, "extract_mp3", spy)
    return extracted


# --------------------------------------------------------------------------- arguments


def test_defaults() -> None:
    args = build_parser().parse_args(["transcribe", "talk.mp4"])
    assert args.input == Path("talk.mp4")
    assert args.mode == "accurate" and args.formats == ["txt"]
    assert args.language is None and args.hotwords_file is None and args.out_dir is None
    assert not args.verbose and not args.quiet


def test_all_options() -> None:
    args = build_parser().parse_args([
        "transcribe", "talk.mp4", "--mode", "fast", "--language", "fr", "--formats", "srt,json",
        "--hotwords-file", "terms.txt", "--out-dir", "out", "-v",
    ])
    assert (args.mode, args.language, args.formats) == ("fast", "fr", ["srt", "json"])
    assert args.hotwords_file == Path("terms.txt") and args.out_dir == Path("out") and args.verbose


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("txt", ["txt"]),
        ("txt,srt,json", ["txt", "srt", "json"]),
        (" SRT , Json ", ["srt", "json"]),
        ("json,txt,json", ["json", "txt"]),
        ("txt,,srt,", ["txt", "srt"]),
    ],
)
def test_parse_formats(given: str, expected: list[str]) -> None:
    assert parse_formats(given) == expected


@pytest.mark.parametrize(("given", "error"), [("pdf", "unknown"), ("txt,docx", "docx"), ("", "no formats"), (" , ", "no formats")])
def test_parse_formats_rejects(given: str, error: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=error):
        parse_formats(given)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--formats", "pdf"], "argument --formats: unknown format(s) pdf"),
        (["--formats", ""], "argument --formats: no formats given"),
        (["--mode", "slow"], "argument --mode: invalid choice"),
        (["--verbose", "--quiet"], "argument -q/--quiet: not allowed with argument -v/--verbose"),
        (["--no-such-option"], "unrecognized arguments: --no-such-option"),
    ],
)
def test_bad_arguments_exit_1_before_doing_anything(
    args: list[str], error: str, video: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    assert run(video, *args) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "usage: studybuddy" in err and f"studybuddy: error: {error}" in err
    assert "Traceback" not in err
    assert not models.created and sorted(p.name for p in video.parent.iterdir()) == ["lecture.mp4"]


def test_missing_command_exits_1(capsys: pytest.CaptureFixture) -> None:
    assert main([]) == 1
    assert "required: COMMAND" in capsys.readouterr().err


def test_help_and_version_exit_0(capsys: pytest.CaptureFixture) -> None:
    assert main(["transcribe", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "usage: studybuddy transcribe" in help_text and "--hotwords-file" in help_text
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("studybuddy ")


# --------------------------------------------------------------------------- hotwords


def test_read_hotwords(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text(
        "﻿# course vocabulary\n"
        "eigenvalue\n"
        "\n"
        "  Fourier   transform  \n"
        "   # an indented comment\n"
        "C#\n"
        "eigenvalue\n"
        "Schrödinger\r\n",
        encoding="utf-8",
    )
    assert read_hotwords(path) == ["eigenvalue", "Fourier transform", "C#", "Schrödinger"]


def test_read_hotwords_empty(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("# nothing yet\n\n", encoding="utf-8")
    assert read_hotwords(path) == []


@pytest.mark.parametrize(
    ("content", "error"), [(None, "No such file"), (b"\xff\xfe\x00bad", "not UTF-8")]
)
def test_unreadable_hotwords_file_exits_1(
    content: bytes | None, error: str, video: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    path = video.parent / "terms.txt"
    if content is not None:
        path.write_bytes(content)
    assert run(video, "--hotwords-file", path) == 1
    err = capsys.readouterr().err
    assert f"cannot read hotwords file {path}" in err and error in err
    assert not models.created and not video.with_suffix(".mp3").exists()


# --------------------------------------------------------------------------- input errors


def test_missing_input_exits_1(tmp_path: Path, models: Models, capsys: pytest.CaptureFixture) -> None:
    assert run(tmp_path / "nope.mp4") == 1
    out, err = capsys.readouterr()
    assert out == "" and err.strip() == f"studybuddy: error: file not found: {tmp_path / 'nope.mp4'}"
    assert not models.created


def test_video_without_audio_exits_1(
    tmp_path: Path, no_audio_mp4: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    src = tmp_path / "silent_film.mp4"
    shutil.copy(no_audio_mp4, src)
    assert run(src) == 1
    err = capsys.readouterr().err
    assert "has no audio stream" in err and "Traceback" not in err
    assert not models.created and sorted(p.name for p in tmp_path.iterdir()) == ["silent_film.mp4"]


@pytest.mark.parametrize("kind", ["directory", "not media"])
def test_unusable_input_exits_1(
    kind: str, tmp_path: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    src = tmp_path / "input.mp4"
    if kind == "directory":
        src.mkdir()
    else:
        src.write_text("not a video", encoding="utf-8")
    assert run(src) == 1
    err = capsys.readouterr().err
    assert ("is a directory" if kind == "directory" else "cannot read") in err
    assert not models.created


def test_unsupported_language_fails_before_extraction(
    video: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    assert run(video, "--language", "Klingon") == 1
    assert "unsupported language 'Klingon'" in capsys.readouterr().err
    assert not models.created and not video.with_suffix(".mp3").exists()


# --------------------------------------------------------------------------- a full run


def test_full_run_writes_every_output(
    video: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    terms = video.parent / "terms.txt"
    terms.write_text("Kenobi\n# a comment\nlightsaber\n", encoding="utf-8")
    assert run(video, "--language", "en", "--hotwords-file", terms, "--formats", "txt,srt,json") == 0
    out, err = capsys.readouterr()

    assert out.splitlines() == outputs(video, "mp3", "txt", "srt", "json")
    assert video.with_suffix(".txt").read_text(encoding="utf-8") == "Hello there.\n\nGeneral Kenobi.\n"
    srt = video.with_suffix(".srt").read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:07,") and "\n\n2\n00:00:07," in srt
    data = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    assert (data["source"], data["model"], data["mode"], data["language"]) == (
        "lecture.mp4", "fake/accurate", "accurate", "English",
    )
    assert data["duration_s"] == pytest.approx(15.5, abs=0.1)
    assert not video.with_name("lecture.partial.jsonl").exists()
    assert audio.probe(video.with_suffix(".mp3")).channels == 1

    (fake,) = models.created
    assert fake.loads == 1  # loaded before the transcribe stage starts
    assert fake.calls == [
        {"offset_s": 0.0, "seconds": pytest.approx(15.5, abs=0.1),
         "language": "English", "hotwords": ["Kenobi", "lightsaber"]},
    ]
    assert "lecture.mp4: 0:15 of audio; accurate mode (Qwen3-ASR 1.7B)" in err
    assert "Done in" in err and "2 segments; language: English" in err
    assert "warning" not in err and "Traceback" not in err


def test_progress_is_reported_in_plain_lines_when_not_a_terminal(
    video: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    assert run(video) == 0
    err = capsys.readouterr().err
    assert "\r" not in err
    assert re.search(r"^extract +100%  done in \d+:\d\d$", err, re.M)
    assert re.search(r"^transcribe   0%  \d+:\d\d elapsed$", err, re.M)
    assert re.search(r"^transcribe 100%  done in \d+:\d\d$", err, re.M)
    assert err.index("extract") < err.index("Loading fake/accurate") < err.index("transcribe ")


def test_out_dir(video: Path, models: Models, capsys: pytest.CaptureFixture) -> None:
    out_dir = video.parent / "out" / "nested"
    assert run(video, "--out-dir", out_dir, "--formats", "srt") == 0
    assert capsys.readouterr().out.splitlines() == [str(out_dir / "lecture.mp3"), str(out_dir / "lecture.srt")]
    assert sorted(p.name for p in out_dir.iterdir()) == ["lecture.mp3", "lecture.srt"]
    assert sorted(p.name for p in video.parent.iterdir()) == ["lecture.mp4", "out"]


def test_fast_mode_ignores_language_and_hotwords_with_a_warning(
    video: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    terms = video.parent / "terms.txt"
    terms.write_text("Kenobi\n", encoding="utf-8")
    assert run(video, "--mode", "fast", "--language", "Ukrainian", "--hotwords-file", terms, "--formats", "json") == 0
    err = capsys.readouterr().err
    assert "warning: fast mode (Parakeet) detects the language itself; ignoring --language Ukrainian" in err
    assert "warning: fast mode (Parakeet) does not support hotwords" in err
    assert models.last.model_id == "fake/fast"
    assert models.last.calls[0]["language"] is None and models.last.calls[0]["hotwords"] is None
    data = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["mode"] == "fast" and data["language"] is None


def test_awkward_file_names(video: Path, models: Models, capsys: pytest.CaptureFixture) -> None:
    src = video.rename(video.with_name("-week 3: café.v2.mp4"))
    assert run(src, "--formats", "txt,json") == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [str(src.with_name(f"-week 3: café.v2.{ext}")) for ext in ("mp3", "txt", "json")]
    assert json.loads(Path(out[2]).read_text(encoding="utf-8"))["source"] == "-week 3: café.v2.mp4"


def test_only_the_first_audio_track_is_used(
    tmp_path: Path, speech_audio: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    other = media.make_silence(tmp_path / "commentary.wav", 5.0)
    src = media.make_mp4(speech_audio, tmp_path / "dual.mp4", extra_audio_tracks=[other], stereo=True)
    assert run(src) == 0
    err = capsys.readouterr().err
    assert "dual.mp4 has 2 audio tracks; using the first" in err
    assert audio.probe(src.with_suffix(".mp3")).channels == 1
    assert models.last.calls[0]["seconds"] == pytest.approx(15.5, abs=0.1)


def test_an_mp3_input_is_transcribed_in_place(
    video: Path, models: Models, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    mp3 = audio.extract_mp3(video, video.parent / "talk.mp3")
    forbid_extraction(monkeypatch)
    assert run(mp3) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == [str(mp3), str(mp3.with_suffix(".txt"))]
    assert "talk.mp3 is already the MP3" in err


def test_no_speech_still_writes_the_outputs_with_a_warning(
    video: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    models.texts = ()
    assert run(video, "--formats", "txt,srt") == 0
    err = capsys.readouterr().err
    assert "warning: no speech was recognized in lecture.mp4" in err
    assert video.with_suffix(".txt").read_text(encoding="utf-8") == ""
    assert video.with_suffix(".srt").read_text(encoding="utf-8") == ""


def test_quiet_prints_only_the_paths(video: Path, models: Models, capsys: pytest.CaptureFixture) -> None:
    assert run(video, "-q") == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == outputs(video, "mp3", "txt")
    assert err == ""


def test_verbose_shows_debug_messages(video: Path, models: Models, capsys: pytest.CaptureFixture) -> None:
    assert run(video, "--verbose") == 0
    err = capsys.readouterr().err
    assert "debug [studybuddy.audio]: running " in err


def test_logging_setup_is_undone(video: Path, models: Models, capsys: pytest.CaptureFixture) -> None:
    root, package = logging.getLogger(), logging.getLogger("studybuddy")
    before = (list(root.handlers), root.level, package.level)
    assert run(video, "-v") == 0
    models.fail = RuntimeError("bug")
    with pytest.raises(RuntimeError, match="bug"):  # unexpected errors keep their traceback
        run(video)
    assert (list(root.handlers), root.level, package.level) == before


# --------------------------------------------------------------------------- errors and Ctrl-C


def test_value_error_from_the_model_exits_1(
    video: Path, models: Models, capsys: pytest.CaptureFixture
) -> None:
    models.fail = ValueError("expected mono audio")
    assert run(video) == 1
    err = capsys.readouterr().err
    assert "studybuddy: error: expected mono audio" in err and "Traceback" not in err


def test_ctrl_c_during_extraction(
    video: Path, models: Models, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "extract_mp3", interrupted)
    assert run(video) == 130
    assert "Interrupted before any transcription was saved." in capsys.readouterr().err
    assert not models.created


_INTERRUPTED_PROGRAM = """
import atexit, sys
from studybuddy import audio, cli

def interrupted(path):
    raise KeyboardInterrupt

atexit.register(print, "clean shutdown")
audio.probe = interrupted
sys.argv = ["studybuddy", "transcribe", "lecture.mp4"]
sys.exit(cli.main())
"""


def test_ctrl_c_ends_the_program_by_sigint(tmp_path: Path) -> None:
    # A shell running studybuddy in a loop or script only stops on Ctrl-C if the child dies
    # of SIGINT; an ordinary exit with status 130 makes bash carry on with the next file.
    proc = subprocess.run(
        [sys.executable, "-c", _INTERRUPTED_PROGRAM],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == -signal.SIGINT, proc.stderr
    assert proc.stderr == "Interrupted before any transcription was saved.\n"  # no traceback
    assert proc.stdout == "clean shutdown\n"  # Python shut down normally before the signal


def test_ctrl_c_saves_progress_and_a_rerun_resumes(
    video: Path, models: Models, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 5 s sections, so the 15 s video has several and some finish before Ctrl-C.
    monkeypatch.setattr(pipeline, "transcribe_file", functools.partial(pipeline.transcribe_file, section_s=5.0))
    partial = video.with_name("lecture.partial.jsonl")
    models.fail, models.fail_after = KeyboardInterrupt(), 1

    assert run(video, "--formats", "json") == 130
    out, err = capsys.readouterr()
    assert out == ""
    assert f"Interrupted. Finished sections are saved in {partial}; run the same command again to resume." in err
    assert partial.exists() and video.with_suffix(".mp3").exists()
    assert not video.with_suffix(".json").exists()
    first = models.last.calls
    assert len(first) == 1

    models.fail = None
    extracted = spy_on_extraction(monkeypatch)
    assert run(video, "--formats", "json") == 0
    out, err = capsys.readouterr()
    # Extracted again, to the same bytes, so the resume key (which hashes the MP3) matches.
    assert extracted == [video.with_suffix(".mp3")]
    assert re.search(r"^resuming lecture\.mp3: 1 of \d+ sections already done$", err, re.M)
    assert "warning" not in err
    assert out.splitlines() == outputs(video, "mp3", "json")
    rest = models.last.calls
    assert len(rest) >= 2 and all(c["offset_s"] > first[0]["offset_s"] for c in rest)  # section 1 skipped
    segments = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))["segments"]
    assert len(segments) == 2 * (1 + len(rest))
    assert segments[0]["start"] == 0.0 and segments[-1]["end"] == pytest.approx(15.5, abs=0.1)
    assert not partial.exists()


def test_another_recording_with_the_same_name_does_not_resume_the_first(
    tmp_path: Path, short_mp4: Path, models: Models, capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two different week-1 lectures, both called lecture.mp4, transcribed into one folder.
    first, second = tmp_path / "course_a" / "lecture.mp4", tmp_path / "course_b" / "lecture.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    shutil.copy(short_mp4, first)
    media.make_mp4(media.make_silence(tmp_path / "quiet.wav", 12.0), second)
    out_dir = tmp_path / "transcripts"
    monkeypatch.setattr(pipeline, "transcribe_file", functools.partial(pipeline.transcribe_file, section_s=5.0))
    models.fail, models.fail_after = KeyboardInterrupt(), 1
    assert run(first, "--out-dir", out_dir) == 130
    assert (out_dir / "lecture.partial.jsonl").exists()

    # The second video is older than the first one's MP3, as after `cp -p`, rsync or unzip.
    old = (out_dir / "lecture.mp3").stat().st_mtime - 3600
    os.utime(second, (old, old))
    capsys.readouterr()
    models.fail = None
    assert run(second, "--out-dir", out_dir, "--formats", "json") == 0
    err = capsys.readouterr().err

    assert "resuming" not in err and "cannot resume" in err  # started over, and said why
    assert models.last.calls[0]["offset_s"] == 0.0  # every section transcribed afresh
    data = json.loads((out_dir / "lecture.json").read_text(encoding="utf-8"))
    assert data["duration_s"] == pytest.approx(12.0, abs=0.1)  # the second recording, not the first
    assert audio.probe(out_dir / "lecture.mp3").duration_s == pytest.approx(12.0, abs=0.1)


# --------------------------------------------------------------------------- the progress display


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_terminal_progress_redraws_a_single_line() -> None:
    stream, clock = io.StringIO(), FakeClock()
    display = cli._Display(stream, interactive=True, clock=clock)
    with display.stage("extract"):
        clock.now += 5
        display.update(0.25)
        display.message("a note")
        clock.now += 5
        display.update(0.5)
    display.close()
    text = stream.getvalue()
    assert text.startswith("\rextract      0%  0:00 elapsed")
    assert "\rextract     25%  0:05 elapsed" in text
    assert "\r" + " " * 29 + "\ra note\n\rextract     25%  0:05 elapsed" in text  # erased, then redrawn
    assert "\rextract     50%  0:10 elapsed, ~0:10 left" in text
    assert text.endswith("\n") and text.rsplit("\r", 1)[1].rstrip() == "extract    100%  done in 0:10"
    assert text.count("\n") == 2  # only the note and the finished stage end a line


def test_terminal_progress_keeps_the_last_state_when_interrupted() -> None:
    stream, clock = io.StringIO(), FakeClock()
    display = cli._Display(stream, interactive=True, clock=clock)
    with pytest.raises(KeyboardInterrupt), display.stage("transcribe"):
        display.update(0.3)
        raise KeyboardInterrupt
    assert stream.getvalue().endswith("\rtranscribe  30%  0:00 elapsed\n")


def test_plain_progress_prints_steps_at_most_every_five_seconds_and_a_heartbeat() -> None:
    stream, clock = io.StringIO(), FakeClock()
    display = cli._Display(stream, interactive=False, clock=clock)
    with display.stage("transcribe"):
        for i in range(100):  # 1% per second
            display.update(i / 100)
            clock.now += 1.0
        clock.now += 30
        display.update(0.99)  # no progress, but 30 s have passed: say so
    lines = stream.getvalue().splitlines()
    assert "\r" not in stream.getvalue()
    assert [line.split()[1] for line in lines[:10]] == [f"{p}%" for p in range(0, 100, 10)]
    assert lines[10].startswith("transcribe  99%  2:10 elapsed")
    assert lines[11] == "transcribe 100%  done in 2:10"
    assert len(lines) == 12


def test_plain_progress_collapses_a_burst_of_updates() -> None:
    stream, clock = io.StringIO(), FakeClock()
    display = cli._Display(stream, interactive=False, clock=clock)
    with display.stage("transcribe"):
        display.update(0.0)
        clock.now += 25
        for i in range(1, 9):  # a section's chunks all reported in the same instant
            display.update(i / 10)
        clock.now += 1
        display.update(0.85)
    lines = stream.getvalue().splitlines()
    assert [line.split()[1] for line in lines] == ["0%", "10%", "100%"]


def test_progress_can_be_switched_off() -> None:
    stream = io.StringIO()
    display = cli._Display(stream, show_progress=False, interactive=True)
    with display.stage("extract"):
        display.update(0.5)
        display.message("still shown")
    assert stream.getvalue() == "still shown\n"


@pytest.mark.parametrize(("seconds", "text"), [(0, "0:00"), (59.4, "0:59"), (75, "1:15"), (3725, "1:02:05"), (36000, "10:00:00")])
def test_clock(seconds: float, text: str) -> None:
    assert cli._clock(seconds) == text


# --------------------------------------------------------------------------- real models (STUDYBUDDY_MODEL_TESTS=1)

_SRT_CUE = re.compile(
    r"(\d+)\n(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2,}):(\d{2}):(\d{2}),(\d{3})\n([^\n]+)"
)


def parse_srt(text: str) -> list[tuple[int, float, float, str]]:
    cues = []
    for block in text.strip("\n").split("\n\n"):
        m = _SRT_CUE.fullmatch(block)
        assert m, f"malformed SRT cue: {block!r}"
        g = m.groups()
        start = int(g[1]) * 3600 + int(g[2]) * 60 + int(g[3]) + int(g[4]) / 1000
        end = int(g[5]) * 3600 + int(g[6]) * 60 + int(g[7]) + int(g[8]) / 1000
        cues.append((int(g[0]), start, end, g[9]))
    return cues


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


@pytest.mark.model
@pytest.mark.parametrize(
    ("mode", "model_id", "language"),
    [("accurate", transcribe.QWEN3_MODEL_ID, "English"), ("fast", transcribe.PARAKEET_MODEL_ID, None)],
)
def test_real_end_to_end(
    mode: str, model_id: str, language: str | None,
    video: Path, speech_text: str, capsys: pytest.CaptureFixture,
) -> None:
    started = time.perf_counter()
    rc = main(["transcribe", str(video), "--mode", mode, "--formats", "txt,srt,json"])
    elapsed = time.perf_counter() - started
    out, err = capsys.readouterr()
    assert rc == 0, err
    assert out.splitlines() == outputs(video, "mp3", "txt", "srt", "json")

    txt = video.with_suffix(".txt").read_text(encoding="utf-8")
    score = jiwer.wer(_norm(speech_text), _norm(txt))
    cues = parse_srt(video.with_suffix(".srt").read_text(encoding="utf-8"))
    data = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"\n[model] cli {mode}: WER {score:.3f}, {len(cues)} cues, {elapsed:.2f} s end to end")

    assert score < 0.1
    duration = data["duration_s"]
    assert duration == pytest.approx(15.5, abs=0.1)
    assert [n for n, *_ in cues] == list(range(1, len(cues) + 1))
    assert all(0.0 <= start < end <= duration + 0.001 for _, start, end, _ in cues)
    assert [start for _, start, _, _ in cues] == sorted(start for _, start, _, _ in cues)
    assert cues[0][1] < 1.0 and cues[-1][2] > duration - 2.0
    assert (data["source"], data["model"], data["mode"], data["language"]) == (
        "lecture.mp4", model_id, mode, language,
    )
    assert [s["text"] for s in data["segments"]] == [text for *_, text in cues]
    assert not video.with_name("lecture.partial.jsonl").exists()
    assert "warning" not in err
