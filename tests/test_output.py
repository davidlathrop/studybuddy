from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from studybuddy import output
from studybuddy.output import write_json, write_outputs, write_srt, write_txt
from studybuddy.types import Segment, Transcript


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        source="lecture.mp4",
        model="mlx-community/Qwen3-ASR-1.7B-8bit",
        mode="accurate",
        language="German",
        duration_s=360_005.0,
        segments=[
            Segment(0.0, 5.25, "  Guten Morgen, schöne Grüße!  ", "German"),
            Segment(5.25, 9.0, "", "German"),  # empty: skipped in txt/srt
            Segment(9.0, 12.5, "   \n ", None),  # whitespace only: skipped too
            Segment(12.5, 61.125, "Line one\n\nline two — naïve café 世界", "German"),
            Segment(3_725.004, 3_725.004, "zero length", None),
            Segment(360_000.5, 360_004.75, "Past one hundred hours.", "German"),
        ],
    )


def test_write_txt(transcript: Transcript, tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    write_txt(transcript, path)
    assert path.read_text(encoding="utf-8") == (
        "Guten Morgen, schöne Grüße!\n"
        "\n"
        "Line one line two — naïve café 世界\n"
        "\n"
        "zero length\n"
        "\n"
        "Past one hundred hours.\n"
    )


def test_write_srt(transcript: Transcript, tmp_path: Path) -> None:
    path = tmp_path / "out.srt"
    write_srt(transcript, path)
    assert path.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:05,250\n"
        "Guten Morgen, schöne Grüße!\n"
        "\n"
        "2\n"
        "00:00:12,500 --> 00:01:01,125\n"
        "Line one line two — naïve café 世界\n"
        "\n"
        "3\n"
        "01:02:05,004 --> 01:02:05,005\n"
        "zero length\n"
        "\n"
        "4\n"
        "100:00:00,500 --> 100:00:04,750\n"
        "Past one hundred hours.\n"
    )


def test_srt_cues_always_advance_and_clamp_negative(tmp_path: Path) -> None:
    t = Transcript("a.mp4", "m", "fast", None, 10.0, [
        Segment(-0.2, 1.0, "before zero"),
        Segment(4.0, 3.0, "backwards"),
        Segment(7.0004, 7.0001, "same millisecond"),
    ])
    path = tmp_path / "a.srt"
    write_srt(t, path)
    times = [line for line in path.read_text().splitlines() if "-->" in line]
    assert times == [
        "00:00:00,000 --> 00:00:01,000",
        "00:00:04,000 --> 00:00:04,001",
        "00:00:07,000 --> 00:00:07,001",
    ]


def test_srt_timestamps_round_to_milliseconds(tmp_path: Path) -> None:
    t = Transcript("a.mp4", "m", "fast", None, 7200.0, [Segment(59.9996, 3599.9996, "x")])
    path = tmp_path / "a.srt"
    write_srt(t, path)
    assert "00:01:00,000 --> 01:00:00,000" in path.read_text()


def test_write_json(transcript: Transcript, tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    write_json(transcript, path)
    raw = path.read_bytes().decode("utf-8")
    assert "世界" in raw and "Grüße" in raw  # not \u-escaped
    assert raw.startswith('{\n  "source": "lecture.mp4",\n')
    assert raw.endswith("}\n")
    data = json.loads(raw)
    assert list(data) == ["source", "model", "mode", "language", "duration_s", "segments"]
    assert data["model"] == "mlx-community/Qwen3-ASR-1.7B-8bit"
    assert data["mode"] == "accurate"
    assert data["language"] == "German"
    assert data["duration_s"] == 360_005.0
    assert data["segments"] == [
        {"start": s.start, "end": s.end, "text": s.text, "language": s.language}
        for s in transcript.segments
    ]


def test_write_json_rounds_times_to_milliseconds(tmp_path: Path) -> None:
    t = Transcript(
        source="a.mp4", model="m", mode="fast", language=None, duration_s=15.5109375,
        segments=[Segment(0.1 + 0.2, 5.6000000000000005, "hi"), Segment(5.6000000000000005, 7.0004999, "yo")],
    )
    path = tmp_path / "out.json"
    write_json(t, path)
    raw = path.read_text(encoding="utf-8")
    assert "0000000" not in raw
    data = json.loads(raw)
    assert data["duration_s"] == 15.511
    assert [(s["start"], s["end"]) for s in data["segments"]] == [(0.3, 5.6), (5.6, 7.0)]


def test_empty_transcript(tmp_path: Path) -> None:
    t = Transcript("a.mp4", "m", "fast", None, 0.0, [])
    paths = write_outputs(t, tmp_path / "a", ["txt", "srt", "json"])
    txt, srt, js = paths
    assert txt.read_text() == ""
    assert srt.read_text() == ""
    assert json.loads(js.read_text())["segments"] == []


def test_write_outputs_paths_and_contents(transcript: Transcript, tmp_path: Path) -> None:
    base = tmp_path / "nested" / "dir" / "lecture.v2"  # parent is created; ".v2" is kept
    paths = write_outputs(transcript, base, ["srt", " JSON ", "txt", "srt"])
    assert paths == [base.parent / "lecture.v2.srt", base.parent / "lecture.v2.json",
                     base.parent / "lecture.v2.txt"]

    for path, writer in zip(paths, (write_srt, write_json, write_txt)):
        expected = tmp_path / f"expected{path.suffix}"
        writer(transcript, expected)
        assert path.read_bytes() == expected.read_bytes()


def test_write_outputs_subset(transcript: Transcript, tmp_path: Path) -> None:
    paths = write_outputs(transcript, tmp_path / "lecture", ["txt"])
    assert paths == [tmp_path / "lecture.txt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lecture.txt"]


@pytest.mark.parametrize("formats", [["txt", "pdf"], ["docx"], []])
def test_write_outputs_rejects_bad_formats_before_writing(
    transcript: Transcript, tmp_path: Path, formats: list[str]
) -> None:
    with pytest.raises(ValueError):
        write_outputs(transcript, tmp_path / "lecture", formats)
    assert list(tmp_path.iterdir()) == []


def test_write_outputs_overwrites_and_leaves_no_temp_files(
    transcript: Transcript, tmp_path: Path
) -> None:
    (tmp_path / "lecture.txt").write_text("stale")
    write_outputs(transcript, tmp_path / "lecture", ["txt", "srt", "json"])
    assert (tmp_path / "lecture.txt").read_text(encoding="utf-8").startswith("Guten Morgen")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["lecture.json", "lecture.srt", "lecture.txt"]


def test_failed_write_keeps_old_file(
    transcript: Transcript, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "lecture.txt"
    target.write_text("previous transcript")

    def broken_replace(src: str | os.PathLike, dst: str | os.PathLike) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(output.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        write_outputs(transcript, tmp_path / "lecture", ["txt"])
    assert target.read_text() == "previous transcript"
    assert [p.name for p in tmp_path.iterdir()] == ["lecture.txt"]


def test_output_files_get_normal_permissions(transcript: Transcript, tmp_path: Path) -> None:
    mask = os.umask(0o022)
    try:
        (path,) = write_outputs(transcript, tmp_path / "lecture", ["txt"])
    finally:
        os.umask(mask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
