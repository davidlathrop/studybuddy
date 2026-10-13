"""Tests for studybuddy.audio (fast: ffmpeg/ffprobe on generated media, no models)."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import media
from studybuddy import audio
from studybuddy.audio import AudioError, MediaInfo, extract_mp3, load_audio, probe
from studybuddy.types import SAMPLE_RATE


@pytest.fixture
def corrupt_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.mp4"
    path.write_text("this is not a video, just some plain text\n" * 50)
    return path


@pytest.fixture(scope="module")
def two_track_mp4(media_dir: Path, speech_audio: Path) -> Path:
    """Stereo MP4 whose first audio track is the speech and whose second is 2 s of silence."""
    silence = media.make_silence(media_dir / "audio_test_silence.wav", 2.0)
    return media.make_mp4(speech_audio, media_dir / "audio_test_two_tracks.mp4",
                          extra_audio_tracks=[silence], stereo=True)


def _files_in(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _audio_stream(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,channels,sample_rate,bit_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)["streams"][0]


def _sound_spans(pcm: np.ndarray, threshold: float = 0.01) -> list[tuple[float, float]]:
    """(start, end) seconds of the stretches whose 100 ms frames are louder than ``threshold`` RMS."""
    frame = SAMPLE_RATE // 10
    frames = pcm[: len(pcm) // frame * frame].reshape(-1, frame)
    loud = np.sqrt(np.mean(frames**2, axis=1)) > threshold
    edges = np.flatnonzero(np.diff(np.concatenate(([0], loud.astype(int), [0]))))
    return [(a / 10, b / 10) for a, b in zip(edges[::2], edges[1::2])]


def _assert_non_decreasing_and_done(calls: list[float]) -> None:
    assert calls, "on_progress was never called"
    assert all(0.0 <= f <= 1.0 for f in calls)
    assert all(a <= b for a, b in zip(calls, calls[1:]))
    assert calls[-1] == 1.0
    assert calls.count(1.0) == 1


# --- probe ------------------------------------------------------------------------------


def test_probe_reads_duration_and_audio_stream(short_mp4: Path) -> None:
    info = probe(short_mp4)
    assert info.path == short_mp4
    assert info.duration_s == pytest.approx(media.media_duration(short_mp4), abs=0.05)
    assert info.audio_streams == 1
    assert info.codec == "aac"
    assert info.channels == 1


def test_probe_counts_tracks_and_describes_the_first(two_track_mp4: Path) -> None:
    info = probe(two_track_mp4)
    assert info.audio_streams == 2
    assert info.channels == 2
    assert info.duration_s > 10.0


def test_probe_without_audio_stream_says_so(no_audio_mp4: Path) -> None:
    with pytest.raises(AudioError, match="no audio stream"):
        probe(no_audio_mp4)


def test_probe_corrupt_file_raises(corrupt_mp4: Path) -> None:
    with pytest.raises(AudioError, match="cannot read") as excinfo:
        probe(corrupt_mp4)
    assert "Invalid data" in str(excinfo.value)


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioError, match="not found"):
        probe(tmp_path / "missing.mp4")


def test_probe_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioError, match="directory"):
        probe(tmp_path)


# --- extract_mp3 ------------------------------------------------------------------------


def test_extract_mp3_writes_mono_mp3_and_nothing_else(short_mp4: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "new" / "dir"
    dst = out_dir / "short.mp3"

    assert extract_mp3(short_mp4, dst) == dst

    info = probe(dst)
    assert info.codec == "mp3"
    assert info.channels == 1
    assert info.duration_s == pytest.approx(probe(short_mp4).duration_s, abs=0.25)
    stream, source = _audio_stream(dst), _audio_stream(short_mp4)
    assert int(stream["bit_rate"]) == 96_000  # plan 5.2: about 43 MB per hour
    assert stream["sample_rate"] == source["sample_rate"]  # not resampled
    assert _files_in(out_dir) == ["short.mp3"]


def test_extract_mp3_is_deterministic(short_mp4: Path, tmp_path: Path) -> None:
    # Resuming depends on this: the resume key hashes the MP3, which is extracted again.
    first = extract_mp3(short_mp4, tmp_path / "first.mp3").read_bytes()
    assert extract_mp3(short_mp4, tmp_path / "second.mp3").read_bytes() == first


def test_extract_mp3_keeps_a_late_audio_start(media_dir: Path, tmp_path: Path) -> None:
    tone = media.make_tone(media_dir / "audio_test_tone3.wav", 3.0)
    src = media.make_late_audio_mp4(tone, tmp_path / "late.mp4", audio_start_s=4.0, seconds=7.0)

    pcm = load_audio(extract_mp3(src, tmp_path / "late.mp3"))

    # Without the timeline the tone would start at 0 and every timestamp would be 4 s early.
    assert len(pcm) / SAMPLE_RATE == pytest.approx(7.0, abs=0.1)
    ((start, end),) = _sound_spans(pcm)
    assert start == pytest.approx(4.0, abs=0.15) and end == pytest.approx(7.0, abs=0.15)


def test_extract_mp3_fills_gaps_in_the_audio_timestamps(tmp_path: Path, media_dir: Path) -> None:
    src = media.make_gapped_audio_mp4(tmp_path / "gap.mp4", seconds=10.0, gap_at_s=3.0, gap_s=3.0)

    pcm = load_audio(extract_mp3(src, tmp_path / "gap.mp3"))

    assert len(pcm) / SAMPLE_RATE == pytest.approx(13.0, abs=0.1)
    (a0, a1), (b0, b1) = _sound_spans(pcm)
    assert a0 == pytest.approx(0.0, abs=0.15) and a1 == pytest.approx(3.0, abs=0.15)
    assert b0 == pytest.approx(6.0, abs=0.15) and b1 == pytest.approx(13.0, abs=0.15)


def test_extract_mp3_replaces_existing_output_and_stale_partial(short_mp4: Path, tmp_path: Path) -> None:
    dst = tmp_path / "short.mp3"
    dst.write_bytes(b"old output")
    audio._partial_path(dst).write_bytes(b"left over from a killed run")

    extract_mp3(short_mp4, dst, duration_s=probe(short_mp4).duration_s)

    assert probe(dst).codec == "mp3"
    assert _files_in(tmp_path) == ["short.mp3"]


def test_extract_mp3_progress_probes_duration_and_ends_at_one(long_mp4: Path, tmp_path: Path) -> None:
    calls: list[float] = []
    extract_mp3(long_mp4, tmp_path / "long.mp3", on_progress=calls.append)
    _assert_non_decreasing_and_done(calls)


def test_extract_mp3_progress_uses_given_duration(long_mp4: Path, tmp_path: Path) -> None:
    real_s = probe(long_mp4).duration_s
    calls: list[float] = []
    extract_mp3(long_mp4, tmp_path / "long.mp3", duration_s=real_s * 10, on_progress=calls.append)

    _assert_non_decreasing_and_done(calls)
    running = calls[:-1]
    assert running, "expected progress while ffmpeg was running"
    assert all(0.0 < f <= 0.11 for f in running)
    assert running[-1] == pytest.approx(0.1, abs=0.01)


def test_extract_mp3_takes_first_track_and_downmixes(two_track_mp4: Path, tmp_path: Path) -> None:
    dst = extract_mp3(two_track_mp4, tmp_path / "two.mp3")
    info = probe(dst)
    assert info.channels == 1
    assert info.duration_s > 10.0  # the speech track, not the 2 s silent one
    pcm = load_audio(dst)
    assert float(np.sqrt(np.mean(pcm**2))) > 0.01


@pytest.mark.parametrize("duration_s", [None, 3.0], ids=["probed", "ffmpeg"])
def test_extract_mp3_without_audio_says_so_and_leaves_nothing(
    no_audio_mp4: Path, tmp_path: Path, duration_s: float | None
) -> None:
    with pytest.raises(AudioError, match="no audio stream"):
        extract_mp3(no_audio_mp4, tmp_path / "out.mp3", duration_s=duration_s, on_progress=lambda f: None)
    assert _files_in(tmp_path) == []


def test_extract_mp3_ffmpeg_failure_reports_stderr_and_cleans_up(corrupt_mp4: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(AudioError, match="exit status") as excinfo:
        extract_mp3(corrupt_mp4, out_dir / "out.mp3", duration_s=10.0)
    assert "Invalid data" in str(excinfo.value)
    assert _files_in(out_dir) == []


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, RuntimeError])
def test_extract_mp3_interrupted_removes_partial(
    long_mp4: Path, tmp_path: Path, exc_type: type[BaseException]
) -> None:
    def interrupt(fraction: float) -> None:
        raise exc_type("stop")

    # A tenfold duration keeps every running fraction below 1.0, so the callback fires
    # while ffmpeg is still running, before the temporary file is renamed.
    duration_s = probe(long_mp4).duration_s * 10
    with pytest.raises(exc_type):
        extract_mp3(long_mp4, tmp_path / "long.mp3", duration_s=duration_s, on_progress=interrupt)
    assert _files_in(tmp_path) == []


def test_extract_mp3_refuses_to_overwrite_its_input(short_mp4: Path, tmp_path: Path) -> None:
    src = tmp_path / "lecture.mp3"
    shutil.copy(short_mp4, src)
    before = src.read_bytes()
    with pytest.raises(AudioError, match="onto itself"):
        extract_mp3(src, src)
    assert src.read_bytes() == before


def test_extract_mp3_refuses_when_temp_file_would_be_the_input(short_mp4: Path, tmp_path: Path) -> None:
    src = tmp_path / "lecture.partial.mp3"
    shutil.copy(short_mp4, src)
    before = src.read_bytes()
    with pytest.raises(AudioError):
        extract_mp3(src, tmp_path / "lecture.mp3")
    assert src.read_bytes() == before


# --- progress parsing -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("out_time_us=1500000\n", 1.5),
        ("out_time_us=0", 0.0),
        ("out_time_us=-23220", 0.0),
        ("out_time_us=N/A", None),
        ("out_time_ms=1500000", None),
        ("out_time=00:00:01.500000", None),
        ("progress=continue", None),
        ("", None),
    ],
)
def test_out_time_parsing(line: str, expected: float | None) -> None:
    assert audio._out_time_s(line) == expected


def test_progress_reporter_is_monotonic_and_holds_back_one() -> None:
    calls: list[float] = []
    reporter = audio._ProgressReporter(10.0, calls.append)
    for us in ["N/A", "0", "2000000", "1000000", "2000000", "5000000", "10000000", "10500000"]:
        reporter.feed(f"out_time_us={us}\n")
        reporter.feed("progress=continue\n")
    assert calls == [0.2, 0.5]
    reporter.finish()
    assert calls == [0.2, 0.5, 1.0]


@pytest.mark.parametrize("duration_s", [None, 0.0, -1.0, float("nan")])
def test_progress_reporter_unknown_duration_only_reports_done(duration_s: float | None) -> None:
    calls: list[float] = []
    reporter = audio._ProgressReporter(duration_s, calls.append)
    reporter.feed("out_time_us=5000000")
    reporter.finish()
    assert calls == [1.0]


# --- load_audio -------------------------------------------------------------------------


def test_load_audio_matches_independent_decode(short_mp4: Path, short_pcm: np.ndarray) -> None:
    pcm = load_audio(short_mp4)
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    assert len(pcm) / SAMPLE_RATE == pytest.approx(probe(short_mp4).duration_s, abs=0.1)
    assert float(np.abs(pcm).max()) <= 1.0
    assert float(np.sqrt(np.mean(pcm**2))) > 0.01
    np.testing.assert_allclose(pcm, np.clip(short_pcm, -1.0, 1.0), atol=1e-6)


def test_load_audio_resamples(short_mp4: Path) -> None:
    pcm = load_audio(short_mp4, sr=8000)
    assert len(pcm) / 8000 == pytest.approx(probe(short_mp4).duration_s, abs=0.1)


def test_load_audio_reads_extracted_mp3(long_mp4: Path, tmp_path: Path) -> None:
    mp3 = extract_mp3(long_mp4, tmp_path / "long.mp3")
    pcm = load_audio(mp3)
    assert pcm.dtype == np.float32
    assert len(pcm) / SAMPLE_RATE == pytest.approx(probe(long_mp4).duration_s, abs=0.15)
    assert float(np.abs(pcm).max()) <= 1.0


def test_load_audio_without_audio_says_so(no_audio_mp4: Path) -> None:
    with pytest.raises(AudioError, match="no audio stream"):
        load_audio(no_audio_mp4)


def test_load_audio_corrupt_file_raises(corrupt_mp4: Path) -> None:
    with pytest.raises(AudioError, match="cannot read") as excinfo:
        load_audio(corrupt_mp4)
    assert "Invalid data" in str(excinfo.value)


def test_load_audio_ffmpeg_failure_reports_stderr(
    corrupt_mp4: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MediaInfo(path=corrupt_mp4, duration_s=1.0, audio_streams=1, codec="aac", channels=1)
    monkeypatch.setattr(audio, "probe", lambda path: fake)
    with pytest.raises(AudioError, match="could not decode") as excinfo:
        load_audio(corrupt_mp4)
    assert "Invalid data" in str(excinfo.value)


def test_load_audio_rejects_bad_sample_rate(short_mp4: Path) -> None:
    with pytest.raises(ValueError):
        load_audio(short_mp4, sr=0)


@pytest.mark.parametrize("capacity", [0, 1, 3, 10, 1000], ids=lambda c: f"capacity={c}")
def test_read_float32_fills_grows_and_drops_partial_sample(capacity: int) -> None:
    samples = np.arange(10, dtype=np.float32) / 10
    stream = io.BytesIO(samples.tobytes() + b"\x01\x02")  # plus a truncated trailing sample
    pcm = audio._read_float32(stream, capacity)
    assert pcm.dtype == np.float32
    np.testing.assert_array_equal(pcm, samples)


def test_read_float32_empty_stream() -> None:
    assert audio._read_float32(io.BytesIO(b""), 16).shape == (0,)


# --- paths and tools --------------------------------------------------------------------


def test_paths_with_spaces_unicode_and_colons(short_mp4: Path, tmp_path: Path) -> None:
    folder = tmp_path / "my lectures é 講義"
    folder.mkdir()
    src = folder / "week 1: intro – draft (v2).mp4"
    shutil.copy(short_mp4, src)

    assert probe(src).audio_streams == 1
    calls: list[float] = []
    dst = extract_mp3(src, folder / "week 1: intro – draft (v2).mp3", on_progress=calls.append)
    assert dst.exists() and calls[-1] == 1.0
    assert len(load_audio(dst)) > 10 * SAMPLE_RATE


def test_relative_paths_starting_with_a_dash(
    short_mp4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shutil.copy(short_mp4, tmp_path / "-lecture.mp4")

    assert probe(Path("-lecture.mp4")).codec == "aac"
    dst = extract_mp3(Path("-lecture.mp4"), Path("-lecture.mp3"))
    assert (tmp_path / "-lecture.mp3").exists()
    assert len(load_audio(dst)) > 10 * SAMPLE_RATE


def test_missing_tools_raise_clear_errors(
    short_mp4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(AudioError, match="ffprobe was not found on PATH"):
        probe(short_mp4)
    with pytest.raises(AudioError, match="ffmpeg was not found on PATH"):
        extract_mp3(short_mp4, tmp_path / "out.mp3")
    with pytest.raises(AudioError, match="ffmpeg was not found on PATH"):
        load_audio(short_mp4)
    assert not (tmp_path / "out.mp3").exists()
