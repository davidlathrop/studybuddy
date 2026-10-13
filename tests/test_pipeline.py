from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import media
from studybuddy import audio, pipeline
from studybuddy.pipeline import split_at_quiet_points, transcribe_file
from studybuddy.types import SAMPLE_RATE, ProgressFn, Segment, Transcript

SR = SAMPLE_RATE


# --------------------------------------------------------------------------- helpers


def noise(seconds: float, *, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(round(seconds * SR))).astype(np.float32)


def noise_with_gaps(total_s: float, *, period_s: float, gap_s: float, seed: int = 0) -> np.ndarray:
    """Loud noise with a digitally silent gap of ``gap_s`` ending every ``period_s``."""
    pcm = noise(total_s, seed=seed)
    period, gap = round(period_s * SR), round(gap_s * SR)
    for end in range(period, len(pcm) + 1, period):
        pcm[end - gap:end] = 0.0
    return pcm


def check_cover(pieces: list[tuple[int, int]], n: int) -> None:
    assert pieces[0][0] == 0
    assert pieces[-1][1] == n
    for (_, end), (start, _) in zip(pieces, pieces[1:]):
        assert end == start
    assert all(end > start for start, end in pieces)


class FakeTranscriber:
    """Deterministic stand-in: text depends only on the section, so reruns match."""

    def __init__(
        self,
        model_id: str = "fake/asr",
        *,
        variant: str = "a",
        fail_on_call: int | None = None,
        respond: Callable[[np.ndarray, float, str | None], list[Segment]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.variant = variant
        self.fail_on_call = fail_on_call
        self.respond = respond
        self.calls: list[dict] = []

    def config(self) -> dict:
        return {"variant": self.variant, "chunk_duration": 60.0}

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_s: float = 0.0,
        language: str | None = None,
        hotwords: list[str] | None = None,
        on_progress: ProgressFn | None = None,
    ) -> list[Segment]:
        self.calls.append(
            {"offset_s": offset_s, "n": len(audio), "language": language, "hotwords": hotwords}
        )
        if self.fail_on_call == len(self.calls) - 1:
            raise RuntimeError("simulated crash")
        if on_progress:
            on_progress(0.5)
        if self.respond is not None:
            segments = self.respond(audio, offset_s, language)
        else:
            end = offset_s + len(audio) / SR
            mid = (offset_s + end) / 2
            # Deliberately out of order: the pipeline sorts by start time.
            segments = [
                Segment(mid, end, f"second half of {offset_s:.4f}", language),
                Segment(offset_s, mid, f"first half of {offset_s:.4f}", language),
            ]
        if on_progress:
            on_progress(1.0)
        return segments

    @property
    def offsets(self) -> list[float]:
        return [c["offset_s"] for c in self.calls]


@pytest.fixture
def install_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
    """Write a stand-in audio file and make ``audio.load_audio`` return ``pcm`` for it."""
    loaded: dict[Path, np.ndarray] = {}

    def fake_load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
        assert sr == SAMPLE_RATE
        return loaded[Path(path)]

    def install(pcm: np.ndarray, name: str = "lecture.mp3") -> Path:
        path = tmp_path / name
        path.write_bytes(pcm.tobytes())
        loaded[path] = pcm
        return path

    monkeypatch.setattr(audio, "load_audio", fake_load_audio)
    return install


@pytest.fixture
def lecture(install_audio: Callable[..., Path]) -> Path:
    """53 s of loud noise with a 1 s silent gap every 7 s (7 sections at section_s=10)."""
    return install_audio(noise_with_gaps(53.0, period_s=7.0, gap_s=1.0))


def run(path: Path, transcriber: FakeTranscriber, **kwargs) -> Transcript:
    kwargs.setdefault("mode", "accurate")
    kwargs.setdefault("section_s", 10.0)
    return transcribe_file(path, transcriber, **kwargs)


def expected_offsets(path: Path, section_s: float = 10.0) -> list[float]:
    pcm = audio.load_audio(path)
    return [start / SR for start, _ in split_at_quiet_points(pcm, SR, section_s)]


def partial_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- splitting


def test_split_empty_audio_gives_no_pieces() -> None:
    assert split_at_quiet_points(np.zeros(0, np.float32), SR, 600.0) == []


@pytest.mark.parametrize("n", [1, 100, SR * 60, SR * 600, SR * 605])
def test_split_short_audio_is_one_piece(n: int) -> None:
    # Up to target + search (600 s + 5 s) stays whole: a cut would only shave off a sliver.
    assert split_at_quiet_points(noise(n / SR), SR, 600.0) == [(0, n)]


@pytest.mark.parametrize(
    "n", [SR * 10 + 1, SR * 15, SR * 15 + 1, SR * 16, SR * 61 + 123, SR * 95, SR * 100 + 7]
)
def test_split_covers_audio_exactly(n: int) -> None:
    pcm = noise(n / SR, seed=n)
    pieces = split_at_quiet_points(pcm, SR, 10.0, search_s=5.0)
    check_cover(pieces, n)
    assert (len(pieces) > 1) == (n > 15 * SR)
    for start, end in pieces:
        assert end - start <= 15 * SR
    for start, end in pieces[:-1]:
        assert end - start >= 5 * SR
    assert pieces[-1][1] - pieces[-1][0] >= SR  # no tiny trailing piece


def test_split_merges_short_tail_when_no_search_room() -> None:
    n = SR * 20 + SR // 2
    assert split_at_quiet_points(noise(n / SR), SR, 10.0, search_s=0.0) == [
        (0, 10 * SR), (10 * SR, n)
    ]


def test_split_without_search_cuts_exactly_at_targets() -> None:
    pieces = split_at_quiet_points(noise(35.0), SR, 10.0, search_s=0.0)
    assert pieces == [(0, 10 * SR), (10 * SR, 20 * SR), (20 * SR, 30 * SR), (30 * SR, 35 * SR)]


def test_split_cuts_land_in_silent_gaps() -> None:
    pcm = noise_with_gaps(300.0, period_s=7.0, gap_s=1.0)
    pieces = split_at_quiet_points(pcm, SR, 30.0, search_s=5.0)
    check_cover(pieces, len(pcm))
    cuts = [start for start, _ in pieces[1:]]
    assert len(cuts) >= 8
    for cut in cuts:
        # Cut near the middle of the 1 s gap, well clear of the noise on either side.
        assert not pcm[cut - SR // 4:cut + SR // 4].any(), f"cut at {cut / SR:.3f}s is not in a gap"
    for start, end in pieces[:-1]:
        assert 25 * SR <= end - start <= 35 * SR


def test_split_prefers_quiet_stretch_nearest_target() -> None:
    pcm = noise(45.0)
    pcm[round(27.0 * SR):round(27.5 * SR)] = 0.0  # 2.5-3 s before the 30 s target
    pcm[round(32.0 * SR):round(32.5 * SR)] = 0.0  # 2-2.5 s after it: nearer
    pieces = split_at_quiet_points(pcm, SR, 30.0, search_s=5.0)
    assert len(pieces) == 2
    assert abs(pieces[0][1] - 32.25 * SR) <= 1  # middle of the nearer gap


def test_split_finds_quietest_window_without_true_silence() -> None:
    pcm = noise(40.0, amp=0.5)
    lo, hi = round(28.0 * SR), round(28.3 * SR)
    pcm[lo:hi] *= 0.02  # quiet, but not silent
    (_, cut), _ = split_at_quiet_points(pcm, SR, 30.0, search_s=5.0)
    assert lo < cut < hi


@pytest.mark.parametrize("target_s", [10.0, 15.0, 20.0])
def test_split_real_speech_cuts_in_pauses(long_speech_wav: Path, target_s: float) -> None:
    pcm = media.decode_pcm(long_speech_wav)  # ~72 s of TTS speech, independent of studybuddy
    pieces = split_at_quiet_points(pcm, SR, target_s)
    check_cover(pieces, len(pcm))
    assert len(pieces) >= 3
    for cut in (start for start, _ in pieces[1:]):
        around = pcm[cut - SR // 20:cut + SR // 20]  # +/- 50 ms
        assert np.sqrt(np.mean(around**2)) < 0.01, f"cut at {cut / SR:.2f}s is in speech"


def test_split_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        split_at_quiet_points(noise(1.0), SR, 0.0)
    with pytest.raises(ValueError):
        split_at_quiet_points(noise(1.0), SR, 10.0, search_s=-1.0)


def test_split_two_and_a_half_hours_is_fast() -> None:
    period = np.concatenate([noise(9.0), np.zeros(SR, np.float32)])  # 9 s noise, 1 s silence
    pcm = np.tile(period, 900)  # 2.5 h, 144M samples
    t0 = time.perf_counter()
    pieces = split_at_quiet_points(pcm, SR, 600.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"split took {elapsed:.2f}s"
    check_cover(pieces, len(pcm))
    assert len(pieces) == 15  # cuts at 599.5 s, 1199.5 s, ...; the last piece is 600.5 s
    for start, _ in pieces[1:]:
        assert not pcm[start - SR // 4:start + SR // 4].any()


# --------------------------------------------------------------------------- transcribe_file


def test_offsets_are_absolute_across_sections(lecture: Path) -> None:
    fake = FakeTranscriber()
    t = run(lecture, fake, language="English", hotwords=["eigenvalue"])

    offsets = expected_offsets(lecture)
    assert offsets == [0.0, 6.5, 13.5, 20.5, 27.5, 34.5, 41.5]  # middles of the gaps
    assert fake.offsets == offsets
    assert all(c["language"] == "English" and c["hotwords"] == ["eigenvalue"] for c in fake.calls)

    pieces = split_at_quiet_points(audio.load_audio(lecture), SR, 10.0)
    assert [c["n"] for c in fake.calls] == [end - start for start, end in pieces]

    starts = [s.start for s in t.segments]
    assert starts == sorted(starts)
    assert len(t.segments) == 14
    assert t.segments[0].start == 0.0
    assert t.segments[-1].end == pytest.approx(53.0)
    for offset, (first, second) in zip(offsets, zip(t.segments[::2], t.segments[1::2])):
        assert first.text == f"first half of {offset:.4f}" and first.start == offset
        assert second.text == f"second half of {offset:.4f}"

    assert t.duration_s == pytest.approx(53.0)
    assert (t.model, t.mode, t.source) == ("fake/asr", "accurate", "lecture.mp3")
    assert t.language == "English"


def test_one_minute_audio_is_a_single_section(install_audio: Callable[..., Path]) -> None:
    path = install_audio(noise(60.0))
    fake = FakeTranscriber()
    t = transcribe_file(path, fake, mode="fast", source="talk.mp4")
    assert fake.calls == [{"offset_s": 0.0, "n": 60 * SR, "language": None, "hotwords": None}]
    assert t.source == "talk.mp4"
    assert t.mode == "fast"
    assert t.language is None


def test_empty_audio(install_audio: Callable[..., Path], tmp_path: Path) -> None:
    path = install_audio(np.zeros(0, np.float32))
    fake = FakeTranscriber()
    progress: list[float] = []
    t = run(path, fake, partial_path=tmp_path / "x.partial.jsonl", on_progress=progress.append)
    assert fake.calls == []
    assert t.segments == [] and t.duration_s == 0.0
    assert progress == [1.0]
    assert len(partial_lines(tmp_path / "x.partial.jsonl")) == 1  # header only


def test_progress_is_monotonic_and_complete(lecture: Path) -> None:
    progress: list[float] = []
    run(lecture, FakeTranscriber(), on_progress=progress.append)
    assert progress == sorted(progress)
    assert all(0.0 <= p <= 1.0 for p in progress)
    assert progress[0] == 0.0 and progress[-1] == 1.0

    pieces = split_at_quiet_points(audio.load_audio(lecture), SR, 10.0)
    total = pieces[-1][1]
    first_len = pieces[0][1] - pieces[0][0]
    assert progress[1] == pytest.approx(0.5 * first_len / total)  # halfway through section 1
    assert pytest.approx(first_len / total) in progress  # section 1 done


def test_progress_ignores_bad_fractions_from_transcriber(lecture: Path) -> None:
    def respond(pcm: np.ndarray, offset_s: float, language: str | None) -> list[Segment]:
        return [Segment(offset_s, offset_s + 1, "x")]

    class Wobbly(FakeTranscriber):
        def transcribe(self, audio, *, on_progress=None, **kwargs):  # type: ignore[override]
            for f in (0.9, 0.2, 1.7, float("nan"), -3.0):
                on_progress(f)
            return super().transcribe(audio, on_progress=on_progress, **kwargs)

    progress: list[float] = []
    run(lecture, Wobbly(respond=respond), on_progress=progress.append)
    assert progress == sorted(progress)
    assert all(0.0 <= p <= 1.0 for p in progress)
    assert progress[-1] == 1.0


def test_resume_skips_finished_sections(lecture: Path, tmp_path: Path) -> None:
    partial = tmp_path / "lecture.partial.jsonl"
    reference = run(lecture, FakeTranscriber())
    offsets = expected_offsets(lecture)

    with pytest.raises(RuntimeError, match="simulated crash"):
        run(lecture, FakeTranscriber(fail_on_call=2), partial_path=partial)
    lines = partial_lines(partial)
    assert len(lines) == 3  # header + sections 0 and 1
    assert [rec["index"] for rec in lines[1:]] == [0, 1]

    fake = FakeTranscriber()
    progress: list[float] = []
    resumed = run(lecture, fake, partial_path=partial, on_progress=progress.append)
    assert fake.offsets == offsets[2:]
    assert resumed == reference
    assert progress[0] == pytest.approx(offsets[2] / reference.duration_s)
    assert progress[-1] == 1.0

    # Partial file is kept (the CLI deletes it) and now holds every section.
    lines = partial_lines(partial)
    assert [rec["index"] for rec in lines[1:]] == list(range(7))

    # A rerun with everything finished transcribes nothing and gives the same result.
    again = FakeTranscriber()
    assert run(lecture, again, partial_path=partial) == reference
    assert again.calls == []


def test_partial_file_format(
    lecture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsyncs: list[int] = []
    real_fsync = pipeline.os.fsync
    monkeypatch.setattr(pipeline.os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd)))

    partial = tmp_path / "lecture.partial.jsonl"
    run(lecture, FakeTranscriber(), partial_path=partial, language="English")
    header, *records = partial_lines(partial)

    assert header["version"] == 1
    assert len(header["key"]) == 64
    assert header["settings"]["audio_sha256"] == hashlib.sha256(lecture.read_bytes()).hexdigest()
    assert header["settings"]["model_id"] == "fake/asr"
    assert header["settings"]["config"] == {"variant": "a", "chunk_duration": 60.0}

    pieces = split_at_quiet_points(audio.load_audio(lecture), SR, 10.0)
    assert [(r["index"], r["start_sample"], r["end_sample"]) for r in records] == [
        (i, start, end) for i, (start, end) in enumerate(pieces)
    ]
    assert set(records[0]["segments"][0]) == {"start", "end", "text", "language"}
    assert len(fsyncs) >= len(pieces)  # every section line is synced to disk


def test_file_sha256_streams_the_file(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    data = np.random.default_rng(1).bytes(3 * 1024 * 1024 + 17)
    path.write_bytes(data)
    assert pipeline.file_sha256(path) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "change",
    [
        {"transcriber": FakeTranscriber(variant="b")},
        {"transcriber": FakeTranscriber(model_id="fake/other")},
        {"language": "German"},
        {"hotwords": ["new", "terms"]},
        {"mode": "fast"},
    ],
    ids=["config", "model-id", "language", "hotwords", "mode"],
)
def test_settings_change_restarts(
    lecture: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture, change: dict
) -> None:
    partial = tmp_path / "lecture.partial.jsonl"
    run(lecture, FakeTranscriber(), partial_path=partial)

    change = dict(change)
    fake = change.pop("transcriber", FakeTranscriber())
    with caplog.at_level(logging.WARNING, logger="studybuddy.pipeline"):
        run(lecture, fake, partial_path=partial, **change)
    assert fake.offsets == expected_offsets(lecture)
    assert "starting over" in caplog.text


def test_section_size_change_restarts(lecture: Path, tmp_path: Path) -> None:
    partial = tmp_path / "lecture.partial.jsonl"
    run(lecture, FakeTranscriber(), partial_path=partial)
    fake = FakeTranscriber()
    run(lecture, fake, partial_path=partial, section_s=20.0)
    assert fake.offsets == expected_offsets(lecture, 20.0)


def test_audio_change_restarts(
    install_audio: Callable[..., Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    partial = tmp_path / "lecture.partial.jsonl"
    path = install_audio(noise_with_gaps(53.0, period_s=7.0, gap_s=1.0, seed=1))
    run(path, FakeTranscriber(), partial_path=partial)

    path = install_audio(noise_with_gaps(53.0, period_s=7.0, gap_s=1.0, seed=2))
    fake = FakeTranscriber()
    with caplog.at_level(logging.WARNING, logger="studybuddy.pipeline"):
        run(path, fake, partial_path=partial)
    assert len(fake.calls) == 7
    assert "changed" in caplog.text


def crash_after_two_sections(lecture: Path, partial: Path) -> None:
    with pytest.raises(RuntimeError):
        run(lecture, FakeTranscriber(fail_on_call=2), partial_path=partial)


def test_truncated_last_line_is_dropped(
    lecture: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    partial = tmp_path / "lecture.partial.jsonl"
    reference = run(lecture, FakeTranscriber())
    crash_after_two_sections(lecture, partial)
    with partial.open("a", encoding="utf-8") as fh:
        fh.write('{"index": 2, "start_sample": 4800, "segm')  # killed mid-write

    fake = FakeTranscriber()
    with caplog.at_level(logging.WARNING, logger="studybuddy.pipeline"):
        resumed = run(lecture, fake, partial_path=partial)
    assert "incomplete last line" in caplog.text
    assert fake.offsets == expected_offsets(lecture)[2:]
    assert resumed == reference
    assert [r["index"] for r in partial_lines(partial)[1:]] == list(range(7))


def test_truncated_multibyte_last_line_is_dropped(lecture: Path, tmp_path: Path) -> None:
    partial = tmp_path / "lecture.partial.jsonl"
    crash_after_two_sections(lecture, partial)
    with partial.open("ab") as fh:
        fh.write('{"index": 2, "segments": [{"text": "世'.encode()[:-1])  # cut inside a character
    fake = FakeTranscriber()
    run(lecture, fake, partial_path=partial)
    assert fake.offsets == expected_offsets(lecture)[2:]


def test_last_line_missing_newline_is_kept(lecture: Path, tmp_path: Path) -> None:
    partial = tmp_path / "lecture.partial.jsonl"
    reference = run(lecture, FakeTranscriber())
    crash_after_two_sections(lecture, partial)
    partial.write_bytes(partial.read_bytes().rstrip(b"\n"))

    fake = FakeTranscriber()
    assert run(lecture, fake, partial_path=partial) == reference
    assert fake.offsets == expected_offsets(lecture)[2:]
    assert len(partial_lines(partial)) == 8  # every line still parses on its own


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda lines: ["not json at all"] + lines[1:],  # bad header
        lambda lines: [lines[0], "garbage", *lines[2:]],  # bad line in the middle
        lambda lines: [lines[0], json.dumps({**json.loads(lines[1]), "end_sample": 1})],
        lambda lines: [lines[0], json.dumps({**json.loads(lines[1]), "index": 99})],
        lambda lines: [lines[0], json.dumps({"index": 0, "start_sample": 0})],
        lambda lines: [lines[0], json.dumps([1, 2, 3])],
        lambda lines: [],
    ],
    ids=["header", "middle-line", "split-mismatch", "index-range", "missing-fields",
         "not-an-object", "empty-file"],
)
def test_untrustworthy_partial_restarts(
    lecture: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture, corrupt: Callable
) -> None:
    partial = tmp_path / "lecture.partial.jsonl"
    reference = run(lecture, FakeTranscriber())
    crash_after_two_sections(lecture, partial)
    lines = partial.read_text(encoding="utf-8").splitlines()
    partial.write_text("".join(line + "\n" for line in corrupt(lines)), encoding="utf-8")

    fake = FakeTranscriber()
    with caplog.at_level(logging.WARNING, logger="studybuddy.pipeline"):
        assert run(lecture, fake, partial_path=partial) == reference
    assert fake.offsets == expected_offsets(lecture)
    assert "starting over" in caplog.text
    assert [r["index"] for r in partial_lines(partial)[1:]] == list(range(7))


def test_warns_when_a_section_with_sound_has_no_text(
    install_audio: Callable[..., Path], caplog: pytest.LogCaptureFixture
) -> None:
    pcm = np.concatenate([noise(10.0), np.zeros(10 * SR, np.float32)])
    path = install_audio(pcm)
    fake = FakeTranscriber(respond=lambda pcm, offset_s, lang: [])
    with caplog.at_level(logging.WARNING, logger="studybuddy.pipeline"):
        t = run(path, fake, section_s=10.0)
    (_, cut), _ = split_at_quiet_points(pcm, SR, 10.0)
    assert 10 * SR < cut < 15 * SR  # the cut lands in the silence
    assert len(fake.calls) == 2
    assert t.segments == []
    # Only the noisy section is reported; the silent one is expected to have no text.
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        f"section 1 (0:00:00-0:00:{cut // SR:02d}) has sound but produced no text"
    ]


def test_warns_about_an_empty_chunk_with_sound(
    install_audio: Callable[..., Path], caplog: pytest.LogCaptureFixture
) -> None:
    path = install_audio(np.concatenate([np.zeros(5 * SR, np.float32), noise(5.0)]))

    def respond(pcm: np.ndarray, offset_s: float, lang: str | None) -> list[Segment]:
        return [Segment(0.0, 5.0, "words over silence?"), Segment(5.0, 10.0, "  ")]

    with caplog.at_level(logging.WARNING, logger="studybuddy.pipeline"):
        t = run(path, FakeTranscriber(respond=respond))
    assert "chunk 0:00:05-0:00:10 has sound but came back empty" in caplog.text
    assert [s.text for s in t.segments] == ["words over silence?"]  # empty segment dropped


def test_segments_are_kept_inside_their_section(install_audio: Callable[..., Path]) -> None:
    path = install_audio(noise(5.0))

    def respond(pcm: np.ndarray, offset_s: float, lang: str | None) -> list[Segment]:
        return [Segment(-0.5, 2.0, "early"), Segment(4.0, 6.0, "padded tail")]

    t = run(path, FakeTranscriber(respond=respond))
    assert [(s.start, s.end) for s in t.segments] == [(0.0, 2.0), (4.0, 5.0)]


def test_language_is_the_most_common_segment_language(install_audio: Callable[..., Path]) -> None:
    path = install_audio(noise(30.0))
    labels = iter(["German", "English", None, "English"])

    def respond(pcm: np.ndarray, offset_s: float, lang: str | None) -> list[Segment]:
        return [Segment(offset_s + i, offset_s + i + 1, "x", next(labels)) for i in range(2)]

    t = run(path, FakeTranscriber(respond=respond), section_s=15.0, language="French")
    assert t.language == "English"


def test_language_falls_back_to_the_argument(install_audio: Callable[..., Path]) -> None:
    path = install_audio(noise(5.0))

    def respond(pcm: np.ndarray, offset_s: float, lang: str | None) -> list[Segment]:
        return [Segment(0.0, 5.0, "no labels")]

    assert run(path, FakeTranscriber(respond=respond), language="French").language == "French"
    assert run(path, FakeTranscriber(respond=respond)).language is None
