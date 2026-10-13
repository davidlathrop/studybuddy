"""Tests for studybuddy.transcribe.

Unit tests swap the mlx-audio models for fakes that return the same shapes (and, for Qwen3,
the same 0.5.8 quirks). Tests marked ``model`` run the real models (STUDYBUDDY_MODEL_TESTS=1).
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jiwer
import mlx.core as mx
import numpy as np
import pytest
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.nemo.alignment import AlignedResult, AlignedSentence, AlignedToken
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import split_audio_into_chunks

import media
from studybuddy import transcribe as tr
from studybuddy.transcribe import (
    ParakeetTranscriber,
    Qwen3Transcriber,
    Transcriber,
    looks_repetitive,
    make_transcriber,
    normalize_language,
)
from studybuddy.types import SAMPLE_RATE, Segment

SR = SAMPLE_RATE


# --- audio helpers -------------------------------------------------------------------------


def tone(seconds: float, amp: float = 0.1, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def tone_blocks(n: int, block_s: float = 10.0, gap_s: float = 0.5) -> np.ndarray:
    """``n`` tone blocks with short silent gaps between them, where chunk cuts will land."""
    parts: list[np.ndarray] = []
    for i in range(n):
        parts += [silence(gap_s)] if i else []
        parts.append(tone(block_s))
    return np.concatenate(parts)


def loop_text(phrase: str = "and then we", copies: int = 60) -> str:
    return " ".join([phrase] * copies)


# --- fake models ---------------------------------------------------------------------------


def sentence(text: str, start: float, end: float) -> AlignedSentence:
    token = AlignedToken(id=1, text=text, start=start, duration=end - start)
    return AlignedSentence(text=text, tokens=[token])


class FakeParakeet:
    """Returns ``sentences`` (times relative to the audio passed in) and calls ``chunk_callback``
    with ``callbacks`` like mlx-audio does. Give ``then`` to answer later calls differently."""

    def __init__(
        self,
        sentences: list[AlignedSentence],
        callbacks: list[tuple[int, int]] = (),
        *,
        then: list[AlignedSentence] | None = None,
    ) -> None:
        self.responses = [sentences, then if then is not None else sentences]
        self.callbacks = callbacks
        self.calls: list[tuple[Any, dict]] = []

    def generate(self, audio: Any, **kwargs: Any) -> AlignedResult:
        sentences = self.responses[min(len(self.calls), 1)]
        self.calls.append((audio, kwargs))
        if kwargs.get("chunk_callback") is not None:
            for current, total in self.callbacks:
                kwargs["chunk_callback"](current, total)
        return AlignedResult(text="".join(s.text for s in sentences), sentences=sentences)


Respond = Callable[[int, np.ndarray, dict], "str | None"]


class FakeQwen:
    """Mimics mlx-audio 0.5.8's Qwen3 ``generate()``.

    It chunks with mlx-audio's own splitter and returns an ``STTOutput`` with one segment per
    chunk. ``respond(chunk_index, chunk_audio, kwargs)`` gives the raw model output for a
    chunk ("language X<asr_text>..." when no language is forced) or ``None`` to leave the
    chunk out, as happens when the token budget runs out. Like 0.5.8, the batched path only
    strips the language prefix from the first chunk and labels every chunk with its language.
    """

    def __init__(self, respond: Respond, *, exhaust_budget: bool = False) -> None:
        self.respond = respond
        self.exhaust_budget = exhaust_budget
        self.calls: list[dict] = []

    def generate(self, audio: np.ndarray, **kwargs: Any) -> STTOutput:
        self.calls.append({"samples": len(audio), **kwargs})
        chunks = split_audio_into_chunks(audio, SR, kwargs["chunk_duration"], kwargs["min_chunk_duration"])
        batched = kwargs["batch_size"] > 1 and len(chunks) > 1
        language = kwargs.get("language")
        segments, tokens = [], 0
        for i, (chunk, offset) in enumerate(chunks):
            raw = self.respond(i, chunk, kwargs)
            if raw is None:
                continue
            tokens += len(raw.split())
            if language is None and not (batched and segments):
                language, text = self._extract_language(raw)
            else:
                text = raw
            segments.append({
                "text": text, "language": language,
                "start": np.float64(offset), "end": np.float64(offset + len(chunk) / SR),
            })
        if self.exhaust_budget:
            tokens = kwargs["max_tokens"]
        return STTOutput(
            text=" ".join(s["text"] for s in segments), segments=segments,
            language=[s["language"] for s in segments], generation_tokens=tokens,
        )

    @staticmethod
    def _extract_language(raw: str) -> tuple[str, str]:
        if raw.startswith("language ") and "<asr_text>" in raw:
            head, _, text = raw.partition("<asr_text>")
            return head[len("language "):].strip(), text
        return "English", raw


def by_kind(batched: Respond, single: Respond) -> Respond:
    """Route a FakeQwen response by call kind: one batched call vs. one-chunk re-runs."""
    return lambda i, chunk, kw: (batched if kw["batch_size"] > 1 else single)(i, chunk, kw)


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], list[str]]:
    """``loads(fake)`` makes load_model return ``fake`` and returns the list of loaded ids."""

    def install(fake: Any) -> list[str]:
        loaded: list[str] = []

        def load_model(model_id: str) -> Any:
            loaded.append(model_id)
            return fake

        monkeypatch.setattr(tr, "load_model", load_model)
        return loaded

    return install


def warnings_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


# --- normalize_language --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("en", "English"), ("EN", "English"), ("en-US", "English"), ("english", "English"),
        (" French ", "French"), ("pt_BR", "Portuguese"), ("zh", "Chinese"), ("zh-Hans", "Chinese"),
        ("yue", "Cantonese"), ("ja", "Japanese"), ("ko", "Korean"), ("fil", "Filipino"),
        ("tl", "Filipino"), ("farsi", "Persian"), ("mk", "Macedonian"),
    ],
)
def test_normalize_language_maps_codes_and_names(given: str, expected: str) -> None:
    assert normalize_language(given) == expected


@pytest.mark.parametrize("given", [None, "", "  ", "auto", "AUTO"])
def test_normalize_language_auto(given: str | None) -> None:
    assert normalize_language(given) is None


@pytest.mark.parametrize("given", ["xx", "Klingon", "e"])
def test_normalize_language_rejects_unknown(given: str) -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        normalize_language(given)


def test_every_language_name_maps_to_itself() -> None:
    for name in tr.QWEN3_LANGUAGES.values():
        assert normalize_language(name) == name
        assert normalize_language(name.upper()) == name


# --- looks_repetitive ----------------------------------------------------------------------


def test_normal_speech_is_not_repetitive() -> None:
    assert not looks_repetitive(media.SPEECH_TEXT, 15.5)
    assert not looks_repetitive(" ".join(media.PASSAGES), 72.5)


def test_fast_speech_under_the_rate_limit_is_not_repetitive() -> None:
    words = " ".join(media.PASSAGES * 2).split()[:280]  # 280 wpm: fast but plausible
    assert not looks_repetitive(" ".join(words), 60.0)


def test_too_many_words_per_minute_is_repetitive() -> None:
    words = " ".join(media.PASSAGES * 2).split()[:350]
    assert looks_repetitive(" ".join(words), 60.0)


def test_short_chunks_are_not_judged_on_rate_alone() -> None:
    assert not looks_repetitive("Right, so let me show you how this part of the proof works out.", 3.0)


@pytest.mark.parametrize(
    "text",
    [
        "thank you " * 6,
        "Okay. Okay. Okay. Okay. Okay. Okay.",
        "so the answer is " + "the the the the the the the the" + " clearly",
        " ".join(["we can see that the result is not"] * 6),  # an 8-word phrase
        loop_text(),
        "谢谢" * 10,
    ],
)
def test_looping_phrases_are_repetitive(text: str) -> None:
    assert looks_repetitive(text, 60.0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "thank you " * 5,
        "No, no, no, that is not what I said at all.",
        "It is very, very, very important to check the units.",
        "今天我们讨论线性代数中的特征值和特征向量。",
    ],
)
def test_ordinary_repetition_is_not_repetitive(text: str) -> None:
    assert not looks_repetitive(text, 60.0)


HINDI = (
    "आज हम रैखिक बीजगणित के बारे में बात करेंगे। मैट्रिक्स और वेक्टर इस विषय की नींव हैं, और हम "
    "देखेंगे कि उनका उपयोग कैसे किया जाता है। पहले हम सरल उदाहरणों से शुरू करेंगे, फिर धीरे धीरे कठिन "
    "प्रश्नों की ओर बढ़ेंगे। यदि आपके मन में कोई सवाल हो तो बीच में पूछ सकते हैं। अगले सप्ताह हम "
    "अभिलक्षणिक मान और अभिलक्षणिक सदिश पर चर्चा करेंगे।"
)
# About 64 syllables: 11 s at a brisk 6 syllables per second.
THAI = (
    "วันนี้เราจะพูดถึงพีชคณิตเชิงเส้น เมทริกซ์และเวกเตอร์เป็นพื้นฐานของวิชานี้ และเราจะดูว่ามันถูกใช้อย่างไร "
    "ก่อนอื่นเราจะเริ่มจากตัวอย่างง่ายๆ แล้วค่อยๆ ไปสู่โจทย์ที่ยากขึ้น ถ้ามีคำถามก็ถามได้เลยนะครับ "
)


def test_words_with_combining_marks_stay_whole() -> None:
    # Devanagari vowel signs and viramas are combining marks, which Python's \w excludes.
    assert tr._tokens("रैखिक बीजगणित।") == ["रैखिक", "बीजगणित"]
    assert len(tr._tokens(HINDI)) == len(HINDI.split())
    assert tr._tokens("café don't don’t 'tis") == ["café", "don't", "don’t", "tis"]


def test_unspaced_scripts_are_split_into_characters_with_their_marks() -> None:
    assert tr._tokens("ขอบคุณครับ") == ["ข", "อ", "บ", "คุ", "ณ", "ค", "รั", "บ"]
    assert tr._tokens("我们用GPU加速") == ["我", "们", "用", "gpu", "加", "速"]


def test_normal_hindi_and_thai_are_not_repetitive() -> None:
    words = HINDI.split()
    assert not looks_repetitive(" ".join((words * 4)[:220]), 60.0)  # 220 words per minute: fast
    assert not looks_repetitive(THAI * 5, 55.0)  # 320 syllables in 55 s


@pytest.mark.parametrize("text", ["धन्यवाद " * 6, " ".join(HINDI.split() * 6)[:4000], "ขอบคุณครับ" * 120, "ขอบคุณครับ " * 6])
def test_hindi_and_thai_loops_are_repetitive(text: str) -> None:
    assert looks_repetitive(text, 60.0)


def test_repetition_thresholds_are_configurable() -> None:
    assert looks_repetitive("thank you " * 3, 60.0, max_repeats=3)
    assert not looks_repetitive(" ".join(["word"] * 100), 60.0, max_wpm=1000, max_repeats=200)


# --- ParakeetTranscriber -------------------------------------------------------------------


def test_parakeet_turns_sentences_into_absolute_segments(loads: Callable) -> None:
    fake = FakeParakeet([
        sentence(" Hello there.", 0.5, 1.75),
        sentence(" ", 2.0, 2.1),
        sentence(" How are you?", 2.25, 4.0),
    ])
    loads(fake)
    audio = np.concatenate([tone(4.0), silence(6.0)])

    segments = ParakeetTranscriber().transcribe(audio, offset_s=600.0)

    assert segments == [
        Segment(600.5, 601.75, "Hello there."),
        Segment(602.25, 604.0, "How are you?"),
    ]
    (passed, kwargs), = fake.calls
    assert isinstance(passed, mx.array) and passed.shape == (len(audio),)
    assert kwargs["chunk_duration"] == 30.0 and kwargs["overlap_duration"] == 2.0


def test_parakeet_passes_its_chunk_settings(loads: Callable) -> None:
    fake = FakeParakeet([])
    loads(fake)
    ParakeetTranscriber(chunk_duration=60.0, overlap_duration=1.0).transcribe(silence(1.0))
    assert fake.calls[0][1]["chunk_duration"] == 60.0
    assert fake.calls[0][1]["overlap_duration"] == 1.0


def test_parakeet_clamps_times_to_the_audio(loads: Callable) -> None:
    loads(FakeParakeet([sentence(" Overrun.", 9.0, 12.0)]))
    (seg,) = ParakeetTranscriber().transcribe(np.concatenate([silence(9.0), tone(1.0)]), offset_s=5.0)
    assert (seg.start, seg.end) == (14.0, 15.0)


def test_parakeet_progress_reports_finished_chunks(loads: Callable) -> None:
    # mlx-audio calls chunk_callback before decoding each chunk, with that chunk's end.
    loads(FakeParakeet([], callbacks=[(30, 100), (60, 100), (100, 100)]))
    progress: list[float] = []
    ParakeetTranscriber().transcribe(silence(5.0), on_progress=progress.append)
    assert progress == [0.0, 0.3, 0.6, 1.0]


def test_parakeet_progress_without_chunking(loads: Callable) -> None:
    loads(FakeParakeet([sentence(" Hi.", 0.0, 1.0)]))  # short input: no callbacks at all
    progress: list[float] = []
    ParakeetTranscriber().transcribe(tone(5.0), on_progress=progress.append)
    assert progress == [0.0, 1.0]


def test_parakeet_ignores_language_and_warns_once_about_hotwords(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeParakeet([sentence(" Hi.", 0.0, 1.0)])
    loads(fake)
    t = ParakeetTranscriber()
    with caplog.at_level(logging.DEBUG, logger="studybuddy.transcribe"):
        t.transcribe(tone(2.0), language="fr", hotwords=["eigenvalue"])
        t.transcribe(tone(2.0), language="fr", hotwords=["eigenvalue"])
    assert warnings_text(caplog).count("hotwords") == 1
    assert all("language" not in kwargs and "hotwords" not in kwargs for _, kwargs in fake.calls)
    assert any("ignoring language" in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


def test_parakeet_warns_when_audio_has_sound_but_no_text(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeParakeet([])
    loads(fake)
    with caplog.at_level(logging.WARNING):
        assert ParakeetTranscriber().transcribe(np.concatenate([silence(5.0), tone(5.0)])) == []
    assert len(fake.calls) == 2  # tried again before giving up
    assert "0:00:00.0-0:00:10.0: no text although the audio is not silent" in warnings_text(caplog)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ParakeetTranscriber().transcribe(silence(5.0))
    assert warnings_text(caplog) == ""


def test_parakeet_transcribes_skipped_speech_again(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    # Three spoken stretches separated by pauses; the first pass skips the middle one.
    audio = np.concatenate([tone(4.0), silence(3.0), tone(4.0), silence(3.0), tone(4.0)])
    fake = FakeParakeet(
        [sentence(" First.", 0.0, 4.0), sentence(" Third.", 14.0, 18.0)],
        then=[sentence(" Second.", 3.0, 7.0)],  # relative to the re-run stretch (4 s to 14 s)
    )
    loads(fake)

    with caplog.at_level(logging.WARNING):
        segments = ParakeetTranscriber().transcribe(audio, offset_s=60.0)

    assert segments == [
        Segment(60.0, 64.0, "First."), Segment(67.0, 71.0, "Second."), Segment(74.0, 78.0, "Third."),
    ]
    (_, first), (rerun_audio, rerun) = fake.calls
    assert rerun_audio.shape == (10 * SR,)
    assert rerun["chunk_duration"] == first["chunk_duration"] and rerun["chunk_callback"] is None
    assert "0:01:04.0-0:01:14.0: Parakeet skipped audible speech" in warnings_text(caplog)


def test_parakeet_leaves_pauses_and_short_gaps_alone(loads: Callable) -> None:
    # 4-7 s is a silent pause; 10-11.5 s has sound but is shorter than MIN_GAP_S.
    audio = np.concatenate([tone(4.0), silence(3.0), tone(4.5)])
    fake = FakeParakeet([sentence(" One.", 0.0, 4.0), sentence(" Two.", 7.0, 10.0)])
    loads(fake)
    ParakeetTranscriber().transcribe(audio)
    assert len(fake.calls) == 1


def test_parakeet_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        ParakeetTranscriber(chunk_duration=2.0, overlap_duration=2.0)


# --- Qwen3Transcriber: conversion ----------------------------------------------------------


def test_qwen_single_chunk_offsets_language_and_hotwords(loads: Callable) -> None:
    fake = FakeQwen(lambda i, chunk, kw: "Hello world.")
    loads(fake)
    audio = tone(15.0)

    segments = Qwen3Transcriber().transcribe(audio, offset_s=120.0, language="en", hotwords=["eigenvalue"])

    assert segments == [Segment(120.0, 135.0, "Hello world.", "English")]
    (call,) = fake.calls
    assert call["samples"] == len(audio)
    assert call["language"] == "English" and call["hotwords"] == ["eigenvalue"]
    assert call["batch_size"] == 1 and call["max_tokens"] == 375  # 25 tokens per second of audio
    assert call["repetition_penalty"] is None
    assert call["chunk_duration"] > 15.0  # kept in one chunk


def test_qwen_detected_language_is_reported(loads: Callable) -> None:
    loads(FakeQwen(lambda i, chunk, kw: "language French<asr_text>Bonjour à tous."))
    (seg,) = Qwen3Transcriber().transcribe(tone(5.0))
    assert (seg.text, seg.language) == ("Bonjour à tous.", "French")


def test_qwen_batched_call_covers_the_audio_and_cleans_language_prefixes(loads: Callable) -> None:
    raw = [
        "language English<asr_text>One.",
        "language French<asr_text>Deux.",
        "language German<asr_text>Drei.",
    ]
    fake = FakeQwen(lambda i, chunk, kw: raw[i])
    loads(fake)
    t = Qwen3Transcriber(chunk_duration=12.0)
    audio = tone_blocks(3)
    bounds = t.chunk_bounds(audio)
    assert len(bounds) == 3

    segments = t.transcribe(audio, offset_s=1000.0)

    assert len(fake.calls) == 1 and fake.calls[0]["batch_size"] == 8
    assert fake.calls[0]["chunk_duration"] == 12.0 and fake.calls[0]["language"] is None
    assert [s.text for s in segments] == ["One.", "Deux.", "Drei."]
    assert [s.language for s in segments] == ["English", "French", "German"]
    assert [(s.start, s.end) for s in segments] == [(1000.0 + a / SR, 1000.0 + b / SR) for a, b in bounds]
    assert segments[0].start == 1000.0 and segments[-1].end == pytest.approx(1000.0 + len(audio) / SR)


def test_qwen_forced_language_in_a_batch(loads: Callable) -> None:
    fake = FakeQwen(lambda i, chunk, kw: f"Satz {i}.")
    loads(fake)
    segments = Qwen3Transcriber(chunk_duration=12.0).transcribe(tone_blocks(3), language="de")
    assert fake.calls[0]["language"] == "German"
    assert [s.language for s in segments] == ["German"] * 3
    assert [s.text for s in segments] == ["Satz 0.", "Satz 1.", "Satz 2."]


def test_qwen_language_none_means_no_speech(loads: Callable, caplog: pytest.LogCaptureFixture) -> None:
    raw = ["language English<asr_text>One.", "language None<asr_text>", "language English<asr_text>Three."]
    loads(FakeQwen(lambda i, chunk, kw: raw[i]))
    with caplog.at_level(logging.WARNING):
        segments = Qwen3Transcriber(chunk_duration=12.0).transcribe(tone_blocks(3))
    assert [s.text for s in segments] == ["One.", "Three."]
    assert "no text although the audio is not silent" in warnings_text(caplog)


def test_qwen_batch_size_one_transcribes_chunk_by_chunk(loads: Callable) -> None:
    langs = iter(["English", "French", "English"])
    fake = FakeQwen(lambda i, chunk, kw: f"language {next(langs)}<asr_text>Part.")
    loads(fake)
    t = Qwen3Transcriber(chunk_duration=12.0, batch_size=1)
    audio = tone_blocks(3)

    segments = t.transcribe(audio)

    assert [c["batch_size"] for c in fake.calls] == [1, 1, 1]
    assert [c["samples"] for c in fake.calls] == [b - a for a, b in t.chunk_bounds(audio)]
    assert [s.language for s in segments] == ["English", "French", "English"]


def test_qwen_chunk_bounds_match_mlx_audio(loads: Callable) -> None:
    audio = tone_blocks(4)
    t = Qwen3Transcriber(chunk_duration=12.0)
    bounds = t.chunk_bounds(audio)
    chunks = split_audio_into_chunks(audio, SR, 12.0, 1.0)
    assert [a for a, _ in bounds] == [round(offset * SR) for _, offset in chunks]
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    assert all(b == c for (_, b), (c, _) in zip(bounds, bounds[1:]))


def test_qwen_progress(loads: Callable) -> None:
    loads(FakeQwen(lambda i, chunk, kw: "Words."))
    progress: list[float] = []
    Qwen3Transcriber(chunk_duration=12.0).transcribe(tone_blocks(3), on_progress=progress.append)
    assert progress == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


# --- Qwen3Transcriber: safety checks -------------------------------------------------------


def test_qwen_reruns_a_repetitive_chunk_with_a_repetition_penalty(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    batched = ["Fine.", loop_text(), "Also fine."]
    fake = FakeQwen(by_kind(lambda i, chunk, kw: batched[i], lambda i, chunk, kw: "Fixed text."))
    loads(fake)
    t = Qwen3Transcriber(chunk_duration=12.0)
    audio = tone_blocks(3)
    a, b = t.chunk_bounds(audio)[1]

    with caplog.at_level(logging.WARNING):
        segments = t.transcribe(audio, language="en")

    assert [s.text for s in segments] == ["Fine.", "Fixed text.", "Also fine."]
    assert len(fake.calls) == 2
    retry = fake.calls[1]
    assert retry["repetition_penalty"] == 1.1 and retry["batch_size"] == 1
    assert retry["samples"] == b - a and retry["language"] == "English"
    assert "looked repetitive; re-ran it with repetition_penalty=1.1" in warnings_text(caplog)


def test_qwen_keeps_the_retry_when_it_is_still_repetitive(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeQwen(
        lambda i, chunk, kw: loop_text("na na") if kw["repetition_penalty"] else loop_text("la la")
    )
    loads(fake)
    with caplog.at_level(logging.WARNING):
        (seg,) = Qwen3Transcriber().transcribe(tone(20.0))
    assert seg.text.startswith("na na")
    assert "still looks repetitive" in warnings_text(caplog)


def test_qwen_keeps_the_original_when_the_retry_is_empty(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    loads(FakeQwen(lambda i, chunk, kw: "" if kw["repetition_penalty"] else loop_text()))
    with caplog.at_level(logging.WARNING):
        (seg,) = Qwen3Transcriber().transcribe(tone(20.0))
    assert seg.text == loop_text()
    assert "returned nothing" in warnings_text(caplog)


def test_qwen_retry_can_be_disabled(loads: Callable, caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeQwen(lambda i, chunk, kw: loop_text())
    loads(fake)
    with caplog.at_level(logging.WARNING):
        Qwen3Transcriber(repetition_penalty=None).transcribe(tone(20.0))
    assert len(fake.calls) == 1
    assert "retry disabled" in warnings_text(caplog)


def test_qwen_reruns_chunks_missing_from_the_batch(loads: Callable, caplog: pytest.LogCaptureFixture) -> None:
    batched = ["One.", "Two.", None]
    fake = FakeQwen(by_kind(lambda i, chunk, kw: batched[i], lambda i, chunk, kw: "Three."))
    loads(fake)
    with caplog.at_level(logging.WARNING):
        segments = Qwen3Transcriber(chunk_duration=12.0).transcribe(tone_blocks(3))
    assert [s.text for s in segments] == ["One.", "Two.", "Three."]
    assert len(fake.calls) == 2
    assert "1 of 3 chunks missing" in warnings_text(caplog)


def test_qwen_reruns_chunks_cut_off_by_the_token_budget(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    # batch_size=2: chunks 0 and 1 decode together and use up the budget; chunk 2 never runs.
    batched = ["Cut o", "Also cut", None]
    reruns = itertools.count()
    fake = FakeQwen(
        by_kind(lambda i, chunk, kw: batched[i], lambda i, chunk, kw: f"Whole chunk {next(reruns)}."),
        exhaust_budget=True,
    )
    loads(fake)
    t = Qwen3Transcriber(chunk_duration=12.0, batch_size=2, max_tokens=50)

    with caplog.at_level(logging.WARNING):
        segments = t.transcribe(tone_blocks(3))

    assert [s.text for s in segments] == ["Whole chunk 0.", "Whole chunk 1.", "Whole chunk 2."]
    assert [c["batch_size"] for c in fake.calls] == [2, 1, 1, 1]
    assert "used up its budget of 50 tokens; re-transcribing" in warnings_text(caplog)


def test_qwen_only_reruns_the_batch_where_the_budget_ran_out(loads: Callable) -> None:
    batched = ["Done.", "Done too.", "Cut o", None]
    fake = FakeQwen(
        by_kind(lambda i, chunk, kw: batched[i], lambda i, chunk, kw: "Rerun."), exhaust_budget=True
    )
    loads(fake)
    segments = Qwen3Transcriber(chunk_duration=12.0, batch_size=2).transcribe(tone_blocks(4))
    assert [s.text for s in segments] == ["Done.", "Done too.", "Rerun.", "Rerun."]


def test_qwen_warns_when_a_single_chunk_hits_max_tokens(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    loads(FakeQwen(lambda i, chunk, kw: "A long lecture that keeps going", exhaust_budget=True))
    with caplog.at_level(logging.WARNING):
        Qwen3Transcriber().transcribe(tone(20.0))
    assert "used up its budget of 500 tokens and may be cut short" in warnings_text(caplog)


def test_token_budget_scales_with_the_audio() -> None:
    t = Qwen3Transcriber()
    assert t.token_budget(60.0) == 1500  # a chunk: enough for the fastest Hindi, bounds a loop
    assert t.token_budget(0.5) == tr.QWEN3_MIN_TOKEN_BUDGET
    assert t.token_budget(600.0) == 8192  # a whole section: capped by max_tokens
    assert Qwen3Transcriber(max_tokens_per_s=10.0).token_budget(60.0) == 600
    with pytest.raises(ValueError):
        Qwen3Transcriber(max_tokens_per_s=0.0)


def test_qwen_bounds_the_rerun_of_a_looping_chunk(loads: Callable) -> None:
    # The batched call loops on chunk 1 until the shared budget runs out; chunk 2 never runs.
    batched = ["Fine.", loop_text(copies=500), None]
    fake = FakeQwen(
        by_kind(lambda i, chunk, kw: batched[i], lambda i, chunk, kw: f"Rerun {kw['max_tokens']}."),
        exhaust_budget=True,
    )
    loads(fake)
    t = Qwen3Transcriber(chunk_duration=12.0)
    audio = tone_blocks(3)
    bounds = t.chunk_bounds(audio)

    t.transcribe(audio)

    batched_call, *reruns = fake.calls
    assert batched_call["max_tokens"] == t.token_budget(len(audio) / SR)  # 775, not 8192
    # Every one-chunk call (re-runs and the repetition-penalty retry) gets its own chunk's budget.
    budgets = {b - a: t.token_budget((b - a) / SR) for a, b in bounds}
    assert len(reruns) == 3 and all(c["max_tokens"] == budgets[c["samples"]] for c in reruns)
    assert all(c["max_tokens"] <= 300 for c in reruns)


def test_qwen_skips_silent_audio_instead_of_inventing_text(loads: Callable) -> None:
    fake = FakeQwen(lambda i, chunk, kw: "So, I'm going to go ahead and start with the first one.")
    loaded = loads(fake)
    assert Qwen3Transcriber().transcribe(silence(5.0), language="en") == []
    assert fake.calls == [] and loaded == [tr.QWEN3_MODEL_ID]


def test_qwen_drops_text_invented_over_a_silent_chunk(
    loads: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    audio = np.concatenate([tone(10.0), silence(11.0), tone(10.0)])
    t = Qwen3Transcriber(chunk_duration=12.0)
    silent = [i for i, (a, b) in enumerate(t.chunk_bounds(audio)) if not np.any(audio[a:b])]
    assert silent, "test audio should produce a fully silent chunk"
    fake = FakeQwen(lambda i, chunk, kw: "Invented words." if i in silent else f"Speech {i}.")
    loads(fake)

    with caplog.at_level(logging.WARNING):
        segments = t.transcribe(audio, language="en")

    assert segments and all(s.text.startswith("Speech") for s in segments)
    assert "dropped text over silent audio" in warnings_text(caplog)


def test_qwen_warns_on_empty_text_over_sound(loads: Callable, caplog: pytest.LogCaptureFixture) -> None:
    loads(FakeQwen(lambda i, chunk, kw: ""))
    with caplog.at_level(logging.WARNING):
        assert Qwen3Transcriber().transcribe(tone(10.0)) == []
    assert "no text although the audio is not silent" in warnings_text(caplog)


def test_qwen_does_not_warn_about_a_quiet_chunk(loads: Callable, caplog: pytest.LogCaptureFixture) -> None:
    loads(FakeQwen(lambda i, chunk, kw: ""))
    with caplog.at_level(logging.WARNING):
        Qwen3Transcriber().transcribe(tone(10.0, amp=0.002))  # about -57 dBFS: room noise
    assert warnings_text(caplog) == ""


# --- shared behaviour ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", [ParakeetTranscriber, Qwen3Transcriber])
def test_model_loads_lazily_once_per_instance(cls: type, loads: Callable) -> None:
    fake = FakeParakeet([]) if cls is ParakeetTranscriber else FakeQwen(lambda i, chunk, kw: "Hi.")
    loaded = loads(fake)
    t = cls()
    assert loaded == []
    t.transcribe(tone(2.0))
    t.transcribe(tone(2.0))
    assert loaded == [t.model_id]
    t.unload()
    t.transcribe(tone(2.0))
    assert loaded == [t.model_id, t.model_id]


@pytest.mark.parametrize("cls", [ParakeetTranscriber, Qwen3Transcriber])
def test_load_preloads_the_model_once(cls: type, loads: Callable) -> None:
    fake = FakeParakeet([]) if cls is ParakeetTranscriber else FakeQwen(lambda i, chunk, kw: "Hi.")
    loaded = loads(fake)
    t = cls()
    t.load()
    assert loaded == [t.model_id]
    t.load()
    t.transcribe(tone(2.0))
    assert loaded == [t.model_id]


@pytest.mark.parametrize("cached", [True, False])
def test_load_model_hides_the_fetch_bar_only_for_cached_weights(
    cached: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    import mlx_audio.stt
    from huggingface_hub.utils import are_progress_bars_disabled

    seen: list[bool] = []

    def fake_load(model_id: str) -> str:
        seen.append(are_progress_bars_disabled())
        return f"model:{model_id}"

    monkeypatch.setattr(mlx_audio.stt, "load", fake_load)
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache",
        lambda repo_id, filename: "/cache/model.safetensors" if cached else None,
    )
    before = are_progress_bars_disabled()
    assert tr.load_model("org/model") == "model:org/model"
    assert seen == [cached or before]
    assert are_progress_bars_disabled() == before  # global setting restored


@pytest.mark.parametrize("cls", [ParakeetTranscriber, Qwen3Transcriber])
def test_empty_audio_needs_no_model(cls: type, loads: Callable) -> None:
    loaded = loads(None)
    progress: list[float] = []
    assert cls().transcribe(np.zeros(0, np.float32), on_progress=progress.append) == []
    assert loaded == [] and progress == [0.0, 1.0]


@pytest.mark.parametrize("cls", [ParakeetTranscriber, Qwen3Transcriber])
def test_stereo_audio_is_rejected(cls: type) -> None:
    with pytest.raises(ValueError, match="mono"):
        cls().transcribe(np.zeros((100, 2), np.float32))


def test_qwen_rejects_an_unknown_language_before_loading(loads: Callable) -> None:
    loaded = loads(None)
    with pytest.raises(ValueError, match="unsupported language"):
        Qwen3Transcriber().transcribe(tone(1.0), language="Klingon")
    assert loaded == []


def test_config_holds_the_output_affecting_settings() -> None:
    assert ParakeetTranscriber().config() == {
        "model_id": tr.PARAKEET_MODEL_ID, "chunk_duration": 30.0, "overlap_duration": 2.0,
    }
    qwen = Qwen3Transcriber().config()
    assert qwen == {
        "model_id": tr.QWEN3_MODEL_ID, "chunk_duration": 60.0, "batch_size": 8, "max_tokens": 8192,
        "max_tokens_per_s": 25.0, "repetition_penalty": 1.1, "max_words_per_minute": 300.0,
        "max_ngram_repeats": 6,
    }
    assert json.loads(json.dumps(qwen)) == qwen
    assert Qwen3Transcriber(chunk_duration=30.0).config() != qwen


def test_transcribers_satisfy_the_protocol() -> None:
    assert isinstance(ParakeetTranscriber(), Transcriber)
    assert isinstance(Qwen3Transcriber(), Transcriber)


def test_make_transcriber() -> None:
    fast = make_transcriber("fast")
    assert isinstance(fast, ParakeetTranscriber) and fast.model_id == tr.PARAKEET_MODEL_ID
    accurate = make_transcriber("accurate", batch_size=4, model_id=None)
    assert isinstance(accurate, Qwen3Transcriber)
    assert accurate.model_id == tr.QWEN3_MODEL_ID and accurate.batch_size == 4
    assert accurate.chunk_duration == 60.0 and accurate.max_tokens == 8192
    with pytest.raises(ValueError, match="unknown mode"):
        make_transcriber("turbo")
    with pytest.raises(TypeError):
        make_transcriber("fast", batch_size=4)  # not a Parakeet setting


# --- real models (STUDYBUDDY_MODEL_TESTS=1) -------------------------------------------------

EXTRA_PASSAGES = (
    "The museum guide led a small group through the gallery of ancient pottery. She "
    "explained how potters shaped the clay by hand, painted bold patterns along the rims, "
    "and fired each bowl slowly in a wood kiln.",
    "Farmers in the valley plant wheat in the autumn and harvest it in early summer. When "
    "the weather turns dry, they open the canals and guide water from the river into long "
    "rows of thirsty crops.",
    "At the train station, travelers hurried across the platform with heavy suitcases. A "
    "voice announced a short delay, so some people bought coffee and newspapers while "
    "others sat on wooden benches and watched the pigeons.",
    "The young chemist carefully poured a blue liquid into a glass flask. She heated it "
    "over a small flame, recorded the color changes in her notebook, and smiled when the "
    "solution finally turned a bright green.",
    "Along the coast, fishing boats return to the harbor before sunset. The crews unload "
    "their nets, sort the silver fish into wooden crates, and sell them at the busy market "
    "near the old stone lighthouse.",
    "Our gardening club meets every spring to plan the community garden. Members choose "
    "which vegetables to grow, share seeds and tools, and build raised beds so that "
    "children from the nearby school can help water the plants.",
    "During the storm, the power went out across the entire neighborhood. Families lit "
    "candles, played card games by the fireplace, and listened to the rain hammer against "
    "the windows until the lights flickered back on.",
)
LONG_PASSAGES = media.PASSAGES + EXTRA_PASSAGES


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def wer(reference: str, segments: list[Segment]) -> float:
    return jiwer.wer(_norm(reference), _norm(" ".join(s.text for s in segments)))


@pytest.fixture(scope="module")
def parakeet() -> ParakeetTranscriber:
    t = ParakeetTranscriber()
    yield t
    t.unload()


@pytest.fixture(scope="module")
def qwen3() -> Qwen3Transcriber:
    t = Qwen3Transcriber()
    yield t
    t.unload()


@pytest.fixture(scope="module")
def long_pcm(media_dir: Path) -> np.ndarray:
    """All twelve distinct passages with 3 s pauses: about 3 minutes."""
    path = media.make_long_speech(media_dir / "twelve_passages.wav", LONG_PASSAGES, gap_s=3.0)
    return media.decode_pcm(path)


def _timed(t: Transcriber, audio: np.ndarray, **kwargs: Any) -> tuple[list[Segment], float]:
    started = time.perf_counter()
    segments = t.transcribe(audio, **kwargs)
    return segments, time.perf_counter() - started


def _check_timeline(segments: list[Segment], offset: float, duration: float) -> None:
    assert segments and all(s.text.strip() for s in segments)
    assert all(offset <= s.start <= s.end <= offset + duration + 1e-6 for s in segments)
    assert [s.start for s in segments] == sorted(s.start for s in segments)
    assert all(b.start >= a.end - 0.5 for a, b in zip(segments, segments[1:]))  # no real overlap
    assert segments[0].start - offset < 5.0 and offset + duration - segments[-1].end < 5.0


@pytest.mark.model
def test_parakeet_real_short(parakeet: ParakeetTranscriber, short_pcm: np.ndarray, speech_text: str) -> None:
    parakeet.transcribe(short_pcm[: SR * 2])  # load and warm up outside the timing
    segments, elapsed = _timed(parakeet, short_pcm, offset_s=100.0)
    score = wer(speech_text, segments)
    print(f"\n[model] parakeet short: WER {score:.3f}, {elapsed:.2f} s for {len(short_pcm) / SR:.1f} s")
    assert score < 0.25
    _check_timeline(segments, 100.0, len(short_pcm) / SR)
    assert all(s.language is None for s in segments)


@pytest.mark.model
@pytest.mark.parametrize(("language", "hotwords"), [(None, None), ("en", ["librarian", "poetry"])])
def test_qwen3_real_short(
    qwen3: Qwen3Transcriber, short_pcm: np.ndarray, speech_text: str,
    language: str | None, hotwords: list[str] | None,
) -> None:
    qwen3.transcribe(short_pcm[: SR * 2], language="en")  # load and warm up outside the timing
    segments, elapsed = _timed(qwen3, short_pcm, offset_s=100.0, language=language, hotwords=hotwords)
    score = wer(speech_text, segments)
    print(f"\n[model] qwen3 short (language={language}, hotwords={hotwords}): "
          f"WER {score:.3f}, {elapsed:.2f} s for {len(short_pcm) / SR:.1f} s")
    assert score < 0.25
    assert len(segments) == 1 and segments[0].language == "English"
    assert segments[0].start == 100.0 and segments[0].end == pytest.approx(100.0 + len(short_pcm) / SR)


@pytest.mark.model
def test_parakeet_real_multi_chunk(parakeet: ParakeetTranscriber, long_pcm: np.ndarray) -> None:
    duration = len(long_pcm) / SR
    assert duration > parakeet.chunk_duration  # mlx-audio has to chunk it
    progress: list[float] = []
    segments, elapsed = _timed(parakeet, long_pcm, offset_s=600.0, on_progress=progress.append)
    score = wer(" ".join(LONG_PASSAGES), segments)
    print(f"\n[model] parakeet multi-chunk: WER {score:.3f}, {len(segments)} segments, "
          f"{elapsed:.2f} s for {duration:.1f} s ({duration / elapsed:.0f}x real time)")
    assert score < 0.05  # a skipped passage alone costs about 0.08
    _check_timeline(segments, 600.0, duration)
    assert progress[0] == 0.0 and progress[-1] == 1.0 and progress == sorted(progress)
    assert len(progress) > 2  # at least one mid-run update from chunk_callback


@pytest.mark.model
def test_parakeet_real_recovers_skipped_speech(
    parakeet: ParakeetTranscriber, long_pcm: np.ndarray,
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    # With 120 s chunks mlx-audio's Parakeet skipped two of these passages outright.
    monkeypatch.setattr(parakeet, "chunk_duration", 120.0)
    with caplog.at_level(logging.WARNING, logger="studybuddy.transcribe"):
        segments, elapsed = _timed(parakeet, long_pcm)
    reruns = warnings_text(caplog).count("skipped audible speech")
    score = wer(" ".join(LONG_PASSAGES), segments)
    print(f"\n[model] parakeet 120 s chunks: WER {score:.3f} after {reruns} re-runs, {elapsed:.2f} s")
    assert score < 0.05
    _check_timeline(segments, 0.0, len(long_pcm) / SR)


@pytest.mark.model
def test_qwen3_real_multi_chunk(qwen3: Qwen3Transcriber, long_pcm: np.ndarray) -> None:
    duration = len(long_pcm) / SR
    bounds = qwen3.chunk_bounds(long_pcm)
    assert len(bounds) > 1  # one batched generate() over several chunks
    progress: list[float] = []
    # language=None exercises the mlx-audio 0.5.8 batched language-prefix bug.
    segments, elapsed = _timed(qwen3, long_pcm, offset_s=600.0, on_progress=progress.append)
    score = wer(" ".join(LONG_PASSAGES), segments)
    print(f"\n[model] qwen3 multi-chunk: WER {score:.3f}, "
          f"{len(segments)} segments over {len(bounds)} chunks, "
          f"{elapsed:.2f} s for {duration:.1f} s ({duration / elapsed:.0f}x real time)")
    assert score < 0.05
    _check_timeline(segments, 600.0, duration)
    by_start = {round((s.start - 600.0) * SR): s for s in segments}
    assert set(by_start) <= {a for a, _ in bounds}  # one segment per chunk, on its bounds
    for a, b in bounds:
        if (seg := by_start.get(a)) is not None:
            assert seg.end == pytest.approx(600.0 + b / SR)
        else:
            assert np.abs(long_pcm[a:b]).max() < 1e-3  # only a silent chunk (the trailing pause) has no text
    assert all("<asr_text>" not in s.text and not s.text.lower().startswith("language ") for s in segments)
    assert all(s.language == "English" for s in segments)
    assert progress[0] == 0.0 and progress[-1] == 1.0 and progress == sorted(progress)


@pytest.mark.model
def test_real_models_return_nothing_for_silence(
    parakeet: ParakeetTranscriber, qwen3: Qwen3Transcriber
) -> None:
    assert parakeet.transcribe(silence(5.0)) == []
    assert qwen3.transcribe(silence(5.0), language="en") == []  # forced language invents text


@pytest.mark.model
def test_language_names_match_the_checkpoint(qwen3: Qwen3Transcriber) -> None:
    model = qwen3._load()
    assert set(tr.QWEN3_LANGUAGES.values()) == set(model.config.support_languages)
