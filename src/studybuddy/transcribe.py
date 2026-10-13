"""Transcriber interface and the mlx-audio backed implementations.

Two backends share one interface:

* ``ParakeetTranscriber`` ("fast"): NVIDIA Parakeet TDT 0.6B v3. Sentence-level timestamps,
  no language option (it detects among its 25 European languages), no hotwords. It can skip
  whole sentences after a pause, so audible stretches it left blank are transcribed again.
* ``Qwen3Transcriber`` ("accurate"): Qwen3-ASR 1.7B. One segment per ~60 s chunk, optional
  language and hotwords. Being LLM based, it can loop or invent text, so every chunk goes
  through the safety checks in plan section 5.6.

Both load their model lazily on the first ``transcribe()`` call and keep it on the instance.
"""

from __future__ import annotations

import gc
import logging
import math
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Protocol, runtime_checkable

import mlx.core as mx
import numpy as np

from studybuddy.types import SAMPLE_RATE, ProgressFn, Segment

log = logging.getLogger(__name__)

PARAKEET_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
QWEN3_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"

MODES = ("accurate", "fast")

# Repetition heuristics (plan 5.6). Lectures run at ~120-180 words per minute.
MAX_WORDS_PER_MINUTE = 300.0
MAX_NGRAM_REPEATS = 6  # this many back-to-back copies of a 1..MAX_NGRAM_WORDS word phrase
MAX_NGRAM_WORDS = 8
MIN_RATE_WINDOW_S = 10.0  # a 3 s tail with 15 words is not 300 wpm in any useful sense

# Level checks, on 100 ms frames. Speech sits around -15..-30 dBFS and real room noise
# around -50..-60, so "silent" here means digital silence: nothing anyone could hear.
FRAME_S = 0.1
SILENT_DBFS = -60.0  # loudest frame below this: there is nothing to transcribe
SOUND_DBFS = -40.0  # at least SOUND_MIN_S of frames above this: something is audible
SOUND_MIN_S = 1.0
MIN_GAP_S = 2.0  # Parakeet: shorter stretches between sentences are not checked for missed speech

QWEN3_MIN_CHUNK_S = 1.0  # mlx-audio pads shorter chunks to this length
# Token budget per second of audio. Measured with Qwen3's tokenizer, fast speech produces
# about 6 tokens/s in English, Chinese or Japanese, 10-11 in Russian or Thai, and 17 in
# Hindi (4.6 tokens per word), so 25 never cuts speech short; it only bounds how long a
# chunk that loops can keep generating.
QWEN3_MAX_TOKENS_PER_S = 25.0
QWEN3_MIN_TOKEN_BUDGET = 64

# ISO 639 codes (and a few common aliases) -> the language names Qwen3-ASR's prompt expects
# (config.json "support_languages" of mlx-community/Qwen3-ASR-1.7B-8bit).
QWEN3_LANGUAGES: dict[str, str] = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
    "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}
_LANGUAGE_LOOKUP = {
    **QWEN3_LANGUAGES,
    **{name.lower(): name for name in QWEN3_LANGUAGES.values()},
    "tl": "Filipino", "tagalog": "Filipino", "mandarin": "Chinese", "farsi": "Persian",
}
_AUTO_LANGUAGE = {"", "auto"}

_NO_TEXT_WARNING = "%s: no text although the audio is not silent (music, or missed speech?)"

# mlx-audio 0.5.8 leaves "language X<asr_text>" on every batched chunk after the first.
_LANGUAGE_PREFIX = re.compile(r"^\s*language\s+([^<]*?)\s*<asr_text>")
_WORD = re.compile(r"\w+(?:['\u2019]\w+)*")
# Scripts written without spaces between words are counted one character at a time. For the
# words-per-minute check, a character is worth half a word in Chinese and Japanese, and a
# quarter of a word in Thai (whose words average 3 to 4 letters), Lao, Myanmar and Khmer.
_CJK = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")  # kana and CJK ideographs
_SOUTHEAST_ASIAN = re.compile("[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff]")  # Thai, Lao, Myanmar, Khmer
_CHAR_WORDS = ((_CJK, 0.5), (_SOUTHEAST_ASIAN, 0.25))


@runtime_checkable
class Transcriber(Protocol):
    model_id: str

    def config(self) -> dict: ...

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_s: float = 0.0,
        language: str | None = None,
        hotwords: list[str] | None = None,
        on_progress: ProgressFn | None = None,
    ) -> list[Segment]: ...


def normalize_language(language: str | None) -> str | None:
    """Map an ISO code ("en", "pt-BR") or a name ("english") to Qwen3-ASR's language name.

    ``None``, ``""`` and ``"auto"`` mean "detect it" and return ``None``.
    Raises ``ValueError`` for a language Qwen3-ASR does not support.
    """
    if language is None or language.strip().lower() in _AUTO_LANGUAGE:
        return None
    key = language.strip().lower()
    primary = re.split(r"[-_]", key, maxsplit=1)[0]  # "pt-BR" -> "pt"
    for candidate in (key, primary):
        if found := _LANGUAGE_LOOKUP.get(candidate):
            return found
    supported = ", ".join(sorted(QWEN3_LANGUAGES.values()))
    raise ValueError(f"unsupported language {language!r}; Qwen3-ASR supports: {supported}")


def looks_repetitive(
    text: str,
    duration_s: float,
    *,
    max_wpm: float = MAX_WORDS_PER_MINUTE,
    max_repeats: int = MAX_NGRAM_REPEATS,
    max_ngram: int = MAX_NGRAM_WORDS,
) -> bool:
    """True if ``text`` is implausibly dense for ``duration_s`` of audio or loops on a phrase.

    Flags more than ``max_wpm`` words per minute, or ``max_repeats`` back-to-back copies of
    any 1..``max_ngram`` word phrase. In scripts written without spaces (Chinese, Japanese,
    Thai, ...) each character is a token, worth a fraction of a word for the rate.
    """
    tokens = _tokens(text)
    words = sum(_word_share(t) for t in tokens)
    if words / (max(duration_s, MIN_RATE_WINDOW_S) / 60.0) > max_wpm:
        return True
    return _has_repeated_ngram(tokens, max_ngram, max_repeats)


def _tokens(text: str) -> list[str]:
    text = text.lower()
    # \w stops at combining marks (Devanagari vowel signs, Thai tone marks, Arabic harakat,
    # ...) and would cut such words into pieces, so find the words in a copy of the text
    # where every mark is "_", which \w matches.
    masked = "".join("_" if unicodedata.category(c)[0] == "M" else c for c in text)
    tokens: list[str] = []
    for m in _WORD.finditer(masked):
        word = text[m.start():m.end()]
        if _CJK.search(word) or _SOUTHEAST_ASIAN.search(word):
            tokens.extend(_split_unspaced(word))
        else:
            tokens.append(word)
    return tokens


def _split_unspaced(word: str) -> list[str]:
    """One token per CJK/Thai/... character (with its marks); other letters stay together."""
    tokens: list[str] = []
    run = ""  # letters of a script that does use spaces, as in "gpu加速"
    for c in word:
        if unicodedata.category(c)[0] == "M" and (run or tokens):
            if run:
                run += c
            else:
                tokens[-1] += c
        elif _CJK.match(c) or _SOUTHEAST_ASIAN.match(c):
            if run:
                tokens.append(run)
                run = ""
            tokens.append(c)
        else:
            run += c
    if run:
        tokens.append(run)
    return tokens


def _word_share(token: str) -> float:
    return next((share for script, share in _CHAR_WORDS if script.match(token)), 1.0)


def _has_repeated_ngram(tokens: Sequence[str], max_ngram: int, copies: int) -> bool:
    # k back-to-back copies of an n-gram is a run of (k - 1) * n positions where
    # tokens[j] == tokens[j + n], so one linear scan per n finds any of them.
    for n in range(1, max_ngram + 1):
        needed, run = (copies - 1) * n, 0
        for a, b in zip(tokens, tokens[n:]):
            run = run + 1 if a == b else 0
            if run >= needed:
                return True
    return False


def load_model(model_id: str) -> Any:
    """Load an mlx-audio STT model (downloads into ~/.cache/huggingface on first use)."""
    from mlx_audio.stt import load

    if not _weights_cached(model_id):
        log.info("Downloading %s into the Hugging Face cache (first use only)", model_id)
        return load(model_id)
    # huggingface_hub draws a "Fetching N files" bar even when every file is cached.
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    already_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        return load(model_id)
    finally:
        if not already_disabled:
            enable_progress_bars()


def _weights_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_id, "model.safetensors"), str)
    except Exception:  # a local path or an unusual repo layout: let the download bar show
        return False


class _LazyModel:
    """Loads ``self.model_id`` on first use and caches it on the instance."""

    model_id: str
    _model: Any

    def load(self) -> None:
        """Load the model now rather than on the first ``transcribe()`` call."""
        self._load()

    def _load(self) -> Any:
        if self._model is None:
            started = time.perf_counter()
            self._model = load_model(self.model_id)
            log.info("Loaded %s in %.1f s", self.model_id, time.perf_counter() - started)
        return self._model

    def unload(self) -> None:
        """Drop the model and give its memory back to the system."""
        self._model = None
        gc.collect()
        mx.clear_cache()


@dataclass
class ParakeetTranscriber(_LazyModel):
    model_id: str = PARAKEET_MODEL_ID
    _: KW_ONLY
    # Measured on this project's test speech (mlx-audio 0.5.8): 45 s, 120 s and unchunked
    # decoding skipped whole sentences after pauses (WER up to 0.17); 20-30 s chunks skipped
    # none, and 30 s was also the fastest (~70x real time on an M4).
    chunk_duration: float = 30.0
    overlap_duration: float = 2.0
    _model: Any = field(default=None, init=False, repr=False, compare=False)
    _warned_hotwords: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_duration < self.chunk_duration:
            raise ValueError("need 0 <= overlap_duration < chunk_duration")

    def config(self) -> dict:
        return {
            "model_id": self.model_id,
            "chunk_duration": self.chunk_duration,
            "overlap_duration": self.overlap_duration,
        }

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_s: float = 0.0,
        language: str | None = None,
        hotwords: list[str] | None = None,
        on_progress: ProgressFn | None = None,
    ) -> list[Segment]:
        audio = _as_mono(audio)
        report = on_progress or _no_progress
        if language:
            log.debug("Parakeet detects the language itself; ignoring language=%r", language)
        if hotwords and not self._warned_hotwords:
            log.warning("Parakeet (fast mode) does not support hotwords; ignoring them")
            self._warned_hotwords = True
        report(0.0)
        if not audio.size:
            report(1.0)
            return []

        model = self._load()
        started = time.perf_counter()
        segments = self._decode(model, audio, offset_s, _parakeet_progress(report))
        for start, end in _missed_speech(audio, segments, offset_s):
            t0 = offset_s + start / SAMPLE_RATE
            where = _span(t0, offset_s + end / SAMPLE_RATE)
            if found := self._decode(model, audio[start:end], t0):
                log.warning(
                    "%s: Parakeet skipped audible speech; transcribed it again on its own: %s",
                    where, _preview(" ".join(s.text for s in found)),
                )
                segments += found
            else:
                log.warning(_NO_TEXT_WARNING, where)
        segments.sort(key=lambda s: s.start)
        _log_speed(len(audio) / SAMPLE_RATE, started)
        report(1.0)
        return segments

    def _decode(
        self,
        model: Any,
        audio: np.ndarray,
        offset_s: float,
        chunk_callback: Callable[[int, int], None] | None = None,
    ) -> list[Segment]:
        # Without chunk_duration mlx-audio encodes the whole input in one pass, and it only
        # accepts mx.array (or a path), not a numpy array.
        result = model.generate(
            mx.array(audio),
            chunk_duration=self.chunk_duration,
            overlap_duration=self.overlap_duration,
            chunk_callback=chunk_callback,
        )
        mx.clear_cache()
        duration = len(audio) / SAMPLE_RATE
        return [
            seg
            for sentence in result.sentences
            if (seg := _sentence_segment(sentence, offset_s, duration))
        ]


def _missed_speech(
    audio: np.ndarray, segments: list[Segment], offset_s: float
) -> list[tuple[int, int]]:
    """(start, end) samples of stretches of at least MIN_GAP_S with sound but no segment."""
    times = [t - offset_s for s in segments for t in (s.start, s.end)]
    edges = [0, *(round(t * SAMPLE_RATE) for t in times), len(audio)]
    return [
        (start, end)
        for start, end in zip(edges[::2], edges[1::2])
        if end - start >= MIN_GAP_S * SAMPLE_RATE and _has_sound(audio[start:end])
    ]


def _sentence_segment(sentence: Any, offset_s: float, duration: float) -> Segment | None:
    text = sentence.text.strip()
    if not text:
        return None
    start = min(max(float(sentence.start), 0.0), duration)
    end = min(max(float(sentence.end), start), duration)
    return Segment(offset_s + start, offset_s + end, text)


def _parakeet_progress(report: ProgressFn) -> Callable[[int, int], None]:
    finished = 0.0

    def callback(current_end: int, total: int) -> None:
        nonlocal finished
        # mlx-audio calls this *before* decoding the chunk that ends at current_end, so
        # only the previous chunk is actually done.
        if finished > 0.0:
            report(finished)
        finished = current_end / total if total else 1.0

    return callback


@dataclass
class _ChunkResult:
    text: str
    language: str | None = None
    truncated: bool = False  # may have been cut off by the token budget


@dataclass
class Qwen3Transcriber(_LazyModel):
    model_id: str = QWEN3_MODEL_ID
    _: KW_ONLY
    chunk_duration: float = 60.0
    batch_size: int = 8
    max_tokens: int = 8192  # the most tokens one generate() call may produce...
    max_tokens_per_s: float = QWEN3_MAX_TOKENS_PER_S  # ...and the most per second of its audio
    repetition_penalty: float | None = 1.1  # used only to re-run chunks that look repetitive
    max_words_per_minute: float = MAX_WORDS_PER_MINUTE
    max_ngram_repeats: int = MAX_NGRAM_REPEATS
    _model: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.chunk_duration <= 0 or self.batch_size < 1 or self.max_tokens < 1:
            raise ValueError("chunk_duration must be > 0, batch_size >= 1 and max_tokens >= 1")
        if self.max_tokens_per_s <= 0:
            raise ValueError("max_tokens_per_s must be > 0")

    def config(self) -> dict:
        return {
            "model_id": self.model_id,
            "chunk_duration": self.chunk_duration,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
            "max_tokens_per_s": self.max_tokens_per_s,
            "repetition_penalty": self.repetition_penalty,
            "max_words_per_minute": self.max_words_per_minute,
            "max_ngram_repeats": self.max_ngram_repeats,
        }

    def chunk_bounds(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """The (start, end) samples of the chunks mlx-audio's generate() will cut ``audio`` into."""
        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import split_audio_into_chunks

        chunks = split_audio_into_chunks(audio, SAMPLE_RATE, self.chunk_duration, QWEN3_MIN_CHUNK_S)
        starts = [round(offset * SAMPLE_RATE) for _, offset in chunks]
        return list(zip(starts, [*starts[1:], len(audio)]))

    def token_budget(self, audio_s: float) -> int:
        """The most tokens one generate() call over ``audio_s`` seconds of audio may produce.

        In a batched call the budget is shared by every chunk, and a chunk that loops keeps
        generating until it runs out; scaling it with the audio bounds what a loop costs, in
        the batched call and in the one-chunk re-runs alike.
        """
        scaled = max(QWEN3_MIN_TOKEN_BUDGET, math.ceil(audio_s * self.max_tokens_per_s))
        return min(self.max_tokens, scaled)

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset_s: float = 0.0,
        language: str | None = None,
        hotwords: list[str] | None = None,
        on_progress: ProgressFn | None = None,
    ) -> list[Segment]:
        audio = _as_mono(audio)
        language = normalize_language(language)
        hotwords = hotwords or None
        report = on_progress or _no_progress
        report(0.0)
        if not audio.size:
            report(1.0)
            return []

        model = self._load()
        started = time.perf_counter()
        bounds = self.chunk_bounds(audio)
        if self.batch_size > 1 and len(bounds) > 1:
            first_pass = self._generate_batched(model, audio, bounds, language, hotwords, offset_s)
        else:
            first_pass = [None] * len(bounds)  # each chunk gets its own generate() call below

        segments: list[Segment] = []
        for i, ((start, end), first) in enumerate(zip(bounds, first_pass)):
            t0, t1 = offset_s + start / SAMPLE_RATE, offset_s + end / SAMPLE_RATE
            piece, where = audio[start:end], _span(t0, t1)
            result = self._finish_chunk(model, piece, first, language, hotwords, where)
            if result.text:
                segments.append(Segment(t0, t1, result.text, result.language))
            report((i + 1) / len(bounds))
        _log_speed(len(audio) / SAMPLE_RATE, started)
        return segments

    def _generate_batched(
        self,
        model: Any,
        audio: np.ndarray,
        bounds: list[tuple[int, int]],
        language: str | None,
        hotwords: list[str] | None,
        offset_s: float,
    ) -> list[_ChunkResult | None]:
        """One batched generate() over all chunks; ``None`` marks a chunk it did not return."""
        budget = self.token_budget(len(audio) / SAMPLE_RATE)
        out = model.generate(
            audio,
            chunk_duration=self.chunk_duration,
            min_chunk_duration=QWEN3_MIN_CHUNK_S,
            batch_size=self.batch_size,
            max_tokens=budget,
            language=language,
            hotwords=hotwords,
        )
        mx.clear_cache()
        index = {start: i for i, (start, _) in enumerate(bounds)}
        results: list[_ChunkResult | None] = [None] * len(bounds)
        for seg in out.segments or []:
            i = index.get(round(float(seg["start"]) * SAMPLE_RATE))
            if i is None:
                start_s = offset_s + float(seg["start"])
                log.warning("Ignoring an unexpected chunk at %.2f s from mlx-audio", start_s)
                continue
            results[i] = _ChunkResult(*_parse_segment(seg, language))

        span = _span(offset_s, offset_s + len(audio) / SAMPLE_RATE)
        # The budget is shared by every chunk of the call. Running out silently cuts off the
        # batch that was decoding and skips every later chunk.
        if out.generation_tokens >= budget:
            present = [i for i, r in enumerate(results) if r is not None]
            last_group = present[-1] // self.batch_size if present else -1
            for i in present:
                if i // self.batch_size == last_group:
                    results[i].truncated = True
            log.warning(
                "%s: used up its budget of %d tokens; re-transcribing the affected chunks one at a time",
                span, budget,
            )
        if missing := sum(r is None for r in results):
            log.warning(
                "%s: %d of %d chunks missing from the batched output; transcribing them one at a time",
                span, missing, len(results),
            )
        return results

    def _finish_chunk(
        self,
        model: Any,
        piece: np.ndarray,
        first: _ChunkResult | None,
        language: str | None,
        hotwords: list[str] | None,
        where: str,
    ) -> _ChunkResult:
        """Run the plan 5.6 checks on one chunk, (re)generating it where needed."""
        if _is_silent(piece):
            # Qwen3 invents text over silence when the language is forced.
            if first is not None and first.text:
                log.warning(
                    "%s: dropped text over silent audio (likely invented): %s",
                    where, _preview(first.text),
                )
            return _ChunkResult("")

        duration = len(piece) / SAMPLE_RATE
        result = first
        if result is None or (result.truncated and not self._repetitive(result.text, duration)):
            result = self._generate_one(model, piece, language, hotwords)
        if self._repetitive(result.text, duration):
            result = self._retry_repetitive(model, piece, result, language, hotwords, where)
        if result.truncated:
            log.warning(
                "%s: output used up its budget of %d tokens and may be cut short",
                where, self.token_budget(duration),
            )
        if not result.text and _has_sound(piece):
            log.warning(_NO_TEXT_WARNING, where)
        return result

    def _retry_repetitive(
        self,
        model: Any,
        piece: np.ndarray,
        original: _ChunkResult,
        language: str | None,
        hotwords: list[str] | None,
        where: str,
    ) -> _ChunkResult:
        if self.repetition_penalty is None:
            log.warning(
                "%s: output looks repetitive (retry disabled): %s", where, _preview(original.text)
            )
            return original
        retry = self._generate_one(
            model, piece, language, hotwords, repetition_penalty=self.repetition_penalty
        )
        penalty = f"repetition_penalty={self.repetition_penalty}"
        if not retry.text and original.text:
            log.warning(
                "%s: output looked repetitive and a re-run with %s returned nothing; keeping it: %s",
                where, penalty, _preview(original.text),
            )
            return original
        if self._repetitive(retry.text, len(piece) / SAMPLE_RATE):
            log.warning(
                "%s: output still looks repetitive after a re-run with %s; keeping the re-run: %s",
                where, penalty, _preview(retry.text),
            )
        else:
            log.warning(
                "%s: output looked repetitive; re-ran it with %s: %s",
                where, penalty, _preview(original.text),
            )
        return retry

    def _generate_one(
        self,
        model: Any,
        piece: np.ndarray,
        language: str | None,
        hotwords: list[str] | None,
        *,
        repetition_penalty: float | None = None,
    ) -> _ChunkResult:
        budget = self.token_budget(len(piece) / SAMPLE_RATE)
        out = model.generate(
            piece,
            chunk_duration=len(piece) / SAMPLE_RATE + 1.0,  # keep it in one chunk
            min_chunk_duration=QWEN3_MIN_CHUNK_S,
            batch_size=1,
            max_tokens=budget,
            language=language,
            hotwords=hotwords,
            repetition_penalty=repetition_penalty,
        )
        mx.clear_cache()
        parsed = [_parse_segment(seg, language) for seg in out.segments or []]
        text = " ".join(t for t, _ in parsed if t)
        detected = next((lang for t, lang in parsed if t), None)
        return _ChunkResult(text, detected, truncated=out.generation_tokens >= budget)

    def _repetitive(self, text: str, duration_s: float) -> bool:
        return looks_repetitive(
            text, duration_s, max_wpm=self.max_words_per_minute, max_repeats=self.max_ngram_repeats
        )


def _parse_segment(seg: dict, forced_language: str | None) -> tuple[str, str | None]:
    """Clean one mlx-audio Qwen3 segment into (text, language)."""
    text = str(seg.get("text") or "").strip()
    language = forced_language or seg.get("language")
    if m := _LANGUAGE_PREFIX.match(text):
        text = text[m.end():].strip()
        if forced_language is None:
            language = m.group(1)
    if not text:
        return "", None
    language = str(language).strip() if language is not None else ""
    return text, None if language.lower() in ("", "none") else language


_TRANSCRIBERS: dict[str, Callable[..., Transcriber]] = {
    "fast": ParakeetTranscriber,
    "accurate": Qwen3Transcriber,
}


def make_transcriber(mode: str, **overrides: Any) -> Transcriber:
    """Build the transcriber for ``mode`` ("fast": Parakeet, "accurate": Qwen3).

    ``overrides`` are constructor arguments; any set to ``None`` keep their defaults.
    """
    try:
        cls = _TRANSCRIBERS[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; expected one of: {', '.join(MODES)}") from None
    return cls(**{k: v for k, v in overrides.items() if v is not None})


def _as_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio as a 1-D array, got shape {audio.shape}")
    return audio


def _frame_dbfs(audio: np.ndarray) -> np.ndarray:
    n = int(FRAME_S * SAMPLE_RATE)
    frames = audio[: len(audio) // n * n].reshape(-1, n) if len(audio) >= n else audio[None, :]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def _is_silent(audio: np.ndarray) -> bool:
    return not audio.size or float(_frame_dbfs(audio).max()) < SILENT_DBFS


def _has_sound(audio: np.ndarray) -> bool:
    if not audio.size:
        return False
    return np.count_nonzero(_frame_dbfs(audio) > SOUND_DBFS) * FRAME_S >= SOUND_MIN_S


def _no_progress(_: float) -> None:
    pass


def _log_speed(audio_s: float, started: float) -> None:
    elapsed = time.perf_counter() - started
    speed = audio_s / max(elapsed, 1e-9)
    log.debug("Transcribed %.1f s of audio in %.1f s (%.0fx real time)", audio_s, elapsed, speed)


def _span(t0: float, t1: float) -> str:
    return f"{_clock(t0)}-{_clock(t1)}"


def _clock(seconds: float) -> str:
    minutes, tenths = divmod(round(max(seconds, 0.0) * 10), 600)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{tenths / 10:04.1f}"


def _preview(text: str, limit: int = 80) -> str:
    return repr(text if len(text) <= limit else text[: limit - 3] + "...")
