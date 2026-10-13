"""Guards the mlx-audio STT APIs studybuddy is built on (run with STUDYBUDDY_MODEL_TESTS=1)."""

from __future__ import annotations

import re

import jiwer
import numpy as np
import pytest

pytestmark = pytest.mark.model

PARAKEET_ID = "mlx-community/parakeet-tdt-0.6b-v3"
QWEN3_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z' ]+", " ", text.lower()).strip()


def _wer(ref: str, hyp: str) -> float:
    return jiwer.wer(_norm(ref), _norm(hyp))


def test_parakeet_generate_shape(short_pcm: np.ndarray, speech_text: str) -> None:
    import mlx.core as mx
    from mlx_audio.stt import load

    model = load(PARAKEET_ID)
    calls: list[tuple[int, int]] = []
    # Parakeet needs an mx.array (or a path); a raw np.ndarray raises TypeError.
    result = model.generate(
        mx.array(short_pcm), chunk_duration=6.0, overlap_duration=1.0,
        chunk_callback=lambda cur, total: calls.append((cur, total)),
    )
    assert _wer(speech_text, result.text) < 0.1
    assert calls and calls[-1][0] == calls[-1][1] == len(short_pcm)
    sentences = result.sentences
    assert sentences and all(s.start <= s.end for s in sentences)
    assert sentences[-1].end <= len(short_pcm) / 16000 + 0.5
    assert {"id", "text", "start", "duration", "end"} <= set(vars(sentences[0].tokens[0]))


def test_qwen3_generate_shape(short_pcm: np.ndarray, speech_text: str) -> None:
    from mlx_audio.stt import load

    model = load(QWEN3_ID)
    result = model.generate(short_pcm, chunk_duration=60.0, batch_size=8, language="English")
    assert _wer(speech_text, result.text) < 0.1
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert set(seg) == {"text", "language", "start", "end"}
    assert seg["start"] == 0.0 and seg["end"] == pytest.approx(len(short_pcm) / 16000)
    assert result.generation_tokens < 8192
