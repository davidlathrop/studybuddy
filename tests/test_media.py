"""Sanity checks for the generated test media (fast; no models)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import media


def _streams(path: Path) -> list[dict]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,channels",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)["streams"]


def test_speech_text_is_plain_words(speech_text: str) -> None:
    assert 35 <= len(speech_text.split()) <= 50
    assert not any(ch.isdigit() for ch in speech_text)


def test_short_mp4_has_video_and_mono_aac(short_mp4: Path) -> None:
    kinds = {(s["codec_type"], s["codec_name"]) for s in _streams(short_mp4)}
    assert kinds == {("video", "h264"), ("audio", "aac")}
    assert 10.0 <= media.media_duration(short_mp4) <= 20.0


def test_short_pcm_is_16k_float32_with_sound(short_pcm: np.ndarray, short_mp4: Path) -> None:
    assert short_pcm.dtype == np.float32 and short_pcm.ndim == 1
    assert abs(len(short_pcm) / media.SAMPLE_RATE - media.media_duration(short_mp4)) < 0.5
    assert float(np.sqrt(np.mean(short_pcm**2))) > 0.01


def test_no_audio_mp4_has_only_video(no_audio_mp4: Path) -> None:
    assert [s["codec_type"] for s in _streams(no_audio_mp4)] == ["video"]
    assert abs(media.media_duration(no_audio_mp4) - 3.0) < 0.5


def test_long_speech_concatenates_passages(long_speech_wav: Path, short_mp4: Path) -> None:
    long_s = media.media_duration(long_speech_wav)
    short_s = media.media_duration(short_mp4)
    # five passages of roughly the short fixture's length, plus gaps
    assert long_s > 4 * short_s
    pcm = media.decode_pcm(long_speech_wav)
    # the 1.5 s gaps really are silent
    window = media.SAMPLE_RATE // 10
    energy = np.convolve(pcm**2, np.ones(window) / window, mode="valid")
    assert float(energy.min()) < 1e-8


def test_multi_track_stereo_mp4(media_dir: Path, speech_audio: Path) -> None:
    silence = media.make_silence(media_dir / "silence.wav", 2.0)
    out = media.make_mp4(speech_audio, media_dir / "two_tracks.mp4",
                         extra_audio_tracks=[silence], stereo=True)
    audio = [s for s in _streams(out) if s["codec_type"] == "audio"]
    assert len(audio) == 2 and all(s["channels"] == 2 for s in audio)
