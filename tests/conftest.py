from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import media
from fakes import FakeClaude

MODEL_TESTS_ENV = "STUDYBUDDY_MODEL_TESTS"
OFFICE_TESTS_ENV = "STUDYBUDDY_OFFICE_TESTS"

# marker -> (environment variable that enables it, why it is off by default)
_GATED = {
    "model": (MODEL_TESTS_ENV, "loads a real ASR model"),
    "office": (OFFICE_TESTS_ENV, "drives PowerPoint/Word"),
}


def model_tests_enabled() -> bool:
    return os.environ.get(MODEL_TESTS_ENV) == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, (env, reason) in _GATED.items():
        if os.environ.get(env) == "1":
            continue
        skip = pytest.mark.skip(reason=f"{reason}; set {env}=1 to run")
        for item in items:
            # get_closest_marker, not `marker in item.keywords`: keywords also hold test names
            # and parametrize ids, so a test merely named "...model..." would be skipped.
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def fake_claude(monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    """No test talks to Claude: study notes (written after add and session) come from a fake."""
    fake = FakeClaude()
    monkeypatch.setattr("studybuddy.study_notes.make_backend", lambda name: fake)
    return fake


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not media.tools_available():
        pytest.skip("test media needs the macOS say command, ffmpeg and ffprobe")
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def speech_text() -> str:
    return media.SPEECH_TEXT


@pytest.fixture(scope="session")
def passages() -> tuple[str, ...]:
    return media.PASSAGES


@pytest.fixture(scope="session")
def speech_audio(media_dir: Path, speech_text: str) -> Path:
    """``speech_text`` spoken by ``say`` (AIFF, mono 22.05 kHz, ~15 s)."""
    return media.make_speech(speech_text, media_dir / "speech.aiff")


@pytest.fixture(scope="session")
def short_mp4(media_dir: Path, speech_audio: Path) -> Path:
    """~15 s MP4: tiny H.264 video + AAC mono speech of ``speech_text``."""
    return media.make_mp4(speech_audio, media_dir / "short.mp4")


@pytest.fixture(scope="session")
def short_pcm(short_mp4: Path) -> np.ndarray:
    """The audio of ``short_mp4`` as float32 mono 16 kHz."""
    return media.decode_pcm(short_mp4)


@pytest.fixture(scope="session")
def no_audio_mp4(media_dir: Path) -> Path:
    """3 s MP4 with a video stream and no audio stream."""
    return media.make_video_only_mp4(media_dir / "no_audio.mp4", seconds=3.0)


@pytest.fixture(scope="session")
def long_speech_wav(media_dir: Path) -> Path:
    """All ``passages`` spoken in order with 1.5 s gaps (WAV, mono 16 kHz, ~72 s)."""
    return media.make_long_speech(media_dir / "long_speech.wav")


@pytest.fixture(scope="session")
def long_mp4(media_dir: Path, long_speech_wav: Path) -> Path:
    """MP4 version of ``long_speech_wav``, for multi-section / multi-chunk tests."""
    return media.make_mp4(long_speech_wav, media_dir / "long.mp4")
