"""ffprobe checks, ffmpeg MP3 extraction with progress, and audio decoding.

Every external command gets an argument list (never a shell) and absolute paths, so
file names with spaces, non-ASCII characters, colons or a leading dash are safe.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np

from studybuddy.types import SAMPLE_RATE, ProgressFn

log = logging.getLogger(__name__)

_STDERR_TAIL_LINES = 15
_STDERR_TAIL_BYTES = 16 * 1024
_DECODE_SLACK = 1.01  # headroom over the probed duration when preallocating decoded audio

# MP3 has no timestamps, so keep the container's timeline in the samples themselves: pad
# with silence when the audio starts after the video and fill gaps in its timestamps, so
# transcript times line up with the video. Continuous audio comes out unchanged.
_KEEP_TIMELINE = "aresample=async=1:first_pts=0"


class AudioError(Exception):
    """Raised when a media file cannot be probed, extracted, or decoded."""


@dataclass
class MediaInfo:
    path: Path
    duration_s: float  # 0.0 when neither the container nor the audio stream reports one
    audio_streams: int
    codec: str | None  # codec and channels describe the first audio stream, the one extracted
    channels: int | None


def probe(path: Path) -> MediaInfo:
    """Describe ``path``'s audio; raise AudioError if it is unreadable or has no audio stream."""
    path = Path(path)
    ffprobe = _tool("ffprobe")
    _check_input(path)
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index,codec_name,channels,duration:format=duration",
        "-of", "json", _arg(path),
    ]
    log.debug("running %s", shlex.join(cmd))
    try:
        proc = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        raise AudioError(f"could not run ffprobe: {exc}") from exc
    if proc.returncode != 0:
        detail = _tail(proc.stderr) or f"ffprobe exit status {proc.returncode}"
        raise AudioError(f"cannot read {path} as a media file:\n{detail}")
    try:
        info: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AudioError(f"ffprobe printed unreadable output for {path}") from exc

    streams: list[dict[str, Any]] = info.get("streams") or []
    if not streams:
        raise AudioError(f"{path} has no audio stream, so there is nothing to extract or transcribe")
    first = streams[0]
    duration = _positive_float(info.get("format", {}).get("duration"))
    if duration is None:
        duration = _positive_float(first.get("duration"))
    if duration is None:
        log.warning("%s does not report a duration; progress will not be shown", path)
    return MediaInfo(
        path=path,
        duration_s=duration or 0.0,
        audio_streams=len(streams),
        codec=first.get("codec_name"),
        channels=_positive_int(first.get("channels")),
    )


def extract_mp3(
    src: Path,
    dst: Path,
    *,
    bitrate: str = "96k",
    duration_s: float | None = None,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Encode the first audio stream of ``src`` as a mono MP3 at ``dst`` and return ``dst``.

    The MP3 follows the container's timeline: audio that starts late is preceded by
    silence, and gaps in its timestamps are filled with silence. Extraction is
    deterministic: the same input and ffmpeg give a byte-identical MP3, which resuming
    relies on.

    ffmpeg writes to a temporary file next to ``dst`` that replaces ``dst`` only on success,
    so ``dst`` never holds a half-written MP3. ``on_progress`` gets non-decreasing fractions
    below 1.0 while ffmpeg runs (when the duration is known; ``src`` is probed for it if
    ``duration_s`` is None) and exactly one 1.0 once ``dst`` is in place.
    """
    src, dst = Path(src), Path(dst)
    ffmpeg = _tool("ffmpeg")
    _check_input(src)
    tmp = _partial_path(dst)
    if _same_file(src, dst) or _same_file(src, tmp):
        raise AudioError(f"refusing to extract {src} onto itself; choose another output path")
    if on_progress is not None and duration_s is None:
        duration_s = probe(src).duration_s
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioError(f"cannot create output directory {dst.parent}: {exc}") from exc

    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", _arg(src), "-map", "0:a:0", "-vn", "-af", _KEEP_TIMELINE, "-ac", "1",
        "-c:a", "libmp3lame", "-b:a", bitrate,
        "-progress", "pipe:1", "-nostats", "-f", "mp3", _arg(tmp),
    ]
    progress = _ProgressReporter(duration_s, on_progress)
    try:
        returncode, stderr = _run_with_progress(cmd, progress)
        if returncode != 0:
            raise AudioError(_failure_message("could not extract audio from", src, returncode, stderr))
        if stderr.strip():
            log.warning("ffmpeg reported problems while extracting %s:\n%s", src, _tail(stderr))
        try:
            os.replace(tmp, dst)
        except OSError as exc:
            raise AudioError(f"could not move the finished MP3 to {dst}: {exc}") from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    progress.finish()
    return dst


@dataclass
class VideoInfo:
    width: int | None
    height: int | None
    codec: str | None
    bit_rate: int | None  # bits per second; None when the file does not say


def video_info(path: Path) -> VideoInfo | None:
    """The first video stream's size and codec, or None when ``path`` has no video."""
    path = Path(path)
    _check_input(path)
    cmd = [
        _tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,bit_rate:format=bit_rate", "-of", "json", _arg(path),
    ]
    log.debug("running %s", shlex.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise AudioError(f"ffprobe could not read {path}:\n{_tail(proc.stderr)}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        return None
    s = streams[0]
    # Containers often omit the stream's bit rate; the overall rate is a close upper bound.
    rate = _positive_int(s.get("bit_rate")) or _positive_int((data.get("format") or {}).get("bit_rate"))
    return VideoInfo(_positive_int(s.get("width")), _positive_int(s.get("height")), s.get("codec_name"), rate)


def has_encoder(name: str) -> bool:
    proc = subprocess.run([_tool("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, text=True)
    return any(line.split()[1:2] == [name] for line in proc.stdout.splitlines() if line.strip())


def make_exam_copy(
    src: Path,
    dst: Path,
    *,
    height: int = 540,
    video_bitrate: str = "500k",
    audio_bitrate: str = "64k",
    duration_s: float | None = None,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Write a small, quickly seekable copy of ``src`` for exam day (plan §3) and return ``dst``.

    Scaled down to ``height`` (never up), H.264 + mono AAC, ``+faststart`` so a browser can
    seek before the whole file loads. Uses the Mac's hardware encoder when ffmpeg has it.
    H.264 video that is already no larger than the target (screen recordings of slides
    often are) is copied as is instead of being re-encoded bigger.
    The audio keeps the container's timeline, like the transcript's MP3, so transcript
    times line up with the copy. Written to a temporary file that replaces ``dst`` only on
    success; ``on_progress`` behaves as in :func:`extract_mp3`.
    """
    src, dst = Path(src), Path(dst)
    ffmpeg = _tool("ffmpeg")
    _check_input(src)
    info = video_info(src)
    if info is None:
        raise AudioError(f"{src} has no video stream to make an exam copy of")
    tmp = _partial_path(dst)
    if on_progress is not None and duration_s is None:
        duration_s = probe(src).duration_s
    dst.parent.mkdir(parents=True, exist_ok=True)

    if _small_h264(info, height, video_bitrate):
        video = ["-c:v", "copy"]
    else:
        scale = f"scale=-2:'min({height},ih)':flags=bicubic,format=yuv420p"
        if has_encoder("h264_videotoolbox"):
            encoder = ["-c:v", "h264_videotoolbox", "-b:v", video_bitrate, "-profile:v", "main"]
        else:
            encoder = ["-c:v", "libx264", "-preset", "veryfast", "-b:v", video_bitrate]
        video = ["-vf", scale, *encoder]
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", _arg(src),
        "-map", "0:v:0", "-map", "0:a:0?", *video,
        "-af", _KEEP_TIMELINE, "-ac", "1", "-c:a", "aac", "-b:a", audio_bitrate,
        "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", "-f", "mp4", _arg(tmp),
    ]
    progress = _ProgressReporter(duration_s, on_progress)
    try:
        returncode, stderr = _run_with_progress(cmd, progress)
        if returncode != 0:
            raise AudioError(_failure_message("could not make an exam copy of", src, returncode, stderr))
        if stderr.strip():
            log.warning("ffmpeg reported problems while making an exam copy of %s:\n%s", src, _tail(stderr))
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    progress.finish()
    return dst


def _small_h264(info: VideoInfo, height: int, video_bitrate: str) -> bool:
    """True when re-encoding ``info``'s video could only make it bigger."""
    target = _bits_per_second(video_bitrate)
    return (
        info.codec == "h264"
        and info.bit_rate is not None
        and target is not None
        and (info.bit_rate <= target or (info.height or 0) <= height)
        and info.bit_rate <= 4 * target  # a small frame at a huge rate is still worth shrinking
    )


def _bits_per_second(rate: str) -> int | None:
    """``"500k"`` -> 500000, ``"2M"`` -> 2000000, ``"800000"`` -> 800000."""
    rate = rate.strip().lower()
    scale = {"k": 1_000, "m": 1_000_000}.get(rate[-1:], 1)
    try:
        return int(float(rate.rstrip("km")) * scale)
    except ValueError:
        return None


def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode the first audio stream of ``path`` to float32 mono at ``sr`` Hz, clipped to [-1, 1]."""
    path = Path(path)
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    ffmpeg = _tool("ffmpeg")
    # The probed duration sizes the output buffer; probing also gives the clearer errors.
    expected_samples = math.ceil(probe(path).duration_s * sr * _DECODE_SLACK) + sr
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", _arg(path), "-map", "0:a:0",
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1",
    ]
    log.debug("running %s", shlex.join(cmd))
    with tempfile.TemporaryFile() as stderr_file:
        proc = _popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file)
        assert proc.stdout is not None
        try:
            audio = _read_float32(proc.stdout, expected_samples)
            returncode = proc.wait()
        except BaseException:
            _kill(proc)
            raise
        finally:
            proc.stdout.close()
        stderr = _read_tail(stderr_file)
    if returncode != 0:
        raise AudioError(_failure_message("could not decode audio from", path, returncode, stderr))
    if stderr.strip():
        log.warning("ffmpeg reported problems while decoding %s:\n%s", path, _tail(stderr))

    if audio.size == 0:
        log.warning("decoded no audio samples from %s", path)
    # Decoded MP3/AAC can overshoot full scale slightly; the models expect [-1, 1].
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio


class _ProgressReporter:
    """Turns ffmpeg ``-progress`` lines into non-decreasing fractions in (0, 1).

    1.0 is held back for :meth:`finish`, so a listener only sees it once the work is done.
    """

    def __init__(self, duration_s: float | None, on_progress: ProgressFn | None) -> None:
        self._duration_s = duration_s if duration_s is not None and duration_s > 0 else None
        self._on_progress = on_progress
        self._last = 0.0

    def feed(self, line: str) -> None:
        if self._on_progress is None or self._duration_s is None:
            return
        done_s = _out_time_s(line)
        if done_s is None:
            return
        fraction = min(done_s / self._duration_s, 1.0)
        if self._last < fraction < 1.0:
            self._last = fraction
            self._on_progress(fraction)

    def finish(self) -> None:
        if self._on_progress is not None:
            self._on_progress(1.0)


def _out_time_s(line: str) -> float | None:
    """Seconds of output written so far, from an ``out_time_us=`` progress line (None otherwise)."""
    key, _, value = line.strip().partition("=")
    if key != "out_time_us":
        return None
    try:
        return max(int(value), 0) / 1e6
    except ValueError:  # "N/A" before the first frame is written
        return None


def _run_with_progress(cmd: list[str], progress: _ProgressReporter) -> tuple[int, str]:
    """Run ffmpeg, feeding its stdout to ``progress``; return (exit status, stderr tail).

    stderr goes to a temporary file rather than a pipe, so ffmpeg can never block on a
    full stderr pipe while this process is busy reading stdout.
    """
    log.debug("running %s", shlex.join(cmd))
    with tempfile.TemporaryFile() as stderr_file:
        proc = _popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr_file,
            text=True, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                progress.feed(line)
            returncode = proc.wait()
        except BaseException:
            _kill(proc)
            raise
        finally:
            proc.stdout.close()
        return returncode, _read_tail(stderr_file)


def _popen(cmd: list[str], **kwargs: Any) -> subprocess.Popen:
    try:
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **kwargs)
    except OSError as exc:
        raise AudioError(f"could not run {Path(cmd[0]).name}: {exc}") from exc


def _kill(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait()


def _read_float32(stream: IO[bytes], capacity: int) -> np.ndarray:
    """Read raw f32le samples from ``stream`` straight into one preallocated array.

    With ``capacity`` at or above the real sample count, the whole decode costs a single
    copy of the audio (about 576 MB for 2.5 hours at 16 kHz); pages past the end are never
    touched, so the slack costs no RAM. The buffer doubles if ``capacity`` falls short.
    """
    audio = np.empty(max(capacity, 1), dtype=np.float32)
    raw = audio.view(np.uint8)
    filled = 0
    while True:
        if filled == raw.size:
            grown = np.empty(2 * audio.size, dtype=np.float32)
            grown.view(np.uint8)[:filled] = raw
            audio, raw = grown, grown.view(np.uint8)
        got = stream.readinto(raw[filled:])
        if not got:
            break
        filled += got
    return audio[: filled // 4]


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise AudioError(f"{name} was not found on PATH; install ffmpeg (for example: brew install ffmpeg)")
    return found


def _check_input(path: Path) -> None:
    if not path.exists():
        raise AudioError(f"file not found: {path}")
    if path.is_dir():
        raise AudioError(f"{path} is a directory, not a media file")


def _arg(path: Path) -> str:
    # Absolute paths keep ffmpeg from reading "-x.mp4" as an option or "a:b.mp4" as a protocol.
    return str(path.absolute())


def _partial_path(dst: Path) -> Path:
    return dst.with_name(f"{dst.stem}.partial{dst.suffix or '.mp3'}")


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or (b.exists() and os.path.samefile(a, b))
    except OSError:
        return False


def _read_tail(f: IO[bytes]) -> str:
    f.seek(0, os.SEEK_END)
    f.seek(max(0, f.tell() - _STDERR_TAIL_BYTES))
    return f.read().decode("utf-8", errors="replace")


def _tail(text: str, lines: int = _STDERR_TAIL_LINES) -> str:
    kept = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(kept[-lines:])


def _failure_message(action: str, path: Path, returncode: int, stderr: str) -> str:
    if "matches no streams" in stderr:
        return f"{path} has no audio stream, so there is nothing to extract or transcribe"
    if returncode < 0:
        status = f"ffmpeg was stopped by signal {-returncode}"
    else:
        status = f"ffmpeg exit status {returncode}"
    detail = _tail(stderr)
    return f"{action} {path} ({status})" + (f":\n{detail}" if detail else "")


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
