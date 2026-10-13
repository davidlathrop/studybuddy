"""Generate small test media with the macOS ``say`` command and ffmpeg.

These helpers depend only on system tools (``say``, ``ffmpeg``, ``ffprobe``) and numpy,
never on ``studybuddy`` itself, so they can be used to test any studybuddy module.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_VOICE = "Samantha"  # en_US; falls back to the system voice if missing

SPEECH_TEXT = (
    "The library opens early on weekdays, and students often gather near the "
    "windows to read quietly. A friendly librarian helps them find old maps, "
    "science journals, and poetry collections. Later in the afternoon, the "
    "reading room becomes calm and peaceful again."
)

# Distinct passages (plain words, no numbers) for building longer, multi-chunk audio.
PASSAGES: tuple[str, ...] = (
    SPEECH_TEXT,
    "Mountain rivers carry cold water from melting snow down through narrow "
    "valleys. Along the banks, tall pine trees shade the trail, and hikers "
    "stop to watch eagles circle above the rocky cliffs before the evening "
    "clouds roll in from the west.",
    "In the kitchen, the baker mixes flour, butter, and honey into a soft "
    "dough. She shapes small loaves, brushes them with milk, and slides them "
    "into the warm oven. Soon the whole street smells of fresh bread and "
    "toasted almonds.",
    "The orchestra tuned their instruments while the audience found their "
    "seats. When the conductor raised her hands, the violins began a gentle "
    "melody, and the cellos answered with a deep and steady rhythm that "
    "filled the concert hall.",
    "Engineers tested the new bridge by driving heavy trucks across it "
    "slowly. They measured how the steel cables stretched and how the concrete "
    "towers moved in the wind, then wrote careful reports for the city "
    "council to review.",
)


class MediaToolError(RuntimeError):
    pass


def tools_available() -> bool:
    return all(shutil.which(tool) for tool in ("say", "ffmpeg", "ffprobe"))


def _run(cmd: Sequence[str | Path]) -> None:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaToolError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()}")


def _ffmpeg(*args: str | Path) -> None:
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args])


def media_duration(path: Path) -> float:
    """Container duration in seconds, via ffprobe."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(proc.stdout.strip())


def make_speech(text: str, path: Path, *, voice: str | None = DEFAULT_VOICE) -> Path:
    """Speak ``text`` into ``path``.

    ``.aiff`` is written by ``say`` directly; any other suffix (``.wav``, ``.m4a``, ...)
    is converted with ffmpeg, keeping say's native mono 22.05 kHz signal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        text_file = Path(tmp) / "text.txt"
        text_file.write_text(text, encoding="utf-8")
        aiff = path if path.suffix.lower() in (".aiff", ".aif") else Path(tmp) / "speech.aiff"
        try:
            _run(["say", *(["-v", voice] if voice else []), "-f", text_file, "-o", aiff])
        except MediaToolError:
            if not voice:
                raise
            _run(["say", "-f", text_file, "-o", aiff])  # voice not installed
        if aiff != path:
            _ffmpeg("-i", aiff, path)
    return path


def make_silence(path: Path, seconds: float, *, sr: int = SAMPLE_RATE) -> Path:
    _ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={sr}:cl=mono", "-t", f"{seconds}", path)
    return path


def make_long_speech(
    path: Path,
    passages: Sequence[str] = PASSAGES,
    *,
    gap_s: float = 1.5,
    repeat: int = 1,
    voice: str | None = DEFAULT_VOICE,
    sr: int = SAMPLE_RATE,
) -> Path:
    """Concatenate several spoken passages, separated by ``gap_s`` of silence.

    The result is a mono ``sr`` Hz file (format from ``path``'s suffix; ``.wav`` is
    cheapest). With the default passages and gap it is about 72 seconds long; use
    ``repeat`` to loop the whole sequence for longer audio.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        tmpdir = Path(tmp)
        pieces = [
            make_speech(text, tmpdir / f"passage{i}.aiff", voice=voice)
            for i, text in enumerate(passages)
        ]
        inputs: list[str | Path] = []
        for piece in pieces:
            inputs += ["-i", piece]
        n = len(pieces)
        # Resample every piece to sr mono, pad each with trailing silence, then concat.
        filters = [
            f"[{i}:a]aresample={sr},aformat=channel_layouts=mono,apad=pad_dur={gap_s}[p{i}]"
            for i in range(n)
        ]
        filters.append("".join(f"[p{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[once]")
        if repeat > 1:
            filters.append(f"[once]aloop=loop={repeat - 1}:size=2147483647[out]")
            label = "[out]"
        else:
            label = "[once]"
        _ffmpeg(*inputs, "-filter_complex", ";".join(filters), "-map", label,
                "-ar", str(sr), "-ac", "1", path)
    return path


def make_mp4(
    audio: Path,
    path: Path,
    *,
    extra_audio_tracks: Sequence[Path] = (),
    stereo: bool = False,
) -> Path:
    """Mux ``audio`` (plus optional extra tracks) with a tiny black H.264 video.

    The video lasts exactly as long as the first audio track.
    """
    seconds = media_duration(audio)
    channels = "2" if stereo else "1"
    inputs: list[str | Path] = ["-f", "lavfi", "-i", f"color=c=black:s=64x64:r=5:d={seconds:.3f}"]
    maps: list[str] = ["-map", "0:v"]
    for i, track in enumerate((audio, *extra_audio_tracks), start=1):
        inputs += ["-i", track]
        maps += ["-map", f"{i}:a"]
    _ffmpeg(
        *inputs, *maps,
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", "-ac", channels,
        "-t", f"{seconds:.3f}", "-movflags", "+faststart", path,
    )
    return path


def make_tone(path: Path, seconds: float, *, freq: float = 440.0, sr: int = SAMPLE_RATE) -> Path:
    _ffmpeg("-f", "lavfi", "-i", f"sine=f={freq}:d={seconds}:r={sr}", path)
    return path


def make_late_audio_mp4(audio: Path, path: Path, *, audio_start_s: float, seconds: float) -> Path:
    """An MP4 whose video starts at 0 and whose audio stream only starts at ``audio_start_s``.

    Lossless trims and some screen recorders produce files like this.
    """
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=5:d={seconds}",
        "-itsoffset", f"{audio_start_s}", "-i", audio, "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", path,
    )
    return path


def make_gapped_audio_mp4(path: Path, *, seconds: float, gap_at_s: float, gap_s: float) -> Path:
    """An MP4 with ``seconds`` of tone whose timestamps jump ahead by ``gap_s`` at ``gap_at_s``.

    That is the shape dropped audio packets leave in long captures; the video covers the
    whole ``seconds + gap_s`` timeline.
    """
    _ffmpeg(
        "-f", "lavfi", "-i", f"sine=f=440:d={seconds}:r={SAMPLE_RATE}",
        "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=5:d={seconds + gap_s}",
        "-map", "1:v", "-map", "0:a",
        "-filter:a", f"asetpts='if(gte(T,{gap_at_s}),PTS+{gap_s}/TB,PTS)'",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", path,
    )
    return path


def make_video_only_mp4(path: Path, seconds: float = 3.0) -> Path:
    """An MP4 with a video stream and no audio stream."""
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=5:d={seconds}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", path,
    )
    return path


def decode_pcm(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any media file to float32 mono at ``sr`` (independent of studybuddy.audio)."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
