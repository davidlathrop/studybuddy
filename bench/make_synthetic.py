"""Build synthetic lecture-like MP4s with known reference text (plan milestone M0).

    .venv/bin/python bench/make_synthetic.py            # the standard set: 1, 10 and 30 min, 2.5 h
    .venv/bin/python bench/make_synthetic.py --name lecture_10m_noise50 --minutes 10 --seed 101 --noise-dbfs -50

Each file is varied English text from ``lecture_text`` spoken by macOS ``say``: a main
lecturer voice that changes every dozen topic blocks (about every 25 minutes), students
asking the occasional question in other voices, 0.6-4 s pauses between paragraphs, and a
few 10-20 s stretches of silence. The audio is AAC 128 kbps stereo at 48 kHz (typical of a
lecture recording) muxed with a tiny still video.

Writes, in ``bench/media/`` (gitignored):

* ``<name>.mp4``   the media
* ``<name>.txt``   the exact reference text, one paragraph per spoken paragraph
* ``<name>.json``  manifest: per-paragraph start/end times, voice, rate, topic; the silent gaps

Rendered paragraphs are cached in ``bench/media/.cache`` keyed on (voice, rate, text), so
rebuilding a file is quick.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np

from lecture_text import LectureTextGenerator

log = logging.getLogger("make_synthetic")

BENCH_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BENCH_DIR / "media"
SAY_RATE_HZ = 22050

LECTURERS = ("Samantha", "Daniel", "Karen", "Moira", "Tessa", "Rishi")
STUDENTS = ("Reed (English (US))", "Flo (English (UK))", "Sandy (English (US))", "Eddy (English (UK))")
BLOCKS_PER_LECTURER = 12
LEAD_IN_S = 1.0
TAIL_S = 2.0
PAUSE_IN_BLOCK_S = (0.6, 1.8)
PAUSE_AFTER_QUESTION_S = (0.5, 1.2)
PAUSE_BETWEEN_BLOCKS_S = (2.0, 4.0)
GAP_S = (10.0, 20.0)
GAP_EVERY_S = 600.0  # one long silent gap per ~10 minutes (at least 2 per file)
GAP_MIN_AFTER_S = 60.0  # no long gap closer than this to the end
_VOICE_LINE = re.compile(r"^(.*?)\s+[a-z]{2,3}_[A-Za-z0-9]+\s+#")  # "Eddy (English (UK)) en_GB    # ..."

STANDARD = (  # name, minutes, seed
    ("lecture_1m", 1.0, 105),
    ("lecture_10m", 10.0, 101),
    ("lecture_30m", 30.0, 102),
    ("lecture_150m", 150.0, 103),
)


@dataclass
class Spoken:
    text: str
    topic: str
    kind: str
    voice: str
    rate: int
    wav: Path | None = None
    duration_s: float = 0.0


@dataclass
class PlacedParagraph:
    index: int
    start_s: float
    end_s: float
    voice: str
    rate: int
    topic: str
    kind: str
    text: str


def build(name: str, minutes: float, seed: int, *, out_dir: Path = MEDIA_DIR,
          noise_dbfs: float | None = None, workers: int = 8) -> Path:
    target_s = minutes * 60.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ".cache"
    cache.mkdir(exist_ok=True)
    lecturers, students = _available_voices()
    voice_rng, pause_rng = random.Random(seed * 7919 + 1), random.Random(seed * 7919 + 2)
    blocks = _voiced_blocks(LectureTextGenerator(seed), voice_rng, lecturers, students)
    gap_targets = _gap_targets(target_s, pause_rng)

    timeline: list[tuple[str, object]] = [("silence", LEAD_IN_S)]
    placed: list[PlacedParagraph] = []
    gaps: list[dict[str, float]] = []
    t = LEAD_IN_S
    done = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while not done:
            remaining = max(target_s - t, 60.0)
            wave_blocks = list(islice(blocks, max(2, int(remaining / 110) + 1)))
            list(pool.map(lambda s: _render(s, cache), [s for block in wave_blocks for s in block]))
            for block in wave_blocks:
                for j, s in enumerate(block):
                    pause = 0.0
                    if placed:
                        pause = _pause_before(block, j, t, gap_targets, target_s, pause_rng)
                    if t + pause + s.duration_s + TAIL_S > target_s:
                        done = True
                        break
                    if pause >= GAP_S[0]:
                        gaps.append({"start_s": round(t, 3), "end_s": round(t + pause, 3)})
                    if pause:
                        timeline.append(("silence", pause))
                        t += pause
                    timeline.append(("speech", s))
                    placed.append(PlacedParagraph(
                        len(placed), round(t, 3), round(t + s.duration_s, 3),
                        s.voice, s.rate, s.topic, s.kind, s.text,
                    ))
                    t += s.duration_s
                if done:
                    break
    timeline.append(("silence", target_s - t))  # pad the end to exactly the target length

    mp4 = out_dir / f"{name}.mp4"
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        raw = Path(tmp) / "audio.s16"
        _write_pcm(timeline, raw, noise_dbfs, seed)
        _mux(raw, target_s, mp4)

    words = sum(len(p.text.split()) for p in placed)
    (out_dir / f"{name}.txt").write_text("\n\n".join(p.text for p in placed) + "\n", encoding="utf-8")
    manifest = {
        "name": name, "seed": seed, "duration_s": target_s, "noise_dbfs": noise_dbfs,
        "words": words, "paragraphs_count": len(placed),
        "voices": sorted({p.voice for p in placed}),
        "gaps": gaps, "paragraphs": [asdict(p) for p in placed],
    }
    (out_dir / f"{name}.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    speech_s = sum(p.end_s - p.start_s for p in placed)
    log.info(
        "%s: %.1f min, %d paragraphs, %d words, %.0f%% speech, %d long gaps, voices: %s",
        mp4.name, target_s / 60, len(placed), words, 100 * speech_s / target_s, len(gaps),
        ", ".join(manifest["voices"]),
    )
    return mp4


def _pause_before(block: list[Spoken], j: int, t: float, gap_targets: list[float],
                  target_s: float, rng: random.Random) -> float:
    """Silence before paragraph ``j`` of ``block``; a long gap once one is due (any paragraph)."""
    if gap_targets and t >= gap_targets[0]:
        gap_targets.pop(0)
        gap = rng.uniform(*GAP_S)
        if t + gap + GAP_MIN_AFTER_S <= target_s:  # keep long gaps inside the lecture
            return gap
    if j == 0:
        return rng.uniform(*PAUSE_BETWEEN_BLOCKS_S)
    if block[j - 1].kind == "question":
        return rng.uniform(*PAUSE_AFTER_QUESTION_S)
    return rng.uniform(*PAUSE_IN_BLOCK_S)


def _voiced_blocks(gen: LectureTextGenerator, rng: random.Random,
                   lecturers: list[str], students: list[str]):
    """Yield blocks as lists of Spoken, with voices and speaking rates assigned."""
    first = rng.randrange(len(lecturers))
    for n, block in enumerate(gen.blocks()):
        lecturer = lecturers[(first + n // BLOCKS_PER_LECTURER) % len(lecturers)]
        if n % BLOCKS_PER_LECTURER == 0:
            base_rate = rng.randint(165, 190)
        out = []
        for p in block.paragraphs:
            if p.kind == "question":
                out.append(Spoken(p.text, p.topic, p.kind, rng.choice(students), rng.randint(170, 195)))
            else:
                out.append(Spoken(p.text, p.topic, p.kind, lecturer, base_rate + rng.randint(-8, 8)))
        yield out


def _gap_targets(target_s: float, rng: random.Random) -> list[float]:
    count = max(2, round(target_s / GAP_EVERY_S))
    step = target_s / count
    return [step * (i + 0.5) + rng.uniform(-0.2, 0.2) * step for i in range(count)]


def _available_voices() -> tuple[list[str], list[str]]:
    listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    installed = {m.group(1) for line in listing.splitlines() if (m := _VOICE_LINE.match(line))}
    lecturers = [v for v in LECTURERS if v in installed]
    students = [v for v in STUDENTS if v in installed] or lecturers
    if not lecturers:
        raise SystemExit(f"none of the lecturer voices are installed: {', '.join(LECTURERS)}")
    if missing := [v for v in (*LECTURERS, *STUDENTS) if v not in installed]:
        log.warning("voices not installed, skipping: %s", ", ".join(missing))
    return lecturers, students


def _render(s: Spoken, cache: Path) -> None:
    key = hashlib.sha1(f"{s.voice}\0{s.rate}\0{s.text}".encode()).hexdigest()
    wav = cache / f"{key}.wav"
    if not wav.exists():
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            text_file, out = Path(tmp) / "text.txt", Path(tmp) / "speech.wav"
            text_file.write_text(s.text, encoding="utf-8")
            subprocess.run(
                ["say", "-v", s.voice, "-r", str(s.rate), "--file-format=WAVE",
                 f"--data-format=LEI16@{SAY_RATE_HZ}", "-f", str(text_file), "-o", str(out)],
                check=True, capture_output=True,
            )
            out.replace(wav)
    with wave.open(str(wav)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAY_RATE_HZ, 1, 2):
            raise RuntimeError(f"unexpected WAV format from say: {wav}")
        s.wav, s.duration_s = wav, w.getnframes() / SAY_RATE_HZ


def _write_pcm(timeline: list[tuple[str, object]], raw: Path, noise_dbfs: float | None, seed: int) -> None:
    noise_rng = np.random.default_rng(seed)
    noise_rms = 32768.0 * 10 ** (noise_dbfs / 20) if noise_dbfs is not None else 0.0
    with open(raw, "wb") as fh:
        for kind, item in timeline:
            if kind == "silence":
                pcm = np.zeros(max(0, round(float(item) * SAY_RATE_HZ)), dtype=np.float32)
            else:
                with wave.open(str(item.wav)) as w:
                    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
            if noise_rms:
                pcm += noise_rng.normal(0.0, noise_rms, pcm.size).astype(np.float32)
            fh.write(np.clip(np.round(pcm), -32768, 32767).astype("<i2").tobytes())


def _mux(raw: Path, duration_s: float, mp4: Path) -> None:
    tmp = mp4.with_name(f".{mp4.stem}.tmp.mp4")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "s16le", "-ar", str(SAY_RATE_HZ), "-ac", "1", "-i", str(raw),
        "-f", "lavfi", "-i", f"color=c=0x1e2a38:s=320x180:r=1:d={duration_s:.3f}",
        "-map", "1:v", "-map", "0:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-t", f"{duration_s:.3f}", "-movflags", "+faststart", "-f", "mp4", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(mp4)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", help="build one file with this name instead of the standard set")
    parser.add_argument("--minutes", type=float, help="length of --name")
    parser.add_argument("--seed", type=int, default=1, help="text and voice seed for --name")
    parser.add_argument("--noise-dbfs", type=float, help="add white noise at this RMS level (e.g. -50)")
    parser.add_argument("--out-dir", type=Path, default=MEDIA_DIR)
    parser.add_argument("--workers", type=int, default=8, help="parallel say processes")
    args = parser.parse_args(argv)
    if not all(shutil.which(t) for t in ("say", "ffmpeg")):
        parser.error("needs the macOS say command and ffmpeg")
    if args.name:
        if not args.minutes:
            parser.error("--name needs --minutes")
        jobs = [(args.name, args.minutes, args.seed)]
    else:
        jobs = list(STANDARD)
    for name, minutes, seed in jobs:
        build(name, minutes, seed, out_dir=args.out_dir, noise_dbfs=args.noise_dbfs, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
