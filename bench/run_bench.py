"""Time studybuddy end to end on the synthetic benchmark media (plan milestone M0).

    .venv/bin/python bench/run_bench.py run bench/media/lecture_10m.mp4 --mode fast
    .venv/bin/python bench/run_bench.py run bench/media/lecture_10m.mp4 --mode accurate --batch-size 16
    .venv/bin/python bench/run_bench.py suite core      # every run in its own process, in order
    .venv/bin/python bench/run_bench.py cli bench/media/lecture_1m.mp4   # the real CLI, end to end
    .venv/bin/python bench/run_bench.py table           # markdown tables from bench/out/*.json

A run goes through the same calls as ``studybuddy transcribe``: ``audio.extract_mp3``, the
transcriber's ``load()``, ``pipeline.transcribe_file`` (with a partial file) and
``output.write_outputs``, and times each step separately. It records:

* ``extract_s``: MP4 -> MP3; ``load_s``: model load; ``transcribe_s``: decode + sections + model
  (``decode_s`` is its audio-decoding part); ``realtime_factor`` = audio seconds / ``transcribe_s``
* memory: MLX peak (``mx.get_peak_memory``) for the load and for the transcription, the
  process's peak physical footprint (includes Metal buffers; what Activity Monitor shows),
  and max RSS (which does not include most GPU memory)
* WER against the reference text (lowercase, no punctuation, whitespace collapsed), also per
  10-minute stretch of the file, and every warning the pipeline logged

Each run writes ``bench/out/<run_id>.json`` and its transcripts to ``bench/out/<run_id>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import metrics

BENCH_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BENCH_DIR / "media"
OUT_DIR = BENCH_DIR / "out"
BUCKET_S = 600.0
ACCURATE_LONG_LIMIT_S = 75 * 60  # skip the 2.5 h accurate run if it would take longer than this

log = logging.getLogger("run_bench")


# --------------------------------------------------------------------------- one run


class _TimedTranscriber:
    """Delegates to a Transcriber and records the wall time of every section."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.sections: list[dict[str, float]] = []

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def config(self) -> dict:
        return self._inner.config()

    def transcribe(self, audio: Any, **kwargs: Any) -> list:
        from studybuddy.types import SAMPLE_RATE

        started = time.perf_counter()
        segments = self._inner.transcribe(audio, **kwargs)
        wall = time.perf_counter() - started
        self.sections.append({
            "offset_s": round(kwargs.get("offset_s", 0.0), 3),
            "audio_s": round(len(audio) / SAMPLE_RATE, 3),
            "wall_s": round(wall, 3),
            "segments": len(segments),
        })
        return segments


class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


def run_id_for(media: Path, mode: str, overrides: dict[str, Any], section_s: float,
               language: str | None, tag: str | None, cache_limit_gb: float | None = None) -> str:
    if tag:
        return tag
    parts = [media.stem, mode]
    short = {"chunk_duration": "cd", "overlap_duration": "ov", "batch_size": "bs", "max_tokens": "mt"}
    parts += [f"{short[k]}{v:g}" for k, v in sorted(overrides.items()) if v is not None]
    if section_s != 600.0:
        parts.append(f"sec{section_s:g}")
    if language:
        parts.append(f"lang-{language}")
    if cache_limit_gb is not None:
        parts.append(f"cache{cache_limit_gb:g}")
    return "-".join(parts)


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from studybuddy import audio, output, pipeline, transcribe

    media: Path = args.media.resolve()
    overrides = {
        "chunk_duration": args.chunk_duration, "overlap_duration": args.overlap,
        "batch_size": args.batch_size, "max_tokens": args.max_tokens,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    run_id = run_id_for(media, args.mode, overrides, args.section_s, args.language, args.tag,
                        args.cache_limit_gb)
    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    work = args.out_dir / run_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    warnings = _WarningCollector()
    logging.getLogger().addHandler(warnings)

    started = time.perf_counter()
    info = audio.probe(media)
    mp3 = work / f"{media.stem}.mp3"
    t = time.perf_counter()
    audio.extract_mp3(media, mp3, duration_s=info.duration_s)
    extract_s = time.perf_counter() - t
    log.info("extracted %.0f s of audio in %.2f s", info.duration_s, extract_s)

    transcriber = transcribe.make_transcriber(args.mode, **overrides)
    mx.reset_peak_memory()
    t = time.perf_counter()
    transcriber.load()
    load_s = time.perf_counter() - t
    model_active = mx.get_active_memory()
    load_peak = mx.get_peak_memory()
    mx.reset_peak_memory()

    decode_s = 0.0
    real_load_audio = audio.load_audio

    def timed_load_audio(*a: Any, **kw: Any) -> Any:
        nonlocal decode_s
        t0 = time.perf_counter()
        try:
            return real_load_audio(*a, **kw)
        finally:
            decode_s += time.perf_counter() - t0

    timed = _TimedTranscriber(transcriber)
    partial = work / f"{media.stem}.partial.jsonl"
    audio.load_audio = timed_load_audio
    try:
        t = time.perf_counter()
        result = pipeline.transcribe_file(
            mp3, timed, mode=args.mode, section_s=args.section_s, language=args.language,
            partial_path=partial, source=media.name,
        )
        transcribe_s = time.perf_counter() - t
    finally:
        audio.load_audio = real_load_audio
    transcribe_peak = mx.get_peak_memory()

    t = time.perf_counter()
    output.write_outputs(result, work / media.stem, ["txt", "srt", "json"])
    write_s = time.perf_counter() - t
    partial.unlink(missing_ok=True)
    total_s = time.perf_counter() - started
    if not args.keep_mp3:
        mp3.unlink(missing_ok=True)

    paragraphs, starts = metrics.load_reference(media)
    hypothesis = " ".join(s.text for s in result.segments)
    wer = metrics.word_error_rate(paragraphs, hypothesis, paragraph_starts=starts, bucket_s=BUCKET_S)
    record = {
        "run_id": run_id,
        "media": media.name,
        "audio_s": round(result.duration_s, 3),
        "mode": args.mode,
        "model_id": transcriber.model_id,
        "config": transcriber.config(),
        "section_s": args.section_s,
        "language": args.language,
        "mlx_cache_limit_gb": args.cache_limit_gb,
        "extract_s": round(extract_s, 3),
        "load_s": round(load_s, 3),
        "decode_s": round(decode_s, 3),
        "transcribe_s": round(transcribe_s, 3),
        "write_s": round(write_s, 3),
        "total_s": round(total_s, 3),
        "realtime_factor": round(result.duration_s / transcribe_s, 2),
        "end_to_end_factor": round(result.duration_s / total_s, 2),
        "wer": round(wer.wer, 5),
        "wer_detail": wer.as_dict(),
        "segments": len(result.segments),
        "detected_language": result.language,
        "sections": timed.sections,
        "memory_gb": {
            "mlx_model_active": metrics.gb(model_active),
            "mlx_peak_load": metrics.gb(load_peak),
            "mlx_peak_transcribe": metrics.gb(transcribe_peak),
            "peak_footprint": metrics.gb(metrics.peak_footprint_bytes()),
            "max_rss": metrics.gb(metrics.max_rss_bytes()),
            "max_rss_children": metrics.gb(metrics.max_rss_children_bytes()),
        },
        "warnings": warnings.messages,
        "machine": metrics.machine_info(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{run_id}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    log.info(
        "%s: transcribe %.1f s (%.1fx real time), WER %.4f, footprint peak %.2f GB, %d warnings",
        run_id, transcribe_s, record["realtime_factor"], wer.wer,
        record["memory_gb"]["peak_footprint"] or 0, len(warnings.messages),
    )
    return record


# --------------------------------------------------------------------------- suites


def _cfg(media: str, mode: str, **settings: Any) -> dict[str, Any]:
    return {"media": media, "mode": mode, **settings}


SUITES: dict[str, list[dict[str, Any]]] = {
    "core": [
        _cfg("lecture_10m", "fast"),
        _cfg("lecture_30m", "fast"),
        _cfg("lecture_150m", "fast"),
        _cfg("lecture_10m", "accurate"),
        _cfg("lecture_30m", "accurate"),
        _cfg("lecture_150m", "accurate", guard="lecture_30m-accurate"),
    ],
    "sweeps": [
        _cfg("lecture_10m", "accurate", batch_size=4),
        _cfg("lecture_10m", "accurate", batch_size=16),
        _cfg("lecture_10m", "fast", chunk_duration=60),
        _cfg("lecture_10m", "fast", chunk_duration=120),
        _cfg("lecture_10m", "fast", chunk_duration=300),
    ],
    # The 10-minute sweeps repeated on the 30-minute file (5,200 words instead of 1,400).
    "confirm": [
        _cfg("lecture_30m", "accurate", batch_size=4),
        _cfg("lecture_30m", "accurate", batch_size=16),
        _cfg("lecture_30m", "fast", chunk_duration=60),
        _cfg("lecture_30m", "fast", chunk_duration=120),
    ],
    # Is the gap between MLX's active peak and the process footprint MLX's buffer cache?
    "cache": [
        _cfg("lecture_10m", "accurate", cache_limit_gb=0),
        _cfg("lecture_10m", "accurate", cache_limit_gb=2),
        _cfg("lecture_10m", "fast", cache_limit_gb=0),
    ],
    # Not part of the M0 brief; cheap checks that help read the results.
    "extras": [
        _cfg("lecture_10m", "accurate", chunk_duration=30),
        _cfg("lecture_10m", "accurate", chunk_duration=120),
        _cfg("lecture_10m", "accurate", language="English"),
        _cfg("lecture_10m_noise50", "fast"),  # same speech, white noise at -50 dBFS
        _cfg("lecture_10m_noise50", "accurate"),
        _cfg("lecture_10m_noise50", "accurate", language="English"),
    ],
}

_FLAGS = {"chunk_duration": "--chunk-duration", "overlap_duration": "--overlap",
          "batch_size": "--batch-size", "max_tokens": "--max-tokens", "section_s": "--section-s",
          "language": "--language", "tag": "--tag", "cache_limit_gb": "--cache-limit-gb"}


def _extrapolated_minutes(guard_id: str, out_dir: Path, audio_s: float) -> float | None:
    path = out_dir / f"{guard_id}.json"
    if not path.exists():
        return None
    guard = json.loads(path.read_text())
    return (guard["transcribe_s"] + guard["load_s"]) * audio_s / guard["audio_s"] / 60


def run_suite(names: list[str], out_dir: Path, force: bool) -> int:
    configs = [c for name in names for c in (SUITES[name] if name != "all" else
                                               [c for s in SUITES.values() for c in s])]
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    failures = 0
    for cfg in configs:
        cfg = dict(cfg)
        media = MEDIA_DIR / f"{cfg.pop('media')}.mp4"
        mode = cfg.pop("mode")
        guard = cfg.pop("guard", None)
        overrides = {k: v for k, v in cfg.items()
                     if k in ("chunk_duration", "overlap_duration", "batch_size", "max_tokens")}
        run_id = run_id_for(media, mode, overrides, cfg.get("section_s", 600.0), cfg.get("language"),
                            cfg.get("tag"), cfg.get("cache_limit_gb"))
        if (out_dir / f"{run_id}.json").exists() and not force:
            log.info("%s: already done, skipping", run_id)
            continue
        if guard:
            minutes = _extrapolated_minutes(guard, out_dir, _media_duration(media))
            if minutes is None or minutes > ACCURATE_LONG_LIMIT_S / 60:
                log.warning("%s: skipped; %s extrapolates to %s min (limit %d)", run_id, guard,
                            "unknown" if minutes is None else f"{minutes:.0f}", ACCURATE_LONG_LIMIT_S // 60)
                continue
            log.info("%s: %s extrapolates to %.1f min; running it", run_id, guard, minutes)
        cmd = [sys.executable, str(Path(__file__).resolve()), "run", str(media), "--mode", mode,
               "--out-dir", str(out_dir)]
        for key, value in cfg.items():
            cmd += [_FLAGS[key], str(value)]
        log.info("running %s", run_id)
        started = time.perf_counter()
        with open(logs / f"{run_id}.log", "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        elapsed = time.perf_counter() - started
        if rc:
            failures += 1
            log.error("%s failed with exit status %d after %.0f s; see %s", run_id, rc, elapsed, logs / f"{run_id}.log")
        else:
            record = json.loads((out_dir / f"{run_id}.json").read_text())
            log.info("%s done in %.0f s: %.1fx real time, WER %.4f, footprint %.2f GB", run_id, elapsed,
                     record["realtime_factor"], record["wer"], record["memory_gb"]["peak_footprint"] or 0)
    return 1 if failures else 0


def _media_duration(media: Path) -> float:
    return json.loads(media.with_suffix(".json").read_text())["duration_s"]


# --------------------------------------------------------------------------- tables


def _settings(r: dict[str, Any]) -> str:
    c = r["config"]
    if r["mode"] == "fast":
        text = f"chunk {c['chunk_duration']:g} s, overlap {c['overlap_duration']:g} s"
    else:
        text = f"chunk {c['chunk_duration']:g} s, batch {c['batch_size']}"
    if r["section_s"] != 600.0:
        text += f", section {r['section_s']:g} s"
    if r.get("language"):
        text += f", language {r['language']}"
    if r.get("mlx_cache_limit_gb") is not None:
        text += f", MLX cache limit {r['mlx_cache_limit_gb']:g} GB"
    return text


def table(out_dir: Path) -> str:
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    records = [r for r in records if "wer_detail" in r]  # skip resume_test.json
    records.sort(key=lambda r: (r["mode"], r["audio_s"], r["run_id"]))
    head = ("| run | audio | settings | extract s | load s | transcribe s | x real time | WER | S/D/I "
            "| MLX peak GB | footprint GB | max RSS GB |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for r in records:
        w, m = r["wer_detail"], r["memory_gb"]
        rows.append(
            f"| {r['run_id']} | {r['audio_s'] / 60:.1f} min | {_settings(r)} | {r['extract_s']:.1f} "
            f"| {r['load_s']:.1f} | {r['transcribe_s']:.1f} | {r['realtime_factor']:.1f} | {r['wer']:.4f} "
            f"| {w['substitutions']}/{w['deletions']}/{w['insertions']} | {m['mlx_peak_transcribe']:.2f} "
            f"| {m['peak_footprint']:.2f} | {m['max_rss']:.2f} |"
        )
    return head + "\n" + "\n".join(rows)


# --------------------------------------------------------------------------- the real CLI


def run_cli(media: Path, mode: str, out_dir: Path) -> dict[str, Any]:
    """Time ``studybuddy transcribe`` itself (a fresh process: imports, model load, everything)."""
    media = media.resolve()
    run_id = f"cli-{media.stem}-{mode}"
    work = out_dir / run_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    studybuddy = BENCH_DIR.parent / ".venv" / "bin" / "studybuddy"
    cmd = ["/usr/bin/time", "-l", str(studybuddy), "transcribe", str(media), "--mode", mode,
           "--formats", "txt,srt,json", "--out-dir", str(work), "-q"]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - started
    if proc.returncode:
        raise SystemExit(f"{run_id}: exit {proc.returncode}\n{proc.stderr}")
    footprint = re.search(r"(\d+)\s+peak memory footprint", proc.stderr)
    rss = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
    transcript = json.loads((work / f"{media.stem}.json").read_text())
    paragraphs, _ = metrics.load_reference(media)
    wer = metrics.word_error_rate(paragraphs, " ".join(s["text"] for s in transcript["segments"]))
    record = {
        "run_id": run_id, "media": media.name, "mode": mode, "audio_s": transcript["duration_s"],
        "wall_s": round(wall, 3), "end_to_end_factor": round(transcript["duration_s"] / wall, 2),
        "wer": round(wer.wer, 5), "segments": len(transcript["segments"]),
        "peak_footprint_gb": metrics.gb(int(footprint.group(1))) if footprint else None,
        "max_rss_gb": metrics.gb(int(rss.group(1))) if rss else None,
        "stderr_tail": proc.stderr.strip().splitlines()[:5],
        "machine": metrics.machine_info(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "cli").mkdir(parents=True, exist_ok=True)
    (out_dir / "cli" / f"{run_id}.json").write_text(json.dumps(record, indent=1) + "\n")
    log.info("%s: %.1f s wall (%.1fx real time end to end), WER %.4f, footprint %.2f GB",
             run_id, wall, record["end_to_end_factor"], wer.wer, record["peak_footprint_gb"] or 0)
    return record


# --------------------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="time one configuration")
    run.add_argument("media", type=Path)
    run.add_argument("--mode", choices=("fast", "accurate"), required=True)
    run.add_argument("--chunk-duration", type=float)
    run.add_argument("--overlap", type=float, help="Parakeet overlap_duration")
    run.add_argument("--batch-size", type=int, help="Qwen3 batch_size")
    run.add_argument("--max-tokens", type=int, help="Qwen3 max_tokens")
    run.add_argument("--section-s", type=float, default=600.0)
    run.add_argument("--language")
    run.add_argument("--tag", help="run id to use instead of the derived one")
    run.add_argument("--cache-limit-gb", type=float,
                     help="mx.set_cache_limit (in units of 10^9 bytes) before loading the model")
    run.add_argument("--keep-mp3", action="store_true")
    run.add_argument("--out-dir", type=Path, default=OUT_DIR)

    suite = sub.add_parser("suite", help="run named suites, each configuration in its own process")
    suite.add_argument("names", nargs="+", choices=[*SUITES, "all"])
    suite.add_argument("--force", action="store_true", help="re-run configurations that already have results")
    suite.add_argument("--out-dir", type=Path, default=OUT_DIR)

    cli = sub.add_parser("cli", help="time the installed studybuddy command end to end")
    cli.add_argument("media", type=Path, nargs="+")
    cli.add_argument("--mode", choices=("fast", "accurate"), nargs="+", default=["fast", "accurate"])
    cli.add_argument("--out-dir", type=Path, default=OUT_DIR)

    tab = sub.add_parser("table", help="print a markdown table of every result")
    tab.add_argument("--out-dir", type=Path, default=OUT_DIR)

    args = parser.parse_args(argv)
    if args.command == "run":
        logging.getLogger("studybuddy").setLevel(logging.INFO)
        run_one(args)
        return 0
    if args.command == "suite":
        return run_suite(args.names, args.out_dir, args.force)
    if args.command == "cli":
        for media in args.media:
            for mode in args.mode:
                run_cli(media, mode, args.out_dir)
        return 0
    print(table(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
