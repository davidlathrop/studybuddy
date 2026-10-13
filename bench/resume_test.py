"""Crash-and-resume test of the real CLI on a benchmark file (plan milestone M3, measured).

    .venv/bin/python bench/resume_test.py                       # lecture_30m, accurate mode
    .venv/bin/python bench/resume_test.py --mode fast --kills 1

Runs ``studybuddy transcribe`` and SIGKILLs its whole process group (no cleanup at all, like
a crash or a closed laptop lid that never comes back) as soon as one more section has
reached ``<name>.partial.jsonl``; repeats that ``--kills`` times, then lets a final run finish.
Checks that every re-run extracts the MP3 again to the same bytes, logs the resume,
transcribes only the sections that were missing, and produces complete output identical to
an uninterrupted run (when one exists in ``bench/out/<name>-<mode>/``). Writes
``bench/out/resume_test.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import metrics

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
STUDYBUDDY = ROOT / ".venv" / "bin" / "studybuddy"


def _sections_written(partial: Path) -> int:
    if not partial.exists():
        return 0
    lines = [line for line in partial.read_bytes().split(b"\n") if line.strip()]
    complete = 0
    for line in lines[1:]:
        try:
            json.loads(line)
            complete += 1
        except ValueError:
            pass
    return complete


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _run(
    cmd: list[str], log: Path, *, kill_after_sections: int | None, partial: Path, mp3: Path
) -> dict[str, Any]:
    started = time.monotonic()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        killed_at = None
        while proc.poll() is None:
            if kill_after_sections is not None and _sections_written(partial) >= kill_after_sections:
                os.killpg(proc.pid, signal.SIGKILL)
                killed_at = time.monotonic() - started
                proc.wait()
                break
            time.sleep(0.1)
    text = log.read_text()
    resumed = re.search(r"resuming \S+: (\d+) of (\d+) sections already done", text)
    return {
        "returncode": proc.returncode,
        "wall_s": round(time.monotonic() - started, 2),
        "killed_at_s": None if killed_at is None else round(killed_at, 2),
        "sections_in_partial_after": _sections_written(partial),
        "resume_logged": resumed.group(0) if resumed else None,
        "sections_transcribed": re.findall(r"section (\d+)/(\d+) \S+ done", text),
        "extracted_mp3": "Extracting the audio" in text,
        "mp3_sha256": _sha256(mp3),
        "log": str(log),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--media", type=Path, default=BENCH_DIR / "media" / "lecture_30m.mp4")
    parser.add_argument("--mode", choices=("accurate", "fast"), default="accurate")
    parser.add_argument("--kills", type=int, default=2, help="how many times to kill the CLI")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "out" / "resume")
    args = parser.parse_args(argv)

    media, out = args.media.resolve(), args.out_dir.resolve()
    stem = media.stem
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{stem}.*"):
        old.unlink()
    partial, mp3 = out / f"{stem}.partial.jsonl", out / f"{stem}.mp3"
    cmd = [str(STUDYBUDDY), "transcribe", str(media), "--mode", args.mode,
           "--formats", "txt,srt,json", "--out-dir", str(out)]

    runs = []
    for k in range(1, args.kills + 1):
        run = _run(
            cmd, out / f"run{k}.log",
            kill_after_sections=_sections_written(partial) + 1, partial=partial, mp3=mp3,
        )
        runs.append(run)
        print(f"run {k}: killed at {run['killed_at_s']} s with {run['sections_in_partial_after']} "
              f"section(s) saved; resume line: {run['resume_logged']}", flush=True)
    final = _run(cmd, out / f"run{args.kills + 1}.log", kill_after_sections=None, partial=partial, mp3=mp3)
    runs.append(final)
    print(f"final run: exit {final['returncode']} in {final['wall_s']} s; resume line: "
          f"{final['resume_logged']}; transcribed sections {final['sections_transcribed']}", flush=True)

    result_json = out / f"{stem}.json"
    transcript = json.loads(result_json.read_text()) if result_json.exists() else None
    checks: dict[str, Any] = {
        "final_exit_0": final["returncode"] == 0,
        "partial_deleted": not partial.exists(),
        "outputs_written": all((out / f"{stem}.{ext}").exists() for ext in ("mp3", "txt", "srt", "json")),
        "every_rerun_logged_resume": all(r["resume_logged"] for r in runs[1:]),
        # A re-run extracts the MP3 again; the resume key hashes it, so it must not change.
        "every_rerun_extracted_mp3_again": all(r["extracted_mp3"] for r in runs[1:]),
        "mp3_identical_across_runs": len({r["mp3_sha256"] for r in runs}) == 1,
        "killed_runs_saved_one_more_section": all(
            r["sections_in_partial_after"] == i + 1 for i, r in enumerate(runs[:-1])
        ),
    }
    # Only sections not already saved may be transcribed again by a re-run.
    for i, r in enumerate(runs[1:], start=1):
        saved = runs[i - 1]["sections_in_partial_after"]
        done = {int(n) for n, _ in r["sections_transcribed"]}
        checks[f"run{i + 1}_skipped_saved_sections"] = all(n > saved for n in done)

    wer = None
    if transcript:
        paragraphs, starts = metrics.load_reference(media)
        hypothesis = " ".join(s["text"] for s in transcript["segments"])
        wer = metrics.word_error_rate(paragraphs, hypothesis, paragraph_starts=starts).wer
        times = [(s["start"], s["end"]) for s in transcript["segments"]]
        checks["segments_ordered_and_within_duration"] = all(
            a <= b <= transcript["duration_s"] + 1e-6 for a, b in times
        ) and times == sorted(times)
        straight = BENCH_DIR / "out" / f"{stem}-{args.mode}" / f"{stem}.json"
        if straight.exists():
            reference_run = json.loads(straight.read_text())
            same = [s["text"] for s in reference_run["segments"]] == [s["text"] for s in transcript["segments"]]
            checks["identical_to_uninterrupted_run"] = same

    report = {
        "media": media.name, "mode": args.mode, "kills": args.kills, "signal": "SIGKILL (process group)",
        "runs": runs, "checks": checks, "wer": wer, "passed": all(checks.values()),
    }
    (BENCH_DIR / "out" / "resume_test.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"checks": checks, "wer": wer, "passed": report["passed"]}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
