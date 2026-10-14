"""Videos: transcript, exam copy and timed passages (plan §5.1).

The transcript and the exam copy are written into the staging folder the caller passes;
both steps are skipped when their output is already there, so re-running ``studybuddy add``
after an interruption resumes (transcription itself also resumes section by section).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from studybuddy import audio, transcription
from studybuddy.enrich.passages import passages_from_transcript
from studybuddy.ingest.model import PassageDraft
from studybuddy.ingest.staging import Staged
from studybuddy.library import Settings
from studybuddy.transcription import NoProgress, Progress

log = logging.getLogger(__name__)


def ingest_video(
    src: Path,
    staging: Path,
    *,
    settings: Settings,
    mode: str,
    hotwords: list[str],
    progress: Progress | None = None,
) -> Staged:
    progress = progress or NoProgress()
    info = audio.probe(src)
    transcript_json = staging / f"{src.stem}.json"
    transcript_txt = staging / f"{src.stem}.txt"
    if transcript_json.is_file() and _matches(transcript_json, src, info):
        log.info("%s: reusing the transcript from an earlier run", src.name)
    else:
        result = transcription.transcribe_media(
            src, staging, mode=mode, formats=["json", "txt"], language=settings.transcribe.language,
            hotwords=hotwords, info=info, progress=progress,
        )
        if not settings.transcribe.keep_mp3:
            result.mp3.unlink(missing_ok=True)

    exam_copy = staging / f"{src.stem}.exam.mp4"
    if not exam_copy.is_file():
        log.info("Making the %dp exam copy", settings.exam_copy.height)
        with progress.stage("exam copy"):
            audio.make_exam_copy(
                src, exam_copy, height=settings.exam_copy.height,
                video_bitrate=settings.exam_copy.video_bitrate, audio_bitrate=settings.exam_copy.audio_bitrate,
                duration_s=info.duration_s, on_progress=progress.update,
            )

    data = json.loads(transcript_json.read_text(encoding="utf-8"))
    passages = [
        PassageDraft(p.text, origin="speech", t_start_ms=round(p.start * 1000), t_end_ms=round(p.end * 1000))
        for p in passages_from_transcript(data)
    ]
    warnings = [] if passages else ["no speech was recognized"]
    files = {"stored": transcript_json, "view": exam_copy}
    if transcript_txt.is_file():
        files["extra"] = transcript_txt
    return Staged(
        passages=passages,
        files=files,
        duration_s=info.duration_s or float(data.get("duration_s") or 0.0) or None,
        page_count=None,
        extractor=f"video/{data.get('mode', mode)}/{data.get('model', '?')}",
        warnings=warnings,
    )


def _matches(transcript_json: Path, src: Path, info: audio.MediaInfo) -> bool:
    """A transcript from an earlier, finished run on this same recording."""
    try:
        data = json.loads(transcript_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    same_name = data.get("source") == src.name
    same_length = abs(float(data.get("duration_s") or 0.0) - (info.duration_s or 0.0)) < 2.0
    return same_name and same_length
