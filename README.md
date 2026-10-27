# studybuddy

Turn lecture videos into an MP3 of the audio and a transcript, locally on an Apple silicon
Mac, and the transcripts into study notes. Transcription runs entirely on the Mac with no API
keys. Only the study notes need the network: Claude writes them from the transcript, through
the `claude` command and your Claude login ([Study notes](#study-notes)). Add `--no-notes`
and nothing leaves the Mac.

```console
$ studybuddy transcribe lecture.mp4 --formats txt,srt
lecture.mp4: 1:02:13 of audio; accurate mode (Qwen3-ASR 1.7B)
Extracting the audio to lecture.mp3
extract    100%  done in 0:09
Loading mlx-community/Qwen3-ASR-1.7B-8bit
Loaded mlx-community/Qwen3-ASR-1.7B-8bit in 1.0 s
section 1/6 (0:00:00-0:10:02) done in 24.6 s
...
transcribe 100%  done in 2:31
Done in 2:43 (23x real time): 63 segments; language: English
lecture.mp3
lecture.txt
lecture.srt
lecture.json
Writing study notes for lecture (1:02:13, 63 passages) with claude-opus-5-5
...
lecture.notes.md
```

## What it does

1. Checks the input with `ffprobe` and stops with a clear error if it has no audio track.
2. Extracts the first audio track to `lecture.mp3` (mono, 96 kbps, about 43 MB per hour).
3. Decodes the MP3 to 16 kHz mono and cuts it into sections of about 10 minutes, at quiet points.
4. Transcribes each section with a speech model running on the GPU through
   [mlx-audio](https://github.com/Blaizzy/mlx-audio). Each finished section is saved right
   away, so an interrupted run can resume (see [Resuming](#resuming)).
5. Writes the transcript as `.txt` and, if you ask for them, `.srt` and `.json`.
6. Has Claude write study notes from the transcript to `lecture.notes.md` (see
   [Study notes](#study-notes)), unless you pass `--no-notes`.

The design, the model choice and the measurements behind it are in
[plans/mp4-transcription-plan.md](plans/mp4-transcription-plan.md).

## One sitting: a folder of videos

When the material for one sitting (a workshop day, a recorded session split into several
files) is in one folder, a single command processes it and opens the search page:

```sh
studybuddy session ~/Videos/econ-workshop-oct-15
```

- **Everything in the folder.** It adds every video in the folder, plus any PowerPoint, PDF
  and Word files, in natural order ("Part 2" before "Part 10"). Hidden files, Office lock
  files and other file types are skipped. `--recursive` includes subfolders.
- **Self-contained.** It keeps the transcripts, exam copies and search index in a
  `.studybuddy` folder inside that folder, which you can copy to another Mac with the
  videos. It doesn't touch your semester library or change your default library.
- **Re-runnable.** Run it again after adding files and only the new ones are processed.
- **Or into a course.** To put a sitting into a course of your semester library instead,
  add the folder there: `studybuddy add bio101 ~/Videos/econ-workshop-oct-15`.
- **Study notes.** It writes study notes for every video into a `Study notes` folder next
  to the videos: one file per video, plus `index.md` for the sitting. See
  [Study notes](#study-notes).
- **Then the search page.** It builds the exam index and opens the search page. Click
  **Study notes** there to read the sitting's notes. Reopen the search later with
  `exam/.venv/bin/studybuddy-exam ~/Videos/econ-workshop-oct-15`.
- **Several sittings together.** Keep the sitting folders in `~/StudyBuddy`, or link to
  them from there, and `studybuddy serve` searches them all from one page (see
  [Exam day](#exam-day-searching-the-library)).
- **Options:**
  - `--title` names the sitting.
  - `--mode fast` transcribes about 3× faster.
  - `--no-notes` skips the study notes, so nothing leaves the Mac.
  - `--no-search` only processes the files.
  - `--port` picks the search page's port.
- **Time:** about 3–4 minutes per hour of video in accurate mode, plus a few seconds per
  document, plus the study notes.

## Building a semester library

For an open-book exam, studybuddy can collect a whole semester of lecture videos, slides,
PDFs and Word files into one library, and then search it during the exam with no AI and no
network ([plans/exam-search-plan.md](plans/exam-search-plan.md)).

```sh
studybuddy library init ~/StudyBuddy/fall-2026          # once per semester
studybuddy course add bio101 --title "Biology 101" --glossary bio-terms.txt
studybuddy add bio101 ~/Downloads/lecture04.mp4 --week 4 --date 2026-09-15
studybuddy add bio101 week04.pptx reading.pdf notes.docx
studybuddy add bio101 ~/Videos/bio-workshop-oct-15 --week 6   # a whole folder
studybuddy status            # every course
studybuddy status bio101     # the files of one course
```

What `add` does with each kind of file:

| File | What is kept in the library | Locations |
|---|---|---|
| Video (MP4, MOV, MKV, M4V, WebM) | The transcript (`.json`, `.txt`), a 540p "exam copy" of the video, about 250 MB per hour, made with the Mac's hardware encoder, and [study notes](#study-notes). The original is not copied; you can move or delete it afterwards. | Start and end time of each ~1-minute passage |
| PowerPoint (.pptx) | A copy, plus a PDF made with PowerPoint for viewing (hidden slides included, so page N is slide N) | Slide number; slide text, speaker notes and text in pictures (read with Apple's OCR) are separate passages |
| PDF | A copy | Page number; scanned pages are read with OCR; long pages are split |
| Word (.docx) | A copy, plus a PDF made with Word | Heading path ("Chapter 2 > 2.1 Glycolysis") and page number |

- A folder adds every video, PowerPoint, PDF and Word file in it, in natural order ("Part 2"
  before "Part 10"), skipping hidden files, Office lock files and other file types.
  `--recursive` includes subfolders. `--week` and `--date` apply to every file added.
- The library is found from `--library PATH`, then `$STUDYBUDDY_LIBRARY`, then the folder
  you are in (or any folder above it), then the library `library init` last created.
- `library.toml` in the library holds the settings: transcription mode and language,
  exam-copy size, OCR and PDF conversion on/off, the enrichment backend and model, and
  whether `add` writes study notes.
- Course terms in `courses/<course>/glossary.txt` are used as transcription hotwords.
- Adding a file that is already in the course does nothing. Adding a different file with
  the same name needs `--replace`.
- An interrupted `add` resumes when you run it again: the transcript, exam copy and
  converted PDF already made are reused.
- Converting PowerPoint and Word files needs Microsoft PowerPoint and Word. The first time,
  macOS asks to let the terminal control them. Without them (or with `--no-convert`) the
  files are still added and searchable, just without a PDF to open.

## Study notes

Claude writes study notes for every video studybuddy transcribes: after `transcribe`,
after `add` (for the videos it adds), after `session` (for every video in the sitting), and
whenever you run `studybuddy notes`.

```sh
studybuddy notes                  # every course: the videos whose notes are missing
studybuddy notes bio101           # one course
studybuddy notes bio101 --redo    # write them again, e.g. after changing the model
studybuddy notes lecture.json     # a transcript outside any library -> lecture.notes.md
```

- **What's in them.** For each video, `courses/<course>/notes/<video>.notes.md` contains:
  - a title and an overview;
  - announcements: exams, assignments, readings, deadlines;
  - one section per topic. Each section has a summary, key points, key terms (each with
    the lecturer's own words), examples, and what the lecturer stressed;
  - review questions, with the answers in a separate part so you can test yourself.

  `index.md` lists every lecture with its overview. It also collects all announcements and
  an A–Z glossary of key terms for the whole course.
- **Where they point.** Every item ends with a time like `[12:40]`: the start of the
  passage (about a minute long) where the lecture says it. The times come from the
  transcript, not from Claude.
- **What is checked.** Claude is told to use only what the lecture says. Each key term's
  quote is checked against the transcript, and any quote that isn't found is left out.
  The notes are still AI-written from a machine transcript, so check anything important
  at its time in the lecture.
- **How.** One call outlines the lecture, then one call per ~20 minutes of lecture writes
  its sections. Both use the `[enrich]` backend and model in `library.toml` (Claude Opus
  5.5 through `claude -p` by default, on your Claude subscription). A 10-minute lecture
  took about 1¼ minutes.
- **Only once.** A video's notes are written again only when its transcript or the notes
  prompts change, or with `--redo`. You can edit a notes file: it is kept until `--redo`.
  `index.md` is rewritten every time.
- **Turning them off.** `--no-notes` skips them for `transcribe`, `add` or `session`. In a
  library, `[notes] auto = false` in `library.toml` turns them off for good. `studybuddy notes`
  still works afterwards.
  `studybuddy status` shows how many videos have notes, and `studybuddy status COURSE`
  marks the videos without notes.
- **Problems.** If `claude` isn't installed, a call fails, or a usage limit is reached,
  the transcripts are still written and the videos still added, and the command doesn't fail
  because of it. studybuddy says which notes are missing and which command writes them
  later. Ctrl-C keeps the notes already written.
- **In the search page.** `studybuddy build-exam` puts the notes in the exam index, and
  `studybuddy notes` updates an index that is already there. In the search page, click
  **Study notes** to see a course's lectures, announcements and glossary. Open a lecture's
  notes and click any time to play the lecture from there, with the transcript below. See
  [Exam day](#exam-day-searching-the-library).
- **For a single file:** `studybuddy transcribe lecture.mp4` writes `lecture.notes.md` next to
  the transcript. It also keeps `lecture.json`, even if you didn't ask for that format,
  because the notes are made from it: if they fail, `studybuddy notes lecture.json` redoes
  them without transcribing again.

## Exam day: searching the library

The search runs as a separate, small program, `studybuddy-exam`, in its own environment.
Only four packages are installed there (a web server, fuzzy string matching and a phonetic
code; listed in [exam/requirements.lock](exam/requirements.lock)) and no AI library at all.

```sh
studybuddy serve                              # every library in ~/StudyBuddy, in one page
studybuddy build-exam                         # after adding files: refresh the exam index
                                              # (studybuddy status says when it is out of date)
exam/.venv/bin/studybuddy-exam                # opens the search page in Chrome
exam/.venv/bin/studybuddy-exam ~/StudyBuddy/fall-2026 --port 8765 --no-browser
```

- **Everything in ~/StudyBuddy at once:** `studybuddy serve` (or `studybuddy serve FOLDER`)
  looks for libraries and session folders in `~/StudyBuddy` and its subfolders.
  - It rebuilds any exam index that is missing or out of date, then opens one search page
    for all of them, study notes included.
  - The course list groups the courses by library.
  - It follows links to folders, so a link brings in session folders kept elsewhere, such as
    in OneDrive: `ln -s ~/Library/CloudStorage/OneDrive-…/"Term 5" ~/StudyBuddy/`.
  - `exam/.venv/bin/studybuddy-exam ~/StudyBuddy` opens the same page without rebuilding
    anything, as on exam day.

- **What it does:** search every course at once or filter by course, kind of file and week.
  Results are ranked so passages containing all your words come first. Misspellings get a
  "did you mean" suggestion. Partial words still match.
- **Where results go:** a video result plays the exam copy from that moment. A slide, PDF or
  Word result opens the bundled PDF viewer at that slide or page with your words
  highlighted. Results without a PDF show their text.
- **Study notes:** click **Study notes** next to the course list (or choose a course with the
  search box empty) to read a course's study notes.
  - The course page lists the lectures in order, each with its overview. It also shows
    the course's announcements and an A–Z glossary of key terms, where each term links to
    where it was explained.
  - A lecture's page has its sections, key points, key terms, examples, and review
    questions (click one to show its answer).
  - Every point has its time in the margin. Clicking the time plays the lecture from there
    and shows that part of the transcript below the video, so you can check the notes
    against what was said.
  - The notes are labelled as written by AI before the exam. They are shown but not
    searched: search results come from the original material only.
  - Typing a search shows results; clearing it brings the notes back. The browser's Back
    button steps back through the pages of notes.
- **How "no AI" is enforced and shown:**
  - The first thing the program does is install a Python audit hook. From then on, importing
    any AI or machine-learning library (or the studybuddy builder) is refused.
  - Any network connection or name lookup that isn't to this computer is refused too.
  - The page's status line shows "No AI · Offline", the index's fingerprint and when it was
    built. With several libraries, hovering over it lists each library's index fingerprint.
  - The page itself may load only files from this computer (Content-Security-Policy), and the
    server listens on 127.0.0.1 only.
  - Keep Wi-Fi off as well. Native code making raw system calls could bypass an audit hook.
- **Rehearse:** before the exam, run it with Wi-Fi off in the browser you will use, and
  try a few questions from your notes.

## Requirements

- A Mac with Apple silicon (M1 or later). MLX does not run on Intel Macs.
- Python 3.12 or later.
- ffmpeg with the `libmp3lame` encoder: `brew install ffmpeg`.
- For study notes: [Claude Code](https://claude.com/claude-code) (the `claude` command),
  signed in to your Claude account. Alternatively, Amazon Bedrock: set
  `[enrich] backend = "bedrock"`.
- About 5 GB of disk space for the two models (about 2.5 GB each). Each one is downloaded
  into `~/.cache/huggingface` the first time its mode is used.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -e ./exam      # the builder (and the exam index format)

python3 -m venv exam/.venv                       # the exam app, separate and AI-free
exam/.venv/bin/pip install -r exam/requirements.lock -e "./exam[dev]"
```

This installs the `studybuddy` command into `.venv/bin` and `studybuddy-exam` into
`exam/.venv/bin`. Either call them by path or activate an environment first
(`source .venv/bin/activate`). `python -m studybuddy` works as well.

## Usage

```
studybuddy transcribe INPUT [--mode accurate|fast] [--language L]
                            [--formats txt,srt,json] [--hotwords-file F]
                            [--out-dir D] [--no-notes] [-v | -q]
```

| Option | Meaning |
|---|---|
| `INPUT` | A video or audio file: MP4, MOV, MKV, M4A, MP3, and anything else ffmpeg reads. |
| `--mode` | `accurate` (the default) or `fast`. See [Modes](#modes). |
| `--language L` | The spoken language, as a name or code: `English`, `en`, `fr`, `pt-BR`, ... By default it is detected. Used in accurate mode only. |
| `--formats` | A comma-separated list of transcript formats from `txt`, `srt` and `json`. The default is `txt`. |
| `--hotwords-file F` | Names and technical terms the model should favour. Used in accurate mode only. See [Hotwords](#hotwords). |
| `--out-dir D` | Where to write the outputs. The default is the folder the input is in. |
| `--no-notes` | Don't write study notes. By default Claude writes them from the transcript (which sends it to Claude); with this option nothing leaves the Mac. See [Study notes](#study-notes). |
| `-v`, `--verbose` | Also show debug messages, such as the exact ffmpeg commands. |
| `-q`, `--quiet` | Show only warnings and errors. The output paths are still printed. |

Examples:

```sh
# A plain-text transcript next to the video
studybuddy transcribe lecture.mp4

# Fast mode, with subtitles and JSON, into another folder
studybuddy transcribe lecture.mp4 --mode fast --formats txt,srt,json --out-dir transcripts/

# A French lecture, with the course's vocabulary as hotwords
studybuddy transcribe cours.mp4 --language fr --hotwords-file vocabulaire.txt

# Every MP4 in a folder
for f in lectures/*.mp4; do studybuddy transcribe "$f" --formats txt,srt; done
```

Progress and messages go to stderr. In a terminal they appear as one line that keeps
updating; when stderr goes to a file or a pipe, a plain line is printed every so often
instead. Once everything has been written, the paths of the output files go to stdout, one
per line.

## Modes

| | `accurate` (default) | `fast` |
|---|---|---|
| Model | [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (`mlx-community/Qwen3-ASR-1.7B-8bit`) | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (`mlx-community/parakeet-tdt-0.6b-v3`) |
| Speed on an M4 (32 GB) | about 22 to 26 times real time: a 2.5-hour file takes about 6 minutes | about 50 to 70 times real time: a 2.5-hour file takes about 2.5 minutes |
| Segments | one per chunk of about 60 seconds | one per sentence |
| Languages | 30, detected automatically or set with `--language` | 25 European languages, always detected automatically |
| Hotwords | supported | not supported (ignored with a warning) |
| License | Apache 2.0 | CC-BY-4.0 (credit NVIDIA if you publish its output) |

Accurate mode has the lowest error rate of the downloadable models compared in the plan.
Because it is LLM-based, its output also goes through some safety checks; see
[Safety checks](#safety-checks). Fast mode is not an LLM, so it cannot get stuck repeating
itself, and its sentence-level timestamps make better subtitles.

`--language` accepts these languages in accurate mode, by English name or ISO code:
Arabic, Cantonese, Chinese, Czech, Danish, Dutch, English, Filipino, Finnish, French,
German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Macedonian, Malay,
Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish and
Vietnamese. Fast mode ignores `--language`, with a warning.

## Output files

For `lecture.mp4` the outputs are written to the same folder, or to `--out-dir`:

| File | Contents |
|---|---|
| `lecture.mp3` | The first audio track, mono, 96 kbps. It keeps the video's timeline: if the audio track starts late or has gaps, they are filled with silence, so transcript times match the video. |
| `lecture.txt` | The transcript, one paragraph per segment. |
| `lecture.srt` | Subtitles, one cue per segment (requested with `--formats ...,srt`). |
| `lecture.json` | `{source, model, mode, language, duration_s, segments: [{start, end, text, language}]}`, with times in seconds (requested with `--formats ...,json`, and always written with the study notes). |
| `lecture.notes.md` | Study notes written by Claude (unless `--no-notes`). See [Study notes](#study-notes). |

Each file is written to a temporary name first and renamed once it is complete, so a file
with the final name is never half-written.

## Resuming

While a run is in progress, finished sections are saved in `lecture.partial.jsonl`. If the
run is stopped (with Ctrl-C, a crash or a reboot), run **the same command** again. It
extracts `lecture.mp3` again (about 11 s per hour of video), skips every section that
is already finished, and continues from there. Ctrl-C says where the progress was saved
and exits with status 130.

Saved sections are reused only if the audio and every setting that affects the transcript
(mode, model, language, hotwords and the model settings) are unchanged. An unchanged video
always gives an identical MP3, while a replaced video, or a different recording that happens
to have the same name and output folder, does not. In that case studybuddy prints a warning
and starts over. `lecture.partial.jsonl` is deleted once all the outputs are written.

## Hotwords

A hotwords file lists names and terms that the model might otherwise misspell, one per
line. Blank lines and lines that start with `#` are ignored.

```
# Linear algebra, week 3
eigenvalue
Gram-Schmidt
Hilbert space
```

Accurate mode adds the terms to the model's prompt. Fast mode does not support hotwords.

## Safety checks

- **Repetition.** An accurate-mode chunk that has more than about 300 words per minute of
  audio, or that repeats the same short phrase many times, is transcribed again with a
  repetition penalty.
- **Missing text.** If a stretch of audio has clear sound but no text, studybuddy logs a
  warning. In fast mode it first transcribes that stretch again on its own.
- **Invented text over silence.** Accurate mode drops any text produced over digitally
  silent audio.
- **Silent truncation.** Accurate mode gives each model call a token budget of 25 tokens per
  second of audio, at most 8192. That is several times what even fast speech needs, and it
  limits how long a chunk that loops can keep generating (to 1500 tokens when a 60-second
  chunk is transcribed again on its own). If a call runs out of its budget, the affected
  chunks are transcribed again one at a time.

These checks report what they did as warnings on stderr.

## Exit status

| Status | Meaning |
|---|---|
| 0 | Success. |
| 1 | An error: a missing or unreadable input, a file with no audio track, a bad option, an unsupported language, and so on. The message says what went wrong. |
| 130 | Interrupted with Ctrl-C. Run the same command again to resume. studybuddy ends by the interrupt signal, as programs should, so a shell script or `for` loop that runs it stops too. |

## Running the tests

```sh
.venv/bin/pytest                                      # unit tests, about 10 s
STUDYBUDDY_MODEL_TESTS=1 .venv/bin/pytest -m model -s  # only the real-model tests, about 35 s
STUDYBUDDY_MODEL_TESTS=1 .venv/bin/pytest              # everything except Office
STUDYBUDDY_OFFICE_TESTS=1 .venv/bin/pytest -m office    # converts files with PowerPoint and Word
exam/.venv/bin/pytest exam/tests                       # the exam app, in its own environment
```

The unit tests replace the speech models with fakes, but run everything else for real. The
tests marked `model` load both real models (which must be downloaded already) and check
the word error rate on generated speech. The `-s` flag shows the measured word error rates
and timings. The test media is generated with the macOS `say` command and ffmpeg, so the
tests need both.

## Project layout

```
src/studybuddy/
  cli.py          # the command line: transcribe, plus the library commands from commands.py
  commands.py     # library init, course add|list, add, notes, status, build-exam, session, serve
  library.py      # the library folder, library.toml settings, the build database
  transcription.py # one file -> MP3 + transcript (shared by transcribe and add)
  audio.py        # ffprobe checks, MP3 extraction, exam copies, decoding to 16 kHz
  transcribe.py   # the Transcriber interface; Qwen3Transcriber and ParakeetTranscriber
  pipeline.py     # sections, saving and resuming, progress, offsets
  output.py       # .txt / .srt / .json writers
  types.py        # Segment, Transcript
  ingest/         # add: video, pptx, pdf, docx, OCR, Office-to-PDF conversion
  enrich/         # AI enrichment backends, schemas and prompts (exam-search plan §6);
                  # notes.py writes, checks and renders study notes
  study_notes.py  # notes: which videos need notes, the run record, the Markdown files
  exam_export.py  # build-exam: library (and its study notes) -> exam/exam.sqlite
exam/             # studybuddy-exam: the separate, AI-free exam app
  src/studybuddy_exam/
    __main__.py   # installs the guard first, then starts the server on 127.0.0.1
    guard.py      # the audit hook: no AI imports, loopback-only network
    index_format.py # exam.sqlite layout, building and opening (FTS5 + trigram + vocabulary)
    locate.py     # finding the library, or every library in a folder
    collection.py # several libraries' indexes combined into one search
    query.py      # safe query building, stopwords
    search.py     # ranking tiers, filters, snippets, "did you mean"
    notes.py      # the study notes pages: a course's lectures and glossary, one lecture
    server.py     # Starlette app: page, JSON API, files with Range, CSP
    static/       # the search page and the bundled PDF.js viewer
  tests/
tests/            # pytest suite; media.py generates the test audio and video
plans/            # the implementation plan
```
