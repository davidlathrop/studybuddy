# Benchmark results (milestone M0)

_Measured 2026-10-13 against the finished pipeline, so every number is end to end: the same
`audio.extract_mp3` → model load → `pipeline.transcribe_file` → `output.write_outputs` path that
`studybuddy transcribe` runs. Scripts and raw JSON: `bench/` (media and results are gitignored)._

## Summary

- **Fast mode (Parakeet TDT 0.6B v3)** transcribes at **66–71× real time**. A 2.5-hour file
  takes **2 min 43 s** end to end, extraction included. WER on the synthetic lectures: 0.3–1.0 %.
  Peak memory 4.1–4.7 GB.
- **Accurate mode (Qwen3-ASR 1.7B, 8-bit)** transcribes at **22–24× real time**. A 2.5-hour file
  takes **7 min 19 s** end to end. WER: 0.14–0.21 %. Peak memory 9.4–10.1 GB, of which about
  5 GB is MLX's buffer cache (see [Memory](#memory)).
- **Neither mode slows down or loses accuracy over 2.5 hours.** Per-section times and per-10-minute
  WER stay flat from the first section to the last.
- Both modes fit easily in 32 GB. Accurate mode is faster than the plan's guess of 10–40 min for
  2.5 h; fast mode is slower than its guess of about 1 min (that guess came from Neural Engine
  numbers; mlx-audio runs on the GPU).
- **Defaults stay as they are.** Parakeet `chunk_duration=30` / `overlap_duration=2` is the fastest,
  leanest setting and the only one tested that never skipped a sentence. Qwen3 `chunk_duration=60`
  / `batch_size=8` is within a few percent of the best setting tested; `batch_size=16` is 2–4 %
  faster with identical output, which is not enough to change it (details in [Decisions](#decisions)).
- **With `--language` forced, Qwen3 can invent text over non-speech.** The pipeline drops it over
  digital silence, but over a -50 dBFS noise floor 3 invented words got through. With the
  language auto-detected (the default), nothing was invented.
- **Resume works on the real CLI.** Killed twice with SIGKILL, re-runs skipped the saved sections
  and the final output was identical to an uninterrupted run.
- **Caveat:** the speech is clean macOS `say` text-to-speech. Real lectures (room echo, a distant
  microphone, hesitations, crosstalk, dense jargon) will have clearly higher WER. Use these
  numbers for **speed, memory and relative comparisons**; the M5 bake-off on real recordings
  decides accuracy and the default mode.

## Machine

| Item | Value |
|---|---|
| Computer | MacBook Pro 14" (Mac16,1), on AC power, no thermal or performance warnings recorded |
| Chip | Apple M4: 4 performance + 6 efficiency CPU cores, 10-core GPU |
| Memory | 32 GB |
| OS | macOS 15.7.4 |
| Python / packages | 3.12.9; mlx 0.32.3, mlx-audio 0.5.8, numpy 2.5.4, jiwer 4.0.0 |
| ffmpeg | 8.0.1 |
| Models | `mlx-community/parakeet-tdt-0.6b-v3` (2.55 GB in memory), `mlx-community/Qwen3-ASR-1.7B-8bit` (2.47 GB in memory), both already in the Hugging Face cache |

Runs were sequential, one process per run, with nothing else using the GPU. Two code reviewers
were reading files at the same time (light CPU use only). One run that overlapped a media build
(`lecture_30m-accurate-bs4`) was repeated on an idle machine; the table shows the repeat.

## Test media

`bench/make_synthetic.py` builds lecture-like MP4s from varied English text
(`bench/lecture_text.py`): 24 topics (biology, history, physics, economics, philosophy, ...), each
with its own terms, scenarios and facts, woven into lecture phrasing ("Notice how closely
evaporation and condensation are linked."), so a long file is thousands of different sentences
rather than one paragraph on repeat. Students ask occasional questions in other voices. Pauses
are 0.6–4 s between paragraphs, plus a 10–20 s silence about every 10 minutes. The audio is AAC
128 kbps stereo at 48 kHz (typical of a lecture recording) with a tiny still video. The exact
reference text and per-paragraph timings are saved next to each file.

| File | Length | Words | Paragraphs | Speech | Long silences | Voices |
|---|---|---|---|---|---|---|
| `lecture_1m` | 1 min | 166 | 2 | 85 % | 0 | Rishi |
| `lecture_10m` | 10 min | 1,432 | 22 | 88 % | 2 (13 s, 18 s) | Tessa; a student: Eddy |
| `lecture_30m` | 30 min | 5,233 | 77 | 91 % | 3 (14–20 s) | Daniel, Samantha; a student: Eddy |
| `lecture_150m` | 2.5 h | 26,281 (1,556 distinct sentences) | 412 | 91 % | 15 (12–20 s) | Daniel, Karen, Moira, Rishi, Samantha, Tessa; students: Eddy, Flo, Reed, Sandy |
| `lecture_10m_noise50` | 10 min | same text as `lecture_10m` | | | | plus white noise at -50 dBFS (about -53 dBFS in the gaps, speech about -21 dBFS) |

The lecturer voices cover US, UK, Irish, Australian, South African and Indian English. `say`'s
speaking rate was set between 157 and 198 words per minute.

## How it was measured

`bench/run_bench.py run` performs one run in a fresh process:

- **extract**: `audio.extract_mp3` (MP4 → mono 96 kbps MP3).
- **load**: the transcriber's `load()` (weights already in the OS file cache).
- **transcribe**: `pipeline.transcribe_file` with a partial file, exactly as the CLI calls it:
  decoding the MP3 to 16 kHz, splitting into 10-minute sections, running the model, saving each
  section. Decoding is about 1 % of it (3.2 s for 2.5 h).
- **× real time** = audio length ÷ transcribe time. **total** adds extraction, model load and
  writing `.txt` / `.srt` / `.json`.
- **WER**: `jiwer` against the reference text after lowercasing, removing punctuation (hyphens
  and slashes become spaces), collapsing whitespace and mapping a few UK spellings to US.
  S/D/I = substituted / deleted / inserted words.
- **Memory**: *MLX model* = `mx.get_active_memory()` after loading; *MLX peak* =
  `mx.get_peak_memory()` during transcription; *footprint* = the process's lifetime peak physical
  footprint (`proc_pid_rusage`; the "Memory" column in Activity Monitor, which includes GPU
  buffers); *max RSS* = `ru_maxrss`, which misses most GPU memory and is shown only for reference.

`bench/run_bench.py cli` times the installed `studybuddy transcribe` command under
`/usr/bin/time -l` instead: a new process each time, so it also includes Python start-up and imports.

## Results

### Default settings

| Run | Extract s | Load s | Transcribe s | × real time | Total s | WER | S/D/I | MLX model GB | MLX peak GB | Footprint GB | Max RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fast, 10 min | 1.8 | 1.1 | 8.5 | **70.6×** | 11.4 (0:11) | 0.98 % | 10/4/0 of 1,432 | 2.55 | 3.45 | 4.07 | 2.70 |
| fast, 30 min | 5.3 | 0.9 | 27.1 | **66.4×** | 33.3 (0:33) | 0.29 % | 13/1/1 of 5,233 | 2.55 | 3.45 | 4.25 | 2.80 |
| fast, 2.5 h | 27.1 | 0.9 | 135.0 | **66.7×** | 163.0 (2:43) | 0.34 % | 78/5/5 of 26,281 | 2.55 | 3.45 | 4.74 | 3.26 |
| accurate, 10 min | 1.7 | 1.0 | 25.3 | **23.8×** | 28.0 (0:28) | 0.21 % | 2/1/0 of 1,432 | 2.47 | 4.11 | 9.39 | 2.95 |
| accurate, 30 min | 5.3 | 1.5 | 82.7 | **21.8×** | 89.5 (1:30) | 0.15 % | 7/1/0 of 5,233 | 2.47 | 4.10 | 9.63 | 3.16 |
| accurate, 2.5 h | 27.3 | 1.5 | 412.4 | **21.8×** | 441.3 (7:21) | 0.14 % | 32/5/0 of 26,281 | 2.47 | 4.11 | 10.12 | 3.70 |

Defaults: fast = Parakeet, 30 s chunks with 2 s overlap; accurate = Qwen3, 60 s chunks, batch 8,
language auto-detected; 600 s sections. No run logged a warning. Extraction runs at about 330× real
time, and the MP3 is 43 MB per hour (108 MB for 2.5 h), as the plan predicted.

### The real CLI, end to end

`studybuddy transcribe FILE --mode M --formats txt,srt,json -q`, timed with `/usr/bin/time -l`
(a new process each time; models already downloaded):

| File | Accurate: wall time | × real time | Footprint GB | Fast: wall time | × real time | Footprint GB |
|---|---|---|---|---|---|---|
| 1 min | 8.0 s | 7.5× | 5.14 | 2.2 s | 27.0× | 3.76 |
| 10 min | 28.1 s | 21.4× | 9.39 | 11.3 s | 53.1× | 4.07 |
| 30 min | 1:29 (88.9 s) | 20.3× | 9.63 | 33.7 s | 53.5× | 4.25 |
| 2.5 h | **7:19** (439.1 s) | 20.5× | 10.12 | **2:43** (163.0 s) | 55.2× | 4.73 |

The transcripts, and so the WERs, match the in-process runs above, and the wall times agree with
their totals to within 1 %. Fixed costs (process start-up, imports, model load, the first GPU
call) dominate short files, so a 1-minute file runs at only 7.5× (accurate) and 27× (fast).

### Long files: no drift over 2.5 hours

WER per 10-minute stretch of `lecture_150m` (%):

| Minutes | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fast | 0.21 | 0.36 | 0.06 | 0.87 | 0.11 | 0.43 | 0.47 | 0.77 | 0.16 | 0.00 | 0.49 | 0.53 | 0.19 | 0.27 | 0.12 |
| accurate | 0.00 | 0.05 | 0.11 | 0.12 | 0.17 | 0.11 | 0.26 | 0.33 | 0.00 | 0.17 | 0.31 | 0.00 | 0.00 | 0.40 | 0.12 |

Time per 10-minute section: fast 8.4–9.1 s (first 9.0 s, last 8.9 s); accurate 24.6–29.8 s
(first 29.8 s, last 27.4 s). There is no upward trend, so no thermal slowdown over the run, and
`pmset` recorded no thermal or performance warnings. Peak memory for 2.5 h is only about 0.7 GB
above that for 10 min, roughly the decoded audio (576 MB of 16 kHz float32).

### Fast mode: Parakeet `chunk_duration`

| File | Chunk | Transcribe s | × real time | WER | S/D/I | Skipped sentences re-transcribed | MLX peak GB | Footprint GB |
|---|---|---|---|---|---|---|---|---|
| 10 min | **30 s (default)** | 8.5 | **70.6×** | 0.98 % | 10/4/0 | **0** | 3.45 | 4.07 |
| 10 min | 60 s | 8.8 | 68.2× | 0.91 % | 10/1/2 | 2 | 3.84 | 4.97 |
| 10 min | 120 s | 9.9 | 60.3× | 0.70 % | 9/1/0 | 5 | 4.34 | 5.45 |
| 10 min | 300 s | 12.7 | 47.1× | 0.77 % | 9/1/1 | 2 | 7.84 | 10.06 |
| 30 min | **30 s (default)** | 27.1 | **66.4×** | 0.29 % | 13/1/1 | **0** | 3.45 | 4.25 |
| 30 min | 60 s | 28.1 | 64.1× | 0.40 % | 14/3/4 | 7 | 3.84 | 5.09 |
| 30 min | 120 s | 30.3 | 59.5× | 0.23 % | 11/1/0 | 7 | 4.35 | 5.61 |

With chunks of 60 s or more, Parakeet skips whole sentences (typically the first sentence after a
pause). `ParakeetTranscriber` notices the audible stretch with no text, transcribes it again on its
own and logs a warning, so the final WER barely moves. The re-transcribed piece loses its
context, though (one came back as "Today I want to spend some time on chemical." for "...chemical
reactions."), and the safety net only fires for gaps of 2 s or more. At 30 s nothing was ever
skipped, and it was also the fastest and used the least memory. The WER differences between
chunk sizes are a few words either way and not significant.

### Accurate mode: Qwen3 `batch_size`

| File | Batch | Transcribe s | × real time | WER | MLX peak GB | Footprint GB |
|---|---|---|---|---|---|---|
| 10 min | 4 | 27.1 | 22.2× | 0.21 % | 3.94 | 10.25 |
| 10 min | **8 (default)** | 25.3 | **23.8×** | 0.21 % | 4.11 | 9.39 |
| 10 min | 16 | 24.6 | 24.4× | 0.21 % | 4.68 | 8.57 |
| 30 min | 4 | 90.8 | 19.8× | 0.15 % | 3.93 | 10.36 |
| 30 min | **8 (default)** | 82.7 | **21.8×** | 0.15 % | 4.10 | 9.63 |
| 30 min | 16 | 79.3 | 22.7× | 0.15 % | 4.68 | 8.63 |

The transcripts are word-for-word identical at every batch size. A 600 s section holds 10 chunks
of 60 s: batch 8 decodes them in two passes (8 + 2), batch 16 in one. One pass instead of two
saves only 2–4 %. So on this 10-core GPU, most of the time is not in what batching speeds up
(sharing the weight reads during decoding). mlx-audio runs the audio encoder chunk by chunk
either way. Where the time goes in detail was not profiled.

### Run-to-run variation

Three runs each of the same 10-minute configuration:

| Configuration | Transcribe s (3 runs) | Mean | Standard deviation |
|---|---|---|---|
| accurate, batch 8 | 25.3, 24.7, 24.6 | 24.8 | 0.36 s (1.5 %) |
| accurate, batch 16 | 24.6, 24.2, 24.2 | 24.3 | 0.20 s (0.8 %) |
| fast, 30 s chunks | 8.50, 8.40, 8.43 | 8.44 | 0.05 s (0.6 %) |

The transcripts, WER and peak footprint were identical in every repeat, because decoding is
deterministic. Differences of under about 3 % between single runs are within noise. Batch 16's
advantage over batch 8 is 2.1 % on the 10-minute file (3 runs each) and 4.1 % on the 30-minute
file (1 run each).

### Other checks

| Run (10 min) | Transcribe s | × real time | WER | S/D/I | Footprint GB | Notes |
|---|---|---|---|---|---|---|
| accurate, 30 s chunks | 23.7 | 25.3× | 0.35 % | 3/1/1 | 8.43 | 21 segments instead of 10 |
| accurate, 60 s chunks (default) | 25.3 | 23.8× | 0.21 % | 2/1/0 | 9.39 | |
| accurate, 120 s chunks | 29.2 | 20.6× | 0.14 % | 1/1/0 | 8.95 | 5 segments; longer chunks are slower, not faster |
| accurate, `--language English` | 25.0 | 24.0× | 0.21 % | 2/1/0 | 9.39 | Qwen3 invented a sentence over the final 13 s of digital silence; the silence check dropped it (1 warning) |
| fast, -50 dBFS noise | 8.6 | 69.8× | 1.05 % | 11/4/0 | 4.07 | one more substitution than the clean file |
| accurate, -50 dBFS noise | 26.1 | 23.0× | 0.21 % | 2/1/0 | 8.21 | same errors as the clean file; nothing invented in the noisy gaps |
| accurate, -50 dBFS noise, `--language English` | 25.0 | 24.0× | 0.42 % | 2/1/3 | 8.19 | **Qwen3 invented "I'm not sure." over the last 2.6 s, which held only noise. Not caught: the silence check only fires below -60 dBFS** |

**Forcing the language makes Qwen3 invent text where there is no speech.** Over digital silence
the pipeline's check drops it; over a -50 dBFS noise floor (ordinary room tone) it gets through.
Here that was 3 words at the very end of the file, but a real recording with long, noisy pauses
and `--language` set could get more. With the language auto-detected (the default), nothing was
invented in either case. If M5 shows this on real recordings, the fix is the speech detector
plan §5.6 already lists, or a silence threshold measured relative to the recording's noise floor.

### Memory

Accurate mode's footprint (9.4–10.1 GB) is more than twice its MLX active peak (4.1 GB). The
difference is MLX's buffer cache: freed GPU buffers kept for reuse during `generate()`
(the pipeline already calls `mx.clear_cache()` after each call). Capping the cache confirms it:

| Run (10 min) | MLX cache limit | Transcribe s | × real time | Footprint GB |
|---|---|---|---|---|
| accurate | none (default) | 25.3 | 23.8× | 9.39 |
| accurate | 2 GB | 25.3 | 23.8× | 6.33 |
| accurate | 0 | 25.6 | 23.5× | 4.35 |
| fast | none (default) | 8.5 | 70.6× | 4.07 |
| fast | 0 | 9.8 | 61.2× | 3.55 |

So capping the cache at about 2 GB in accurate mode (`mx.set_cache_limit(2_000_000_000)`) would cut
its memory by a third at no measurable cost. On this 32 GB machine it doesn't matter; on a 16 GB Mac it might. Fast mode
needs its cache: without it, it runs 13 % slower.

### Resume after a crash (CLI)

`bench/resume_test.py` ran `studybuddy transcribe lecture_30m.mp4 --mode accurate --formats
txt,srt,json` and SIGKILLed the whole process group (no clean-up at all) as soon as one more
section had reached `lecture_30m.partial.jsonl`, twice, then let a third run finish:

| Run | What happened | Wall time |
|---|---|---|
| 1 | extracted the MP3, transcribed section 1/3, killed | 34.2 s |
| 2 | extracted the MP3 again (byte-identical), logged "resuming lecture_30m.mp3: 1 of 3 sections already done", transcribed section 2/3 only, killed | 34.2 s |
| 3 | extracted the MP3 again (byte-identical), logged "resuming lecture_30m.mp3: 2 of 3 sections already done", transcribed section 3/3 only, exit 0 | 36.4 s |

A re-run extracts the MP3 again rather than reusing the one on disk, so a replaced video, or a
different recording with the same name, can never resume against old audio. That costs about 6 s
per re-run on this 30-minute file. (The first measurement reused the MP3 and took 28.7 s and
30.1 s for runs 2 and 3.)

Checks, all passed: every re-run extracted a byte-identical MP3 and logged the resume; no saved
section was transcribed twice; the partial file was deleted at the end; `.mp3`, `.txt`, `.srt` and
`.json` were written; segments are in order and inside the file's duration; the transcript is
identical to the uninterrupted run's (WER 0.15 %). Sections are cheap to lose: at most about 30 s of work
in accurate mode and 9 s in fast mode.

## What the remaining errors are

- **Rare or technical words**, mostly in fast mode: chloroplasts → "chloroplists", Calvin cycle →
  "Culvin"/"Kelvin cycle", stomata → "stomach", glucose → "glycolysis" (accurate).
- **Grammar the models fix.** The text templates sometimes produce agreement errors ("Roman
  roads starts to matter", "ocean currents connects to"), and both models write the grammatical
  form. Together with the spelling variant "tune in to" → "tune into", these artifacts of the
  synthetic text are 40–75 % of accurate mode's errors: 2 of 3 on the 10-minute file, 6 of 8 on
  the 30-minute file, and 15 of 37 on the 2.5-hour file (11 agreement fixes, 4 from "tune into").
  Without them, accurate mode's WER on the 2.5-hour file is about 0.08 %.
- **Accent-dependent slips** in accurate mode: "core of the sun" → "call of the sun" 4 times, all
  while the South African voice (Tessa) was lecturing (fast mode got these right); "Kant" → "hemp";
  "weather fronts" → "weather France".
- Very few insertions in either mode. With the language auto-detected (the default), nothing was
  invented anywhere. With `--language English`, Qwen3 invented text over non-speech twice: the
  pipeline dropped it over digital silence but not over a noise floor (see [Other checks](#other-checks)).

## Decisions

| Setting | Default | Decision | Why |
|---|---|---|---|
| Parakeet `chunk_duration` | 30 s | **keep** | fastest (70.6× vs 47–68×), least memory (4.1 GB vs 5–10 GB), and the only size tested that skipped no sentences |
| Parakeet `overlap_duration` | 2 s | keep | not swept |
| Qwen3 `chunk_duration` | 60 s | **keep** | 30 s is 6 % faster and 120 s 13 % slower, with error counts within 2 words of the default (5, 3 and 2 errors); 60 s gives one timestamp per minute and stays under the 3-minute limit of the word-timestamp model |
| Qwen3 `batch_size` | 8 | **keep** | 16 is 2.1–4.1 % faster with identical text (one decode pass per 10-chunk section). That margin doesn't justify a change, and `tests/test_transcribe.py` pins 8. It might pay off more on chips with more GPU cores (not tested) |
| `section_s` | 600 s | keep | not swept; a section costs 9 s (fast) or 25–30 s (accurate), which is the most a crash can lose |

No code defaults changed; the unit suite still passes (261 passed, 12 model tests skipped).

Follow-ups suggested by the data (not done here):

1. Cap MLX's buffer cache at about 2 GB in accurate mode (a third less memory, no speed cost).
2. Run the M5 bake-off on real recordings. On clean speech accurate mode makes 2–5× fewer
   errors (3 vs 14, 8 vs 15, 37 vs 88), but both are under 1 %, so this data cannot settle the
   default mode.
3. Optionally try batch 16 again on real recordings, together with the M5 bake-off.
4. Watch for invented text when `--language` is forced on recordings with a noise floor (see
   [Other checks](#other-checks)); add the speech detector from plan §5.6 if it shows up.

## Reproduce

```sh
.venv/bin/python bench/make_synthetic.py                       # 1, 10, 30 min and 2.5 h (about 3 min)
.venv/bin/python bench/make_synthetic.py --name lecture_10m_noise50 --minutes 10 --seed 101 --noise-dbfs -50
.venv/bin/python bench/run_bench.py suite core sweeps confirm extras cache   # about 25 min
.venv/bin/python bench/run_bench.py cli bench/media/lecture_1m.mp4 bench/media/lecture_10m.mp4 \
    bench/media/lecture_30m.mp4 bench/media/lecture_150m.mp4
.venv/bin/python bench/resume_test.py
.venv/bin/python bench/run_bench.py table                      # every run as one markdown table
```

Rendered speech is cached in `bench/media/.cache`, so rebuilding the media is quick after the
first time. Each run's JSON in `bench/out/` also holds per-section times, per-10-minute WER, every
logged warning and the machine details.
