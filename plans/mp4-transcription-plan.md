# MP4 → MP3 → Transcript: Implementation Plan

_Status: M0–M4 implemented (see the project README); M5, the bake-off on real recordings, is still to do. Research done 2026-10-13._

## 1. Goal

- **Input:** MP4 files, anywhere from 1 minute to 2.5 hours long.
- **Output:** an MP3 of the audio track, plus a transcript (`.txt`) with optional timestamped `.srt` / `.json`.
- **Where it runs:** locally on the development Mac.

## 2. Target environment (checked on this machine)

| Item | Value |
|---|---|
| Machine | Apple M4, 32 GB RAM |
| OS | macOS 15.7.4 |
| ffmpeg | 8.0.1, includes `libmp3lame` (MP3 encoder) |
| Python | 3.12.9 (pyenv) |

## 3. Model decision

### 3.1 Summary

| Mode | Model | Runtime on Mac | Why |
|---|---|---|---|
| **Accurate** (default until the bake-off in §7) | [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | [mlx-audio](https://github.com/Blaizzy/mlx-audio) with `mlx-community/Qwen3-ASR-1.7B-8bit` | Lowest error rate of any downloadable model. 52 languages, automatic language detection, Apache 2.0 |
| **Fast** | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | mlx-audio with `mlx-community/parakeet-tdt-0.6b-v3` | Much faster. Word timestamps built in. Not an LLM, so it can't get stuck repeating itself |
| Backup | [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | transformers / vLLM | Best measured downloadable model on the leaderboard's long-audio test |

Both main models run through the same Python library (mlx-audio), so switching modes means changing a model name and its settings.

### 3.2 Evidence: Open ASR Leaderboard (data as of 2026-10-09)

The leaderboard's own "avg" column mixes different sets of tests. Some models skip the hardest, noisiest test set, which makes their average look better. The table below re-averages every model over the **8 test sets all of them completed**. WER is word error rate; lower is better.

| Model | WER (8 shared sets) | Noisy set (URGENT2024) | Long-audio test avg | Languages | License |
|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | **4.31** | 9.93 | not tested | 52 | Apache 2.0 |
| Granite Speech 4.1 2B | 4.62 | 11.25 | not tested | 6 | Apache 2.0 |
| Cohere Transcribe | 4.67 | **9.50** | **9.73** | 14 | Apache 2.0 (must accept terms on HF) |
| **Parakeet TDT 0.6B v3** | 4.71–4.84 | 11.5 | 10.72 | 25 European | CC-BY-4.0 |
| Granite Speech 5.0 TurboCTC | 5.03 | 11.79 | not tested | English | Apache 2.0 |
| Phonon-2 | 5.17 | 14.9 | not tested | English | CC-BY-4.0 |
| Whisper large-v3 | 5.78 | 13.73 | 11.23 | 99 | Apache 2.0 |

Qwen3-ASR-1.7B is also the only LLM-based model in this group. It pairs an audio encoder with a Qwen3 language model.

### 3.3 Considered and rejected

- **Whisper large-v3.** Clearly worse on meeting audio (13.63 WER vs 8.31 for Qwen3) and on noisy audio. Known to invent text during silence.
- **Phi-4-multimodal, Granite Speech 3.3.** They rank high on the leaderboard's official average only because they skip the noisy test set. On the 8 shared sets they score 5.02 and 5.26.
- **Phonon-2.** A compressed copy of Parakeet v3. It's less accurate on most test sets, much worse on noisy audio, and English only. Its extra speed doesn't matter at these file lengths.
- **Granite Speech 5.0 TurboCTC.** A workable English-only fast option, and the best of this group on meeting audio (7.71). Its model card doesn't say whether the output has punctuation and capitalization; check before using it. Avoid the `-nc` variant, which is licensed for non-commercial use only.
- **EmbeddingGemma 2.** An embedding model: it turns audio into a list of numbers, not text, so it can't transcribe. It's a good fit for a future search feature (§11).

## 4. Architecture

```
MP4 ─ffprobe─▶ validate ─ffmpeg─▶ MP3 ─decode to 16 kHz mono─▶ sections (~10 min, cut at quiet points)
                                                                   │
                                            chunks (~60 s), several at once ─▶ speech model
                                                                   │
                                  <name>.partial.jsonl (saved after each section) ─▶ .txt / .srt / .json
```

### 4.1 Project layout

```
studybuddy/
  pyproject.toml
  plans/
  src/studybuddy/
    cli.py          # `studybuddy transcribe …` command
    audio.py        # ffprobe checks + ffmpeg MP3 extraction with progress
    transcribe.py   # Transcriber interface; Qwen3Transcriber, ParakeetTranscriber
    pipeline.py     # sections, saving and resuming, progress, offset bookkeeping
    output.py       # .txt / .srt / .json writers
  tests/
```

### 4.2 Command line

```
studybuddy transcribe lecture.mp4 \
    [--mode accurate|fast] [--language English] \
    [--formats txt,srt,json] [--hotwords-file terms.txt] [--out-dir DIR]
```

Outputs go next to the input unless `--out-dir` is given: `lecture.mp3`, `lecture.txt`, and optionally `lecture.srt` and `lecture.json`.

## 5. Pipeline details

### 5.1 Check the input

```
ffprobe -v error -select_streams a \
  -show_entries stream=index,codec_name,channels:format=duration -of json in.mp4
```

- If the file has no audio track, stop with a clear error.
- Keep the duration for the progress display and time estimates.

### 5.2 Extract the MP3

```
ffmpeg -nostdin -y -i in.mp4 -map 0:a:0 -vn -af aresample=async=1:first_pts=0 -ac 1 \
  -c:a libmp3lame -b:a 96k -progress pipe:1 -nostats out.mp3.tmp
```

- **Size:** mono at 96 kbps is about 43 MB per hour, so a 2.5-hour file is about 108 MB. That's plenty for speech; the model works at 16 kHz anyway.
- **Progress:** read `out_time_us` from the `-progress` output and compare it with the duration to show a percentage.
- **Safe writes:** write to a temporary file and rename it when ffmpeg succeeds, so a half-written MP3 never looks finished.
- **Multiple audio tracks:** `-map 0:a:0` takes the first audio track.
- **Timeline:** MP3 has no timestamps, so `aresample=async=1:first_pts=0` writes the container's timeline into the samples: an audio track that starts after the video gets leading silence, and gaps in its timestamps (dropped packets) are filled with silence. Without it, every transcript time after the offset or gap would be early. Continuous audio is unchanged.
- **Deterministic:** the same input gives a byte-identical MP3, so a resumed run extracts it again and still matches the resume key (§5.4).

### 5.3 Load the audio

- Decode the MP3 to 16 kHz mono with `mlx_audio.stt.utils.load_audio`.
- A 2.5-hour file uses about 576 MB of RAM once decoded, which is fine with 32 GB.

### 5.4 Accurate mode (Qwen3-ASR-1.7B)

mlx-audio's default settings don't suit long files. By default it:

- cuts audio into **20-minute chunks**, and returns only **one timestamp per chunk** (one every 20 minutes)
- gives no progress until the whole file is done, and loses everything if it crashes
- transcribes one chunk at a time (`batch_size=1`)

Instead:

1. Split the audio into **~10-minute sections** at quiet points. mlx-audio's own `split_audio_into_chunks` can do this with `chunk_duration=600`.
2. For each section:
   ```python
   model.generate(section, chunk_duration=60, batch_size=8,
                  language=lang, hotwords=terms)
   ```
   - **60-second chunks** match the clip lengths where the leaderboard measured Qwen3 as best. They also give roughly one timestamp per minute and fit within the 3-minute limit of the word-timestamp model (§9).
   - **`batch_size=8`** transcribes several chunks of a section at the same time.
3. Add the section's start time to each returned segment's `start` / `end`.
4. Append the section's results to `<name>.partial.jsonl`. On a re-run, skip sections that are already done. Use the file's hash plus the settings as the key.

A 1-minute file is a single section with a single chunk, so it uses the same code.

### 5.5 Fast mode (Parakeet TDT 0.6B v3)

- **You must pass `chunk_duration`.** mlx-audio's Parakeet `generate()` does **no chunking by default**, so it would try to process all 2.5 hours in one pass.
  ```python
  model.generate(audio, chunk_duration=30, overlap_duration=2,
                 chunk_callback=on_progress)
  ```
  M0 (§6) chose 30 s. With 45 s, 60 s or 120 s chunks Parakeet skipped whole sentences after pauses; 30 s skipped none and was also the fastest and used the least memory ([benchmark-results.md](benchmark-results.md)).
- It returns word-level tokens and sentences with timestamps.
- **If GPU speed disappoints:** [FluidAudio](https://github.com/FluidInference/FluidAudio) runs Parakeet v3 on the Neural Engine, at about 156× real time on an M4 Pro. It's a Swift library with a macOS command-line tool (`fluidaudiocli transcribe`) and has no Python package, so the app would call it as a separate process. FluidAudio also offers "Parakeet Ultra", which scored better than v3 in their own 60-minute test (13.5% vs 16.5% WER). It isn't on the HF leaderboard and its origin is unclear, so include it in the bake-off if you go this route.

### 5.6 Safety checks

- **Repetition (accurate mode).** Flag a chunk if it has more than about 300 words per minute of audio, or the same short phrase repeated many times. Re-run that chunk with `repetition_penalty≈1.1`.
- **Missing text.** If a chunk comes back empty but its audio clearly has sound, log a warning.
- **Invented text over silence or music.** If this happens with real files, add a speech detector (Silero VAD) to drop the silent stretches before transcription. Add it only if it actually happens.
- **Language.** Pass `--language` when it's known. Otherwise both models detect it automatically (Parakeet only among its 25 languages).

### 5.7 Output

- **`.txt`:** one paragraph per segment.
- **`.srt`:** one subtitle line per segment. In accurate mode that's about one per minute, which is good for jumping around but long for on-screen subtitles. Fast mode gives one per sentence.
- **`.json`:** `{source, model, mode, language, duration_s, segments: [{start, end, text}]}`.
- Delete `<name>.partial.jsonl` once the final files are written.

## 6. Milestones

| # | Milestone | Done when |
|---|---|---|
| **M0** | **Speed test, before any app code.** Install mlx-audio in a scratch virtual environment. Make test audio with `say -f text.txt -o test.aiff`, looped to 10 minutes and to 2.5 hours. Time both models. | Real-time speed and peak memory recorded for both models; `chunk_duration` / `batch_size` chosen |
| M1 | Project skeleton, `audio.py` (ffprobe + MP3 extraction with progress), and tests that use a short MP4 generated with ffmpeg | `studybuddy transcribe x.mp4` writes `x.mp3`; a file with no audio track fails with a clear error |
| M2 | Fast mode end to end | A 2.5-hour file produces `.txt` with progress shown |
| M3 | Accurate mode with sections, batching and resume | Killing a run partway and restarting it skips finished sections |
| M4 | `.srt` / `.json` output and the safety checks | Timestamps line up when spot-checked against the video |
| M5 | Bake-off on real recordings (§7) | Default mode chosen and written down |

### Time estimates (measured in M0)

Wall-clock time of `studybuddy transcribe` on this Apple M4 / 32 GB, end to end (MP3 extraction, model load, transcription, writing `.txt` / `.srt` / `.json`), with the models already downloaded. Details in [benchmark-results.md](benchmark-results.md).

| File length | Accurate mode | Fast mode |
|---|---|---|
| 1 min | 8 s | 2 s |
| 10 min | 28 s | 11 s |
| 30 min | 1 min 29 s | 34 s |
| 2.5 h | **7 min 19 s** | **2 min 43 s** |

- Rule of thumb: accurate mode takes about 3 s per minute of audio (22× real time while transcribing), fast mode about 1.1 s per minute (67×). Extracting the MP3 accounts for about 11 s per hour of video in both modes.
- Peak memory: accurate 9–10 GB (about 5 GB of that is MLX's reusable buffer cache), fast 4–5 GB.
- Neither mode slows down or loses accuracy over a 2.5-hour file.
- The first run of each mode also downloads its model once.
- Measured on clean synthetic speech, where WER was 0.14–0.21 % (accurate) and 0.3–1.0 % (fast). Real recordings should take about as long but will have clearly higher error rates; the M5 bake-off measures those.

## 7. Evaluation (bake-off)

1. Pick 3 typical MP4s plus **the longest one** you have.
2. Hand-correct a 5-minute excerpt from each. For the long file, take three excerpts (start, middle, end) to catch any drop in quality later in the recording.
3. Score each model with `jiwer`. Before scoring, normalize the text (lowercase, strip punctuation), or use Whisper's `EnglishTextNormalizer` to match how the leaderboard scores.
4. Compare Qwen3-ASR-1.7B, Parakeet v3, and optionally Cohere Transcribe (and Parakeet Ultra if using FluidAudio). Record the time each file takes.
5. **Suggested rule:** if Parakeet makes no more than about 10% more errors than Qwen3 on your audio, make fast mode the default and keep accurate mode as an option.

## 8. Dependencies

- `mlx-audio` (0.5.8 when researched). Provides both models and audio loading.
- `ffmpeg` / `ffprobe` (system, already installed).
- Development only: `pytest`, `jiwer`.
- Don't install the official `qwen-asr` package. It isn't needed on the Mac (see §9).

## 9. Pitfalls found during research

- **Truncated transcripts with `qwen-asr`.** The official `qwen-asr` package stops after `max_new_tokens=512` by default. A 20-minute chunk needs several thousand tokens, so long transcripts get cut short without any error. mlx-audio's default (8192) is fine; raise the limit if you ever use `qwen-asr`.
- **Version conflict.** `qwen-asr` requires `transformers==4.57.6`, but the `Qwen3-ASR-1.7B-hf` checkpoints need `transformers>=5.13`. Never install both in one environment.
- **Parakeet doesn't chunk by default** in mlx-audio (§5.5).
- **Word timestamps for Qwen3** need the separate `Qwen3-ForcedAligner-0.6B` model. It only handles pieces up to 3 minutes and supports 11 languages.
- **Qwen3-ASR has never been scored on the leaderboard's long-audio test.** The bake-off on your longest file is how to check it.
- **No official Mac support for the PyTorch version of Qwen3-ASR.** Its examples only cover NVIDIA GPUs. Use MLX on the Mac.
- **Cohere Transcribe** requires accepting terms on Hugging Face, must be told the language, and has no timestamps.
- **The first run downloads the model weights** (a few GB) into `~/.cache/huggingface`.
- **Licensing.** Parakeet's CC-BY-4.0 license requires giving credit. Avoid any `-nc` variant if the app could be used commercially.

## 10. Open questions

1. **What languages are the videos in?** English only makes Granite TurboCTC an option. Anything outside English and the European languages rules out Parakeet.
2. **Local Mac only, or a server or web app later?** A server would use `qwen-asr` with vLLM on NVIDIA GPUs instead of MLX.
3. **Are timestamps needed, and how detailed?** Per segment is the default. Per word means Parakeet, or Qwen3 with the separate timestamp model.
4. **Could the app be used commercially?** That decides which licenses are acceptable.

## 11. Future: searching across lectures

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) can't transcribe, but it can make transcripts searchable:

- After transcription, split each transcript into passages and store an embedding for each passage with its timestamp.
- A student can then ask "where did they explain eigenvalues?" and jump to the matching moment in the video.
- It runs on the Mac and is Apache 2.0.

## 12. Sources

- Open ASR Leaderboard: [space](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) · [short-form results](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results) · [long-form results](https://huggingface.co/datasets/hf-audio/leaderboard_longform)
- Qwen3-ASR: [model card (-hf)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) · [GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Cohere Transcribe model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio). Source checked: `stt/models/qwen3_asr/qwen3_asr.py` and `stt/models/parakeet/parakeet.py`
- [FluidAudio](https://github.com/FluidInference/FluidAudio) · [FluidAudio benchmarks](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md)
- [Granite Speech 5.0 TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) · [Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) · [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
- Apple chip speed references: [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr/) · [OminiX-MLX Qwen3-ASR](https://www.mintlify.com/OminiX-ai/OminiX-MLX/asr/qwen3-asr) · [soniqo.audio benchmarks](https://soniqo.audio/benchmarks)
- Background articles: [MarkTechPost: Best Open ASR Models 2026](https://www.marktechpost.com/2026/07/23/best-open-speech-recognition-asr-models-in-2026-wer-languages-latency-and-license-compared/) · [Gladia](https://www.gladia.io/blog/best-open-source-speech-to-text-models) · [Northflank](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
