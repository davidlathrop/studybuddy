# Exam Search: Implementation Plan

_Status: E0 (enrichment pilot), E1 (library and ingestion) and E2 (AI-free exam app) done 2026-10-14; see [e0-results.md](e0-results.md). Study notes (§6.8), built on the E0 backends, added 2026-10-22; shown in the exam app 2026-10-26. Every library in a folder searched as one (`studybuddy serve`) 2026-10-27. E3 onwards not started._
_Builds on the transcription pipeline in [mp4-transcription-plan.md](mp4-transcription-plan.md), which is implemented on branch `feature/mp4-transcription`._

## 1. Goal and hard constraints

**Goal.** Over the semester, index every lecture video and every course document (PowerPoint, PDF, Word). During an open-book, open-notes final exam, type a question and get back **which video and timestamp** or **which file and slide/page** to look at. One click opens the original at that spot.

**Hard constraints**

| # | Constraint | What it means for the design |
|---|---|---|
| C1 | **No AI during the exam** | The exam app runs only classic search code (word-matching search with relevance ranking, string matching, graph lookups). It cannot load or call any AI library, and it can prove that (§8). |
| C2 | AI is allowed **before** the exam | Transcription (already built), enrichment, and knowledge-graph building all happen during the semester. Enrichment uses Claude Opus 5.5 through `claude -p` (Bedrock optional). |
| C3 | **Offline during the exam** | Videos, documents, the index and every web asset live on the laptop. The app works with Wi-Fi off. |
| C4 | **Every result leads back to the original** | Each passage, concept, relation and AI-written text records which video timestamp or slide/page it came from. |
| C5 | **AI-written text is labelled and checked** | Every AI claim must quote the source word for word, and the quote is checked by string matching. Claims whose quote isn't found are hidden. AI text is always shown labelled, next to the original. |

Before relying on this in an exam, confirm with the instructor that a laptop search tool over your own materials is allowed.

## 2. Architecture

The work splits into two halves that share only data files.

```
                 ┌──────────────── BUILDER (semester; may use AI) ────────────────┐
 MP4 ──transcribe (Qwen3-ASR, existing)──▶ transcript.json ─┐
 PPTX/PDF/DOCX ──extract + OCR + convert to PDF────────────┴─▶ passages ──▶ library.sqlite
                                                                  │            (everything, incl.
                                  Claude Opus 5.5 (claude -p) ──enrich┘             every AI run + its
                                         Titan embeddings ──merge duplicate concepts   provenance)
                                                                  │
                                                       build-exam (no AI)
                                                                  ▼
                 └──────────────────────────── exam.sqlite (read-only) ─────────────┘
                                                                  │
                 ┌──────────────── EXAM APP (exam day; no AI, offline) ───────────┐
   browser ◀──▶ 127.0.0.1 web server (Starlette) ── text search + graph ── exam.sqlite
                 │ video player (jumps to timestamps)   PDF.js viewer (jumps to slide/page)
                 └───────────────────────────────────────────────────────────────────┘
```

- **Builder:** the existing `studybuddy` package, extended. It has the AI dependencies (mlx-audio for transcription; the `claude` CLI, plus anthropic and boto3 for the optional Bedrock backend).
- **Exam app:** a **separate package and separate virtualenv** (`studybuddy-exam`) whose only dependencies are starlette, uvicorn, rapidfuzz and jellyfish. Its code cannot import the builder package.
- **The hand-off file:** `exam.sqlite`, a read-only file holding only checked content. Its SHA-256 is shown in the exam app.

## 3. Library layout

```
~/StudyBuddy/fall-2026/                 # one "library" per semester
  library.toml                          # courses, Bedrock settings, model choice
  courses/bio101/
    videos/   lecture04.mp4  lecture04.json (transcript)  lecture04.exam.mp4 (smaller exam copy)
    docs/     week04.pptx  week04.pdf (converted for viewing)  notes.docx  notes.pdf
    glossary.txt                        # course terms → better transcription and aliases
  build/library.sqlite                  # full build database
  exam/exam.sqlite                      # read-only index for exam day
```

**Video storage.**
- About 170 hours of 1080p lecture video is roughly 85–255 GB, and this Mac has about 67 GB free.
- At build time (no AI), make an **exam copy**: 540p H.264/AAC with `-movflags +faststart`, about 200–250 MB per hour, so about 35–40 GB for a semester.
- **The 1080p original is only needed while it's being added** (transcription and making the exam copy). After that, nothing in the builder or the exam app reads it again.
- Where the originals go afterwards is your choice: keep them, move them to an external SSD, back them up to S3 (optional milestone E7), or delete them.

## 4. Data model (`library.sqlite`)

Adapted from the knowledge-graph research (provenance first: every AI output points to its run and its source passage):

```
course(id, slug, title, created_at)                          -- E1: built (schema v1)
source(id, course_id, kind[video|pptx|pdf|docx], title, week, date,
       original_path, stored_path, view_path,               -- stored: library copy / transcript
       duration_s, page_count, sha256, extractor, warnings, added_at)
passage(id, source_id, ord, t_start_ms, t_end_ms,          -- video
        slide_no, page_no, heading_path,                     -- documents
        origin[speech|text|notes|ocr], text, text_sha)
run(id, task, prompt_version, model_id, params_json, schema_ver, input_sha, created_at,
    input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, cost_usd)  -- built (schema v2)
notes(source_id, run_id, data)                               -- §6.8: built (schema v2)
active_run(task, run_id)                                     -- which run feeds the exam index
enrichment(id, passage_id, run_id, field[question|keyword|context|summary],
           value, quote, q_start, q_end, verified)
chapter(id, source_id, run_id, title, first_passage_id, last_passage_id)
concept(id, course_id, pref_label, norm_key UNIQUE, status[active|merged], merged_into)
alias(concept_id, label, norm_key, kind[acronym|synonym|spelling], origin[rule|llm|user], run_id)
mention(id, concept_id, passage_id, run_id, surface, char_start, char_end,
        role[defines|explains|example|mentions], verified)
definition(id, concept_id, passage_id, run_id, quote, char_start, char_end, gloss_ai, verified)
relation(id, head_id, type, tail_id, run_id, status)        -- type ∈ 7-relation vocabulary (§6.4)
relation_evidence(relation_id, passage_id, quote, char_start, char_end, verified)
passage_link(from_passage, to_passage, type[slide_for|revisits], score, method, run_id)
merge_log(from_id, into_id, method[exact|alias|llm|user], score, run_id, ts)
```

`exam.sqlite` is **exported** from this database (§7.1). It contains only active, verified content plus the full-text indexes and a spelling vocabulary.

## 5. Ingestion (no AI except transcription and OCR)

### 5.1 Videos

- `studybuddy add bio101 lecture04.mp4` runs the existing transcription pipeline:
  - accurate mode;
  - `--hotwords-file courses/bio101/glossary.txt`, so course terms are spelled correctly, which matters for search.
- **Passages:**
  - Accurate mode already gives segments of about 60 s; each becomes a passage.
  - Fast-mode sentences are merged into passages of 45–75 s.
  - Passage times come from the transcript.
- **Jump precision.** A result jumps to the passage start, which can be up to about 60 s before the relevant sentence. Optional upgrade in E5: word timestamps from `Qwen3-ForcedAligner-0.6B` (AI before the exam, which is allowed). Its 3-minute limit fits 60 s passages.
- **Exam copy.** Remux or transcode to the exam copy (§3). This is plain ffmpeg, no AI.

### 5.2 Documents (tested on this Mac)

| Format | Extract with | Location stored | View at exam time |
|---|---|---|---|
| **PPTX** | python-pptx 1.0.2 (MIT): titles, text frames, nested groups, tables, chart labels, **speaker notes**. SmartArt is read from `ppt/diagrams/data*.xml` with lxml. Pictures go to OCR. | `slide_no` | Convert to PDF with **PowerPoint (AppleScript)**. **Unhide hidden slides on a copy first**, otherwise PDF page N ≠ slide N. |
| **PDF** | pypdfium2 5.14 (Apache/BSD; fastest, correct reading order). Fallback: pypdf. NFKC-normalize the text and clean `\r\n` and `\x02`. Long pages are split into chunks of about 150–300 words. | `page_no` (physical page) | As is |
| **DOCX** | python-docx 1.2.0 (MIT): `iter_inner_content()` gives paragraphs and tables in order, with a heading-path stack (fallback: `w:outlineLvl`). Headers, footers and text boxes need extra handling. | `heading_path`, plus `page_no` found by matching chunk text against the converted PDF | Convert to PDF with **Word (JXA)** |

- **OCR:** `ocrmac` 1.0.1 (MIT) uses Apple's built-in Vision framework. It is local, needs no download, and takes about 0.5 s per page.
  - Use it on scanned pages: pypdfium2 finds almost no characters on a page that has images.
  - Also use it on images inside slides.
- **Diagrams and equations:** optionally describe them with Claude's vision during enrichment, labelled as AI text.
- **Office automation pitfalls:**
  - Office is sandboxed and may show a one-time "Grant File Access" dialog. Converting from inside `~/Library/Containers/com.microsoft.{Word,Powerpoint}/Data/` avoids it.
  - macOS asks once for Automation permission.
  - Keynote couldn't open PPTX files reliably when scripted; don't use it.
  - Fallback: `brew install --cask libreoffice` with `ExportHiddenSlides=true`. This is untested and renders less faithfully.
- Avoid PyMuPDF because of its AGPL license.

## 6. Enrichment with Claude (before the exam)

**Decided 2026-10-14, after the E0 pilot ([e0-results.md](e0-results.md)):** enrichment runs on **Claude Opus 5.5 through `claude -p`**, using your Claude subscription. Bedrock stays available as an optional backend (§6.7).

### 6.1 Default: Opus 5.5 through `claude -p`

- **The call** (implemented in `enrich/backends/claude_cli.py`):
  ```bash
  claude -p --safe-mode --tools "" --no-session-persistence --output-format json \
    --model claude-opus-5-5 --effort medium \
    --system-prompt-file <instructions + source transcript> --json-schema '<schema>'  < task prompt
  ```
  - **`--safe-mode`** skips CLAUDE.md, hooks, plugins and MCP servers but keeps your login. `--bare` would require an API key.
  - **The transcript goes in the system prompt.** Claude Code caches the system prompt, so every window of a lecture reads it from the cache. E0 measured about 11k cache-read tokens per window.
  - **Output:** `--json-schema` returns a validated object in `structured_output`. It's validated again locally with Pydantic, and every quote is checked against the source (§6.6).
- **Measured in E0** (30-minute lecture, effort `medium`):
  - **Time:** 6 minutes, about 12 minutes per lecture hour.
  - **Output:** 268 questions and 236 concept mentions, all with verified quotes. 82% of the 60 relations passed the check.
  - **No failed calls.**
- **Subscription allowance.**
  - Every call reports your 5-hour and weekly usage. A lecture hour used about **4% of the 5-hour allowance**. The weekly figure didn't move at 1% resolution for half an hour.
  - When a limit is hit, enrichment stops with the reset time (`UsageLimitError`). A re-run picks up where it stopped.
- **Run it after each lecture, not all at once.** Four courses at about 12 lecture hours a week fit easily into regular use. A whole semester in one go would hit the limits, and Anthropic's terms say subscription limits assume "ordinary, individual usage".
- **The model is a setting** (`[enrich] model` in `library.toml`). Sonnet 5.5 is 3× faster but produced about half as many questions.

### 6.2 Optional: Bedrock backend (verified in E0)

- **Path:** `AnthropicBedrock` (bedrock-runtime) with the `global.anthropic.claude-{opus,sonnet,haiku}-5-5` inference profiles, `aws_profile="bedrock"`, `aws_region="us-east-1"`. The region must be passed explicitly.
  - The newer Mantle endpoint (`AnthropicBedrockMantle`) serves **only Opus 5.5** on this account. Sonnet and Haiku return 404 there.
- **Corrections to the earlier research:**
  - **JSON output works on both paths** for all three 5.5 models, through `output_config.format`.
  - Forced tool use is still rejected, but it isn't needed.
- **Batch inference** doesn't support the 5.5 models, so use on-demand requests.
- **Caching works:** windows after the first read about 11k cached tokens each.
- **Measured cost per lecture hour:** Haiku $0.10, Sonnet $0.78, Opus about $1.90 (extrapolated from one window).
- **Titan embeddings** (`amazon.titan-embed-text-v2:0`, 1024 dimensions) work through the same profile. They're used for merging duplicate concepts (§6.4) on either backend.

### 6.3 Calls per source

**Call A, once per lecture or document.**
- Input: the full source, about 13k tokens per lecture hour.
- Output:
  - **chapters**: titles plus passage-id ranges, used for outlines and the context prefix;
  - **glossary candidates**: term, definition quote, passage id;
  - a short source summary.

**Call B, one per window.**
- Window size: **6–10 passages, overlapping the next window by one**.
- Cached prefix: instructions, schema, the full source and its chapters.
- The model refers to passages by id and **never copies timestamps**.
- Output for each passage:
  - **8–12 questions** the passage answers (definition, why/how, exam-style, plain-language paraphrase), each with an `answer_quote`. This document-expansion technique has the strongest evidence for helping keyword search (§12).
  - **5–10 keywords and aliases**, including acronym expansions.
  - A **1–2 sentence context line**: course, lecture, chapter, topic.
  - A **one-line summary**, to help you skim results.
  - **Concept mentions:** wording used, preferred label, aliases, role (`defines`, `explains`, `example`, `mentions`), quote.
  - **Relations** from the fixed vocabulary (§6.4), each with an evidence quote.
- An optional second "what did you miss?" pass finds more concepts.

**Call C, concept merging per course** (§6.4). **Call D, linking lecture moments to slides** (§6.5).

### 6.4 Knowledge graph

- **Relation types (7):** `is_a`, `part_of`, `prerequisite_of`, `causes`, `used_for`, `example_of`, `contrasts_with` (symmetric).
  - Definitions go in the `definition` table, and synonyms in `alias`, not as edges.
  - There is no AI-written `related_to`. "Mentioned together" edges are counted deterministically instead.
- **Merging duplicate concepts:**
  1. **Normalize the name:** Unicode-normalize (NFC), lowercase, strip punctuation and leading articles, singularize. An exact match or alias match merges automatically.
  2. **Aliases come from extraction.** "ATP" and "adenosine triphosphate" don't match as strings, and embedding similarity is unreliable for acronyms.
  3. **Find candidates:** a text-search match on the label, plus the top 10 by Titan embedding similarity of at least 0.85.
  4. **Claude decides on candidates in batches of 20–50 pairs**, each with its definition quotes. **Never merge on embedding similarity alone**: mitosis and meiosis look alike to embeddings.
  5. **Every merge is logged in `merge_log` and can be undone.**
- **Weekly updates:** new lectures add new nodes and edges, and their new concepts are checked against the existing graph for duplicates.

### 6.5 Linking lecture moments to slides

- Align each lecture's passages, in order, to that week's slides by text similarity (TF-IDF cosine plus a monotonic alignment, deterministic).
- Claude confirms the top candidates.
- Store the result as `passage_link(type=slide_for)`.
- In the exam app, a video result shows "slide 14 of week04.pptx", and a slide result shows "explained at 23:40 in Lecture 4".

### 6.6 Checking AI claims, versioning and cost

- **Quote checking.** Every quote must be found in its passage. Try in order:
  1. an exact match;
  2. a match after normalizing (lowercase, punctuation and whitespace removed);
  3. a rapidfuzz `partial_ratio` of at least 90.

  Items that don't pass are stored with `verified=0` and **never exported**. A relation is shown only if its evidence quote, or the passages next to it, names both concepts.
- **Versioning.**
  - Each run records the prompt version, model id, parameters, schema version, input hash, tokens and cost.
  - `active_run` chooses what gets exported.
  - Re-running skips anything whose input hash is unchanged.
- **Cost guard.** `studybuddy enrich --dry-run` estimates tokens and dollars. Actual cost is shown per run and per course.

**Measured usage (E0, 30-minute transcript).** The semester figures assume 4 courses, about 170 lecture hours plus about 3,000 document pages, roughly 220 lecture-hour equivalents.

| Model | Output tokens per lecture hour | Time per lecture hour | `claude -p` subscription (default) | Bedrock, per-token |
|---|---|---|---|---|
| **Claude Opus 5.5 (chosen)** | about 100k | about 12 min | **about 4% of the 5-hour allowance**; no per-token bill | about $1.90/h, about $420/semester |
| Claude Sonnet 5.5 | about 43k | about 4 min | allowance change too small to measure | $0.78/h, about $170/semester |
| Claude Haiku 5.5 | about 300k (mostly reasoning) | about 18 min | about 2% of the 5-hour allowance | $0.10/h, about $22/semester |

`claude -p` reports a list-price equivalent ($2.47 per lecture hour for Opus), but on a subscription that's an estimate, not a bill. Ways to use less: fewer questions per passage, effort `low`, or skipping relations for documents.

### 6.7 Swappable enrichment backends

All enrichment calls go through one small interface, so the backend can be changed without touching windows, quote checking, versioning or export:

```python
class EnrichmentBackend(Protocol):
    name: str
    def call(self, *, system: str, cached_context: str, prompt: str,
             schema: dict, model: str) -> BackendResult: ...
    # BackendResult: parsed JSON, input/output/cache tokens, cost_usd, raw response
```

It's chosen with `[enrich] backend = "claude-cli"` in `library.toml` or `studybuddy enrich --backend …`, and recorded on every `run`.

| Backend | How it calls Claude | Use it for |
|---|---|---|
| **`claude-cli`** (default) | `claude -p` as in §6.1. Uses your Claude login, caches the system prompt, returns JSON through `--json-schema`, and reports allowance usage | All enrichment |
| **`bedrock`** | `AnthropicBedrock` as in §6.2. JSON through `output_config.format`, explicit caching, exact token counts, per-token billing on your AWS account | If the subscription allowance runs short, or you want exact per-token costs |

Both backends were implemented and tested against the live services in E0, using the same prompts, schemas and quote checking.

### 6.8 Study notes (built 2026-10-22)

Study notes for each lecture video, for learning and revising before the exam (and as
open-notes material). This step is separate from the search enrichment above.

- **Calls** (`enrich/notes.py`, prompts `notes_{system,outline,sections}_v1.md`). Every call shares the cached prefix of §6.1.
  - An outline: title, overview, consecutive sections, and announcements (logistics only).
  - One call per group of sections, about 20 passages each. Each section gets a summary, key points with details, key terms with a verbatim quote, examples, what was stressed, and review questions with answers.
- **Checks** (deterministic):
  - Sections are made to cover every passage in order.
  - A group that skips a section is asked once more for the missing ones.
  - A key term's quote is kept only if `check_quote` finds it in that section's passages, or in the passages on either side.
  - Times are looked up from passage ids; the model never writes them.
- **Storage** (`study_notes.py`). Each set of notes is a `run` row (task `notes`), plus the checked notes as JSON in `notes`.
  - A video is redone only when its `input_sha` (the transcript as sent) or the prompt version changes, or with `--redo`.
  - Markdown is written to `courses/<slug>/notes/<video>.notes.md`, plus `index.md`: lectures, announcements and an A–Z glossary. A session writes to `FOLDER/Study notes/`.
- **When.**
  - After `transcribe`, next to the transcript (which is then also kept as `.json`, so that `studybuddy notes FILE.json` can redo the notes).
  - After `add`, for the videos it adds.
  - After `session`, for the whole sitting.
  - With `studybuddy notes [COURSE...] [--redo]`.
  - Turned off with `--no-notes` or `[notes] auto = false`.
- **Measured.** On the synthetic 10-minute transcript, Opus 5.5 through `claude -p` took 75 s for 2 calls, and every key-term quote verified.
- **In the exam app (added 2026-10-26).** `build-exam` exports the notes whose `input_sha` still matches the transcript into a `notes` table (index format 2), with every passage number replaced by the exam passage's id and start time.
  - Choosing a course with an empty search box shows the course page: its lectures in order with overviews, the announcements, and an A–Z glossary merged across lectures (§7.3).
  - A lecture's page shows its sections. Every point's time sits in the margin and plays the lecture from there, with that passage of the transcript below (C4).
  - Each page is labelled as AI-written (C5). The notes are browsed only and not indexed for search, so results stay original text.
  - `studybuddy notes` rebuilds an existing index. `status` reports an index that is out of date, using the newest notes run, or that was built for an older format.

## 7. Exam-time search (no AI)

### 7.1 Building `exam.sqlite` (`studybuddy build-exam`, deterministic)

- Copy the active, verified rows from `library.sqlite`.
- Copy the study notes that match their transcripts into `notes`, for browsing only (§6.8).
- **Full-text search tables** (SQLite FTS5 with external content, built once, no triggers):
  - `passage_fts(text, aliases, questions, context)` with `tokenize='porter unicode61'`;
  - `passage_tri(text)` with `tokenize='trigram'`, a fallback for partial words;
  - `concept_fts(pref_label, aliases, definition_quote)`.
- **Spelling vocabulary:** `vocab(term, doc_freq, metaphone)` from an unstemmed table plus `fts5vocab`, used for "did you mean".
- Precompute the graph layout positions, for the graph view.
- Run `INSERT INTO passage_fts(passage_fts) VALUES('optimize')`, record the file's SHA-256, then open it read-only (`mode=ro&immutable=1`).
- **Measured:** 100k passages build in about 12 s, and a search with snippets takes about 7 ms.

### 7.2 Answering a question

1. **Tokenize** and drop stopwords.
2. **"Did you mean".** Any word not in the vocabulary is matched against it with rapidfuzz normalized similarity ≥ 0.75 or the same Metaphone sound-alike code (jellyfish). It's suggested, not applied silently.
3. **Expand with aliases.** Word sequences in the question are looked up in the stored alias table, and matches add the concept's label and top aliases. Capped at about 30–50 extra OR'd terms: 500 terms took 323 ms in testing.
4. **Build a safe query.**
   - Wrap every word in quotes (doubling any embedded `"`) and add the AND/OR/NEAR operators yourself. Raw input such as `AND`, `c++` or `x:y` would break FTS.
   - Join words with **OR**, because FTS's default AND fails on long questions.
   - Pass the query as a bound `?` parameter.
5. **Rank** with `bm25(passage_fts, 1.0, 0.8, 0.5, 0.3)`, weighting text, aliases, questions and context. These starting weights get tuned in E6.
   - Boost passages that **define** a matched concept.
   - Show at most 3 results per source, to keep variety.
   - Fall back to the trigram table when there are few hits.
6. **Show results** grouped by course and source:
   - title, timestamp range or slide/page, a **highlighted snippet of the original text**, and the matched concepts;
   - AI fields labelled;
   - linked slide or lecture moment (§6.5).

### 7.3 Browsing

- **Concept page:** the definition as the original quote plus its source link, the AI gloss labelled, aliases, relations with evidence links, and every mention with its timestamp or page.
- **Lecture outline:** chapters with timestamps. The study notes' sections provide this now (§6.8).
- **Glossary:** an A–Z list for each course. The study notes' key terms provide this now (§6.8).
- **Course map:** a Cytoscape.js graph using the precomputed positions.
- **Filters:** course, week or date range, and source type.

## 8. The exam app

- **Server:** Starlette ≥ 1.3 with uvicorn, bound to **127.0.0.1** only.
  - Starlette's `FileResponse` handles HTTP Range requests, which seeking in a `<video>` needs. Python's built-in `http.server` doesn't, which was verified.
  - Add `TrustedHostMiddleware(["127.0.0.1","localhost"])` and the header `Content-Security-Policy: default-src 'self'`, so the browser also refuses outside requests.
- **Vendored assets, no CDN:**
  - **PDF.js 6.4.299** (Apache-2.0), trimmed to about 8 MB. Opening `viewer.html?file=/docs/x.pdf#page=N&search=term&phrase=true` lands on the page with the term highlighted (verified).
  - **Cytoscape.js 3.34.x** (MIT).
- **Video:** one `<video>` element.
  - A click sets `currentTime`, waiting for `loadedmetadata` when the source changes.
  - Media fragments (`#t=`) only for deep links. They're verified in Chrome but untested in Safari.
- **Proving there's no AI** (constraint C1):
  1. **Separate package and virtualenv** (`exam/pyproject.toml`, lockfile committed). Dependencies are limited to starlette, uvicorn, rapidfuzz and jellyfish.
  2. **Runtime guard:** `sys.addaudithook` (PEP 578) is the first line of the entry point.
     - It refuses to import `mlx`, `torch`, `transformers`, `anthropic`, `boto3`, `botocore`, `openai` and similar.
     - It refuses any network connection or DNS lookup that isn't to loopback.
     - Tested: blocks `import boto3`/`anthropic` and connections to 1.1.1.1; allows 127.0.0.1.
  3. **Status page:** "AI libraries loaded: none · network: loopback only · index SHA-256: …".
  4. **Tests:**
     - pytest-socket (`--disable-socket --allow-hosts=127.0.0.1`);
     - an import-linter rule forbidding `studybuddy_exam` from importing `studybuddy`;
     - a subprocess check of `sys.modules` after startup.
- **Launch:** `studybuddy-exam ~/StudyBuddy/fall-2026` opens the browser at `http://127.0.0.1:8765`.
- **Several libraries (added 2026-10-27):** `studybuddy-exam ~/StudyBuddy` searches every library and session folder in that folder.
  - Each library keeps its own `exam.sqlite`. At startup the exam app copies them into one index in a temporary folder, with new ids. Libraries without an index, or with an index in an older format, are left out with a message.
  - Format 3 of the index added a `library` table: every course belongs to a library, and the files are served from that library's folder.
  - The course list groups the courses by library. The status line's tooltip lists each library's index fingerprint, since those are the hand-off files.
  - `studybuddy serve [FOLDER]` (default `~/StudyBuddy`) first rebuilds every index that is missing or out of date, then starts the exam app on the folder in place of itself.

## 9. Project layout and commands

```
src/studybuddy/                 # builder (existing transcription + new)
  library.py                    # library.toml, courses, sources registry
  ingest/{video,pptx,pdf,docx,convert,ocr}.py
  enrich/{schemas,windows,verify,resolve,links,costs}.py
  enrich/backends/{base,bedrock,claude_cli}.py   # swappable backends (§6.7)
  sync.py                       # optional S3 backup (E7)
  enrich/prompts/*.md           # versioned prompts
  exam_export.py                # builds exam.sqlite
exam/                           # separate distribution: studybuddy-exam
  pyproject.toml  (starlette, uvicorn, rapidfuzz, jellyfish only)
  src/studybuddy_exam/{__main__,guard,query,search,server}.py
  src/studybuddy_exam/static/   # UI + vendored PDF.js, Cytoscape.js
```

```
studybuddy library init ~/StudyBuddy/fall-2026
studybuddy course add bio101 --title "Biology 101" [--glossary terms.txt]
studybuddy add bio101 lecture04.mp4 [--date 2026-09-03] [--title "Cell energy"]
studybuddy add bio101 week04.pptx notes.docx reading.pdf
studybuddy enrich [--course bio101] [--backend bedrock|claude-cli] [--model anthropic.claude-sonnet-5-5] [--dry-run]
studybuddy build-exam
studybuddy status                   # what's ingested, enriched, exported; cost so far
studybuddy sync push|pull           # optional (E7): back up / restore the library via S3
studybuddy-exam ~/StudyBuddy/fall-2026
studybuddy serve [~/StudyBuddy]     # every library in the folder: update the indexes, then search them all
```

## 10. Milestones

| # | Milestone | Done when |
|---|---|---|
| **E0** ✅ | **Enrichment pilot (done 2026-10-14).** Both backends (`claude -p` and Bedrock) built and tested live. JSON output, caching and Titan confirmed. A real 2-minute clip and a 30-minute transcript run through Opus, Sonnet and Haiku 5.5. | Done: Opus 5.5 through `claude -p` chosen. Measurements and side-by-side samples in [e0-results.md](e0-results.md) |
| E1 ✅ | **Library and ingestion (done 2026-10-14).** `library init`, `course add/list`, `add` for MP4 (transcribe + passages + exam copy) and for PPTX/PDF/DOCX (extraction, OCR, Office→PDF conversion, slide/page mapping), `status`. Tests use generated sample files; Office tests drive the real PowerPoint and Word. | Done: real-Office tests confirm PDF page N == slide N with a hidden slide, and that every Word passage is on its page. See *E1 notes* below |
| **E2** ✅ | **AI-free exam app (done 2026-10-14).** `build-exam` over the original text only; safe query building; ranking; snippets; video jump; PDF.js page jump; guard and offline tests. | Done. Tests in a fresh process confirm the guard blocks AI imports and non-loopback network, and the running app reports "AI libraries loaded: none · network: loopback only". Checked in headless Chrome: zero console errors or CSP violations, slide and video jumps land correctly. See *E2 notes* below |
| E3 | **Enrichment.** Calls A and B, schema checking, quote checking, versioning, cost guard; questions, aliases and context indexed as extra columns. | A re-run skips unchanged input; unchecked items are never exported |
| E4 | **Knowledge graph.** Concepts, mentions, relations, duplicate merging (Titan + Claude); concept pages; alias expansion; course map. | Duplicates merged with an undo log; every relation has a checked quote |
| E5 | **Polish.** Lecture↔slide links, "did you mean", trigram fallback, filters, optional word-level timestamps. | — |
| E6 | **Evaluation and rehearsal.** 30–50 practice questions per course (from quizzes), each with its known answer location. Measure recall@5 and @10; tune the ranking weights, the alias cap and the number of questions per passage. A full rehearsal with Wi-Fi off a week before the final. | Recall@5 target agreed and met; exam-day checklist written |
| E7 | **Optional: cloud backup.** `studybuddy sync push`/`pull` wraps `aws s3 sync` to a private, encrypted, versioned S3 bucket: originals, documents, transcripts, `library.sqlite` snapshots and the exam set. A lifecycle rule moves originals to Glacier Instant Retrieval after 30 days (about $0.80/month for 200 GB). The exam app never uses the cloud. | A restore from S3 onto an empty folder rebuilds a working library |

**E1 notes: what was built differently from the plan**

- **IDs are never reused** (`AUTOINCREMENT`), because enrichment will point at source and passage ids.
- **Passage origin.** Each passage records where its text came from: `speech`, `text`, `notes` (speaker notes) or `ocr`. On a slide, the text, the speaker notes and the text read from pictures are separate passages, so results can be labelled.
- **Staging folder.** `add` works in `courses/<slug>/.staging/<file hash>/`. Only when everything succeeded does it move files into place and record the source in one transaction. An interrupted run resumes from what is already there.
- **Replacing.** A different file with the same name needs `--replace`, and the old source is removed only after the new one is ingested.
- **View PDF names.** View PDFs are named `<file name>.pdf` (`week04.pptx.pdf`), so slides, a handout and notes that share a stem never collide.
- **Exam copies.** They use the Mac's hardware H.264 encoder: about 44× real time, roughly 3.5 minutes for 2.5 hours of video. H.264 video already at or below the target bitrate is copied as is, since re-encoding would only make it bigger.
- **Text extraction beyond the plan:**
  - **Equations.** PowerPoint text boxes that hold equations are read; python-pptx can't see them.
  - **Charts.** Labels are read from the chart XML, so 3-D and stock charts work.
  - **Repeated logos.** OCR text repeated on 3 or more slides is dropped.
  - **Word.** Tracked insertions, content controls and simple fields are included; deleted text and table-of-contents paragraphs are skipped.
  - **PDF.** Running headers and footers are removed. Words hyphenated across lines are rejoined (pdfium marks them with U+FFFE or `\x02`).
- **Transparent images.** OCR flattens them onto white first. Otherwise dark text on a transparent PNG reads as black on black.

**E2 notes: what was built differently from the plan**

- **The exam app owns the index format** (`studybuddy_exam/index_format.py`). `build-exam` passes it rows; the exam app never imports the builder. The builder environment installs the exam package (`pip install -e ./exam`) to write the index.
- **Three ranking tiers instead of one OR query:**
  1. passages containing every word;
  2. then passages containing some of the words;
  3. then partial-word (trigram) matches, only when the first two find fewer than 5.

  BM25 ranks within each tier. This keeps two-word questions from being led by passages that contain only the rarer word.
- **"Did you mean".**
  - It only checks words that match nothing, even after stemming, so "tariff" is never "corrected" when only "tariffs" occurs.
  - Swapping two letters counts as one edit ("cycel" → "cycle").
  - When the question finds nothing at all, the corrected question is searched and the page says so.
- **Questions made only of common words** ("what is it") search nothing. A single common word ("IT") is still searched.
- **Guard coverage:**
  - Imports refused: AI and machine-learning libraries, model clients, Apple's Vision OCR, and the studybuddy builder.
  - Network refused: connections, name lookups, UDP sends and binds that aren't to loopback.
  - Fetches through `urllib` are refused too.
- **Lockfile.** `exam/requirements.lock` is generated from a clean environment holding only the four runtime dependencies. A test checks it and the installed environment for AI packages.
- **PDF.js internal patch.** PDF.js's `#search=` scrolls the first match to the top of the viewer, which can show the next slide or jump to another page. The page wraps PDF.js's internal `findController.scrollMatchIntoView` to return to the hit's page. That internal could change in a future PDF.js release; the bundled version is pinned at 6.4.299.
- **Later:** the API has no per-source passage list yet. A lecture outline, and a transcript that follows playback, come with chapters in E3/E5.

E2 deliberately comes before enrichment, so a working AI-free search exists early and every later step can be measured against it.

## 11. Pitfalls found during research

- **`claude -p` sign-in and limits.**
  - `--bare` refuses a subscription login; use `--safe-mode`.
  - Usage limits stop a run until the reset time, so enrich after each lecture instead of in bulk.
- **Bedrock (optional backend):**
  - Mantle serves only Opus 5.5 on this account, so use bedrock-runtime with `global.` inference profiles.
  - There's no batch discount for the 5.5 models.
  - The region is never read from `~/.aws/config`, and the `bedrock` profile has no region set.
- **Python environment.**
  - The pyenv Python can't load SQLite extensions, so spellfix1 is out.
  - It has no `lzma`, so the `wn` package is out.
  - Neither is needed: rapidfuzz, jellyfish and the AI-generated aliases replace them.
- **Leave WordNet out** at exam time. Its synonyms are poor for course terms: "heart" returns "eye, kernel, marrow".
- **Long OR queries slow down sharply** (5,000 terms take 7 s). Expand aliases mostly on the index side and cap query-side expansion.
- **Hidden slides** shift the PDF page numbers. **SmartArt** isn't read by python-pptx. **DOCX has no page numbers** (map chunk text against the converted PDF).
- **Office sandbox** file-access dialogs; Keynote can't be scripted reliably.
- **AI summaries are wrong roughly 2–24% of the time** depending on the model (about 10–12% for recent Claude models on Vectara's leaderboard). Hence quote checking, labels, and always showing the original.
- **Accurate-mode passages are about 60 s**, so a jump can land up to a minute early (word timestamps are optional, in E5).
- **Disk space** for a semester of video (§3).

## 12. Open questions

**Answered 2026-10-14**

1. **Four courses, many lecture hours.** Plan for about 170 hours (4 courses × ~3 h/week × 14 weeks). Enrichment cost scales with hours (§6.6).
2. **No practice questions yet.** You'll add quizzes and practice exams as they become available, so the E6 evaluation set grows during the semester. Until then, use a starter set of about 20 questions per course, written by you from your notes.
3. **Chrome on exam day.** Chrome is where seeking, media fragments and PDF.js deep links were verified, so the Safari caveats in §8 don't apply.
4. **Videos are 1080p MP4s downloaded to this laptop.**
   - At an assumed 0.5–1.5 GB per hour, 170 hours is roughly 85–255 GB, against about 67 GB free.
   - So the 540p exam copies (§3), about 35–40 GB, are needed.
   - The originals must live somewhere else: an external SSD or cloud storage.

**Decided**

- **Enrichment backend:** Bedrock is the main path, with `claude -p` as a swappable option (§6.7).
- **Where the originals live:** not a design dependency, because nothing reads them after `add` (§3). Optional S3 backup is milestone E7.

## 13. Sources

- **Bedrock:**
  - [Claude in Amazon Bedrock](https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock)
  - [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) ([price list API](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/us-east-1/index.json))
  - [Batch-supported models](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html)
  - [Batch limitations](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html)
  - [Mantle quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-mantle.html)
  - [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
  - [Titan embeddings V2](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
  - [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)
- **Retrieval and knowledge graphs:**
  - [doc2query](https://ar5iv.labs.arxiv.org/html/1904.08375), [docTTTTTquery](https://github.com/castorini/docTTTTTquery), [Doc2Query--](https://ar5iv.labs.arxiv.org/html/2301.03266), [Doc2Query++](https://arxiv.org/html/2510.09557v1), [When expansions fail](https://ar5iv.labs.arxiv.org/html/2309.08541)
  - [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
  - [GraphRAG](https://arxiv.org/html/2404.16130v2), [LightRAG](https://arxiv.org/html/2410.05779v3), [Zep/Graphiti](https://arxiv.org/html/2501.13956v1)
  - [Graphusion](https://arxiv.org/html/2407.10794), [LectureBank](https://arxiv.org/abs/1811.12181), [MaViLS](https://arxiv.org/pdf/2409.16765)
  - [Neo4j entity resolution](https://neo4j.com/labs/agent-memory/explanation/resolution-deduplication/), [BatchER](https://arxiv.org/abs/2312.03987)
  - [LangExtract](https://github.com/google/langextract), [Vectara hallucination leaderboard](https://github.com/vectara/hallucination-leaderboard/)
- **Search and serving:**
  - [SQLite FTS5](https://www.sqlite.org/fts5.html)
  - [Starlette release notes](https://starlette.dev/release-notes/), [CPython http.server Range issue](https://github.com/python/cpython/issues/86809)
  - [PEP 578 audit hooks](https://peps.python.org/pep-0578/)
  - [Cytoscape.js](https://blog.js.cytoscape.org/2025/01/13/3.31.0-release/)
  - [PDF.js v6.4.299](https://github.com/mozilla/pdf.js/releases/tag/v6.4.299), [PDF.js viewer options](https://github.com/mozilla/pdf.js/wiki/Viewer-options)
  - [Media fragments support](https://caniuse.com/media-fragments)
- **Documents:**
  - [python-pptx](https://github.com/scanny/python-pptx) ([SmartArt issue #83](https://github.com/scanny/python-pptx/issues/83)), [python-docx](https://github.com/python-openxml/python-docx)
  - [pypdfium2](https://github.com/pypdfium2-team/pypdfium2), [pypdf](https://github.com/py-pdf/pypdf)
  - [ocrmac](https://github.com/straussmaximilian/ocrmac), [mammoth](https://github.com/mwilliamson/python-mammoth)
  - [PyMuPDF license](https://pymupdf.readthedocs.io/en/latest/about.html)
