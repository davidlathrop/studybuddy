# Exam Search: Implementation Plan

_Status: plan only, nothing implemented yet. Research done 2026-10-14._
_Builds on the transcription pipeline in [mp4-transcription-plan.md](mp4-transcription-plan.md), which is implemented on branch `feature/mp4-transcription`._

## 1. Goal and hard constraints

**Goal.** Over the semester, index every lecture video and every course document (PowerPoint, PDF, Word). During an open-book, open-notes final exam, type a question and get back **which video and timestamp** or **which file and slide/page** to look at. One click opens the original at that spot.

**Hard constraints**

| # | Constraint | What it means for the design |
|---|---|---|
| C1 | **No AI during the exam** | The exam app runs only classic search code (word-matching search with relevance ranking, string matching, graph lookups). It cannot load or call any AI library, and it can prove that (§8). |
| C2 | AI is allowed **before** the exam | Transcription (already built), enrichment, and knowledge-graph building all happen during the semester. Enrichment uses Claude on AWS Bedrock. |
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
                                         Claude on Bedrock ──enrich┘             every AI run + its
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

- **Builder:** the existing `studybuddy` package, extended. It has the AI dependencies (mlx-audio, anthropic, boto3).
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
course(id, slug, title)
source(id, course_id, kind[video|pptx|pdf|docx], title, week, date, path, view_path,
       duration_s, page_count, sha256)
passage(id, source_id, ord, t_start_ms, t_end_ms,          -- video
        slide_no, page_no, heading_path,                     -- documents
        text, text_sha)
run(id, task, prompt_version, model_id, params_json, schema_ver, input_sha, created_at,
    input_tokens, output_tokens, cache_read_tokens, cost_usd)
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

## 6. Enrichment with Claude on AWS Bedrock (before the exam)

### 6.1 Client and setup (checked against this account)

- **Install:** `pip install "anthropic[bedrock]>=1.14"` into the builder venv. `AnthropicBedrockMantle` needs at least SDK 0.91, and 1.x uses `httpx2`.
- **Client:** `AnthropicBedrockMantle(aws_profile="bedrock", aws_region="us-east-1")`.
  - **The region must be passed explicitly.** The client doesn't read the region from `~/.aws/config`, and the `bedrock` profile has none set.
  - Endpoint: `https://bedrock-mantle.{region}.api.aws/anthropic`.
- **Models:** `anthropic.claude-opus-5-5`, `anthropic.claude-sonnet-5-5`, `anthropic.claude-haiku-5-5`.
  - All three are listed as available for the `bedrock` profile in us-east-1 and us-west-2.
  - The use-case agreement isn't on file, but Mantle doesn't need it.
- **Default model:** Claude Opus 5.5. The model is a setting in `library.toml`. See §6.6 for cost per model; choose after the E0 pilot.
- **Embeddings for merging duplicate concepts:**
  - `amazon.titan-embed-text-v2:0`, 1024 dimensions, $0.02 per million tokens, through boto3 `bedrock-runtime`.
  - Confirm access to it in E0.

### 6.2 Verified Bedrock constraints that shape the design

| Fact | Consequence |
|---|---|
| The JSON-schema output mode (`output_config.format`) is documented for the legacy Bedrock integration only, not confirmed on Mantle for the 5.5 models | Get JSON through a **client tool** (`submit_enrichment` with a JSON schema), validate it locally with Pydantic, and retry with the validation error if it fails. E0 tests whether `output_config.format` works on Mantle; use it if so. |
| Opus 5.5 and Sonnet 5.5 reject forced tool use (`tool_choice` `any` or `tool`) with a 400 | Use `tool_choice: auto`, tell the model in the prompt to call `submit_enrichment`, and treat a reply without the tool call as a retry. |
| Bedrock batch inference **doesn't support the 5.5 models** and doesn't allow tool use | Use **on-demand** requests with modest parallelism (about 4). The SDK retries 429s. The volume is far below the quotas. |
| Prompt caching works on Bedrock; cache reads cost 0.05× of input on Opus and Sonnet 5.5 | Cache the instructions plus the whole lecture transcript, then run the passage windows back to back within the 5-minute cache lifetime. |

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

**Rough cost.** These numbers are estimates, to be replaced with E0 measurements.
- **Per lecture hour:** about 60 passages × about 1,000 output tokens (enrichment JSON plus thinking) is about 60k output tokens, plus about 30k input tokens, mostly cache reads.
- **Prices:** Bedrock global on-demand prices as of 2026-10-12 (regional endpoints cost 10% more).
- **Semester assumption:** 4 courses, about 170 lecture hours plus about 3,000 document pages, roughly 220 lecture-hour equivalents.

| Model | Output price per million tokens | Per lecture hour | Semester |
|---|---|---|---|
| Claude Opus 5.5 (default) | $20 | about $1.40 | about $310 |
| Claude Sonnet 5.5 | $10 | about $0.70 | about $155 |
| Claude Haiku 5.5 | $0.50 | about $0.04 | about $9 |

Output tokens dominate the cost. Ways to cut it: fewer questions per passage, skip relations for documents, or a cheaper model. The E0 pilot runs one real lecture through each model, so you can compare quality before choosing.

### 6.7 Swappable enrichment backends

All enrichment calls go through one small interface, so the backend can be changed without touching windows, quote checking, versioning or export:

```python
class EnrichmentBackend(Protocol):
    name: str
    def call(self, *, system: str, cached_context: str, prompt: str,
             schema: dict, model: str) -> BackendResult: ...
    # BackendResult: parsed JSON, input/output/cache tokens, cost_usd, raw response
```

It's chosen with `[enrich] backend = "bedrock"` in `library.toml` or `studybuddy enrich --backend …`, and recorded on every `run`.

| Backend | How it calls Claude | Use it for |
|---|---|---|
| **`bedrock`** (default) | `AnthropicBedrockMantle` as in §6.1–6.2: client tool plus local validation and retry, explicit prompt caching of the shared transcript, exact token counts from `usage` | The semester-scale enrichment. Full control over caching and cost, per-token billing on your AWS account. |
| **`claude-cli`** | Runs `claude -p` as a subprocess, one call per window | Small runs and trying out prompts |

**How the `claude-cli` backend works**

- **The call:**
  ```bash
  claude -p --bare --tools "" --no-session-persistence --model <model> \
    --system-prompt-file prompts/enrich_vN.md --json-schema '<schema>' --output-format json
  ```
  The window goes in on stdin.
- **Output:** the validated object comes back in `structured_output`. Claude Code re-prompts when the output doesn't match the schema, and reports `error_max_structured_output_retries` when it gives up.
- **Sign-in:**
  - Bedrock: set `CLAUDE_CODE_USE_BEDROCK=1`, `AWS_PROFILE=bedrock`, `AWS_REGION`, and `ANTHROPIC_MODEL` set to a Bedrock model ID.
  - Or an `ANTHROPIC_API_KEY`.
  - `--bare` never reads a subscription login.
- **Why not a Pro or Max subscription for the semester run:**
  - Anthropic's terms say subscription limits assume "ordinary, individual usage", and direct programmatic use to API keys.
  - Session and weekly limits would stall a run of thousands of calls.
  - It's fine for trying out prompts by hand.
- **Costs:** `total_cost_usd` is a client-side estimate, and the shared transcript is only cached if calls reuse it within about 5 minutes. Measure per-call tokens in E0 before using this backend for bulk work.
- **Unverified:** whether `--json-schema` works on Bedrock. Check it in E0.

## 7. Exam-time search (no AI)

### 7.1 Building `exam.sqlite` (`studybuddy build-exam`, deterministic)

- Copy the active, verified rows from `library.sqlite`.
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
- **Lecture outline:** chapters with timestamps.
- **Glossary:** an A–Z list for each course.
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
```

## 10. Milestones

| # | Milestone | Done when |
|---|---|---|
| **E0** | **Bedrock pilot.** SDK and client set up. Test tool-based JSON and `output_config.format` on Mantle. Confirm cache hits. Run one real lecture through Opus, Sonnet and Haiku 5.5 and measure tokens and cost. Confirm Titan access. | Measured cost per lecture hour per model, sample outputs side by side, model chosen |
| E1 | **Library and ingestion.** `library init`, `course add`, `add` for MP4 (transcribe + passages + exam copy) and for PPTX/PDF/DOCX (extraction, OCR, Office→PDF conversion, slide/page mapping). Tests use generated sample files. | Every passage has a correct location; PDF page N == slide N even with hidden slides |
| **E2** | **AI-free exam app (no enrichment yet).** `build-exam` over the original text only; safe query building; ranking; snippets; video jump; PDF.js page jump; guard and offline tests. | Works with Wi-Fi off; the guard tests pass. **Already useful at this point.** |
| E3 | **Enrichment.** Calls A and B, schema checking, quote checking, versioning, cost guard; questions, aliases and context indexed as extra columns. | A re-run skips unchanged input; unchecked items are never exported |
| E4 | **Knowledge graph.** Concepts, mentions, relations, duplicate merging (Titan + Claude); concept pages; alias expansion; course map. | Duplicates merged with an undo log; every relation has a checked quote |
| E5 | **Polish.** Lecture↔slide links, "did you mean", trigram fallback, filters, optional word-level timestamps. | — |
| E6 | **Evaluation and rehearsal.** 30–50 practice questions per course (from quizzes), each with its known answer location. Measure recall@5 and @10; tune the ranking weights, the alias cap and the number of questions per passage. A full rehearsal with Wi-Fi off a week before the final. | Recall@5 target agreed and met; exam-day checklist written |
| E7 | **Optional: cloud backup.** `studybuddy sync push`/`pull` wraps `aws s3 sync` to a private, encrypted, versioned S3 bucket: originals, documents, transcripts, `library.sqlite` snapshots and the exam set. A lifecycle rule moves originals to Glacier Instant Retrieval after 30 days (about $0.80/month for 200 GB). The exam app never uses the cloud. | A restore from S3 onto an empty folder rebuilds a working library |

E2 deliberately comes before enrichment, so a working AI-free search exists early and every later step can be measured against it.

## 11. Pitfalls found during research

- **JSON output on Mantle.**
  - The JSON-schema output mode isn't confirmed on Mantle for the 5.5 models.
  - Forced `tool_choice` is a 400 on Opus and Sonnet 5.5.
  - So use a tool with `tool_choice: auto`, validate locally, and retry.
- **No batch discount** for the 5.5 models, and batch jobs can't use tools.
- **The region is never read from `~/.aws/config`**, and the `bedrock` profile has no region set.
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
