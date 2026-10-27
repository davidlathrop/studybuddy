// StudyBuddy exam search: the browser side.
//
// Talks only to the local exam server (see server.py for the API). No AI, no outside
// requests, no frameworks, no build step. The Content-Security-Policy forbids inline
// scripts and eval, so everything happens here and all DOM is built with createElement.

const KINDS = {
  video: { label: "Video", noun: ["video", "videos"] },
  pptx: { label: "Slides", noun: ["slide deck", "slide decks"] },
  pdf: { label: "PDF", noun: ["PDF", "PDFs"] },
  docx: { label: "Word", noun: ["Word file", "Word files"] },
};
const KIND_ORDER = Object.keys(KINDS);
const ORIGINS = { notes: "Speaker notes", ocr: "Text read from an image" };
const DEBOUNCE_MS = 250;
const STATUS_EVERY_MS = 30_000;
const PIN_FOR_MS = 6_000;
const FIND_NOT_FOUND = 1; // PDF.js FindState.NOT_FOUND

const $ = (id) => document.getElementById(id);
const narrowScreen = window.matchMedia("(max-width: 860px)"); // the split view stacks here (app.css)
const el = {
  guardAlert: $("guard-alert"),
  libraryName: $("library-name"),
  form: $("search-form"),
  q: $("q"),
  course: $("f-course"),
  notesButton: $("notes-button"),
  weekFrom: $("f-week-from"),
  weekTo: $("f-week-to"),
  kindButtons: [...document.querySelectorAll(".kind-toggle")],
  clearFilters: $("clear-filters"),
  resultsPane: $("results-pane"),
  summary: $("results-summary"),
  results: $("results"),
  viewerHead: $("viewer-head"),
  viewerFlag: $("viewer-flag"),
  viewerKind: $("viewer-kind"),
  viewerTitle: $("viewer-title"),
  viewerMeta: $("viewer-meta"),
  stageEmpty: $("stage-empty"),
  videoWrap: $("video-wrap"),
  video: $("video"),
  back10: $("back10"),
  replayHit: $("replay-hit"),
  videoRange: $("video-range"),
  videoError: $("video-error"),
  frame: $("doc-frame"),
  noView: $("no-view"),
  passage: $("passage"),
  passageWhere: $("passage-where"),
  passageText: $("passage-text"),
  pPrev: $("p-prev"),
  pNext: $("p-next"),
  pGo: $("p-go"),
  statusLine: $("status-line"),
};

const state = {
  courses: [],
  kinds: new Set(),
  timer: 0,
  controller: null,
  shownKey: null, // the request whose results are on screen
  pendingKey: null,
  terms: [], // words to highlight, from the last search
  hits: [], // in display order
  cursor: 0, // the hit that takes keyboard focus (roving tabindex)
  openHit: null,
  passage: null,
  passageSeq: 0,
  videoSource: null,
  pendingSeek: null,
  docSource: null,
  docPin: null, // {page, until}: keep this page in view after PDF.js finds the words
  coursesProblem: null,
  viewSeq: 0, // which request's page the results pane should show
  lectureId: null, // the lecture whose study notes are open (no search, a course chosen)
  returnTo: null, // the lecture to bring back into view on the course's notes page
  lectures: new Map(), // source id -> lecture, for the times in the notes
  replay: null, // {sourceId, fileUrl, ms}: what "Play again" plays
};

// ---------- Small helpers ----------

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  node.append(...children.flat().filter((c) => c != null && c !== false));
  return node;
}

const pad = (n) => String(n).padStart(2, "0");

function clock(ms) {
  const total = Math.max(0, Math.floor((ms || 0) / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  return hours ? `${hours}:${pad(minutes)}:${pad(seconds)}` : `${minutes}:${pad(seconds)}`;
}

const timeRange = (start, end) => (end != null && end > start ? `${clock(start)}–${clock(end)}` : clock(start));
const plural = (n, [one, many]) => `${n} ${n === 1 ? one : many}`;
const headingPath = (item) => (item.heading_parts ?? []).join(" › ");

function listWords(items) {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

function weeksText(weeks) {
  if (!weeks.length) return "";
  if (weeks.length === 1) return `Week ${weeks[0]}`;
  const contiguous = weeks.every((w, i) => i === 0 || w === weeks[i - 1] + 1);
  return contiguous && weeks.length > 2 ? `Weeks ${weeks[0]}–${weeks.at(-1)}` : `Weeks ${listWords(weeks.map(String))}`;
}

/** The server's snippet HTML is escaped text with <mark> around matches; keep only that. */
function snippetNodes(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  const copy = (from, into) => {
    for (const child of from.childNodes) {
      // Blank lines between headings and paragraphs only spread a snippet out.
      if (child.nodeType === Node.TEXT_NODE) into.append(child.textContent.replace(/\n\s*\n/g, "\n"));
      else if (child.nodeName === "MARK") copy(child, into.appendChild(document.createElement("mark")));
      else copy(child, into);
    }
    return into;
  };
  return copy(template.content, document.createDocumentFragment());
}

/** Plain text with the search words marked. */
function highlighted(text, terms) {
  const fragment = document.createDocumentFragment();
  const words = [...new Set(terms)].filter((t) => t.length > 1).sort((a, b) => b.length - a.length);
  if (!words.length) {
    fragment.append(text);
    return fragment;
  }
  const pattern = new RegExp(words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "giu");
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    fragment.append(text.slice(last, match.index), h("mark", {}, match[0]));
    last = match.index + match[0].length;
  }
  fragment.append(text.slice(last));
  return fragment;
}

class ServerError extends Error {}

async function getJSON(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error("unreachable");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok || data == null) throw new ServerError(data?.error || `the server answered ${response.status}`);
  return data;
}

function problemText(err, doing) {
  if (err instanceof ServerError) return `${doing} didn't work: ${err.message}.`;
  return "Can't reach the exam server. Check that studybuddy-exam is still running, then try again.";
}

// ---------- Filters ----------

function fillCourses(courses) {
  state.courses = courses;
  const option = (course) => h("option", { value: course.slug }, course.title);
  const groups = libraryGroups(courses);
  if (groups.length > 1) el.course.append(...groups.map((g) => h("optgroup", { label: g.library }, g.courses.map(option))));
  else el.course.append(...courses.map(option));
  fillWeeks();
}

// Started on a folder of libraries (~/StudyBuddy), the server lists every library's courses.
function libraryGroups(courses) {
  const groups = new Map();
  for (const course of courses) {
    if (!groups.has(course.library)) groups.set(course.library, []);
    groups.get(course.library).push(course);
  }
  return [...groups].map(([library, list]) => ({ library, courses: list }));
}

// A course's name, with its library's when there are several and they differ ("Fall 2026 · Biology 101").
function courseName(slug, title) {
  const course = state.courses.find((c) => c.slug === slug);
  const several = new Set(state.courses.map((c) => c.library)).size > 1;
  return several && course?.library && course.library !== title ? `${course.library} · ${title}` : title;
}

function fillWeeks() {
  const slug = el.course.value;
  const weeks = new Set();
  for (const course of state.courses) {
    if (!slug || course.slug === slug) for (const w of course.weeks || []) weeks.add(w);
  }
  const sorted = [...weeks].sort((a, b) => a - b);
  for (const select of [el.weekFrom, el.weekTo]) {
    const keep = select.value;
    select.replaceChildren(h("option", { value: "" }, "Any"), ...sorted.map((w) => h("option", { value: w }, String(w))));
    select.value = sorted.includes(Number(keep)) ? keep : "";
  }
  const disabled = sorted.length === 0;
  el.weekFrom.disabled = el.weekTo.disabled = disabled;
}

function filtersActive() {
  return Boolean(el.course.value || state.kinds.size || el.weekFrom.value || el.weekTo.value);
}

function filtersText() {
  const parts = [];
  if (el.course.value) parts.push(el.course.selectedOptions[0].textContent);
  if (state.kinds.size) parts.push(listWords(KIND_ORDER.filter((k) => state.kinds.has(k)).map((k) => kindButton(k).textContent.trim())));
  const from = el.weekFrom.value;
  const to = el.weekTo.value;
  if (from && to) parts.push(from === to ? `week ${from}` : `weeks ${from}–${to}`);
  else if (from) parts.push(`week ${from} onwards`);
  else if (to) parts.push(`up to week ${to}`);
  return parts.join(", ");
}

const kindButton = (kind) => el.kindButtons.find((b) => b.dataset.kind === kind);

function syncFilterControls() {
  for (const button of el.kindButtons) button.setAttribute("aria-pressed", String(state.kinds.has(button.dataset.kind)));
  el.clearFilters.hidden = !filtersActive();
  const course = notesCourse();
  const showing = Boolean(course) && !el.q.value.trim() && el.course.value === course.slug;
  el.notesButton.hidden = !course;
  el.notesButton.setAttribute("aria-pressed", String(showing));
  el.notesButton.title = course ? `Read the study notes of ${course.title}` : "";
}

/** The course whose notes the Study notes button opens: the chosen one, or the only one there is. */
function notesCourse() {
  const slug = el.course.value || (state.courses.length === 1 ? state.courses[0].slug : "");
  const course = state.courses.find((c) => c.slug === slug);
  return course?.notes ? course : null;
}

function clearFilters() {
  el.course.value = "";
  state.lectureId = null;
  state.kinds.clear();
  fillWeeks();
  el.weekFrom.value = el.weekTo.value = "";
  runSearch();
}

// ---------- Searching ----------

function requestParams() {
  const params = new URLSearchParams();
  const q = el.q.value.trim();
  if (q) params.set("q", q);
  if (el.course.value) params.set("course", el.course.value);
  if (state.kinds.size) params.set("kind", KIND_ORDER.filter((k) => state.kinds.has(k)).join(","));
  if (el.weekFrom.value) params.set("week_from", el.weekFrom.value);
  if (el.weekTo.value) params.set("week_to", el.weekTo.value);
  if (!q && el.course.value && state.lectureId != null) params.set("lecture", state.lectureId);
  return params;
}

function scheduleSearch() {
  clearTimeout(state.timer);
  state.timer = setTimeout(runSearch, DEBOUNCE_MS);
}

async function runSearch() {
  clearTimeout(state.timer);
  syncFilterControls();
  const params = requestParams();
  const key = params.toString();
  history.replaceState(null, "", key ? `?${key}` : location.pathname);
  if (key === state.shownKey || key === state.pendingKey) return;

  state.controller?.abort();
  state.controller = null;
  state.pendingKey = null;
  const seq = ++state.viewSeq;
  if (!params.has("q")) {
    state.shownKey = key;
    if (el.course.value) showNotes(seq);
    else showHome();
    return;
  }

  const controller = new AbortController();
  state.controller = controller;
  state.pendingKey = key;
  el.resultsPane.setAttribute("aria-busy", "true");
  try {
    const data = await getJSON(`/api/search?${key}`, { signal: controller.signal });
    state.shownKey = key;
    renderResults(data);
  } catch (err) {
    if (err.name === "AbortError") return;
    state.shownKey = null;
    renderProblem("The search didn't work", problemText(err, "The search"));
  } finally {
    if (state.controller === controller) {
      state.controller = null;
      state.pendingKey = null;
      el.resultsPane.removeAttribute("aria-busy");
    }
  }
}

function searchFor(text) {
  el.q.value = text;
  runSearch();
  el.q.focus();
}

// ---------- Results ----------

function whereShort(hit) {
  if (hit.kind === "video") return timeRange(hit.t_start_ms, hit.t_end_ms);
  if (hit.slide_no != null) return `Slide ${hit.slide_no}`;
  if (hit.page_no != null) return `Page ${hit.page_no}`;
  return "";
}

function courseLine(item) {
  const when = item.week != null ? `week ${item.week}` : item.date || "";
  return [courseName(item.course_slug, item.course_title), when].filter(Boolean).join(", ");
}

function renderResults(data) {
  state.terms = data.terms || [];
  const groups = new Map();
  for (const hit of data.results || []) {
    if (!groups.has(hit.source_id)) groups.set(hit.source_id, []);
    groups.get(hit.source_id).push(hit);
  }
  state.hits = [...groups.values()].flat();
  state.cursor = 0;

  renderSummary(data, groups.size);

  if (!state.hits.length) {
    renderNothingFound(data);
    return;
  }

  let index = 0;
  const list = h("ol", { class: "groups", role: "list" });
  for (const hits of groups.values()) {
    const first = hits[0];
    list.append(
      h(
        "li",
        { class: "group" },
        h(
          "div",
          { class: "group-head" },
          h("span", { class: "group-kind" }, KINDS[first.kind]?.label || first.kind, h("span", { class: "flag", "data-kind": first.kind })),
          h("span", { class: "group-title" }, h("span", { class: "title" }, first.title), h("span", { class: "meta" }, courseLine(first))),
        ),
        h("ul", { class: "hits", role: "list" }, hits.map((hit) => h("li", {}, hitButton(hit, index++)))),
      ),
    );
  }
  el.results.replaceChildren(list);
  // Keep the open result marked when it is still in the list.
  const open = state.openHit ? state.hits.findIndex((x) => x.passage_id === state.openHit.passage_id) : -1;
  if (open >= 0) state.cursor = open;
  markOpen(open);
  setCursor(state.cursor);
  el.resultsPane.scrollTop = 0;
}

function hitButton(hit, index) {
  const tags = h("span", { class: "tags" });
  if (hit.heading_parts?.length) tags.append(h("span", { class: "heading-path" }, headingPath(hit)));
  if (ORIGINS[hit.origin]) tags.append(h("span", { class: "tag" }, ORIGINS[hit.origin]));
  if (hit.match === "partial") tags.append(h("span", { class: "tag partial", title: "Your word appears inside a longer word" }, "Part of a word"));
  const snippet = h("span", { class: "snippet" });
  snippet.append(snippetNodes(hit.snippet_html || ""));
  return h(
    "button",
    { type: "button", class: "hit", tabindex: -1, "data-index": index },
    h("span", { class: "where" }, whereShort(hit)),
    h("span", { class: "body" }, tags, snippet),
  );
}

function renderSummary(data, files) {
  const lines = [];
  if (data.corrected && data.suggestion) {
    lines.push(h("p", {}, `No results for “${data.query}”. Showing results for “`, h("strong", {}, data.suggestion), "”."));
  } else if (data.suggestion) {
    const link = h("button", { type: "button", class: "text-button" }, data.suggestion);
    link.addEventListener("click", () => searchFor(data.suggestion));
    lines.push(h("p", {}, "Did you mean ", link, "?"));
  }
  if (data.count) {
    const shown = data.limited ? "The first " : "";
    const more = data.limited ? " are shown; add words or filters to narrow them down." : "";
    lines.push(h("p", {}, shown, h("strong", {}, plural(data.count, ["result", "results"])), ` in ${plural(files, ["file", "files"])}${more}`));
  }
  el.summary.replaceChildren(...lines);
}

function renderNothingFound(data) {
  const note = h(
    "div",
    { class: "note" },
    h("h2", {}, `Nothing found for “${data.query}”`),
    h("p", {}, "Check the spelling, try fewer words, or try a word the lecturer would have used."),
  );
  if (filtersActive()) {
    const all = h("button", { type: "button", class: "text-button" }, "Search all files");
    all.addEventListener("click", clearFilters);
    note.append(h("p", {}, `Only searching ${filtersText()}. `, all));
  }
  el.results.replaceChildren(note);
}

function renderProblem(title, text) {
  state.hits = [];
  el.summary.replaceChildren();
  el.results.replaceChildren(h("div", { class: "note error", role: "alert" }, h("h2", {}, title), h("p", {}, text)));
}

function showHome() {
  state.hits = [];
  el.summary.replaceChildren();
  const library = el.libraryName.dataset.library;
  const note = h(
    "div",
    { class: "note home" },
    h("h2", {}, library ? `Search everything in ${library}` : "Search everything in your courses"),
    h("p", {}, "Type a word or a “quoted phrase”. Each result shows the moment it was said in a lecture, or the slide or page it is on."),
  );
  if (state.coursesProblem) note.append(h("p", { class: "alert", role: "alert" }, state.coursesProblem));
  const groups = libraryGroups(state.courses);
  for (const group of groups) {
    if (groups.length > 1) note.append(h("h3", { class: "library-name" }, group.library));
    note.append(
      h(
        "ul",
        { role: "list" },
        group.courses.map((course) =>
          h(
            "li",
            {},
            h("button", { type: "button", class: "course-name", "data-course": course.slug }, course.title),
            h("span", { class: "course-files" }, courseFacts(course, { notes: false })),
            course.notes
              ? h("button", { type: "button", class: "text-button course-notes", "data-course": course.slug }, `Read the study notes (${plural(course.notes, ["lecture", "lectures"])})`)
              : null,
          ),
        ),
      ),
    );
  }
  const notes = state.courses.some((c) => c.notes) ? " The Study notes button next to the course list opens a course's notes." : "";
  note.append(h("p", { class: "tips" }, `Misspelled words get a “Did you mean” suggestion. Filters narrow the search to one course, kind of file or range of weeks.${notes}`));
  el.results.replaceChildren(note);
}

function courseFacts(course, { notes: withNotes = true } = {}) {
  const files = KIND_ORDER.filter((k) => course.files?.[k]).map((k) => plural(course.files[k], KINDS[k].noun));
  const notes = withNotes && course.notes ? `Study notes for ${plural(course.notes, ["lecture", "lectures"])}` : "";
  return [listWords(files), weeksText(course.weeks || []), notes].filter(Boolean).join(". ") + ".";
}

// ---------- Study notes ----------
//
// With a course chosen and nothing typed, the left pane shows the course's study notes:
// its lectures, then one lecture's notes. They were written by AI before the exam, so they
// are labelled as such, and every time in them plays the lecture from that moment with the
// transcript below: the original, next to the notes.

const NOTE_LISTS = [
  ["key_points", "Key points"],
  ["key_terms", "Key terms"],
  ["examples", "Examples"],
  ["emphasis", "Stressed in the lecture"],
];

async function showNotes(seq) {
  const lectureId = state.lectureId;
  state.hits = [];
  el.summary.replaceChildren();
  el.resultsPane.setAttribute("aria-busy", "true");
  const url = lectureId != null ? `/api/notes/${lectureId}` : `/api/notes?${new URLSearchParams({ course: el.course.value })}`;
  try {
    const data = await getJSON(url);
    if (seq !== state.viewSeq) return;
    if (lectureId != null) renderLecture(data);
    else renderCourse(data);
  } catch (err) {
    if (seq !== state.viewSeq) return;
    state.shownKey = null;
    if (lectureId != null && err instanceof ServerError) {
      // An old link to a lecture whose notes are not in this index: show the course instead.
      state.lectureId = null;
      runSearch();
      return;
    }
    renderProblem("The study notes didn't load", problemText(err, "Loading the study notes"));
  } finally {
    if (seq === state.viewSeq) el.resultsPane.removeAttribute("aria-busy");
  }
}

/** Show a course's notes, or one lecture's, as a step the browser's Back button undoes. */
function goToNotes(slug, lectureId = null) {
  if (el.course.value !== slug) {
    el.course.value = slug;
    fillWeeks();
  }
  if (lectureId == null) state.returnTo = state.lectureId;
  state.lectureId = lectureId;
  const key = requestParams().toString();
  history.pushState(null, "", key ? `?${key}` : location.pathname);
  runSearch();
}

const lectureName = (sourceId) => state.lectures.get(sourceId)?.title ?? "";

function lectureFacts(lecture) {
  const when = lecture.week != null ? `week ${lecture.week}` : "";
  const length = lecture.duration_s ? clock(lecture.duration_s * 1000) : "";
  return [lecture.title, when, lecture.date, length].filter(Boolean).join(" · ");
}

function aiLabel(text) {
  return h("p", { class: "ai-label" }, h("span", { class: "ai-tag" }, "AI"), text);
}

/** A time in the notes: it plays the lecture from there. Text only when the notes point nowhere. */
function timeLink(sourceId, item, label = null) {
  if (item.t_ms == null) return label;
  const at = clock(item.t_ms);
  return h(
    "button",
    {
      type: "button",
      class: "time-link",
      "data-source": sourceId,
      "data-passage": item.passage_id,
      "data-ms": item.t_ms,
      "aria-label": `Play ${label ? `${label} ` : "the lecture "}from ${at}`,
    },
    label ? `${label}, ${at}` : at,
  );
}

/** One point of the notes: its time in the margin, the text to the right. */
function noteRow(sourceId, item, ...body) {
  return h("li", { class: "note-row" }, h("span", { class: "where" }, timeLink(sourceId, item)), h("div", { class: "body" }, ...body));
}

function notesBlock(title, ...content) {
  return h("section", { class: "notes-block" }, h("h3", { class: "notes-label" }, title), ...content);
}

function renderCourse(data) {
  for (const lecture of data.lectures) state.lectures.set(lecture.source_id, lecture);
  const course = state.courses.find((c) => c.slug === data.course.slug);
  const head = h("header", { class: "notes-head" }, h("h2", { tabindex: -1 }, data.course.title));
  if (course) head.append(h("p", { class: "notes-facts" }, courseFacts(course)));
  const page = h("article", { class: "notes" }, head);

  const missing = data.videos - data.lectures.length;
  if (!data.lectures.length) {
    head.append(
      h("p", { class: "notes-missing" }, data.videos ? `None of its ${plural(data.videos, ["video", "videos"])} has study notes in this index. Write them before the exam with ` : "This course has no lecture videos, so there are no study notes.", data.videos ? [h("code", {}, `studybuddy notes ${data.course.slug}`), "."] : null),
      h("p", { class: "notes-missing" }, "Type in the search box to search the course."),
    );
    showPage(page);
    return;
  }
  head.append(aiLabel(`Study notes written by AI from the lecture transcripts before the exam. Each time plays the lecture from that moment, with the transcript below it, so you can check the notes against what was said.`));
  if (missing > 0) head.append(h("p", { class: "notes-missing" }, `${plural(missing, ["video has", "videos have"])} no study notes.`));

  page.append(
    notesBlock(
      "Lectures",
      h(
        "ol",
        { class: "lecture-list", role: "list" },
        data.lectures.map((lecture, i) =>
          h(
            "li",
            {},
            h(
              "button",
              { type: "button", class: "lecture", "data-lecture": lecture.source_id },
              h("span", { class: "where" }, lecture.week != null ? `Week ${lecture.week}` : String(i + 1)),
              h(
                "span",
                { class: "body" },
                h("span", { class: "lecture-title" }, lecture.notes_title),
                h("span", { class: "lecture-meta" }, lectureFacts(lecture)),
                h("span", { class: "lecture-overview" }, lecture.overview),
              ),
            ),
          ),
        ),
      ),
    ),
  );
  if (data.announcements.length) {
    page.append(
      notesBlock(
        "Announcements",
        h(
          "ul",
          { class: "note-rows", role: "list" },
          data.announcements.map((a) => noteRow(a.source_id, a, h("p", { class: "note-text" }, a.text), h("span", { class: "note-source" }, lectureName(a.source_id)))),
        ),
      ),
    );
  }
  if (data.glossary.length) page.append(glossary(data.glossary));
  showPage(page);

  const back = state.returnTo != null && el.results.querySelector(`.lecture[data-lecture="${state.returnTo}"]`);
  state.returnTo = null;
  if (back) {
    back.scrollIntoView({ block: "center" });
    back.focus({ preventScroll: true });
  }
}

function glossary(entries) {
  const letters = new Map();
  for (const entry of entries) {
    if (!letters.has(entry.letter)) letters.set(entry.letter, []);
    letters.get(entry.letter).push(entry);
  }
  const jumps = h(
    "nav",
    { class: "letters", "aria-label": "Key terms by letter" },
    [...letters.keys()].map((letter) => h("button", { type: "button", class: "letter-jump", "data-letter": letter }, letter)),
  );
  const groups = h(
    "ul",
    { class: "note-rows", role: "list" },
    [...letters].map(([letter, terms]) =>
      h(
        "li",
        { class: "note-row letter-group", id: `letter-${letter}` },
        h("span", { class: "where letter" }, letter),
        h(
          "ul",
          { class: "body terms", role: "list" },
          terms.map((term) =>
            h(
              "li",
              {},
              h("p", { class: "note-text" }, h("strong", {}, term.term), `: ${term.definition}`),
              h("span", { class: "places" }, term.places.map((place) => timeLink(place.source_id, place, lectureName(place.source_id)))),
            ),
          ),
        ),
      ),
    ),
  );
  return notesBlock(`Key terms A–Z (${entries.length})`, jumps, groups);
}

function renderLecture(data) {
  state.lectures.set(data.source_id, data);
  const id = data.source_id;
  const written = data.written_at ? new Date(data.written_at) : null;
  const writtenText = written && !Number.isNaN(written.getTime()) ? `, ${written.toLocaleDateString(undefined, { dateStyle: "medium" })}` : "";
  const step = (lectureId, text) => h("button", { type: "button", class: "tool-button", "data-lecture": lectureId, disabled: lectureId == null }, text);
  const page = h(
    "article",
    { class: "notes" },
    h(
      "nav",
      { class: "notes-nav", "aria-label": "Study notes" },
      h("button", { type: "button", class: "text-button", "data-course": data.course_slug, title: `All the study notes of ${data.course_title}` }, "All lectures"),
      h("span", { class: "notes-steps" }, step(data.previous_id, "Previous lecture"), step(data.next_id, "Next lecture")),
    ),
    h(
      "header",
      { class: "notes-head" },
      h("h2", { tabindex: -1 }, data.notes_title),
      h("p", { class: "notes-facts" }, lectureFacts(data)),
      aiLabel(`Written by AI (${data.model}) from the transcript${writtenText}. Each time plays the lecture from that moment, with the transcript below it: check anything important there.`),
      h("p", { class: "note-text overview" }, data.overview),
    ),
  );
  if (data.announcements.length) {
    page.append(notesBlock("Announcements", h("ul", { class: "note-rows", role: "list" }, data.announcements.map((a) => noteRow(id, a, h("p", { class: "note-text" }, a.text))))));
  }
  if (data.sections.length > 1) {
    page.append(
      notesBlock(
        "Contents",
        h(
          "ol",
          { class: "note-rows", role: "list" },
          data.sections.map((s, i) => noteRow(id, s, h("button", { type: "button", class: "text-button", "data-section": `notes-s${i + 1}` }, `${i + 1}. ${s.title}`))),
        ),
      ),
    );
  }
  data.sections.forEach((s, i) => page.append(noteSection(id, s, i + 1)));
  showPage(page);
}

function noteSection(id, s, number) {
  const section = h(
    "section",
    { class: "notes-section", id: `notes-s${number}` },
    h("h3", { class: "note-row section-title" }, h("span", { class: "where" }, timeLink(id, s)), h("span", { class: "body" }, `${number}. ${s.title}`)),
  );
  if (s.summary) section.append(h("p", { class: "note-text section-summary" }, s.summary));
  const empty = !s.summary && NOTE_LISTS.every(([key]) => !s[key].length);
  if (empty) section.append(h("p", { class: "notes-missing section-summary" }, "No notes were written for this section."));
  for (const [key, title] of NOTE_LISTS) {
    if (!s[key].length) continue;
    section.append(
      h("h4", { class: "notes-label" }, title),
      h(
        "ul",
        { class: "note-rows", role: "list" },
        s[key].map((item) => {
          if (key === "key_terms") {
            return noteRow(
              id,
              item,
              h("p", { class: "note-text" }, h("strong", {}, item.term), `: ${item.definition}`),
              item.quote ? h("blockquote", { class: "note-quote" }, `“${item.quote}”`) : null,
            );
          }
          const details = item.details?.length ? h("ul", { class: "note-details" }, item.details.map((d) => h("li", {}, d))) : null;
          return noteRow(id, item, h("p", { class: "note-text" }, item.text), details);
        }),
      ),
    );
  }
  if (s.review_questions.length) {
    section.append(
      h("h4", { class: "notes-label" }, "Review questions"),
      h(
        "ul",
        { class: "note-rows", role: "list" },
        s.review_questions.map((q) => noteRow(id, q, h("details", { class: "question" }, h("summary", {}, q.question), h("p", { class: "answer" }, q.answer)))),
      ),
    );
  }
  return section;
}

function showPage(page) {
  // Keep the keyboard where it was unless the button that was used has just been replaced.
  const moveFocus = document.activeElement === document.body || el.results.contains(document.activeElement);
  el.results.replaceChildren(page);
  el.resultsPane.scrollTop = 0;
  if (moveFocus) page.querySelector("h2")?.focus({ preventScroll: true });
}

/** A time in the notes was chosen: play the lecture from there, and show the transcript. */
function openMoment(button) {
  const lecture = state.lectures.get(Number(button.dataset.source));
  if (!lecture) return;
  const ms = Number(button.dataset.ms);
  state.openHit = null;
  for (const other of el.results.querySelectorAll(".time-link.is-open")) {
    other.classList.remove("is-open");
    other.removeAttribute("aria-current");
  }
  button.classList.add("is-open");
  button.setAttribute("aria-current", "true");
  showHead(lecture);
  if (!lecture.has_view) {
    state.replay = null;
    showStage("none");
  } else {
    showStage("video");
    state.replay = { sourceId: lecture.source_id, fileUrl: lecture.file_url, ms };
    el.replayHit.textContent = "Play this moment again";
    el.videoRange.textContent = `From the notes: ${clock(ms)}`;
    playVideo(lecture.source_id, lecture.file_url, ms);
  }
  if (button.dataset.passage) loadPassage(Number(button.dataset.passage));
  if (narrowScreen.matches) document.querySelector(".viewer-pane").scrollIntoView({ block: "start" });
}

function setCursor(index) {
  state.cursor = index;
  el.results.querySelectorAll(".hit").forEach((b, i) => (b.tabIndex = i === index ? 0 : -1));
}

function focusHit(index) {
  const buttons = el.results.querySelectorAll(".hit");
  if (!buttons.length) return;
  const i = Math.max(0, Math.min(index, buttons.length - 1));
  setCursor(i);
  buttons[i].focus();
  buttons[i].scrollIntoView({ block: "nearest" });
}

function markOpen(index) {
  el.results.querySelectorAll(".hit").forEach((b, i) => {
    b.classList.toggle("is-open", i === index);
    if (i === index) b.setAttribute("aria-current", "true");
    else b.removeAttribute("aria-current");
  });
}

function focusSearch(selectAll = true) {
  el.q.focus();
  if (selectAll) el.q.select();
  else el.q.setSelectionRange(el.q.value.length, el.q.value.length);
}

// ---------- Viewer ----------

function showStage(which) {
  el.stageEmpty.hidden = which !== "empty";
  el.videoWrap.hidden = which !== "video";
  el.frame.hidden = which !== "doc";
  el.noView.hidden = which !== "none";
  if (which !== "video") el.video.pause();
}

function showHead(item) {
  el.viewerHead.hidden = false;
  el.viewerFlag.dataset.kind = item.kind;
  el.viewerKind.textContent = KINDS[item.kind]?.label || item.kind;
  el.viewerTitle.textContent = item.title ?? item.source_title;
  el.viewerMeta.textContent = courseLine(item);
}

function openHit(index) {
  const hit = state.hits[index];
  if (!hit) return;
  state.openHit = hit;
  markOpen(index);
  setCursor(index);
  showHead(hit);
  if (!hit.has_view) {
    showStage("none");
  } else if (hit.kind === "video") {
    showStage("video");
    state.replay = { sourceId: hit.source_id, fileUrl: hit.file_url, ms: hit.t_start_ms };
    el.replayHit.textContent = "Play match again";
    el.videoRange.textContent = `Match at ${timeRange(hit.t_start_ms, hit.t_end_ms)}`;
    playVideo(hit.source_id, hit.file_url, hit.t_start_ms);
  } else {
    showStage("doc");
    showDocument(hit.source_id, hit.open_url, hit.slide_no ?? hit.page_no ?? 1, hit.kind);
  }
  loadPassage(hit.passage_id);
  if (narrowScreen.matches) document.querySelector(".viewer-pane").scrollIntoView({ block: "start" });
}

// One <video> for every video hit: change the source only when the file changes, and seek
// once its metadata has loaded.
function playVideo(sourceId, fileUrl, startMs) {
  el.videoError.hidden = true;
  state.pendingSeek = Math.max(0, (startMs || 0) / 1000);
  if (state.videoSource !== sourceId) {
    state.videoSource = sourceId;
    el.video.src = fileUrl;
  } else if (el.video.readyState >= HTMLMediaElement.HAVE_METADATA) {
    applySeek();
  }
}

function applySeek() {
  if (state.pendingSeek == null) return;
  el.video.currentTime = state.pendingSeek;
  state.pendingSeek = null;
  el.video.play().catch(() => {
    // Chrome may refuse to start playing without a click; the controls are there.
  });
}

function seekBy(seconds) {
  if (el.video.readyState < HTMLMediaElement.HAVE_METADATA) return;
  el.video.currentTime = Math.max(0, el.video.currentTime + seconds);
}

const pdfApp = () => {
  try {
    return el.frame.contentWindow?.PDFViewerApplication || null;
  } catch {
    return null;
  }
};

// PDF.js opens at #page=N and then finds the words. Finding scrolls the first match to the
// top of the viewer, and jumps to another page when page N has no exact match (a scanned
// page, a stemmed word). The pin brings page N back once that scroll has happened.
function showDocument(sourceId, url, page, kind) {
  let [base, fragment = ""] = url.split("#");
  if (kind === "pptx" && !/(^|&)zoom=/.test(fragment)) fragment += `${fragment ? "&" : ""}zoom=page-fit`;
  state.docPin = { page: Number(page) || 1, until: Date.now() + PIN_FOR_MS };
  const app = pdfApp();
  if (state.docSource === sourceId && app?.pdfDocument) {
    app.pdfLinkService.setHash(fragment);
    return;
  }
  state.docSource = sourceId;
  el.frame.src = `${base}#${fragment}`;
}

function documentUrl(fileUrl, page) {
  const params = new URLSearchParams({ page });
  if (state.terms.length) params.set("search", state.terms.join(" "));
  const fragment = params.toString().replace(/\+/g, "%20");
  return `/pdfjs/web/viewer.html?file=${encodeURIComponent(fileUrl)}#${fragment}`;
}

function keepPinnedPage(app) {
  const pin = state.docPin;
  state.docPin = null;
  // Always set it: PDF.js updates app.page only on its next scroll event, so it may still
  // read as the pinned page although the match scroll has just moved the view.
  if (pin && Date.now() < pin.until) app.page = pin.page;
}

async function onFrameLoad() {
  const app = pdfApp();
  if (!app) return;
  try {
    el.frame.contentWindow.addEventListener("keydown", onFrameKeydown);
    await app.initializedPromise;
    const finder = app.findController;
    if (!finder || finder.studybuddyPinned) return;
    finder.studybuddyPinned = true;
    const scroll = finder.scrollMatchIntoView.bind(finder);
    finder.scrollMatchIntoView = (args) => {
      const willScroll = finder._scrollMatches === true;
      scroll(args);
      if (willScroll && !finder._scrollMatches) keepPinnedPage(app);
    };
    app.eventBus.on("updatefindcontrolstate", (event) => {
      if (event.state === FIND_NOT_FOUND) state.docPin = null;
    });
  } catch {
    // An unexpected PDF.js version: the viewer still works, just without the pin.
  }
}

function onFrameKeydown(event) {
  const target = event.target;
  if (event.key === "/" && !target.closest?.("input, textarea, select, [contenteditable]")) {
    event.preventDefault();
    focusSearch();
  }
}

// ---------- Passage panel ----------

async function loadPassage(id) {
  const seq = ++state.passageSeq;
  el.passage.hidden = false;
  el.passageText.classList.add("is-loading");
  try {
    const passage = await getJSON(`/api/passage/${id}`);
    if (seq === state.passageSeq) renderPassage(passage);
  } catch (err) {
    if (seq !== state.passageSeq) return;
    state.passage = null;
    el.passageText.classList.remove("is-loading");
    el.passageText.replaceChildren(h("p", { role: "alert" }, problemText(err, "Loading the full text")));
  }
}

function passageWhere(p) {
  if (p.kind === "video") return [h("strong", {}, "Transcript"), ` ${timeRange(p.t_start_ms, p.t_end_ms)}`];
  const place = p.slide_no != null ? `Slide ${p.slide_no}` : p.page_no != null ? `Page ${p.page_no}` : "Text";
  const extra = [];
  if (ORIGINS[p.origin]) extra.push(ORIGINS[p.origin].toLowerCase());
  if (p.heading_parts?.length) extra.push(headingPath(p));
  return [h("strong", {}, place), extra.length ? `, ${extra.join(", ")}` : ""];
}

function renderPassage(p) {
  state.passage = p;
  el.passageWhere.replaceChildren(...passageWhere(p));
  el.passageText.classList.remove("is-loading");
  el.passageText.replaceChildren(h("p", {}, highlighted(p.text || "", state.terms)));
  el.passageText.scrollTop = 0;
  el.pPrev.disabled = p.previous_id == null;
  el.pNext.disabled = p.next_id == null;

  if (!p.has_view) {
    el.pGo.hidden = true;
  } else if (p.kind === "video") {
    el.pGo.hidden = false;
    el.pGo.textContent = `Play from ${clock(p.t_start_ms)}`;
  } else {
    const n = p.slide_no ?? p.page_no;
    el.pGo.hidden = n == null;
    el.pGo.textContent = p.slide_no != null ? `Show slide ${n}` : `Show page ${n}`;
  }
}

function goToPassage() {
  const p = state.passage;
  if (!p?.has_view) return;
  showHead(p);
  if (p.kind === "video") {
    showStage("video");
    playVideo(p.source_id, p.file_url, p.t_start_ms);
  } else {
    const page = p.slide_no ?? p.page_no ?? 1;
    showStage("doc");
    showDocument(p.source_id, documentUrl(p.file_url, page), page, p.kind);
  }
}

// ---------- Status line ----------

async function loadStatus() {
  try {
    renderStatus(await getJSON("/api/status"));
  } catch {
    el.statusLine.classList.add("is-warning");
    el.statusLine.textContent = "Can't reach the exam server. Check that studybuddy-exam is still running.";
  }
}

function renderStatus(s) {
  const guard = s.guard || {};
  const loaded = guard.ai_modules_loaded || [];
  const library = s.library?.name || "";
  if (library && el.libraryName.dataset.library !== library) {
    el.libraryName.textContent = library;
    el.libraryName.dataset.library = library;
    if (!el.q.value.trim() && el.results.querySelector(".note.home")) showHome();
  }

  if (!guard.installed || loaded.length) {
    const text = loaded.length
      ? `Warning: AI libraries are loaded in the exam server (${loaded.join(", ")}). Quit it and start it again with studybuddy-exam.`
      : "Warning: the no-AI guard is not running, so AI libraries and outside network access are not blocked. Quit the server and start it again with studybuddy-exam.";
    el.guardAlert.textContent = text;
    el.guardAlert.hidden = false;
    el.statusLine.classList.add("is-warning");
    el.statusLine.textContent = text;
    return;
  }

  el.guardAlert.hidden = true;
  el.statusLine.classList.remove("is-warning");
  const sha = s.index?.sha256 || "";
  // Several libraries searched as one: each has its own index, and the tooltip lists their fingerprints.
  const parts = s.libraries || [];
  el.statusLine.replaceChildren(
    h("span", { class: "ok" }, "No AI"),
    parts.length
      ? ` · Offline · ${library} · ${plural(parts.length, ["library", "libraries"])}`
      : ` · Offline · ${library} · index ${sha.slice(0, 12)} · built ${builtTime(s.index?.built_at)}`,
  );
  el.statusLine.title = [
    `AI libraries loaded: none`,
    `Network: ${guard.network || "loopback only"}`,
    `Listening on ${guard.listening || ""}`,
    ...(parts.length
      ? parts.map((p) => `${p.name}: index SHA-256 ${p.sha256.slice(0, 16)}…, built ${builtTime(p.built_at)}`)
      : [`Index SHA-256: ${sha}`]),
    s.index ? `${s.index.courses} courses, ${s.index.sources} files, ${s.index.passages} passages, study notes for ${s.index.notes ?? 0} lectures` : "",
  ]
    .filter(Boolean)
    .join("\n");
}

function builtTime(iso) {
  const built = iso ? new Date(iso) : null;
  return built && !Number.isNaN(built.getTime()) ? built.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "unknown";
}

// ---------- Keyboard ----------

const isField = (node) => Boolean(node?.closest?.("input, select, textarea, [contenteditable]"));

function onKeydown(event) {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
  const target = event.target;

  if (event.key === "/" && !isField(target)) {
    event.preventDefault();
    focusSearch();
    return;
  }

  if (target === el.q) {
    if (event.key === "ArrowDown" && state.hits.length) {
      event.preventDefault();
      focusHit(state.cursor);
    }
    return;
  }

  const hit = target.closest?.(".hit");
  const inResults = hit || target === document.body || target === el.resultsPane;
  if (!inResults || !state.hits.length) return;
  const index = Number(hit?.dataset.index);
  switch (event.key) {
    case "ArrowDown":
      event.preventDefault();
      focusHit(hit ? index + 1 : state.cursor);
      break;
    case "ArrowUp":
      event.preventDefault();
      if (!hit) focusHit(state.cursor);
      else if (index === 0) focusSearch(false);
      else focusHit(index - 1);
      break;
    case "Home":
      event.preventDefault();
      focusHit(0);
      break;
    case "End":
      event.preventDefault();
      focusHit(state.hits.length - 1);
      break;
    case "Escape":
      event.preventDefault();
      focusSearch();
      break;
    default:
      // Typing a letter while in the results goes on with the search.
      if (hit && event.key.length === 1 && event.key !== " ") focusSearch(false);
  }
}

// ---------- Wiring ----------

function applyUrl() {
  const params = new URLSearchParams(location.search);
  el.q.value = params.get("q") || "";
  const course = params.get("course") || "";
  el.course.value = [...el.course.options].some((o) => o.value === course) ? course : "";
  const lecture = Number(params.get("lecture"));
  state.lectureId = el.course.value && Number.isInteger(lecture) && lecture > 0 ? lecture : null;
  fillWeeks();
  state.kinds.clear();
  for (const kind of (params.get("kind") || "").split(",")) if (KINDS[kind]) state.kinds.add(kind);
  for (const [name, select] of [["week_from", el.weekFrom], ["week_to", el.weekTo]]) {
    const week = params.get(name) || "";
    if ([...select.options].some((o) => o.value === week)) select.value = week;
  }
}

function wire() {
  el.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const key = requestParams().toString();
    // Enter searches; when the results for this search are already showing, it opens the first.
    if (key === state.shownKey && state.hits.length) openHit(state.cursor);
    else runSearch();
  });
  el.q.addEventListener("input", scheduleSearch);

  el.course.addEventListener("change", () => {
    state.lectureId = null;
    fillWeeks();
    runSearch();
  });
  el.weekFrom.addEventListener("change", () => {
    if (el.weekTo.value && Number(el.weekFrom.value) > Number(el.weekTo.value)) el.weekTo.value = el.weekFrom.value;
    runSearch();
  });
  el.weekTo.addEventListener("change", () => {
    if (el.weekFrom.value && el.weekTo.value && Number(el.weekTo.value) < Number(el.weekFrom.value)) el.weekFrom.value = el.weekTo.value;
    runSearch();
  });
  for (const button of el.kindButtons) {
    button.addEventListener("click", () => {
      const kind = button.dataset.kind;
      if (state.kinds.has(kind)) state.kinds.delete(kind);
      else state.kinds.add(kind);
      runSearch();
    });
  }
  el.clearFilters.addEventListener("click", clearFilters);
  el.notesButton.addEventListener("click", () => {
    const course = notesCourse();
    if (!course) return;
    el.q.value = "";
    goToNotes(course.slug);
  });

  el.results.addEventListener("click", (event) => {
    const target = event.target;
    const hit = target.closest(".hit");
    if (hit) return openHit(Number(hit.dataset.index));
    const time = target.closest(".time-link");
    if (time) return openMoment(time);
    const lecture = target.closest("[data-lecture]");
    if (lecture) return goToNotes(el.course.value, Number(lecture.dataset.lecture));
    const course = target.closest("[data-course]");
    if (course) return goToNotes(course.dataset.course);
    const jump = target.closest("[data-section]") || target.closest("[data-letter]");
    if (jump) {
      const id = jump.dataset.section ?? `letter-${jump.dataset.letter}`;
      document.getElementById(id)?.scrollIntoView({ block: "start" });
    }
  });
  el.results.addEventListener("focusin", (event) => {
    const hit = event.target.closest(".hit");
    if (hit) setCursor(Number(hit.dataset.index));
  });

  el.video.addEventListener("loadedmetadata", applySeek);
  el.video.addEventListener("error", () => {
    if (!el.video.getAttribute("src")) return;
    el.videoError.textContent = "This video can't be played. The file may be missing from the library.";
    el.videoError.hidden = false;
  });
  el.back10.addEventListener("click", () => seekBy(-10));
  el.replayHit.addEventListener("click", () => {
    const replay = state.replay;
    if (replay) playVideo(replay.sourceId, replay.fileUrl, replay.ms);
  });

  el.frame.addEventListener("load", onFrameLoad);

  // Back and Forward move between the pages of study notes (goToNotes).
  window.addEventListener("popstate", () => {
    state.returnTo = state.lectureId;
    applyUrl();
    runSearch();
  });

  el.pPrev.addEventListener("click", () => state.passage?.previous_id != null && loadPassage(state.passage.previous_id));
  el.pNext.addEventListener("click", () => state.passage?.next_id != null && loadPassage(state.passage.next_id));
  el.pGo.addEventListener("click", goToPassage);

  document.addEventListener("keydown", onKeydown);
}

async function start() {
  wire();
  focusSearch();
  showStage("empty");
  loadStatus();
  setInterval(loadStatus, STATUS_EVERY_MS);
  try {
    fillCourses(await getJSON("/api/courses"));
  } catch (err) {
    state.coursesProblem = problemText(err, "Loading the course list");
  }
  applyUrl();
  if (el.q.value) focusSearch(false);
  state.shownKey = null;
  await runSearch();
}

start();
