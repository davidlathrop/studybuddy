"""Study notes from a lecture transcript, written by Claude and checked before use.

Every call shares one cached prefix (the instructions plus the whole transcript as numbered
passages, as in plan §6.1):

1. **Outline**, once per lecture: a title, an overview, consecutive sections and any
   announcements about exams or assignments.
2. **Sections**, once per group of sections covering about 20 minutes of lecture: a
   summary, key points, key terms with a verbatim quote, examples, what the lecturer
   stressed, and review questions with answers.

The model refers to passages by id only; times are looked up from the transcript when the
notes are rendered. Checking is deterministic: sections are made to cover every passage
in order, a group that skips sections is asked once more for them, and a key term's quote
is kept only when it is found in the transcript (plan §6.6).
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from studybuddy.enrich import load_prompt
from studybuddy.enrich.backends import BackendResult, EnrichmentBackend, EnrichmentError, Usage
from studybuddy.enrich.backends.base import T
from studybuddy.enrich.passages import Passage, clock, render_transcript
from studybuddy.enrich.schemas import (
    Announcement,
    KeyPoint,
    KeyTerm,
    NoteItem,
    NotesOutline,
    NotesSectionPlan,
    NotesWindow,
    ReviewQuestion,
    SectionNotes,
)
from studybuddy.enrich.verify import check_quote, normalize
from studybuddy.types import ProgressFn

log = logging.getLogger(__name__)

VERSION = "notes_v1"  # names the prompts below; notes made with another version are redone
SCHEMA_VERSION = "1"
PROMPTS = {"system": "notes_system_v1", "outline": "notes_outline_v1", "sections": "notes_sections_v1"}
GROUP_PASSAGES = 20  # passages per sections call: about 20 minutes of lecture


class Section(BaseModel):
    title: str
    first_passage_id: int
    last_passage_id: int
    summary: str = ""
    key_points: list[KeyPoint] = []
    key_terms: list[KeyTerm] = []
    examples: list[NoteItem] = []
    emphasis: list[NoteItem] = []
    review_questions: list[ReviewQuestion] = []


class StudyNotes(BaseModel):
    """The checked notes of one lecture, as stored and rendered."""

    title: str
    overview: str
    announcements: list[Announcement]
    sections: list[Section]
    times: list[tuple[float, float]]  # start and end (seconds) of P1, P2, ...

    def at(self, passage_id: int) -> str:
        """`` [12:40]``, the start of the passage, or nothing for an id that isn't one."""
        if 1 <= passage_id <= len(self.times):
            return f" [{clock(self.times[passage_id - 1][0])}]"
        return ""

    def span(self, section: Section) -> str:
        start = self.times[section.first_passage_id - 1][0]
        end = self.times[section.last_passage_id - 1][1]
        return f"{clock(start)}–{clock(end)}"


@dataclass
class NotesResult:
    notes: StudyNotes
    model: str
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    calls: int = 0
    seconds: float = 0.0
    quotes_dropped: int = 0  # key-term quotes not found in the transcript
    meta: dict[str, Any] = field(default_factory=dict)  # the last call's, e.g. allowance used

    def add(self, result: BackendResult[Any]) -> None:
        self.usage += result.usage
        if result.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + result.cost_usd
        self.calls += 1
        self.seconds += result.seconds
        self.meta = result.meta


def input_sha(title: str, passages: Sequence[Passage]) -> str:
    """Identifies the transcript as sent: notes made from the same input are not made again."""
    return hashlib.sha256(render_transcript(title, passages).encode("utf-8")).hexdigest()


def write_notes(
    passages: Sequence[Passage],
    title: str,
    *,
    backend: EnrichmentBackend,
    model: str,
    effort: str = "medium",
    on_progress: ProgressFn | None = None,
) -> NotesResult:
    """Have ``backend`` write study notes on ``passages`` (ids 1..n, in order) and check them."""
    if not passages:
        raise EnrichmentError(f"{title}: no speech to take notes on")
    if [p.id for p in passages] != list(range(1, len(passages) + 1)):
        raise ValueError("passages must be numbered 1, 2, ... in order")
    system = load_prompt(PROMPTS["system"])
    context = render_transcript(title, passages)
    progress = on_progress or (lambda fraction: None)
    result = NotesResult(notes=StudyNotes(title=title, overview="", announcements=[], sections=[], times=[]),
                         model=model)

    def ask(prompt: str, output: type[T], max_tokens: int) -> T:
        reply = backend.call(model=model, system=system, context=context, prompt=prompt, output=output,
                             effort=effort, max_tokens=max_tokens)
        result.add(reply)
        result.model = reply.model
        return reply.data

    outline = ask(load_prompt(PROMPTS["outline"]), NotesOutline, 16000)
    plans = fit_sections(outline.sections, len(passages), fallback_title=outline.title or title)
    groups = section_groups(plans, GROUP_PASSAGES)
    listing = "\n".join(f"- S{i} (P{s.first_passage_id}-P{s.last_passage_id}): {s.title}"
                        for i, s in enumerate(plans, start=1))
    progress(1 / (1 + len(groups)))

    written: dict[int, SectionNotes] = {}
    for done, group in enumerate(groups, start=1):
        for _ in range(2):  # a group that skips sections is asked once more for them
            missing = [n for n in group if n not in written]
            if not missing:
                break
            prompt = load_prompt(PROMPTS["sections"]).format(first=missing[0], last=missing[-1], sections=listing)
            for entry in ask(prompt, NotesWindow, 32000).sections:
                if entry.section in missing:
                    written.setdefault(entry.section, entry)
        progress((1 + done) / (1 + len(groups)))

    sections: list[Section] = []
    for number, plan in enumerate(plans, start=1):
        entry = written.get(number)
        if entry is None:
            log.warning("%s: no notes came back for section %d (%s)", title, number, plan.title)
            sections.append(Section(**plan.model_dump()))
            continue
        nearby = " ".join(p.text for p in passages[max(plan.first_passage_id - 2, 0): plan.last_passage_id + 1])
        terms = []
        for term in entry.key_terms:
            if term.quote and not check_quote(term.quote, nearby).verified:
                result.quotes_dropped += 1
                term = term.model_copy(update={"quote": ""})
            terms.append(term)
        sections.append(Section(
            **plan.model_dump(), summary=entry.summary, key_points=entry.key_points, key_terms=terms,
            examples=entry.examples, emphasis=entry.emphasis, review_questions=entry.review_questions,
        ))
    result.notes = StudyNotes(
        title=outline.title.strip() or title,
        overview=outline.overview,
        announcements=outline.announcements,
        sections=sections,
        times=[(p.start, p.end) for p in passages],
    )
    return result


def fit_sections(plans: Sequence[NotesSectionPlan], count: int, *, fallback_title: str) -> list[NotesSectionPlan]:
    """Sections that cover passages 1..``count`` in order, with no gaps or overlaps.

    Each section starts where the model said (moved into range) and ends where the next one
    starts; a section that doesn't start after the one before it is dropped.
    """
    starts: list[tuple[int, str]] = []
    for plan in plans:
        first = min(max(plan.first_passage_id, 1), count)
        if not starts or first > starts[-1][0]:
            starts.append((first, plan.title.strip() or fallback_title))
    if not starts:
        return [NotesSectionPlan(title=fallback_title, first_passage_id=1, last_passage_id=count)]
    starts[0] = (1, starts[0][1])
    ends = [first - 1 for first, _ in starts[1:]] + [count]
    return [NotesSectionPlan(title=t, first_passage_id=f, last_passage_id=e) for (f, t), e in zip(starts, ends)]


def section_groups(sections: Sequence[NotesSectionPlan], size: int) -> list[list[int]]:
    """Section numbers (from 1) grouped so that each call covers about ``size`` passages."""
    groups: list[list[int]] = []
    current: list[int] = []
    passages = 0
    for number, s in enumerate(sections, start=1):
        n = s.last_passage_id - s.first_passage_id + 1
        if current and passages + n > size:
            groups.append(current)
            current, passages = [], 0
        current.append(number)
        passages += n
    if current:
        groups.append(current)
    return groups


# --------------------------------------------------------------------------- Markdown


def _line(text: str) -> str:
    """Text on one line, so it can't break the list or heading it is in."""
    return " ".join(text.split())


def render_lecture(notes: StudyNotes, *, source: str, duration_s: float | None, model: str, written: str) -> str:
    """One lecture's notes as Markdown: overview, sections, then review questions and answers."""
    facts = " · ".join(x for x in (f"**{_line(source)}**", clock(duration_s) if duration_s else "",
                                   f"study notes by {model}, {written}") if x)
    out = [
        f"# {_line(notes.title)}", "", facts, "",
        "> Written by AI from a machine transcript. Times like [12:40] point to the moment in the "
        "lecture: check anything important there.", "",
        "## Overview", "", _line(notes.overview), "",
    ]
    if notes.announcements:
        out += ["## Announcements", ""]
        out += [f"- {_line(a.text)}{notes.at(a.passage_id)}" for a in notes.announcements] + [""]
    if len(notes.sections) > 1:
        out += ["## Contents", ""]
        out += [f"{i}. {_line(s.title)} [{notes.span(s)}]" for i, s in enumerate(notes.sections, start=1)] + [""]

    for i, s in enumerate(notes.sections, start=1):
        out += [f"## {i}. {_line(s.title)} [{notes.span(s)}]", ""]
        if not (s.summary or s.key_points or s.key_terms or s.examples):
            out += ["_No notes were written for this section._", ""]
            continue
        if s.summary:
            out += [_line(s.summary), ""]
        if s.key_points:
            out += ["### Key points", ""]
            for p in s.key_points:
                out.append(f"- {_line(p.point)}{notes.at(p.passage_id)}")
                out += [f"  - {_line(d)}" for d in p.details if d.strip()]
            out.append("")
        if s.key_terms:
            out += ["### Key terms", ""]
            for t in s.key_terms:
                out.append(f"- **{_line(t.term)}**: {_line(t.definition)}{notes.at(t.passage_id)}")
                if t.quote:
                    out.append(f"  > “{_line(t.quote)}”")
            out.append("")
        for heading, items in (("Examples", s.examples), ("Stressed in the lecture", s.emphasis)):
            if items:
                out += [f"### {heading}", ""] + [f"- {_line(x.text)}{notes.at(x.passage_id)}" for x in items] + [""]

    asked = [(i, s) for i, s in enumerate(notes.sections, start=1) if s.review_questions]
    if asked:
        out += ["## Review questions", ""]
        for i, s in asked:
            out += [f"### {i}. {_line(s.title)}", ""]
            out += [f"{n}. {_line(q.question)}" for n, q in enumerate(s.review_questions, start=1)] + [""]
        out += ["## Answers", ""]
        for i, s in asked:
            out += [f"### {i}. {_line(s.title)}", ""]
            out += [f"{n}. {_line(q.answer)}{notes.at(q.passage_id)}"
                    for n, q in enumerate(s.review_questions, start=1)] + [""]
    return "\n".join(out).rstrip() + "\n"


@dataclass(frozen=True)
class IndexEntry:
    source: str  # the video's title in the library
    link: str  # its notes file, relative to the index
    notes: StudyNotes
    week: int | None = None
    date: str | None = None
    duration_s: float | None = None


def render_index(course: str, entries: Sequence[IndexEntry]) -> str:
    """A course's notes at a glance: each lecture's overview, announcements and an A-Z glossary."""
    count = f"{len(entries)} lecture{'' if len(entries) == 1 else 's'}"
    out = [
        f"# {_line(course)}: study notes", "",
        f"{count}, with notes written by AI from machine transcripts. Each lecture's notes give the "
        "times to check against the recording.", "",
        "_studybuddy rewrites this page whenever it writes notes; edit the lecture files instead._", "",
        "## Lectures", "",
    ]
    for n, e in enumerate(entries, start=1):
        facts = [e.source, f"week {e.week}" if e.week else "", e.date or "", clock(e.duration_s) if e.duration_s else ""]
        out += [f"### {n}. [{_line(e.notes.title)}]({quote(e.link)})", "", " · ".join(_line(f) for f in facts if f), "",
                _line(e.notes.overview), ""]

    announced = [(e, a) for e in entries for a in e.notes.announcements]
    if announced:
        out += ["## Announcements", ""]
        out += [f"- {_line(a.text)} ({_line(e.source)}{e.notes.at(a.passage_id)})" for e, a in announced] + [""]

    terms: dict[str, list[tuple[IndexEntry, KeyTerm]]] = defaultdict(list)
    for e in entries:
        for s in e.notes.sections:
            for t in s.key_terms:
                if key := normalize(t.term):
                    terms[key].append((e, t))
    if terms:
        out += ["## Key terms A–Z", ""]
        letter = None
        for key in sorted(terms, key=str.casefold):
            initial = key[0].upper() if key[0].isalpha() else "#"
            if initial != letter:
                out += [f"### {initial}", ""] if letter is None else ["", f"### {initial}", ""]
                letter = initial
            (first, term), *more = terms[key]
            places = [f"{_line(first.source)}{first.notes.at(term.passage_id)}"]
            places += [f"{_line(e.source)}{e.notes.at(t.passage_id)}" for e, t in more]
            out.append(f"- **{_line(term.term)}**: {_line(term.definition)} ({'; '.join(dict.fromkeys(places))})")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
