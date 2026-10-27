"""Browsing the study notes in the exam index: a course's lectures, announcements and
glossary, and one lecture's notes.

The notes were written by AI before the exam (plan §6.8); here they are only read. The page
labels them as AI-written and links every point to the moment of the lecture it comes from,
so it can be checked against the recording (plan C4, C5). They are not searched.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

_NOT_WORD = re.compile(r"[^\w]+")


def course_notes(conn: sqlite3.Connection, slug: str) -> dict[str, Any] | None:
    """The course's lectures with notes (in lecture order), their announcements, and an A-Z glossary."""
    course = conn.execute("SELECT id, slug, title FROM course WHERE slug = ?", (slug,)).fetchone()
    if course is None:
        return None
    videos = conn.execute(
        "SELECT COUNT(*) FROM source WHERE course_id = ? AND kind = 'video'", (course["id"],)
    ).fetchone()[0]
    lectures, announcements = [], []
    terms: dict[str, dict[str, Any]] = {}
    for row in _lectures(conn, "s.course_id = ?", course["id"]):
        data = json.loads(row["data"])
        lectures.append(_lecture(row, data))
        announcements += [{**a, "source_id": row["source_id"]} for a in data["announcements"]]
        for section in data["sections"]:
            for term in section["key_terms"]:
                if not (key := _term_key(term["term"])):
                    continue
                entry = terms.setdefault(key, {
                    "term": term["term"], "definition": term["definition"],
                    "letter": key[0].upper() if key[0].isalpha() else "#", "places": [],
                })
                place = {"source_id": row["source_id"], "passage_id": term["passage_id"], "t_ms": term["t_ms"]}
                if place not in entry["places"]:
                    entry["places"].append(place)
    return {
        "course": {"slug": course["slug"], "title": course["title"]},
        "videos": videos,
        "lectures": lectures,
        "announcements": announcements,
        "glossary": [terms[key] for key in sorted(terms)],
    }


def lecture_notes(conn: sqlite3.Connection, source_id: int) -> dict[str, Any] | None:
    """One lecture's notes, with the lectures before and after it in its course."""
    rows = _lectures(conn, "s.id = ?", source_id)
    if not rows:
        return None
    row = rows[0]
    data = json.loads(row["data"])
    order = [r[0] for r in conn.execute(
        "SELECT n.source_id FROM notes n JOIN source s ON s.id = n.source_id WHERE s.course_id = ? ORDER BY n.position",
        (row["course_id"],),
    )]
    at = order.index(source_id)
    return {
        **_lecture(row, data),
        "previous_id": order[at - 1] if at > 0 else None,
        "next_id": order[at + 1] if at + 1 < len(order) else None,
        "announcements": data["announcements"],
        "sections": data["sections"],
    }


def _lectures(conn: sqlite3.Connection, where: str, value: int) -> list[sqlite3.Row]:
    return conn.execute(
        f"""SELECT n.source_id, n.title AS notes_title, n.model, n.written_at, n.data,
                   s.course_id, s.title, s.week, s.date, s.duration_s, s.view_path,
                   c.slug AS course_slug, c.title AS course_title
              FROM notes n JOIN source s ON s.id = n.source_id JOIN course c ON c.id = s.course_id
             WHERE {where} ORDER BY n.position""",
        (value,),
    ).fetchall()


def _lecture(row: sqlite3.Row, data: dict[str, Any]) -> dict[str, Any]:
    has_view = row["view_path"] is not None
    return {
        "source_id": row["source_id"],
        "kind": "video",
        "title": row["title"],
        "notes_title": row["notes_title"],
        "overview": data["overview"],
        "course_slug": row["course_slug"],
        "course_title": row["course_title"],
        "week": row["week"],
        "date": row["date"],
        "duration_s": row["duration_s"],
        "has_view": has_view,
        "file_url": f"/files/{row['source_id']}" if has_view else None,
        "model": row["model"],
        "written_at": row["written_at"],
    }


def _term_key(term: str) -> str:
    """"Net Present Value (NPV)" and "net present value (NPV)" are one glossary entry."""
    return " ".join(_NOT_WORD.sub(" ", term.casefold()).split())
