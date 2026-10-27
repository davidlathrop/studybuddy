"""Output schemas for the enrichment calls (plan §6.3) and their JSON Schema form.

Claude returns these through structured outputs (``output_config.format``), which needs a
schema where every object is closed (``additionalProperties: false``), every property is
required and nothing is referenced through ``$defs``. ``strict_json_schema`` derives that
from the Pydantic models, so the models stay the single definition.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RelationType = Literal[
    "is_a", "part_of", "prerequisite_of", "causes", "used_for", "example_of", "contrasts_with"
]
MentionRole = Literal["defines", "explains", "example", "mentions"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratedQuestion(_Strict):
    question: str
    answer_quote: str = Field(description="Verbatim excerpt of the passage that answers the question.")


class ConceptMention(_Strict):
    label: str = Field(description="Standard name of the concept.")
    surface: str = Field(description="Wording used in the passage.")
    aliases: list[str]
    role: MentionRole
    quote: str = Field(description="Verbatim excerpt of the passage where the concept appears.")


class Relation(_Strict):
    head: str
    type: RelationType
    tail: str
    quote: str = Field(description="Verbatim excerpt of the passage stating the relationship.")


class PassageEnrichment(_Strict):
    passage_id: int
    context: str
    summary: str
    keywords: list[str]
    questions: list[GeneratedQuestion]
    concepts: list[ConceptMention]
    relations: list[Relation]


class WindowEnrichment(_Strict):
    """Call B: one window of passages."""

    passages: list[PassageEnrichment]


class Chapter(_Strict):
    title: str
    first_passage_id: int
    last_passage_id: int


class GlossaryEntry(_Strict):
    term: str
    passage_id: int
    definition_quote: str = Field(description="Verbatim excerpt of the passage defining the term.")


class SourceOutline(_Strict):
    """Call A: one per lecture or document."""

    summary: str
    chapters: list[Chapter]
    glossary: list[GlossaryEntry]


# --- study notes (enrich/notes.py) ---------------------------------------------------------


class NotesSectionPlan(_Strict):
    title: str
    first_passage_id: int
    last_passage_id: int


class Announcement(_Strict):
    text: str
    passage_id: int


class NotesOutline(_Strict):
    """Notes call 1: one per lecture."""

    title: str
    overview: str
    sections: list[NotesSectionPlan]
    announcements: list[Announcement]


class KeyPoint(_Strict):
    point: str
    details: list[str]
    passage_id: int


class KeyTerm(_Strict):
    term: str
    definition: str
    passage_id: int
    quote: str = Field(description="Verbatim excerpt of the passage where the term is defined or explained.")


class NoteItem(_Strict):
    text: str
    passage_id: int


class ReviewQuestion(_Strict):
    question: str
    answer: str
    passage_id: int


class SectionNotes(_Strict):
    section: int
    summary: str
    key_points: list[KeyPoint]
    key_terms: list[KeyTerm]
    examples: list[NoteItem]
    emphasis: list[NoteItem]
    review_questions: list[ReviewQuestion]


class NotesWindow(_Strict):
    """Notes call 2: one per group of consecutive sections."""

    sections: list[SectionNotes]


_DROP = {"title", "default"}


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Inline ``$defs``, close every object and mark every property required."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROP:
                continue
            # Property names (e.g. Chapter.title) are data, not schema keywords.
            out[key] = {k: walk(v) for k, v in value.items()} if key == "properties" else walk(value)
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}))
        return out

    return walk(raw)
