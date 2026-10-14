"""Enrichment: AI-generated search aids built before the exam (plans/exam-search-plan.md §6)."""

from __future__ import annotations

from importlib import resources


def load_prompt(name: str) -> str:
    """Return the text of ``prompts/<name>.md`` (names carry their version, e.g. ``passages_v1``)."""
    return (resources.files(__package__) / "prompts" / f"{name}.md").read_text(encoding="utf-8")
