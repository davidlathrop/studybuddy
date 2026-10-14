"""Swappable enrichment backends (plan §6.7)."""

from studybuddy.enrich.backends.base import (
    BackendResult,
    EnrichmentBackend,
    EnrichmentError,
    Usage,
    UsageLimitError,
)

__all__ = ["BackendResult", "EnrichmentBackend", "EnrichmentError", "Usage", "UsageLimitError"]
