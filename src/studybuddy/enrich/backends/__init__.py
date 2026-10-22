"""Swappable enrichment backends (plan §6.7)."""

import shutil

from studybuddy.enrich.backends.base import (
    BackendResult,
    EnrichmentBackend,
    EnrichmentError,
    Usage,
    UsageLimitError,
)

BACKENDS = ("claude-cli", "bedrock")

__all__ = [
    "BACKENDS", "BackendResult", "EnrichmentBackend", "EnrichmentError", "Usage", "UsageLimitError",
    "make_backend", "model_id",
]


def make_backend(name: str) -> EnrichmentBackend:
    """The backend ``[enrich] backend`` names; the Bedrock SDK is only imported when chosen."""
    if name == "claude-cli":
        from studybuddy.enrich.backends.claude_cli import ClaudeCliBackend

        if shutil.which("claude") is None:
            raise EnrichmentError(
                "the claude command was not found: install Claude Code and sign in, "
                "or set [enrich] backend = \"bedrock\" in library.toml"
            )
        return ClaudeCliBackend()
    if name == "bedrock":
        from studybuddy.enrich.backends.bedrock import BedrockBackend

        return BedrockBackend()
    raise EnrichmentError(f"unknown enrichment backend {name!r}; choose from {', '.join(BACKENDS)}")


def model_id(backend: str, model: str) -> str:
    """``claude-opus-5-5`` as the backend names it: Bedrock wants an inference profile."""
    if backend == "bedrock" and "anthropic." not in model:
        return f"global.anthropic.{model}"
    return model
