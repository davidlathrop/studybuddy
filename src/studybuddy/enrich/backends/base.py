"""The backend interface, token usage and the price table used for cost tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class EnrichmentError(RuntimeError):
    """A call that produced no usable output (refusal, cut off, or invalid after retries)."""


class UsageLimitError(EnrichmentError):
    """A subscription usage limit was hit; nothing more can run until ``resets_at``."""

    def __init__(self, message: str, resets_at: int | None = None) -> None:
        super().__init__(message)
        self.resets_at = resets_at  # Unix time, when known


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_write: float  # 5-minute cache
    cache_read: float


# Bedrock global cross-region prices, AWS price list 2026-10-12. Geographic profiles
# ("us.", "eu.", ...) and in-region endpoints cost 10% more.
PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": Price(2.00, 10.00, 2.50, 0.10),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.125, 0.01),  # prompts up to 100K tokens
}
REGIONAL_MARKUP = 1.10


def price_for(model_id: str) -> Price | None:
    base = next((p for name, p in PRICES.items() if name in model_id), None)
    if base is None or model_id.startswith("global."):
        return base
    m = REGIONAL_MARKUP
    return Price(base.input * m, base.output * m, base.cache_write * m, base.cache_read * m)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __iadd__(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        return self

    def cost(self, model_id: str) -> float | None:
        p = price_for(model_id)
        if p is None:
            return None
        return (
            self.input_tokens * p.input
            + self.output_tokens * p.output
            + self.cache_creation_input_tokens * p.cache_write
            + self.cache_read_input_tokens * p.cache_read
        ) / 1_000_000


@dataclass
class BackendResult(Generic[T]):
    data: T
    model: str
    usage: Usage
    cost_usd: float | None  # billed cost, or the list-price equivalent on a subscription
    seconds: float
    attempts: int
    meta: dict[str, Any] = field(default_factory=dict)  # backend-specific, e.g. rate-limit state


class EnrichmentBackend(Protocol):
    name: str

    def call(
        self,
        *,
        model: str,
        system: str,
        context: str,
        prompt: str,
        output: type[T],
        effort: str = "medium",
        max_tokens: int = 32000,
    ) -> BackendResult[T]:
        """Send ``system`` + cached ``context`` + ``prompt``; return ``output`` validated."""
        ...
