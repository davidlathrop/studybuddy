"""Claude on Amazon Bedrock.

E0 findings for this account (plans/e0-results.md): the Mantle endpoint serves only Opus 5.5,
while the bedrock-runtime path serves Opus, Sonnet and Haiku 5.5 through cross-region
inference profiles (``global.anthropic.claude-*-5-5``). Both accept structured outputs, so
JSON comes back through ``output_config.format`` and is validated again locally.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import anthropic
from anthropic import AnthropicBedrock, AnthropicBedrockMantle
from pydantic import ValidationError

from studybuddy.enrich.backends.base import BackendResult, EnrichmentError, T, Usage
from studybuddy.enrich.schemas import strict_json_schema

log = logging.getLogger(__name__)

Endpoint = Literal["runtime", "mantle"]


def _usage(message: Any) -> Usage:
    u = message.usage
    return Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )


class BedrockBackend:
    name = "bedrock"

    def __init__(
        self,
        *,
        profile: str | None = "bedrock",
        region: str = "us-east-1",
        endpoint: Endpoint = "runtime",
        attempts: int = 2,
        client: Any = None,
    ) -> None:
        # The region is passed explicitly: the SDK does not read it from ~/.aws/config.
        if client is None:
            cls = AnthropicBedrockMantle if endpoint == "mantle" else AnthropicBedrock
            client = cls(aws_profile=profile, aws_region=region, max_retries=4)
        self._client = client
        self._attempts = attempts

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
        schema = strict_json_schema(output)
        messages = [
            {
                "role": "user",
                "content": [
                    # system + context form the cached prefix shared by every call on a source.
                    {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        usage = Usage()
        started = time.monotonic()
        for attempt in range(1, self._attempts + 1):
            try:
                with self._client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                ) as stream:
                    message = stream.get_final_message()
            except anthropic.APIStatusError as e:
                raise EnrichmentError(f"{model}: HTTP {e.status_code}: {e.message}") from e
            usage += _usage(message)
            if message.stop_reason == "refusal":
                raise EnrichmentError(f"{model} declined the request")
            if message.stop_reason == "max_tokens":
                raise EnrichmentError(f"{model} output was cut off at max_tokens={max_tokens}")
            text = next((b.text for b in message.content if b.type == "text"), "")
            try:
                data = output.model_validate_json(text)
            except ValidationError as e:
                if attempt == self._attempts:
                    raise EnrichmentError(f"{model} returned invalid {output.__name__}: {e}") from e
                log.warning("%s returned invalid %s; retrying", model, output.__name__)
                continue
            return BackendResult(
                data=data,
                model=model,
                usage=usage,
                cost_usd=usage.cost(model),
                seconds=time.monotonic() - started,
                attempts=attempt,
            )
        raise AssertionError("unreachable")
