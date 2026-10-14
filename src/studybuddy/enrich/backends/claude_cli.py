"""Claude through the Claude Code CLI (``claude -p``), using whatever login ``claude`` has.

Each call is one headless run:

    claude -p --safe-mode --tools "" --no-session-persistence --output-format json \
        --model M --effort E --system-prompt-file F --json-schema S  < prompt

* ``--safe-mode`` skips CLAUDE.md, hooks, plugins and MCP servers but keeps the login, so a
  Claude subscription works (``--bare`` would demand an API key).
* The system prompt file holds the instructions *and* the source transcript. Claude Code
  caches the system prompt, so every window of the same source reads it from the cache;
  only the short task prompt changes between calls.
* ``--json-schema`` makes Claude Code return a validated object in ``structured_output``.
* On a subscription, ``total_cost_usd`` is a list-price estimate, not a bill. The result
  stream also reports how much of the 5-hour and weekly allowance is used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from studybuddy.enrich.backends.base import BackendResult, EnrichmentError, T, Usage, UsageLimitError
from studybuddy.enrich.schemas import strict_json_schema

DEFAULT_TIMEOUT_S = 20 * 60


def _usage(result: dict[str, Any]) -> Usage:
    u = result.get("usage") or {}
    return Usage(
        input_tokens=u.get("input_tokens", 0) or 0,
        output_tokens=u.get("output_tokens", 0) or 0,
        cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=u.get("cache_read_input_tokens", 0) or 0,
    )


def parse_events(stdout: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the final ``result`` event and the last ``rate_limit_info`` from ``--output-format json``."""
    try:
        events = json.loads(stdout)
    except json.JSONDecodeError:
        return None, None
    if isinstance(events, dict):  # older CLIs print just the result object
        events = [events]
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    limit = next((e["rate_limit_info"] for e in reversed(events) if e.get("type") == "rate_limit_event"), None)
    return result, limit


class ClaudeCliBackend:
    name = "claude-cli"

    def __init__(self, *, executable: str = "claude", attempts: int = 2,
                 timeout_s: float = DEFAULT_TIMEOUT_S, runner: Any = subprocess.run) -> None:
        self._exe = shutil.which(executable) or executable
        self._attempts = attempts
        self._timeout_s = timeout_s
        self._run = runner

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
        schema = json.dumps(strict_json_schema(output))
        usage = Usage()
        cost = 0.0
        started = time.monotonic()
        # A neutral working directory: nothing from the project leaks into the run.
        with tempfile.TemporaryDirectory(prefix="studybuddy-enrich-") as tmp:
            system_file = Path(tmp) / "system.md"
            system_file.write_text(f"{system}\n\n{context}", encoding="utf-8")
            cmd = [
                self._exe, "-p", "--safe-mode", "--tools", "", "--no-session-persistence",
                "--output-format", "json", "--model", model, "--effort", effort,
                "--system-prompt-file", str(system_file), "--json-schema", schema,
            ]
            env = {**os.environ, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_tokens)}
            for attempt in range(1, self._attempts + 1):
                try:
                    proc = self._run(cmd, input=prompt, capture_output=True, text=True, cwd=tmp,
                                     env=env, timeout=self._timeout_s)
                except subprocess.TimeoutExpired as e:
                    raise EnrichmentError(f"claude -p timed out after {self._timeout_s:.0f}s") from e
                result, limit = parse_events(proc.stdout)
                rejected = bool(limit) and limit.get("status") == "rejected"
                if result is None:
                    tail = (proc.stderr or proc.stdout or "").strip()[-500:]
                    if rejected:
                        raise UsageLimitError(f"Claude usage limit reached: {tail}", limit.get("resetsAt"))
                    raise EnrichmentError(f"claude -p exited {proc.returncode} without a result: {tail}")
                usage += _usage(result)
                cost += result.get("total_cost_usd") or 0.0
                if result.get("is_error") or result.get("subtype") != "success":
                    message = str(result.get("result") or result.get("subtype"))
                    if rejected or ("limit" in message.lower() and "reset" in message.lower()):
                        raise UsageLimitError(message, limit.get("resetsAt") if limit else None)
                    if attempt < self._attempts and result.get("subtype") == "error_max_structured_output_retries":
                        continue
                    raise EnrichmentError(f"claude -p failed ({result.get('subtype')}): {message[:500]}")
                try:
                    data = output.model_validate(result.get("structured_output"))
                except ValidationError as e:
                    if attempt == self._attempts:
                        raise EnrichmentError(f"{model} returned invalid {output.__name__}: {e}") from e
                    continue
                windows = (limit or {}).get("unifiedWindows", {})
                return BackendResult(
                    data=data,
                    model=model,
                    usage=usage,
                    cost_usd=cost,
                    seconds=time.monotonic() - started,
                    attempts=attempt,
                    meta={
                        "cost_basis": "estimate",
                        "five_hour_utilization": windows.get("five_hour", {}).get("utilization"),
                        "seven_day_utilization": windows.get("seven_day", {}).get("utilization"),
                    },
                )
        raise AssertionError("unreachable")
