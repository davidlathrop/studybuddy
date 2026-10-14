"""Unit tests for the enrichment building blocks (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from studybuddy.enrich import load_prompt
from studybuddy.enrich.backends import EnrichmentError
from studybuddy.enrich.backends.base import PRICES, Usage, price_for
from studybuddy.enrich.backends.bedrock import BedrockBackend
from studybuddy.enrich.passages import Passage, clock, passages_from_transcript, render_transcript, windows
from studybuddy.enrich.schemas import SourceOutline, WindowEnrichment, strict_json_schema
from studybuddy.enrich.verify import check_quote, names_appear, normalize


# --- schemas -------------------------------------------------------------------------------


def _objects(node: Any):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for key, value in node.items():
            yield from (_objects(v) for v in value.values()) if key == "properties" else [_objects(value)]
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def _flatten(gen):
    for item in gen:
        if isinstance(item, dict):
            yield item
        else:
            yield from _flatten(item)


@pytest.mark.parametrize("model", [WindowEnrichment, SourceOutline])
def test_strict_schema_closes_every_object_and_inlines_refs(model):
    schema = strict_json_schema(model)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text
    objects = list(_flatten(_objects(schema)))
    assert objects, "expected nested objects"
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])


def test_strict_schema_keeps_a_property_named_title():
    chapter = strict_json_schema(SourceOutline)["properties"]["chapters"]["items"]
    assert "title" in chapter["properties"]
    assert "title" in chapter["required"]


def test_relation_type_is_a_closed_vocabulary():
    schema = strict_json_schema(WindowEnrichment)
    rel = schema["properties"]["passages"]["items"]["properties"]["relations"]["items"]
    assert set(rel["properties"]["type"]["enum"]) == {
        "is_a", "part_of", "prerequisite_of", "causes", "used_for", "example_of", "contrasts_with"
    }


def test_prompts_load_and_format():
    assert "verbatim" in load_prompt("system_v1")
    text = load_prompt("passages_v1").format(first_id=3, last_id=9, chapters="- P1-P9: Intro")
    assert "P3-P9" in text and "- P1-P9: Intro" in text


# --- passages ------------------------------------------------------------------------------


def _seg(start: float, end: float, text: str = "words") -> dict[str, Any]:
    return {"start": start, "end": end, "text": text, "language": "English"}


def test_accurate_segments_map_one_to_one():
    data = {"segments": [_seg(0, 55), _seg(55, 115), _seg(115, 170)]}
    ps = passages_from_transcript(data)
    assert [(p.id, p.start, p.end) for p in ps] == [(1, 0, 55), (2, 55, 115), (3, 115, 170)]


def test_sentences_merge_until_min_length_and_short_tail_joins_previous():
    segs = [_seg(i * 10, i * 10 + 10, f"s{i}") for i in range(11)]  # 110 s of 10 s sentences
    ps = passages_from_transcript({"segments": segs}, min_s=45)
    assert [(p.start, p.end) for p in ps] == [(0, 50), (50, 110)]  # 100-110 tail (<22.5 s) joined
    assert ps[0].text == "s0 s1 s2 s3 s4"


def test_long_tail_stands_alone_and_blank_segments_are_skipped():
    segs = [_seg(0, 50), _seg(50, 52, "  "), _seg(52, 80)]
    ps = passages_from_transcript({"segments": segs}, min_s=45)
    assert [(p.start, p.end) for p in ps] == [(0, 50), (52, 80)]


def test_render_and_windows():
    ps = [Passage(i, (i - 1) * 60.0, i * 60.0, f"text {i}") for i in range(1, 21)]
    rendered = render_transcript("Lecture 4", ps)
    assert rendered.splitlines()[0] == "<source title='Lecture 4'>"
    assert "[P2 01:00-02:00] text 2" in rendered
    assert [len(w) for w in windows(ps, 8)] == [8, 8, 4]
    with pytest.raises(ValueError):
        windows(ps, 0)
    assert clock(3725) == "1:02:05"


# --- quote verification --------------------------------------------------------------------

SOURCE = "Clothing prices rose 9% over a three-month window, and spending fell 7%. That’s elastic."


@pytest.mark.parametrize(
    ("quote", "method"),
    [
        ("spending fell 7%", "exact"),
        ("CLOTHING PRICES ROSE 9%  over a three month window", "normalized"),
        ("That's elastic", "normalized"),  # curly vs straight apostrophe
        ("clothing prices rose 9 % over a three-month windows", "fuzzy"),
    ],
)
def test_check_quote_accepts(quote, method):
    result = check_quote(quote, SOURCE)
    assert result.verified and result.method == method


@pytest.mark.parametrize("quote", ["", "   ", "demand-pull inflation is rising fast"])
def test_check_quote_rejects(quote):
    assert not check_quote(quote, SOURCE).verified


def test_short_quote_is_exact_or_nothing():
    assert check_quote("PRICES", SOURCE).method == "normalized"
    assert not check_quote("prizes", SOURCE).verified  # too short for fuzzy


def test_names_appear_matches_whole_words():
    assert names_appear(["ATP", "adenosine triphosphate"], "Adenosine triphosphate stores energy.")
    assert not names_appear(["ATP"], "The ATPase enzyme")
    assert normalize("Cost-push  inflation!") == "cost push inflation"


# --- prices and usage ----------------------------------------------------------------------


def test_prices_global_and_regional():
    assert price_for("global.anthropic.claude-opus-5-5") == PRICES["claude-opus-5-5"]
    assert price_for("us.anthropic.claude-opus-5-5").output == pytest.approx(22.0)
    assert price_for("some-other-model") is None
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_creation_input_tokens=0,
                  cache_read_input_tokens=2_000_000)
    assert usage.cost("global.anthropic.claude-sonnet-5-5") == pytest.approx(2.0 + 1.0 + 0.2)


# --- Bedrock backend with a fake client ----------------------------------------------------


def _message(text: str, stop: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_creation_input_tokens=1000,
                              cache_read_input_tokens=0),
    )


class _FakeClient:
    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def stream(self, **kwargs: Any):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)

        class _Stream:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def get_final_message(self_inner):
                return reply

        return _Stream()


OUTLINE_JSON = json.dumps({"summary": "s", "chapters": [{"title": "t", "first_passage_id": 1,
                                                          "last_passage_id": 2}], "glossary": []})


def test_backend_sends_cached_context_and_schema_and_parses():
    client = _FakeClient([_message(OUTLINE_JSON)])
    result = BedrockBackend(client=client).call(
        model="global.anthropic.claude-haiku-5-5", system="sys", context="CTX", prompt="go",
        output=SourceOutline, effort="low")
    assert result.data.chapters[0].title == "t"
    assert result.attempts == 1 and result.usage.cache_creation_input_tokens == 1000
    assert result.cost_usd == pytest.approx((100 * 0.10 + 50 * 0.50 + 1000 * 0.125) / 1e6)
    req = client.requests[0]
    first = req["messages"][0]["content"][0]
    assert first == {"type": "text", "text": "CTX", "cache_control": {"type": "ephemeral"}}
    assert req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["schema"] == strict_json_schema(SourceOutline)


def test_backend_retries_invalid_json_once_and_sums_usage():
    client = _FakeClient([_message("{not json"), _message(OUTLINE_JSON)])
    result = BedrockBackend(client=client).call(
        model="global.anthropic.claude-haiku-5-5", system="s", context="c", prompt="p", output=SourceOutline)
    assert result.attempts == 2 and result.usage.output_tokens == 100


def test_backend_gives_up_after_attempts():
    client = _FakeClient([_message("{}"), _message("[]")])
    with pytest.raises(EnrichmentError, match="invalid SourceOutline"):
        BedrockBackend(client=client).call(
            model="m", system="s", context="c", prompt="p", output=SourceOutline)


@pytest.mark.parametrize(("stop", "match"), [("refusal", "declined"), ("max_tokens", "cut off")])
def test_backend_raises_on_refusal_or_truncation(stop, match):
    client = _FakeClient([_message(OUTLINE_JSON, stop=stop)])
    with pytest.raises(EnrichmentError, match=match):
        BedrockBackend(client=client).call(model="m", system="s", context="c", prompt="p", output=SourceOutline)


# --- claude -p backend with a fake runner --------------------------------------------------

from studybuddy.enrich.backends import UsageLimitError  # noqa: E402
from studybuddy.enrich.backends.claude_cli import ClaudeCliBackend, parse_events  # noqa: E402


def _cli_stdout(result: dict[str, Any], status: str = "allowed") -> str:
    limit = {"type": "rate_limit_event", "rate_limit_info": {
        "status": status, "resetsAt": 1792017600, "rateLimitType": "five_hour",
        "unifiedWindows": {"five_hour": {"utilization": 0.12}, "seven_day": {"utilization": 0.07}}}}
    return json.dumps([{"type": "system", "subtype": "init"}, limit, {"type": "result", **result}])


def _ok(structured: dict[str, Any]) -> dict[str, Any]:
    return {"subtype": "success", "is_error": False, "structured_output": structured, "total_cost_usd": 0.01,
            "usage": {"input_tokens": 5, "output_tokens": 70, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 900}}


class _FakeRunner:
    def __init__(self, outputs: list[tuple[int, str]]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, cmd, **kwargs):
        system_file = cmd[cmd.index("--system-prompt-file") + 1]
        self.calls.append({"cmd": cmd, "system": Path(system_file).read_text(), **kwargs})
        code, out = self.outputs.pop(0)
        return SimpleNamespace(returncode=code, stdout=out, stderr="")


OUTLINE = json.loads(OUTLINE_JSON)


def test_cli_backend_builds_command_and_parses_structured_output():
    runner = _FakeRunner([(0, _cli_stdout(_ok(OUTLINE)))])
    result = ClaudeCliBackend(runner=runner).call(
        model="claude-haiku-5-5", system="SYS", context="CTX", prompt="TASK", output=SourceOutline, effort="low")
    assert result.data.chapters[0].title == "t"
    assert result.usage.cache_read_input_tokens == 900 and result.cost_usd == pytest.approx(0.01)
    assert result.meta["five_hour_utilization"] == 0.12 and result.meta["seven_day_utilization"] == 0.07
    call = runner.calls[0]
    cmd = call["cmd"]
    for flag in ("-p", "--safe-mode", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-5-5" and cmd[cmd.index("--effort") + 1] == "low"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == strict_json_schema(SourceOutline)
    assert call["system"] == "SYS\n\nCTX"  # transcript rides in the cached system prompt
    assert call["input"] == "TASK"
    assert "--bare" not in cmd  # --bare would refuse a subscription login


def test_cli_backend_retries_structured_output_failure_then_succeeds():
    fail = {"subtype": "error_max_structured_output_retries", "is_error": True, "usage": {}}
    runner = _FakeRunner([(1, _cli_stdout(fail)), (0, _cli_stdout(_ok(OUTLINE)))])
    result = ClaudeCliBackend(runner=runner).call(
        model="m", system="s", context="c", prompt="p", output=SourceOutline)
    assert result.attempts == 2


def test_cli_backend_reports_usage_limit():
    limited = {"subtype": "error_during_execution", "is_error": True, "usage": {},
               "result": "You've hit your session limit · resets 3:45pm"}
    runner = _FakeRunner([(1, _cli_stdout(limited, status="rejected"))])
    with pytest.raises(UsageLimitError) as info:
        ClaudeCliBackend(runner=runner).call(model="m", system="s", context="c", prompt="p", output=SourceOutline)
    assert info.value.resets_at == 1792017600


def test_cli_backend_raises_on_garbage_output():
    runner = _FakeRunner([(1, "not json at all")])
    with pytest.raises(EnrichmentError, match="without a result"):
        ClaudeCliBackend(runner=runner).call(model="m", system="s", context="c", prompt="p", output=SourceOutline)


def test_parse_events_accepts_a_bare_result_object():
    result, limit = parse_events(json.dumps({"type": "result", "subtype": "success"}))
    assert result == {"type": "result", "subtype": "success"} and limit is None
