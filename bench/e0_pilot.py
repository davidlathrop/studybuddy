"""E0 Bedrock pilot (plans/exam-search-plan.md §10): run one transcript through each model.

For every model: one outline call (Call A), then one call per window of passages (Call B),
all sharing a cached prefix (system prompt + full transcript). Records tokens, cost, time,
cache hits, subscription allowance used (claude-cli) and how many AI quotes verify against
the source, then writes a side-by-side report.

    .venv/bin/python bench/e0_pilot.py TRANSCRIPT.json --out bench/out/e0/NAME \
        [--backend claude-cli|bedrock] [--models haiku,sonnet,opus] [--max-minutes 60]
        [--window 8] [--effort medium] [--route global] [--max-usd 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from studybuddy.enrich import load_prompt
from studybuddy.enrich.backends import BackendResult, EnrichmentBackend, EnrichmentError, UsageLimitError
from studybuddy.enrich.backends.bedrock import BedrockBackend
from studybuddy.enrich.backends.claude_cli import ClaudeCliBackend
from studybuddy.enrich.passages import Passage, clock, passages_from_transcript, render_transcript, windows
from studybuddy.enrich.schemas import SourceOutline, WindowEnrichment
from studybuddy.enrich.verify import check_quote, names_appear

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}


def model_id(backend: str, short: str, route: str) -> str:
    return f"{route}.anthropic.{MODELS[short]}" if backend == "bedrock" else MODELS[short]
PROMPTS = {"system": "system_v1", "outline": "outline_v1", "passages": "passages_v1"}


def call_record(kind: str, r: BackendResult[Any]) -> dict[str, Any]:
    return {"kind": kind, "seconds": round(r.seconds, 2), "attempts": r.attempts,
            "cost_usd": r.cost_usd, **asdict(r.usage), **r.meta}


def score_window(result: WindowEnrichment, by_id: dict[int, Passage], expected: list[int]) -> dict[str, Any]:
    """Count items and how many of their quotes verify against their own passage."""
    c: Counter[str] = Counter()
    returned = [p.passage_id for p in result.passages]
    c["passages_expected"] += len(expected)
    c["passages_missing"] += len(set(expected) - set(returned))
    c["passages_unexpected"] += len(set(returned) - set(expected))
    for pe in result.passages:
        passage = by_id.get(pe.passage_id)
        if passage is None:
            continue
        c["keywords"] += len(pe.keywords)
        for q in pe.questions:
            c["questions"] += 1
            c["questions_verified"] += check_quote(q.answer_quote, passage.text).verified
        aliases: dict[str, list[str]] = {}
        for m in pe.concepts:
            c["concepts"] += 1
            c["concepts_verified"] += check_quote(m.quote, passage.text).verified
            aliases[m.label.lower()] = [m.label, m.surface, *m.aliases]
        for rel in pe.relations:
            c["relations"] += 1
            quote_ok = check_quote(rel.quote, passage.text).verified
            names_ok = all(
                names_appear(aliases.get(n.lower(), [n]), rel.quote, passage.text) for n in (rel.head, rel.tail)
            )
            c["relations_verified"] += quote_ok and names_ok
    return dict(c)


def run_model(short: str, model: str, args: argparse.Namespace, passages: list[Passage],
              title: str, backend: EnrichmentBackend, out: Path, spent: float) -> dict[str, Any]:
    system = load_prompt(PROMPTS["system"])
    context = render_transcript(title, passages)
    by_id = {p.id: p for p in passages}
    calls: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    model_dir = out / short
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    outline = backend.call(model=model, system=system, context=context, prompt=load_prompt(PROMPTS["outline"]),
                           output=SourceOutline, effort=args.effort, max_tokens=16000)
    calls.append(call_record("outline", outline))
    glossary_ok = sum(
        check_quote(g.definition_quote, by_id[g.passage_id].text).verified if g.passage_id in by_id else 0
        for g in outline.data.glossary
    )
    totals.update(glossary=len(outline.data.glossary), glossary_verified=glossary_ok)
    (model_dir / "outline.json").write_text(outline.data.model_dump_json(indent=2))
    chapters = "\n".join(f"- P{c.first_passage_id}-P{c.last_passage_id}: {c.title}" for c in outline.data.chapters)

    enriched: list[dict[str, Any]] = []
    for win in windows(passages, args.window):
        cost_so_far = spent + sum(c["cost_usd"] or 0 for c in calls)
        if cost_so_far > args.max_usd:
            print(f"  budget ${args.max_usd:.2f} reached; stopping {short}", file=sys.stderr)
            break
        prompt = load_prompt(PROMPTS["passages"]).format(first_id=win[0].id, last_id=win[-1].id, chapters=chapters)
        try:
            r = backend.call(model=model, system=system, context=context, prompt=prompt,
                             output=WindowEnrichment, effort=args.effort, max_tokens=32000)
        except UsageLimitError:
            raise
        except EnrichmentError as e:
            print(f"  window P{win[0].id}-P{win[-1].id} failed: {e}", file=sys.stderr)
            calls.append({"kind": "window", "error": str(e)})
            continue
        calls.append(call_record("window", r))
        totals.update(score_window(r.data, by_id, [p.id for p in win]))
        enriched += [p.model_dump() for p in r.data.passages]
        print(f"  {short} P{win[0].id}-P{win[-1].id}: {r.seconds:.0f}s ${r.cost_usd or 0:.4f} "
              f"out={r.usage.output_tokens} cache_read={r.usage.cache_read_input_tokens}", file=sys.stderr)
    (model_dir / "passages.json").write_text(json.dumps(enriched, indent=2, ensure_ascii=False))

    ok = [c for c in calls if "error" not in c]
    audio_h = (passages[-1].end - passages[0].start) / 3600
    cost = sum(c["cost_usd"] or 0 for c in ok)
    summary = {
        "model": model,
        "effort": args.effort,
        "audio_hours": round(audio_h, 3),
        "passages": len(passages),
        "calls": len(calls),
        "failed_calls": len(calls) - len(ok),
        "seconds": round(time.monotonic() - started, 1),
        "cost_usd": round(cost, 4),
        "cost_per_audio_hour": round(cost / audio_h, 3) if audio_h else None,
        **{k: sum(c.get(k, 0) for c in ok) for k in
           ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")},
        **dict(totals),
        **{f"{w}_{edge}": next((c.get(f"{w}_utilization") for c in (ok if edge == "start" else ok[::-1])
                                if c.get(f"{w}_utilization") is not None), None)
           for w in ("five_hour", "seven_day") for edge in ("start", "end")},
        "calls_detail": calls,
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def allowance(s: dict[str, Any]) -> str:
    """Subscription allowance in percent at the start and end of the model's run (claude-cli only)."""
    if s.get("five_hour_start") is None:
        return "-"
    span = lambda w: f"{100 * s[f'{w}_start']:.0f}->{100 * s[f'{w}_end']:.0f}%"  # noqa: E731
    return f"{span('five_hour')} / {span('seven_day')}"


def write_report(out: Path, title: str, passages: list[Passage], summaries: dict[str, dict[str, Any]]) -> None:
    lines = [f"# E0 pilot: {title}", "",
             f"{len(passages)} passages, {clock(passages[-1].end - passages[0].start)} of audio.", "",
             "| Model | Cost | Per audio hour | Time | Output tok | Cache read tok | 5-hour / weekly allowance "
             "| Questions (verified) | Concepts (verified) | Relations (verified) | Glossary (verified) "
             "| Missing passages |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for short, s in summaries.items():
        lines.append(
            f"| {short} | ${s['cost_usd']:.3f} | ${s['cost_per_audio_hour']:.2f} | {s['seconds']:.0f}s "
            f"| {s['output_tokens']:,} | {s['cache_read_input_tokens']:,} | {allowance(s)} "
            f"| {s.get('questions', 0)} ({pct(s.get('questions_verified', 0), s.get('questions', 0))}) "
            f"| {s.get('concepts', 0)} ({pct(s.get('concepts_verified', 0), s.get('concepts', 0))}) "
            f"| {s.get('relations', 0)} ({pct(s.get('relations_verified', 0), s.get('relations', 0))}) "
            f"| {s.get('glossary', 0)} ({pct(s.get('glossary_verified', 0), s.get('glossary', 0))}) "
            f"| {s.get('passages_missing', 0)} |")
    sample = passages[min(1, len(passages) - 1)]
    lines += ["", f"## Side by side: P{sample.id} ({clock(sample.start)}-{clock(sample.end)})", "",
              f"> {sample.text}", ""]
    for short in summaries:
        data = json.loads((out / short / "passages.json").read_text())
        pe = next((p for p in data if p["passage_id"] == sample.id), None)
        lines += [f"### {short}", ""]
        if pe is None:
            lines += ["(no output for this passage)", ""]
            continue
        lines += [f"**Context:** {pe['context']}", "", f"**Summary:** {pe['summary']}", "",
                  f"**Keywords:** {', '.join(pe['keywords'])}", "", "**Questions:**"]
        lines += [f"- {q['question']}" for q in pe["questions"]]
        lines += ["", "**Concepts:** " + "; ".join(
            f"{c['label']} ({c['role']}; aliases: {', '.join(c['aliases']) or '-'})" for c in pe["concepts"])]
        lines += ["", "**Relations:** " + ("; ".join(
            f"{r['head']} --{r['type']}--> {r['tail']}" for r in pe["relations"]) or "-"), ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="haiku,sonnet,opus")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--backend", choices=["claude-cli", "bedrock"], default="claude-cli")
    ap.add_argument("--route", default="global", help="bedrock only: inference profile prefix, global or us")
    ap.add_argument("--profile", default="bedrock")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--max-usd", type=float, default=10.0)
    args = ap.parse_args()

    data = json.loads(args.transcript.read_text())
    passages = [p for p in passages_from_transcript(data) if p.start < args.max_minutes * 60]
    title = Path(data.get("source", args.transcript.stem)).stem
    args.out.mkdir(parents=True, exist_ok=True)
    backend: EnrichmentBackend = (
        BedrockBackend(profile=args.profile, region=args.region) if args.backend == "bedrock" else ClaudeCliBackend()
    )
    summaries: dict[str, dict[str, Any]] = {}
    spent = 0.0
    for short in args.models.split(","):
        model = model_id(args.backend, short, args.route)
        print(f"{short}: {model} via {backend.name}", file=sys.stderr)
        try:
            summaries[short] = run_model(short, model, args, passages, title, backend, args.out, spent)
        except UsageLimitError as e:
            print(f"stopping: {e} (resets at {e.resets_at})", file=sys.stderr)
            break
        spent += summaries[short]["cost_usd"]
    (args.out / "summary.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "calls_detail"} for k, v in summaries.items()}, indent=2))
    write_report(args.out, title, passages, summaries)
    print(f"total ${spent:.3f}; report: {args.out / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
