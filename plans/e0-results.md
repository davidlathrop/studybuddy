# E0 Results: Enrichment Pilot

_Run 2026-10-14 on the M4 MacBook (32 GB). Part of [exam-search-plan.md](exam-search-plan.md) §10._

## Decision

**Enrichment uses Claude Opus 5.5 through `claude -p`, on your Claude subscription.**

- Bedrock is kept as a tested, optional backend.
- Both backends share the same prompts (`src/studybuddy/enrich/prompts/*_v1.md`), output schemas, and quote checking.

## What was tested

| Check | `claude -p` (subscription) | Bedrock |
|---|---|---|
| Sign-in | claude.ai login; `--safe-mode` keeps it (`--bare` would require an API key) | `bedrock` profile, region `us-east-1` passed explicitly |
| Opus / Sonnet / Haiku 5.5 | all three work | all three work through bedrock-runtime `global.` profiles. Mantle serves **only Opus 5.5** (Sonnet and Haiku return 404) |
| Guaranteed JSON output | `--json-schema` returns a validated object in `structured_output` | `output_config.format` works on both Bedrock paths, which corrects the earlier research. Strict tool use also works. |
| Prompt caching across windows | yes: the transcript is in the system prompt, about 11k cache-read tokens per window | yes: about 11k cache-read tokens per window |
| Usage reporting | tokens, a list-price cost estimate, and 5-hour and weekly allowance on every call | exact tokens, and cost from the Bedrock price list |
| Titan embeddings (`amazon.titan-embed-text-v2:0`) | — | works, 1024 dimensions |
| Token counting | — | runtime path: not supported (Mantle: works) |
| Failed calls | 0 | 0 |

## Results

All runs used effort `medium`, windows of 8 passages, and one outline call per source.

### Real lecture clip: "Applied Micro 2 Price Elasticity" (2:08, 2 passages)

Every model produced usable output, and **100% of question and concept quotes verified** on both backends.

| Model (`claude -p`) | Questions | Concepts | Relations (verified) | Glossary | Time |
|---|---|---|---|---|---|
| Opus 5.5 | 17 | 10 | 6 (100%) | 5 | 68 s |
| Sonnet 5.5 | 14 | 7 | 3 (100%) | 3 | 15 s |
| Haiku 5.5 | 18 | 9 | 4 (100%) | 5 | 36 s |

### 30-minute benchmark transcript (synthetic, 30 passages) through `claude -p`

| Model | Time | Output tokens | 5-hour allowance | Questions | Concepts | Relations (verified) | Glossary (verified) |
|---|---|---|---|---|---|---|---|
| **Opus 5.5** | 364 s | 50,654 | 12% → 14% | **268** (100%) | **236** (100%) | **60** (82%) | **40** (100%) |
| Sonnet 5.5 | 112 s | 21,376 | 12% → 12% | 143 (100%) | 96 (100%) | 16 (56%) | 8 (100%) |
| Haiku 5.5 | 532 s | 158,445 | 11% → 12% | 237 (100%) | 220 (100%) | 19 (79%) | 20 (95%) |

- **Opus** gives the most complete coverage: about 9 questions and 8 concepts per passage, and the most verified relations and glossary entries.
- **Sonnet** stays under the requested 8–12 questions per passage.
- **Haiku** spends most of its output on reasoning, which makes it slower than Opus.
- Unverified relations are hidden at exam time by design (constraint C5).

### Same transcript through Bedrock (per-token, global prices)

| Model | Cost per lecture hour | Semester (about 220 lecture-hour equivalents) |
|---|---|---|
| Opus 5.5 | about $1.90 (run stopped after 1 of 4 windows when switching to `claude -p`; extrapolated) | about $420 |
| Sonnet 5.5 | $0.78 | about $170 |
| Haiku 5.5 | $0.10 | about $22 |

Bedrock charges during E0 came to about **$0.85** in total: probes, the real clip, and the partial 30-minute run.

## What Opus 5.5 produced for the real clip (passage P2, 0:58–2:08)

> Furniture, same pattern. These categories are behaving like classic elastic goods. Consumers have substitutes, they can delay purchases… Now flip to services: travel, healthcare, financial services… This isn't demand-pull inflation… It's cost-push driven by tariffs…

- **Keywords:** elastic goods, inelastic demand, substitutes, delaying purchases, services, travel healthcare, cost-push inflation, demand-pull inflation, tariffs, supply chain
- **Sample questions:** Why do durable goods like furniture behave as elastic goods? · Why is demand for services less sensitive to price than demand for goods? · What is the difference between demand-pull and cost-push inflation in this case? · What is causing the current price increases?
- **Concepts:** elastic demand (aliases: elastic goods, price-elastic demand) · inelastic demand · substitute goods · demand-pull inflation (*defines*) · cost-push inflation (aliases: supply-side inflation) · tariffs · cost pass-through
- **Relations:** substitute goods → *causes* → elastic demand · services → *example_of* → inelastic demand · cost-push inflation ↔ *contrasts_with* ↔ demand-pull inflation · tariffs → *causes* → cost-push inflation

## Subscription allowance projection (Opus 5.5)

- **Measured rate:** about 4% of the 5-hour allowance per lecture hour, about 12 minutes of processing per lecture hour. The weekly allowance didn't move at 1% resolution over a 30-minute run.
- **At about 12 lecture hours a week** (4 courses), enriching after each lecture uses roughly half of one 5-hour window spread across the week. That's comfortable alongside normal use.
- **Bulk runs aren't advisable.** A whole semester at once (about 220 lecture-hour equivalents) would hit the limits. It would also go beyond the "ordinary, individual usage" that Anthropic's terms say subscription limits assume.
- **Not yet measured:** weekly usage at real semester volume. The `run` table will record allowance on every call from E3 on, so this becomes measured rather than projected.

## Caveats

- **Synthetic transcript.** The 30-minute transcript is text-to-speech from generated paragraphs. Real lectures are messier, so expect somewhat different output volume. The 2-minute real clip behaved the same as the synthetic one.
- **Short clip.** The only real recording on the laptop is 2 minutes long. Re-run the pilot on a full real lecture when you download one: `bench/e0_pilot.py <transcript.json> --models opus`.
- **Cost estimates.** `claude -p` cost figures are list-price estimates (`costBasis: list`), not charges on a subscription.

## Files

- **Code:** `src/studybuddy/enrich/` (schemas, passages, quote checking, prompts, and the `claude_cli` and `bedrock` backends) and `bench/e0_pilot.py`.
- **Tests:** `tests/test_enrich.py`, 29 unit tests, no network.
- **Raw outputs (not in git):** `bench/out/e0/`, holding a per-model `outline.json`, `passages.json`, `summary.json` and `report.md` for each run.
