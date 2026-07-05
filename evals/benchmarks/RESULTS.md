# Public Memory Benchmarks — LoCoMo & LongMemEval (S3)

The credibility yardsticks for the "Stripe for AI memory" positioning. This
harness runs both benchmarks end-to-end through a **live Smritikosh server**
(SDK → REST API → full encode/retrieve pipeline), so the numbers measure the
product as deployed, and are methodologically comparable with published
Mem0 / Zep / Memobase results.

## Benchmarks

| Benchmark | Shape | What it stresses |
|---|---|---|
| **LoCoMo** ([snap-research/locomo](https://github.com/snap-research/locomo), ACL 2024) | 10 two-speaker conversations, ~5,900 turns, **1,540 scored questions** | single-hop (cat 1), temporal (2), multi-hop (3), open-domain (4); cat 5 (adversarial) excluded by default, matching published setups |
| **LongMemEval** ([xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval), ICLR 2025) | 500 instances, each with its own timestamped session haystack | information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention (`_abs` ids); variants: `oracle` (evidence-only) / `s` (~115K tokens) / `m` (~1.5M tokens) |

## Method

1. **Ingest** — every turn becomes a date-stamped, speaker-attributed event
   (`[date] speaker: text`) stored via `POST /memory/event`, i.e. the full
   Hippocampus pipeline (importance, embedding, fact extraction). LoCoMo image
   turns keep their BLIP caption. `--chunk-turns N` groups N consecutive turns
   per event to cut extraction cost (record the value used — it's part of the
   method). Ingestion is resumable (`data/state-*.json`).
2. **Answer** — for each question: `POST /context` (hybrid retrieval + profile)
   → answer with the configured LLM from the retrieved context only, with an
   explicit "I don't know" instruction.
3. **Judge** — binary LLM-as-judge (the standard protocol for both benchmarks);
   abstention questions are scored as correct iff the system declines.
4. **Report** — judge accuracy (the "J" metric) overall + per category, plus
   retrieval/answer latency p50/p95.

Run (see `python -m evals.benchmarks --help`):

```bash
export SMRITIKOSH_API_KEY=<admin key>     # benchmarks create synthetic users
python -m evals.benchmarks locomo --json locomo-results.json
python -m evals.benchmarks longmemeval --variant oracle --json lme-results.json
```

Answering and judging default to the server's `.env` LLM; for publishable runs
override them independently of the product under test (API keys come from the
provider's standard env var, e.g. `OPENAI_API_KEY`):

```bash
python -m evals.benchmarks locomo \
    --answer-model openai:gpt-4o --judge-model openai:gpt-4o-mini \
    --ingest-concurrency 8 --json locomo-results.json
```

`--ingest-concurrency N` parallelizes ingestion **across users only** (turn
order within a user is part of the method); keep it 1 on a local single-GPU
model, raise it against cloud providers where sequential ingestion of a full
benchmark takes days. The judge model is recorded in the report JSON
(`judge_model`) alongside `answer_model`.

**`scripts/run_publishable_benchmarks.sh`** wraps all of this (preflight
checks, chunk-turns 1, timestamped JSON reports in `evals/benchmarks/results/`)
— see its header for knobs and rough cost/time estimates per benchmark.

Full LoCoMo is ~5,900 ingested turns + 1,540×2 LLM calls — start with
`--users 2 --questions 100` and scale up. `--cleanup` removes the synthetic
users afterwards.

> **Run the server with `RECONSOLIDATION_ON_RECALL=0`.** Every `/context` call
> otherwise schedules a background reconsolidation LLM call; under benchmark
> load on a single local model these saturate the provider and stall subsequent
> `/context` calls behind them (observed: ~5 min per request, client timeouts).
> Disabling it for benchmark runs is fair — competitors' harnesses don't do
> post-recall write-backs either — but note it in the writeup. (Since
> 2026-07-01 the durable fix is in: with Redis configured, reconsolidation
> runs on the ARQ taskworker instead of the API process — the flag remains
> the simplest way to keep benchmark runs free of write-back LLM traffic.)
> On a single-GPU local LLM also use `--concurrency 1`.

## Published numbers to compare against

From the [Mem0 paper (arXiv:2504.19413, ECAI 2025)](https://arxiv.org/abs/2504.19413)
and the [Memobase LoCoMo report](https://github.com/memodb-io/memobase/tree/main/docs/experiments/locomo-benchmark)
— overall LLM-judge score (J) on LoCoMo, GPT-4o-class answer/judge models:

| System | LoCoMo overall J |
|---|---|
| Mem0 | ~66.9% |
| Zep (initial report / updated) | ~66.0% / ~75.1% |
| Memobase v0.0.37 | ~75.8% |
| Full-context baseline | ~72.9% (per Mem0 paper) |

> ⚠️ Before publishing a comparison: re-verify these numbers against the
> sources, and match the answer/judge model class. A LoCoMo score produced
> with a small local model is **not** comparable with the table above — the
> published comparisons all use GPT-4o-class models for answering and judging.

For LongMemEval, the [leaderboard in the paper repo](https://github.com/xiaowu0162/LongMemEval)
is the reference point.

## Smritikosh results

| Date | Benchmark | Scope | Answer model | Judge | Chunk | J (overall) | Notes |
|---|---|---|---|---|---|---|---|
| 2026-07-01 | LoCoMo | 1 conv / 12 q (smoke) | ollama gemma4:e4b | same | 10 | 8.3% (1/12, 0 errors) | harness validation only — see analysis below |

_Add a row per run; keep the `--json` reports next to this file._

**Smoke-run analysis (why 8.3% is not the product's score):** all 11 misses were
"I don't know" answers. Probing the retrieved context showed two distinct causes:
(a) episodic retrieval misses — e.g. "Sweden" (the gold answer) absent from the
retrieved context; likely hurt by `--chunk-turns 10` diluting per-event embeddings,
and worth re-testing at chunk 1 with a stronger embedding model; (b) answer-model
weakness — e.g. `friends_duration=4 years` was present in the context and gemma4:e4b
still abstained. Retrieval p50 was 8.3s / answer p50 3.0s on local hardware.
Both causes argue for GPT-4o-class models (answering *and* embedding) before any
publishable run, plus a chunk-1 ingestion pass.

## Publication checklist

Harness-side prerequisites are all in place (separate `--answer-model` /
`--judge-model`, `--ingest-concurrency`, `scripts/run_publishable_benchmarks.sh`).
The runs themselves are **blocked on a cloud API key** (OPENAI_API_KEY or
ANTHROPIC_API_KEY) — both for answering/judging and for the server's own
extraction + embedding config, which the smoke run showed dominates the score.

- [ ] Point the server at its recommended cloud config (GPT-4o-class
      `LLM_MODEL`, `text-embedding-3-small`+ `EMBEDDING_MODEL`) and re-ingest
- [ ] Full LoCoMo run (10 conversations, 1,540 questions) with a GPT-4o-class
      answer model and a *different* (or at least GPT-4o-class) judge model
- [ ] Full LongMemEval `s` run (500 questions) — estimate cost from a
      10-user partial first (~57M haystack tokens through extraction)
- [ ] `--chunk-turns 1` (product-native granularity) for the headline number
- [ ] Latency table (retrieval p50/p95) alongside accuracy
- [ ] Re-verify competitor numbers from primary sources at publication time
- [ ] Writeup: method, prompts (in `common.py`), limitations, cost
