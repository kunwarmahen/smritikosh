"""
CLI for the public memory benchmarks.

    python -m evals.benchmarks locomo                      # full LoCoMo (slow!)
    python -m evals.benchmarks locomo --users 2 --questions 40
    python -m evals.benchmarks locomo --chunk-turns 10     # cheaper ingestion
    python -m evals.benchmarks longmemeval --variant oracle --users 50
    python -m evals.benchmarks locomo --skip-ingest --json out.json
    python -m evals.benchmarks locomo --reset-state        # force re-ingestion
    python -m evals.benchmarks locomo \
        --answer-model openai:gpt-4o --judge-model openai:gpt-4o-mini

Needs: a running Smritikosh server (SMRITIKOSH_BASE_URL) and an **admin** API
key (SMRITIKOSH_API_KEY). Answering/judging default to the .env LLM the server
uses; --answer-model / --judge-model override them with any provider:model
(keys from the provider's standard env var, e.g. OPENAI_API_KEY) — required
for publishable runs, which need GPT-4o-class answering and judging. Full runs
make thousands of LLM calls — start with --users/--questions limits and scale up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from evals.benchmarks.datasets import DATA_DIR, load_locomo, load_longmemeval
from evals.benchmarks.runner import BenchConfig, apply_limits, llm_for, run_benchmark


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.benchmarks", description=__doc__)
    parser.add_argument("benchmark", choices=["locomo", "longmemeval"])
    parser.add_argument("--variant", default="oracle", choices=["oracle", "s", "m"],
                        help="LongMemEval haystack size (default: oracle)")
    parser.add_argument("--include-adversarial", action="store_true",
                        help="LoCoMo: include category-5 (adversarial) questions")
    parser.add_argument("--users", type=int, help="limit number of benchmark users")
    parser.add_argument("--questions", type=int, help="limit total questions")
    parser.add_argument("--chunk-turns", type=int, default=1,
                        help="turns per ingested event (1 = product-native; "
                             "higher = fewer extraction calls)")
    parser.add_argument("--answer-model", metavar="PROVIDER:MODEL",
                        help="answer LLM, e.g. openai:gpt-4o "
                             "(default: the server's .env LLM)")
    parser.add_argument("--judge-model", metavar="PROVIDER:MODEL",
                        help="judge LLM, e.g. openai:gpt-4o-mini "
                             "(default: the answer model)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="QA concurrency (use 1 for a local single-GPU LLM)")
    parser.add_argument("--ingest-concurrency", type=int, default=1,
                        help="users ingested in parallel; per-user turn order "
                             "is always preserved (raise for cloud providers)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-request timeout in seconds (server calls are LLM-bound)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="assume history is already ingested")
    parser.add_argument("--cleanup", action="store_true",
                        help="delete benchmark users' memory after the run")
    parser.add_argument("--reset-state", action="store_true",
                        help="clear the ingestion-resume state first")
    parser.add_argument("--json", type=Path, help="write full report to this path")
    parser.add_argument("--quiet", action="store_true", help="no per-question progress")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    if args.benchmark == "locomo":
        users = load_locomo(DATA_DIR, include_adversarial=args.include_adversarial)
    else:
        users = load_longmemeval(DATA_DIR, variant=args.variant)
    users = apply_limits(users, max_users=args.users, max_questions=args.questions)

    total_q = sum(len(u.questions) for u in users)
    total_turns = sum(u.total_turns for u in users)
    config = BenchConfig.from_env(args.benchmark, chunk_turns=args.chunk_turns,
                                  qa_concurrency=args.concurrency,
                                  ingest_concurrency=args.ingest_concurrency,
                                  timeout_s=args.timeout)
    if args.reset_state:
        config.state().reset()

    print(f"{args.benchmark}: {len(users)} users, {total_turns} turns, {total_q} questions")
    print(f"server={config.base_url} app_id={config.app_id} chunk_turns={config.chunk_turns}\n")

    report = asyncio.run(
        run_benchmark(
            users,
            config,
            llm=llm_for(args.answer_model) if args.answer_model else None,
            judge_llm=llm_for(args.judge_model) if args.judge_model else None,
            skip_ingest=args.skip_ingest,
            cleanup=args.cleanup,
            progress=not args.quiet,
        )
    )

    data = report.to_json()
    print(f"\n{args.benchmark} — model={data['answer_model']} judge={data['judge_model']}")
    print(f"Judge accuracy (J): {data['accuracy']:.1%} "
          f"({data['questions']} questions, {data['errors']} errors)")
    for category, entry in data["by_category"].items():
        print(f"  {category:<24} {entry['correct']}/{entry['total']}  ({entry['accuracy']:.0%})")
    lat = data["latency"]
    print(f"Latency: retrieval p50={lat['retrieval_p50_ms']}ms p95={lat['retrieval_p95_ms']}ms · "
          f"answer p50={lat['answer_p50_ms']}ms p95={lat['answer_p95_ms']}ms")
    if not args.skip_ingest:
        print(f"Ingest: {data['ingested_events']} events in {data['ingest_s']}s · "
              f"QA: {data['qa_s']}s")

    if args.json:
        args.json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
