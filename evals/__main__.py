"""
CLI for the extraction-quality eval harness.

    python -m evals                         # run the full golden set
    python -m evals --list                  # show cases without running
    python -m evals --filter diet           # only cases whose id contains "diet"
    python -m evals --kind session          # only session (delta) cases
    python -m evals --judge                 # LLM judge for near-miss values
    python -m evals --json report.json      # write machine-readable report
    python -m evals --baseline report.json  # compare against a previous run
    python -m evals --min-f1 0.75           # exit 1 below threshold (CI gate)

Requires a configured LLM provider (same .env the server uses). Costs tokens —
run manually or nightly, not per-commit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from evals.runner import GOLDEN_DIR, build_report, category_breakdown, load_cases, run_eval


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_DIR, help="golden-set directory")
    parser.add_argument("--filter", default="", help="only run cases whose id contains this")
    parser.add_argument("--kind", choices=["event", "session"], help="only run this case kind")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="LLM judge for near-miss values")
    parser.add_argument("--json", type=Path, help="write the full report to this path")
    parser.add_argument("--baseline", type=Path, help="previous --json report to compare against")
    parser.add_argument("--min-f1", type=float, help="exit non-zero if aggregate F1 is below this")
    parser.add_argument("--list", action="store_true", help="list matching cases and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    cases = load_cases(args.golden)
    if args.filter:
        cases = [c for c in cases if args.filter in c.id]
    if args.kind:
        cases = [c for c in cases if c.kind == args.kind]
    if not cases:
        print("no cases match the given filters", file=sys.stderr)
        return 2

    if args.list:
        for case in cases:
            print(f"{case.id:<40} {case.kind:<8} expected={len(case.expected)} {case.notes}")
        return 0

    from smritikosh.llm.adapter import LLMAdapter  # deferred: needs configured env

    llm = LLMAdapter()
    model = llm._chat_model  # noqa: SLF001 — reporting only
    print(f"Running {len(cases)} cases against {model} "
          f"(concurrency={args.concurrency}, judge={args.judge}) …\n")

    start = time.monotonic()
    agg = asyncio.run(run_eval(cases, llm=llm, concurrency=args.concurrency, judge=args.judge))
    duration = time.monotonic() - start

    # ── Per-case table ────────────────────────────────────────────────────────
    print(f"{'case':<40} {'P':>6} {'R':>6} {'F1':>6}  detail")
    for score in agg.cases:
        detail = []
        if score.error:
            detail.append("ERROR")
        if score.fn:
            detail.append("missed: " + "; ".join(e["value"] for e in score.unmatched_expected))
        if score.fp:
            detail.append(
                "spurious: "
                + "; ".join(str(f.get("value")) for f in score.unmatched_predicted[:3])
            )
        if score.violations:
            detail.append(f"VIOLATIONS={score.violations}")
        print(
            f"{score.case_id:<40} {score.precision:>6.2f} {score.recall:>6.2f} "
            f"{score.f1:>6.2f}  {' | '.join(detail)}"
        )

    # ── Category recall ───────────────────────────────────────────────────────
    print("\nPer-category recall (required expected facts):")
    for cat, entry in category_breakdown(cases, agg).items():
        print(f"  {cat:<14} {entry['found']}/{entry['expected']}  ({entry['recall']:.0%})")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(
        f"\nAggregate: precision={agg.precision:.3f} recall={agg.recall:.3f} "
        f"F1={agg.f1:.3f}  (TP={agg.tp} FP={agg.fp} FN={agg.fn} "
        f"violations={agg.violations} errors={agg.errors})  [{duration:.0f}s]"
    )

    report = build_report(cases, agg, model=model, duration_s=duration)

    if args.baseline:
        base = json.loads(args.baseline.read_text())["aggregate"]
        print(
            f"vs baseline ({base.get('f1')}): "
            f"ΔP={agg.precision - base['precision']:+.3f} "
            f"ΔR={agg.recall - base['recall']:+.3f} "
            f"ΔF1={agg.f1 - base['f1']:+.3f}"
        )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"report written to {args.json}")

    if args.min_f1 is not None and agg.f1 < args.min_f1:
        print(f"FAIL: F1 {agg.f1:.3f} < --min-f1 {args.min_f1}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
