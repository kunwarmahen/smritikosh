"""
Eval runner — executes the golden set against the production extraction path.

For each golden case it builds the exact prompt production builds:

  kind="event"    → smritikosh.memory.hippocampus._build_extraction_prompt
                    (the POST /memory/event encode pipeline)
  kind="session"  → prepare_transcript + build_delta_prompt
                    (the POST /ingest/session passive-extraction pipeline,
                    including assistant-turn filtering and sentinel stripping)

then calls LLMAdapter.extract_structured with the production schema/example
and scores the returned facts with evals.matcher.

The optional LLM judge re-examines leftover (expected, predicted) pairs in the
same category whose values lexical matching rejected, and upgrades them to TPs
when the model deems them equivalent — useful for paraphrase-heavy values.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.hippocampus import (
    _EXTRACTION_EXAMPLE,
    _EXTRACTION_SCHEMA,
    _build_extraction_prompt,
)
from smritikosh.memory.semantic import FactRecord
from smritikosh.processing.transcript_utils import build_delta_prompt, prepare_transcript

from evals.matcher import AggregateScore, CaseScore, ExpectedFact, score_case

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).parent / "golden"

_JUDGE_PROMPT = (
    "Two systems extracted a fact about the same user from the same text.\n"
    "Fact A: {a}\n"
    "Fact B: {b}\n"
    "Do A and B express the same underlying fact about the user "
    "(same attribute, equivalent value)? Answer with a single word: yes or no."
)


@dataclass
class GoldenCase:
    """One golden-set case, loaded from evals/golden/*.json."""

    id: str
    kind: str  # "event" | "session"
    expected: list[ExpectedFact]
    forbidden: list[ExpectedFact]
    notes: str = ""
    content: str = ""  # kind == "event"
    turns: list[dict] = field(default_factory=list)  # kind == "session"
    existing_facts: list[dict] = field(default_factory=list)  # kind == "session"

    @classmethod
    def from_json(cls, obj: dict) -> "GoldenCase":
        kind = obj["kind"]
        if kind not in ("event", "session"):
            raise ValueError(f"case {obj.get('id')}: unknown kind {kind!r}")
        return cls(
            id=obj["id"],
            kind=kind,
            notes=obj.get("notes", ""),
            content=obj.get("content", ""),
            turns=obj.get("turns", []),
            existing_facts=obj.get("existing_facts", []),
            expected=[ExpectedFact.from_json(e) for e in obj.get("expected", [])],
            forbidden=[ExpectedFact.from_json(e) for e in obj.get("forbidden", [])],
        )


def load_cases(golden_dir: Path = GOLDEN_DIR) -> list[GoldenCase]:
    """Load every case from every .json file in the golden dir (sorted, stable)."""
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for path in sorted(golden_dir.glob("*.json")):
        for obj in json.loads(path.read_text()):
            case = GoldenCase.from_json(obj)
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
            cases.append(case)
    return cases


def _to_fact_records(raw: list[dict]) -> list[FactRecord]:
    """Golden existing_facts ({category,key,value,confidence}) → FactRecord."""
    return [
        FactRecord(
            category=f["category"],
            key=f.get("key", "fact"),
            value=f["value"],
            confidence=float(f.get("confidence", 0.9)),
            frequency_count=1,
            first_seen_at="2026-01-01T00:00:00+00:00",
            last_seen_at="2026-01-01T00:00:00+00:00",
        )
        for f in raw
    ]


async def extract_for_case(llm: LLMAdapter, case: GoldenCase) -> list[dict]:
    """Run the production extraction path for one case, return predicted facts."""
    if case.kind == "event":
        prompt = _build_extraction_prompt(case.content)
    else:
        transcript = prepare_transcript(case.turns)
        if not transcript.combined_text.strip():
            # Production skips extraction when no user content survives filtering.
            return []
        prompt = build_delta_prompt(
            transcript.user_turns, _to_fact_records(case.existing_facts)
        )
    result = await llm.extract_structured(
        prompt=prompt,
        schema_description=_EXTRACTION_SCHEMA,
        example_output=_EXTRACTION_EXAMPLE,
    )
    facts = result.get("facts", [])
    return facts if isinstance(facts, list) else []


async def _judge_equivalent(llm: LLMAdapter, a: str, b: str) -> bool:
    answer = await llm.complete(
        messages=[{"role": "user", "content": _JUDGE_PROMPT.format(a=a, b=b)}],
        temperature=0.0,
    )
    return answer.strip().lower().startswith("yes")


async def _apply_judge(llm: LLMAdapter, case: GoldenCase, score: CaseScore) -> None:
    """Upgrade same-category FN/FP pairs that the LLM judge deems equivalent."""
    if not score.unmatched_expected or not score.unmatched_predicted:
        return
    for exp in list(score.unmatched_expected):
        exp_categories = exp["category"].split("|")
        for pred in list(score.unmatched_predicted):
            if pred.get("category") not in exp_categories:
                continue
            a = f"{exp['category']}: {exp['value']}"
            b = f"{pred.get('category')}: {pred.get('value')}"
            try:
                same = await _judge_equivalent(llm, a, b)
            except Exception:
                logger.exception("judge call failed for case %s", case.id)
                return
            if same:
                score.tp += 1
                score.fn -= 1
                score.fp -= 1
                score.unmatched_expected.remove(exp)
                score.unmatched_predicted.remove(pred)
                score.matched.append({"expected": exp["value"], "predicted": pred, "via": "judge"})
                break


async def run_case(
    llm: LLMAdapter, case: GoldenCase, *, judge: bool = False
) -> CaseScore:
    try:
        predicted = await extract_for_case(llm, case)
    except Exception as exc:
        logger.exception("extraction failed for case %s", case.id)
        score = CaseScore(case_id=case.id, error=str(exc))
        score.fn = sum(1 for e in case.expected if not e.optional)
        return score
    score = score_case(case.id, predicted, case.expected, case.forbidden)
    if judge:
        await _apply_judge(llm, case, score)
    return score


async def run_eval(
    cases: list[GoldenCase],
    *,
    llm: LLMAdapter | None = None,
    concurrency: int = 4,
    judge: bool = False,
) -> AggregateScore:
    """Run all cases with bounded concurrency; preserves case order."""
    llm = llm or LLMAdapter()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(case: GoldenCase) -> CaseScore:
        async with semaphore:
            return await run_case(llm, case, judge=judge)

    scores = await asyncio.gather(*(bounded(c) for c in cases))
    return AggregateScore(cases=list(scores))


# ── Reporting ─────────────────────────────────────────────────────────────────


def category_breakdown(cases: list[GoldenCase], agg: AggregateScore) -> dict[str, dict]:
    """Per-category recall over expected facts (TP/FN attribution by spec category)."""
    by_case = {s.case_id: s for s in agg.cases}
    stats: dict[str, dict] = {}
    for case in cases:
        score = by_case.get(case.id)
        if score is None:
            continue
        missed = {(e["category"], e["value"]) for e in score.unmatched_expected}
        for spec in case.expected:
            if spec.optional:
                continue
            cat = spec.categories[0]
            entry = stats.setdefault(cat, {"expected": 0, "found": 0})
            entry["expected"] += 1
            if ("|".join(spec.categories), spec.value) not in missed:
                entry["found"] += 1
    for entry in stats.values():
        entry["recall"] = entry["found"] / entry["expected"] if entry["expected"] else 1.0
    return dict(sorted(stats.items()))


def build_report(
    cases: list[GoldenCase], agg: AggregateScore, *, model: str, duration_s: float
) -> dict:
    """Full machine-readable report (also the --baseline comparison input)."""
    return {
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": round(duration_s, 1),
        "aggregate": {
            "cases": len(agg.cases),
            "tp": agg.tp,
            "fp": agg.fp,
            "fn": agg.fn,
            "violations": agg.violations,
            "errors": agg.errors,
            "precision": round(agg.precision, 4),
            "recall": round(agg.recall, 4),
            "f1": round(agg.f1, 4),
        },
        "categories": category_breakdown(cases, agg),
        "cases": [
            {
                "id": s.case_id,
                "tp": s.tp,
                "fp": s.fp,
                "fn": s.fn,
                "violations": s.violations,
                "precision": round(s.precision, 4),
                "recall": round(s.recall, 4),
                "f1": round(s.f1, 4),
                "missed": s.unmatched_expected,
                "spurious": [
                    {"category": f.get("category"), "key": f.get("key"), "value": f.get("value")}
                    for f in s.unmatched_predicted
                ],
                "violation_facts": s.violation_facts,
                **({"error": s.error} if s.error else {}),
            }
            for s in agg.cases
        ],
    }
