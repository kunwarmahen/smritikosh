"""
Benchmark orchestration: ingest → answer → judge → report.

Scores are micro-averaged LLM-judge accuracy ("J" in the Mem0 paper), overall
and per category, plus retrieval/answer latency percentiles — the numbers the
published Mem0 / Zep / Memobase comparisons report.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from smritikosh.llm.adapter import LLMAdapter
from smritikosh.sdk.client import SmritikoshClient

from evals.benchmarks.adapter import (
    IngestState,
    QAResult,
    answer_question,
    cleanup_user,
    ingest_user,
    judge_result,
)
from evals.benchmarks.common import BenchUser
from evals.benchmarks.datasets import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class BenchConfig:
    benchmark: str  # "locomo" | "longmemeval"
    base_url: str = "http://localhost:8080"
    api_key: str = ""
    app_id: str = ""
    chunk_turns: int = 1
    qa_concurrency: int = 4
    timeout_s: float = 300.0  # /context and encode are LLM-bound server-side
    data_dir: Path = DATA_DIR

    @classmethod
    def from_env(cls, benchmark: str, **overrides) -> "BenchConfig":
        api_key = os.environ.get("SMRITIKOSH_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "SMRITIKOSH_API_KEY is not set — benchmarks need an *admin* API key "
                "(they create many synthetic users)."
            )
        return cls(
            benchmark=benchmark,
            base_url=os.environ.get("SMRITIKOSH_BASE_URL", "http://localhost:8080"),
            api_key=api_key,
            app_id=f"bench-{benchmark}",
            **overrides,
        )

    def make_client(self) -> SmritikoshClient:
        return SmritikoshClient(
            base_url=self.base_url,
            app_id=self.app_id,
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def state(self) -> IngestState:
        suffix = f"-chunk{self.chunk_turns}" if self.chunk_turns != 1 else ""
        return IngestState(self.data_dir / f"state-{self.benchmark}{suffix}.json")


@dataclass
class BenchReport:
    benchmark: str
    results: list[QAResult] = field(default_factory=list)
    ingested_events: int = 0
    ingest_s: float = 0.0
    qa_s: float = 0.0
    answer_model: str = ""

    @property
    def scored(self) -> list[QAResult]:
        return [r for r in self.results if not r.error]

    @property
    def accuracy(self) -> float:
        scored = self.scored
        return sum(r.correct for r in scored) / len(scored) if scored else 0.0

    def by_category(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.scored:
            entry = out.setdefault(r.category, {"total": 0, "correct": 0})
            entry["total"] += 1
            entry["correct"] += int(r.correct)
        for entry in out.values():
            entry["accuracy"] = entry["correct"] / entry["total"]
        return dict(sorted(out.items()))

    def latency(self) -> dict[str, float]:
        def pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            return ordered[min(int(p * len(ordered)), len(ordered) - 1)]

        retrieval = [r.retrieval_ms for r in self.scored]
        answer = [r.answer_ms for r in self.scored]
        return {
            "retrieval_p50_ms": round(pct(retrieval, 0.50), 1),
            "retrieval_p95_ms": round(pct(retrieval, 0.95), 1),
            "answer_p50_ms": round(pct(answer, 0.50), 1),
            "answer_p95_ms": round(pct(answer, 0.95), 1),
        }

    def to_json(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "answer_model": self.answer_model,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "questions": len(self.results),
            "errors": sum(1 for r in self.results if r.error),
            "accuracy": round(self.accuracy, 4),
            "by_category": self.by_category(),
            "latency": self.latency(),
            "ingested_events": self.ingested_events,
            "ingest_s": round(self.ingest_s, 1),
            "qa_s": round(self.qa_s, 1),
            "results": [r.to_json() for r in self.results],
        }


async def run_benchmark(
    users: list[BenchUser],
    config: BenchConfig,
    *,
    llm: LLMAdapter | None = None,
    skip_ingest: bool = False,
    cleanup: bool = False,
    progress: bool = True,
) -> BenchReport:
    llm = llm or LLMAdapter()
    report = BenchReport(benchmark=config.benchmark)
    report.answer_model = getattr(llm, "_chat_model", "unknown")
    state = config.state()

    async with config.make_client() as client:
        # ── Ingest (sequential: encode is already LLM-bound server-side) ─────
        if not skip_ingest:
            started = time.monotonic()
            for i, user in enumerate(users, 1):
                stored = await ingest_user(
                    client, user, state, chunk_turns=config.chunk_turns
                )
                report.ingested_events += stored
                if progress and stored:
                    print(
                        f"  ingested {user.user_id} "
                        f"({stored} events, {i}/{len(users)} users)",
                        flush=True,
                    )
            report.ingest_s = time.monotonic() - started

        # ── QA with bounded concurrency ───────────────────────────────────────
        started = time.monotonic()
        semaphore = asyncio.Semaphore(config.qa_concurrency)
        pairs = [(user, q) for user in users for q in user.questions]

        async def run_one(user: BenchUser, q) -> QAResult:
            async with semaphore:
                try:
                    result = await answer_question(client, llm, user.user_id, q)
                    result.correct = await judge_result(llm, result)
                except Exception as exc:  # noqa: BLE001 — record, keep going
                    logger.exception("QA failed for %s", q.question_id)
                    # str(exc) alone can be empty (httpx timeouts) — keep the type
                    result = QAResult(
                        question_id=q.question_id,
                        category=q.category,
                        question=q.question,
                        gold_answer=q.gold_answer,
                        is_abstention=q.is_abstention,
                        error=f"{type(exc).__name__}: {exc}".strip(": "),
                    )
                if progress:
                    mark = "✓" if result.correct else ("!" if result.error else "✗")
                    print(f"  {mark} [{result.category}] {q.question[:70]}", flush=True)
                return result

        report.results = list(
            await asyncio.gather(*(run_one(user, q) for user, q in pairs))
        )
        report.qa_s = time.monotonic() - started

        if cleanup:
            for user in users:
                await cleanup_user(client, user)
                state.reset()

    return report


def apply_limits(
    users: list[BenchUser],
    *,
    max_users: int | None = None,
    max_questions: int | None = None,
) -> list[BenchUser]:
    """Deterministic subsetting for cheap partial runs (interleaves categories)."""
    subset = users[:max_users] if max_users else list(users)
    if max_questions is None:
        return subset
    # Round-robin across users and categories so a small cap stays representative.
    out: list[BenchUser] = [
        BenchUser(user_id=u.user_id, sessions=u.sessions, questions=[]) for u in subset
    ]
    remaining = max_questions
    index = 0
    queues = [list(u.questions) for u in subset]
    while remaining > 0 and any(queues):
        if queues[index % len(queues)]:
            out[index % len(out)].questions.append(queues[index % len(queues)].pop(0))
            remaining -= 1
        index += 1
    return [u for u in out if u.questions or not max_questions]
