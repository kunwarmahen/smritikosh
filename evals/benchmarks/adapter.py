"""
Smritikosh adapter — ingestion, retrieval-augmented answering, and judging.

Everything goes through the public API via the Python SDK against a live
server, so benchmark numbers measure the product as deployed (auth, quotas,
and audit included), not a shortcut through internals.

Requires an **admin** API key (benchmarks create many synthetic user_ids).

Ingestion is resumable: a state file in the data dir records which users are
fully ingested; interrupted users are wiped and re-ingested for consistency.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from smritikosh.llm.adapter import LLMAdapter
from smritikosh.sdk.client import SmritikoshClient, SmritikoshError

from evals.benchmarks.common import (
    ANSWER_SYSTEM,
    JUDGE_ABSTENTION_PROMPT,
    JUDGE_PROMPT,
    BenchQuestion,
    BenchUser,
    build_answer_prompt,
    format_turn_content,
)

logger = logging.getLogger(__name__)

ABSTAIN_MARKERS = ("i don't know", "i dont know", "not mentioned", "no information")


# ── Ingestion state ───────────────────────────────────────────────────────────


class IngestState:
    """Tracks fully-ingested users so interrupted runs resume cheaply."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._done: set[str] = set()
        if path.exists():
            self._done = set(json.loads(path.read_text()).get("done", []))

    def is_done(self, user_id: str) -> bool:
        return user_id in self._done

    def mark_done(self, user_id: str) -> None:
        self._done.add(user_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"done": sorted(self._done)}, indent=0))

    def reset(self) -> None:
        self._done.clear()
        if self.path.exists():
            self.path.unlink()


# ── Ingestion ─────────────────────────────────────────────────────────────────


def chunk_session_events(user: BenchUser, chunk_turns: int = 1) -> list[str]:
    """
    Render a user's history as ingestible event strings.

    chunk_turns=1 → one event per turn (product-native granularity).
    chunk_turns=N → N consecutive turns of a session per event (cheaper:
    fewer extraction calls; the per-line date+speaker prefixes are kept).
    """
    events: list[str] = []
    for session in user.sessions:
        lines = [format_turn_content(session, t) for t in session.turns if t.content.strip()]
        for i in range(0, len(lines), max(1, chunk_turns)):
            chunk = "\n".join(lines[i : i + max(1, chunk_turns)])
            if chunk.strip():
                events.append(chunk)
    return events


async def ingest_user(
    client: SmritikoshClient,
    user: BenchUser,
    state: IngestState,
    *,
    chunk_turns: int = 1,
) -> int:
    """Ingest one user's history; returns events stored (0 if already done)."""
    if state.is_done(user.user_id):
        return 0
    # Partial ingestion from an interrupted run would skew retrieval — start clean.
    try:
        await client.delete_user_memory(user_id=user.user_id)
    except SmritikoshError as exc:
        if exc.status_code != 404:
            raise
    events = chunk_session_events(user, chunk_turns)
    for content in events:
        await client.encode(user_id=user.user_id, content=content)
    state.mark_done(user.user_id)
    return len(events)


async def cleanup_user(client: SmritikoshClient, user: BenchUser) -> None:
    try:
        await client.delete_user_memory(user_id=user.user_id)
    except SmritikoshError as exc:
        if exc.status_code != 404:
            raise


# ── Question answering ────────────────────────────────────────────────────────


@dataclass
class QAResult:
    question_id: str
    category: str
    question: str
    gold_answer: str
    answer: str = ""
    correct: bool = False
    is_abstention: bool = False
    retrieval_ms: float = 0.0
    answer_ms: float = 0.0
    memories_used: int = 0
    error: str = ""
    judge_raw: str = ""

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


async def answer_question(
    client: SmritikoshClient,
    llm: LLMAdapter,
    user_id: str,
    question: BenchQuestion,
) -> QAResult:
    """Retrieve memory context for the question, then answer from it."""
    result = QAResult(
        question_id=question.question_id,
        category=question.category,
        question=question.question,
        gold_answer=question.gold_answer,
        is_abstention=question.is_abstention,
    )
    started = time.monotonic()
    context = await client.build_context(user_id=user_id, query=question.question)
    result.retrieval_ms = (time.monotonic() - started) * 1000
    result.memories_used = context.total_memories

    started = time.monotonic()
    result.answer = (
        await llm.complete(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": build_answer_prompt(
                        context.context_text, question.question, question.question_date
                    ),
                },
            ],
            temperature=0.0,
        )
    ).strip()
    result.answer_ms = (time.monotonic() - started) * 1000
    return result


# ── Judging ───────────────────────────────────────────────────────────────────


def looks_like_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in ABSTAIN_MARKERS)


async def judge_result(llm: LLMAdapter, result: QAResult) -> bool:
    """Binary LLM-as-judge (the standard LoCoMo/LongMemEval protocol)."""
    if result.is_abstention:
        # Fast path: an explicit abstention is correct by definition.
        if looks_like_abstention(result.answer):
            result.judge_raw = "abstained"
            return True
        prompt = JUDGE_ABSTENTION_PROMPT.format(
            question=result.question, answer=result.answer
        )
    else:
        prompt = JUDGE_PROMPT.format(
            question=result.question, gold=result.gold_answer, answer=result.answer
        )
    raw = await llm.complete(
        messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    result.judge_raw = raw.strip()
    return "correct" in raw.strip().lower()[:20]
