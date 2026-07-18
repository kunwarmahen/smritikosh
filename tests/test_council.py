"""
Tests for the Deliberation Council (E4, FUTURE.md #4):
    CouncilAgent — four specialists + judge over shared memory, citation
    validation per specialist, quorum guard, verdict logged as memory.

All tests run offline with mocked sessions and LLMs.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from smritikosh.cognition.council import (
    COUNCIL_ROLES,
    CouncilAgent,
    CouncilOpinion,
)
from smritikosh.db.models import Event, SourceType, UserBelief
from smritikosh.memory.episodic import SearchResult
from smritikosh.memory.semantic import FactRecord, UserProfile
from smritikosh.retrieval.context_builder import MemoryContext


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(text: str = "some memory") -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        raw_text=text,
        importance_score=0.7,
        source_type=SourceType.API_EXPLICIT,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )


def make_context(events: list[Event] | None = None) -> MemoryContext:
    events = events if events is not None else [make_event()]
    return MemoryContext(
        user_id="u1",
        query="decision",
        similar_events=[SearchResult(event=e, hybrid_score=0.9) for e in events],
        user_profile=UserProfile(
            user_id="u1",
            app_id="default",
            facts=[FactRecord(
                category="role", key="current", value="entrepreneur", confidence=0.9,
                frequency_count=1, first_seen_at="2026-01-01", last_seen_at="2026-06-01",
            )],
        ),
        beliefs=[UserBelief(
            id=uuid.uuid4(), user_id="u1", app_id="default",
            statement="values iteration", category="value", confidence=0.8,
            evidence_count=2, evidence_event_ids=[],
        )],
        complexity="complex",
    )


def opinion_response(cited: list[str] | None = None, position: str = "support") -> dict:
    return {
        "position": position,
        "argument": "Grounded argument.",
        "confidence": 0.7,
        "cited_event_ids": cited or [],
    }


def verdict_response() -> dict:
    return {
        "recommendation": "Proceed, with a fallback plan.",
        "reasoning": "Three of four specialists support it.",
        "confidence": 0.66,
        "dissent": "The risk specialist warns the cost is unrecoverable.",
        "open_questions": ["Runway after the spend?"],
    }


def make_council(
    ctx: MemoryContext,
    llm_responses: list,   # 4 specialist responses (dict | Exception), then judge
) -> CouncilAgent:
    llm = AsyncMock()

    def _next(*args, **kwargs):
        item = llm_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    llm.extract_structured = AsyncMock(side_effect=_next)
    llm.embed = AsyncMock(return_value=[0.1] * 8)

    builder = AsyncMock()
    builder.build = AsyncMock(return_value=ctx)

    episodic = AsyncMock()
    episodic.store = AsyncMock(return_value=make_event("council log"))

    return CouncilAgent(llm=llm, context_builder=builder, episodic=episodic)


# ── CouncilAgent ──────────────────────────────────────────────────────────────


class TestCouncilAgent:
    @pytest.mark.asyncio
    async def test_skips_on_empty_memory(self):
        ctx = MemoryContext(user_id="u1", query="d")
        agent = make_council(ctx, [])

        result = await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?"
        )

        assert result.skipped is True
        assert "No memory" in result.skip_reason
        agent.llm.extract_structured.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_full_deliberation_produces_verdict_and_opinions(self):
        event = make_event()
        ctx = make_context([event])
        responses = [opinion_response([str(event.id)])] * 4 + [verdict_response()]
        agent = make_council(ctx, responses)

        result = await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Should I raise?"
        )

        assert result.skipped is False
        assert len(result.opinions) == 4
        assert {op.role for op in result.opinions} == set(COUNCIL_ROLES)
        assert result.recommendation == "Proceed, with a fallback plan."
        assert result.dissent.startswith("The risk specialist")
        assert result.confidence == pytest.approx(0.66)
        # 4 specialists + 1 judge
        assert agent.llm.extract_structured.await_count == 5

    @pytest.mark.asyncio
    async def test_specialist_citations_validated_and_unioned(self):
        event = make_event()
        ctx = make_context([event])
        invented = str(uuid.uuid4())
        responses = [
            opinion_response([str(event.id), invented]),   # invented id dropped
            opinion_response([str(event.id)]),             # duplicate not repeated
            opinion_response([]),
            opinion_response([invented]),                  # all invented → empty
            verdict_response(),
        ]
        agent = make_council(ctx, responses)

        result = await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?"
        )

        assert result.cited_event_ids == [str(event.id)]
        assert result.opinions[0].cited_event_ids == [str(event.id)]
        assert result.opinions[3].cited_event_ids == []

    @pytest.mark.asyncio
    async def test_failed_specialist_dropped_not_fatal(self):
        ctx = make_context()
        responses = [
            RuntimeError("provider hiccup"),
            opinion_response(),
            opinion_response(),
            opinion_response(),
            verdict_response(),
        ]
        agent = make_council(ctx, responses)

        result = await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?"
        )

        assert result.skipped is False
        assert len(result.opinions) == 3
        assert "risk" not in {op.role for op in result.opinions}

    @pytest.mark.asyncio
    async def test_below_quorum_skips_without_judge(self):
        ctx = make_context()
        responses = [
            RuntimeError("down"),
            RuntimeError("down"),
            RuntimeError("down"),
            opinion_response(),
        ]
        agent = make_council(ctx, responses)

        result = await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?"
        )

        assert result.skipped is True
        assert "1 of 4" in result.skip_reason
        # judge never called: 4 specialist attempts only
        assert agent.llm.extract_structured.await_count == 4
        agent.episodic.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_judge_failure_skips(self):
        ctx = make_context()
        responses = [opinion_response()] * 4 + [RuntimeError("judge down")]
        agent = make_council(ctx, responses)

        result = await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?"
        )

        assert result.skipped is True
        assert "judge down" in result.skip_reason
        agent.episodic.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_position_normalised(self):
        ctx = make_context()
        responses = [
            opinion_response(position="strongly-agree"),
            opinion_response(),
            opinion_response(),
            opinion_response(),
            verdict_response(),
        ]
        agent = make_council(ctx, responses)

        result = await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?"
        )

        assert result.opinions[0].position == "conditional"

    @pytest.mark.asyncio
    async def test_verdict_logged_as_episodic_event(self):
        ctx = make_context()
        responses = [opinion_response()] * 4 + [verdict_response()]
        agent = make_council(ctx, responses)

        result = await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Raise a round?"
        )

        agent.episodic.store.assert_awaited_once()
        kwargs = agent.episodic.store.await_args.kwargs
        assert kwargs["source_type"] == SourceType.AGENT_COUNCIL
        assert "Raise a round?" in kwargs["raw_text"]
        assert "Dissent:" in kwargs["raw_text"]
        assert kwargs["source_meta"]["positions"]["risk"] == "support"
        assert result.logged_event_id is not None

    @pytest.mark.asyncio
    async def test_builds_context_at_complex_tier(self):
        from smritikosh.retrieval.intent_classifier import ComplexityTier

        ctx = make_context()
        responses = [opinion_response()] * 4 + [verdict_response()]
        agent = make_council(ctx, responses)

        await agent.deliberate(
            AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?"
        )

        kwargs = agent.context_builder.build.await_args.kwargs
        assert kwargs["complexity_override"] == ComplexityTier.COMPLEX
        # memory assembled once, shared by all specialists
        agent.context_builder.build.assert_awaited_once()


class TestCouncilOpinion:
    def test_defaults(self):
        op = CouncilOpinion(role="risk", position="support", argument="a")
        assert op.confidence == 0.5
        assert op.cited_event_ids == []
