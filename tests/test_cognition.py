"""
Tests for the cognitive agent layer (item E4):
    PredictionEngine — predict-observe-learn scoring + importance nudges
    DecisionAgent    — memory-grounded recommendations with citation validation
    ReflectionAgent  — drift/contradiction cycles, guards, insight validation

All tests run offline with mocked sessions and LLMs.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from smritikosh.cognition.decision import DecisionAgent
from smritikosh.cognition.prediction import Prediction, PredictionEngine
from smritikosh.cognition.reflection import ReflectionAgent, ReflectionResult
from smritikosh.db.models import (
    Event,
    MemoryPrediction,
    Reflection,
    ReflectionKind,
    SourceType,
    UserBelief,
)
from smritikosh.memory.episodic import SearchResult
from smritikosh.memory.semantic import FactRecord, UserProfile
from smritikosh.retrieval.context_builder import MemoryContext


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(text: str = "some memory", source_type: str = SourceType.API_EXPLICIT) -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        raw_text=text,
        importance_score=0.7,
        source_type=source_type,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )


def make_belief(statement: str = "values iteration", confidence: float = 0.8) -> UserBelief:
    return UserBelief(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        statement=statement,
        category="value",
        confidence=confidence,
        evidence_count=2,
        evidence_event_ids=[],
    )


def make_fact() -> FactRecord:
    return FactRecord(
        category="role", key="current", value="entrepreneur", confidence=0.9,
        frequency_count=1, first_seen_at="2026-01-01", last_seen_at="2026-06-01",
    )


# ── PredictionEngine.predict ──────────────────────────────────────────────────


class TestPredict:
    @pytest.mark.asyncio
    async def test_predict_persists_and_returns_prediction(self):
        engine = PredictionEngine()
        session = AsyncMock()
        cluster_rows = [SimpleNamespace(cluster_id=3, recalls=10)]
        eid = uuid.uuid4()
        event_rows = [SimpleNamespace(id=eid)]
        session.execute = AsyncMock(side_effect=[cluster_rows, event_rows])

        p = await engine.predict(session, user_id="u1", query="what should I build?")

        session.add.assert_called_once()
        added = session.add.call_args.args[0]
        assert isinstance(added, MemoryPrediction)
        assert added.predicted_cluster_ids == [3]
        assert added.predicted_event_ids == [str(eid)]
        assert isinstance(p, Prediction)
        assert p.predicted_event_ids == [str(eid)]

    @pytest.mark.asyncio
    async def test_predict_truncates_query_preview(self):
        engine = PredictionEngine()
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=[[], []])

        await engine.predict(session, user_id="u1", query="x" * 500)

        added = session.add.call_args.args[0]
        assert len(added.query_preview) == 300


# ── PredictionEngine.record_outcome ───────────────────────────────────────────


def make_prediction_row(predicted: list[str]) -> MemoryPrediction:
    return MemoryPrediction(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        query_preview="q",
        intent="general",
        predicted_event_ids=predicted,
        predicted_cluster_ids=[],
        actual_event_ids=[],
    )


class TestRecordOutcome:
    @pytest.mark.asyncio
    async def test_hit_rate_computed(self):
        engine = PredictionEngine()
        a, b, c = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        row = make_prediction_row(predicted=[a, b])
        session = AsyncMock()
        session.get = AsyncMock(return_value=row)

        # actual = {a, c}: one of two surfaced events was predicted → 0.5
        hit_rate = await engine.record_outcome(session, str(row.id), [a, c])

        assert hit_rate == pytest.approx(0.5)
        assert row.hit_rate == pytest.approx(0.5)
        assert row.scored_at is not None
        assert set(row.actual_event_ids) == {a, c}

    @pytest.mark.asyncio
    async def test_importance_nudges_applied_for_hits_and_misses(self):
        engine = PredictionEngine()
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        row = make_prediction_row(predicted=[a, b])
        session = AsyncMock()
        session.get = AsyncMock(return_value=row)

        await engine.record_outcome(session, str(row.id), [a])

        # one UPDATE for the hit set, one for the miss set
        assert session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_actual_scores_zero_without_nudges(self):
        engine = PredictionEngine()
        row = make_prediction_row(predicted=[str(uuid.uuid4())])
        session = AsyncMock()
        session.get = AsyncMock(return_value=row)

        hit_rate = await engine.record_outcome(session, str(row.id), [])

        assert hit_rate == 0.0
        # only the miss-decay UPDATE fires (no hits)
        assert session.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_already_scored_is_noop(self):
        engine = PredictionEngine()
        row = make_prediction_row(predicted=[])
        row.scored_at = datetime.now(timezone.utc)
        session = AsyncMock()
        session.get = AsyncMock(return_value=row)

        assert await engine.record_outcome(session, str(row.id), ["x"]) is None
        session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_prediction_is_noop(self):
        engine = PredictionEngine()
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)

        assert await engine.record_outcome(session, str(uuid.uuid4()), ["x"]) is None

    @pytest.mark.asyncio
    async def test_nudges_disabled_when_zero(self):
        engine = PredictionEngine(hit_bump=0.0, miss_decay=0.0)
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        row = make_prediction_row(predicted=[a, b])
        session = AsyncMock()
        session.get = AsyncMock(return_value=row)

        await engine.record_outcome(session, str(row.id), [a])

        session.execute.assert_not_awaited()


# ── DecisionAgent ─────────────────────────────────────────────────────────────


def make_context(events: list[Event] | None = None, beliefs=None) -> MemoryContext:
    events = events if events is not None else [make_event()]
    return MemoryContext(
        user_id="u1",
        query="decision",
        similar_events=[SearchResult(event=e, hybrid_score=0.9) for e in events],
        user_profile=UserProfile(user_id="u1", app_id="default", facts=[make_fact()]),
        beliefs=beliefs or [make_belief()],
        complexity="complex",
    )


def make_decision_agent(ctx: MemoryContext, llm_response: dict | Exception) -> DecisionAgent:
    llm = AsyncMock()
    if isinstance(llm_response, Exception):
        llm.extract_structured = AsyncMock(side_effect=llm_response)
    else:
        llm.extract_structured = AsyncMock(return_value=llm_response)
    llm.embed = AsyncMock(return_value=[0.1] * 8)

    builder = AsyncMock()
    builder.build = AsyncMock(return_value=ctx)

    episodic = AsyncMock()
    episodic.store = AsyncMock(return_value=make_event("decision log"))

    return DecisionAgent(llm=llm, context_builder=builder, episodic=episodic)


class TestDecisionAgent:
    @pytest.mark.asyncio
    async def test_skips_on_empty_memory(self):
        ctx = MemoryContext(user_id="u1", query="d")
        agent = make_decision_agent(ctx, {})

        result = await agent.decide(AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?")

        assert result.skipped is True
        assert "No memory" in result.skip_reason
        agent.llm.extract_structured.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_parses_recommendation_and_alignment(self):
        event = make_event()
        ctx = make_context([event])
        response = {
            "recommendation": "Do it.",
            "reasoning": "Memory supports it.",
            "confidence": 0.8,
            "belief_alignment": [
                {"belief": "values iteration", "alignment": "supports", "note": "n"},
                {"belief": "x", "alignment": "bogus-value", "note": ""},
            ],
            "risks": ["It may fail."],
            "cited_event_ids": [str(event.id)],
            "open_questions": ["Budget?"],
        }
        agent = make_decision_agent(ctx, response)

        result = await agent.decide(AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?")

        assert result.skipped is False
        assert result.recommendation == "Do it."
        assert result.confidence == pytest.approx(0.8)
        assert result.belief_alignment[0].alignment == "supports"
        assert result.belief_alignment[1].alignment == "neutral"  # invalid → neutral
        assert result.risks == ["It may fail."]
        assert result.open_questions == ["Budget?"]

    @pytest.mark.asyncio
    async def test_hallucinated_citations_filtered(self):
        event = make_event()
        ctx = make_context([event])
        response = {
            "recommendation": "Do it.",
            "reasoning": "r",
            "confidence": 0.7,
            "cited_event_ids": [str(event.id), str(uuid.uuid4())],  # 2nd is invented
        }
        agent = make_decision_agent(ctx, response)

        result = await agent.decide(AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?")

        assert result.cited_event_ids == [str(event.id)]

    @pytest.mark.asyncio
    async def test_decision_logged_as_episodic_event(self):
        ctx = make_context()
        agent = make_decision_agent(ctx, {"recommendation": "Yes", "reasoning": "r", "confidence": 0.6})

        result = await agent.decide(AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?")

        agent.episodic.store.assert_awaited_once()
        kwargs = agent.episodic.store.await_args.kwargs
        assert kwargs["source_type"] == SourceType.AGENT_DECISION
        assert "Should I?" in kwargs["raw_text"]
        assert result.logged_event_id is not None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_skipped(self):
        ctx = make_context()
        agent = make_decision_agent(ctx, RuntimeError("provider down"))

        result = await agent.decide(AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?")

        assert result.skipped is True
        assert "provider down" in result.skip_reason
        agent.episodic.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_builds_context_at_complex_tier(self):
        from smritikosh.retrieval.intent_classifier import ComplexityTier

        ctx = make_context()
        agent = make_decision_agent(ctx, {"recommendation": "Yes", "reasoning": "r", "confidence": 0.6})

        await agent.decide(AsyncMock(), AsyncMock(), user_id="u1", decision="Should I?")

        kwargs = agent.context_builder.build.await_args.kwargs
        assert kwargs["complexity_override"] == ComplexityTier.COMPLEX


# ── ReflectionAgent ───────────────────────────────────────────────────────────


def make_reflection_agent(
    recent: list[Event],
    llm_response: dict | Exception | None = None,
    beliefs: list[UserBelief] | None = None,
    min_events: int = 3,
) -> ReflectionAgent:
    llm = AsyncMock()
    if isinstance(llm_response, Exception):
        llm.extract_structured = AsyncMock(side_effect=llm_response)
    else:
        llm.extract_structured = AsyncMock(return_value=llm_response or {"insights": []})

    semantic = AsyncMock()
    semantic.get_user_profile = AsyncMock(
        return_value=UserProfile(user_id="u1", app_id="default", facts=[make_fact()])
    )
    episodic = AsyncMock()
    episodic.get_recent = AsyncMock(return_value=recent)
    episodic.store = AsyncMock(return_value=make_event())

    return ReflectionAgent(
        llm=llm, semantic=semantic, episodic=episodic, min_events=min_events
    )


def make_reflect_session(beliefs=None, open_insights=None) -> AsyncMock:
    session = AsyncMock()
    beliefs_result = MagicMock()
    beliefs_result.scalars.return_value.all.return_value = beliefs or []
    open_result = MagicMock()
    open_result.scalars.return_value.all.return_value = open_insights or []
    session.execute = AsyncMock(side_effect=[beliefs_result, open_result])
    session.add = MagicMock()
    return session


class TestReflectionAgent:
    @pytest.mark.asyncio
    async def test_skips_below_min_events(self):
        agent = make_reflection_agent(recent=[make_event()], min_events=3)

        result = await agent.reflect(AsyncMock(), AsyncMock(), user_id="u1")

        assert result.skipped is True
        agent.llm.extract_structured.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_own_reflection_events_dont_count_toward_guard(self):
        # 3 events, but 2 are the agent's own summaries → only 1 substantive
        recent = [
            make_event(),
            make_event(source_type=SourceType.AGENT_REFLECTION),
            make_event(source_type=SourceType.AGENT_REFLECTION),
        ]
        agent = make_reflection_agent(recent=recent, min_events=3)

        result = await agent.reflect(AsyncMock(), AsyncMock(), user_id="u1")

        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_valid_insights_stored_and_logged(self):
        recent = [make_event(f"event {i}") for i in range(3)]
        response = {
            "insights": [
                {
                    "kind": "drift",
                    "insight": "You stated a goal but logged nothing recently.",
                    "severity": "notice",
                    "evidence_event_ids": [str(recent[0].id), str(uuid.uuid4())],
                }
            ]
        }
        agent = make_reflection_agent(recent=recent, llm_response=response)
        session = make_reflect_session()

        result = await agent.reflect(session, AsyncMock(), user_id="u1")

        assert result.insights_found == 1
        assert result.insights_stored == 1
        stored = session.add.call_args.args[0]
        assert isinstance(stored, Reflection)
        assert stored.kind == "drift"
        # invented evidence id filtered; only the real event survives
        assert stored.evidence == {"event_ids": [str(recent[0].id)]}
        # the cycle logs its own reasoning as memory
        agent.episodic.store.assert_awaited_once()
        assert (
            agent.episodic.store.await_args.kwargs["source_type"]
            == SourceType.AGENT_REFLECTION
        )

    @pytest.mark.asyncio
    async def test_invalid_kind_and_severity_normalised(self):
        recent = [make_event(f"e{i}") for i in range(3)]
        response = {
            "insights": [
                {"kind": "nonsense", "insight": "Something.", "severity": "critical"}
            ]
        }
        agent = make_reflection_agent(recent=recent, llm_response=response)
        session = make_reflect_session()

        await agent.reflect(session, AsyncMock(), user_id="u1")

        stored = session.add.call_args.args[0]
        assert stored.kind == ReflectionKind.OBSERVATION
        assert stored.severity == "info"

    @pytest.mark.asyncio
    async def test_empty_insights_stores_nothing(self):
        recent = [make_event(f"e{i}") for i in range(3)]
        agent = make_reflection_agent(recent=recent, llm_response={"insights": []})
        session = make_reflect_session()

        result = await agent.reflect(session, AsyncMock(), user_id="u1")

        assert result.insights_stored == 0
        agent.episodic.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_skipped(self):
        recent = [make_event(f"e{i}") for i in range(3)]
        agent = make_reflection_agent(recent=recent, llm_response=RuntimeError("down"))
        session = make_reflect_session()

        result = await agent.reflect(session, AsyncMock(), user_id="u1")

        assert result.skipped is True
        assert "down" in result.skip_reason

    @pytest.mark.asyncio
    async def test_acknowledge_scopes_to_user(self):
        agent = make_reflection_agent(recent=[])
        row = Reflection(
            id=uuid.uuid4(), user_id="u1", app_id="default",
            kind="drift", insight="i", severity="info", evidence={},
        )
        session = AsyncMock()
        session.get = AsyncMock(return_value=row)

        assert await agent.acknowledge(session, "u1", row.id) is True
        assert row.acknowledged is True
        # another user cannot acknowledge someone else's insight
        row.acknowledged = False
        assert await agent.acknowledge(session, "intruder", row.id) is False
        assert row.acknowledged is False


class TestReflectionResult:
    def test_defaults(self):
        r = ReflectionResult(user_id="u1", app_id="default")
        assert r.insights_found == 0
        assert r.insights_stored == 0
        assert r.skipped is False
