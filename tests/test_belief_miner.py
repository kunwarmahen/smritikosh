"""
Tests for BeliefMiner, _build_belief_prompt, and UserIdentity belief integration.

Unit tests mock AsyncSession, NeoSession, LLMAdapter, and SemanticMemory.
DB integration tests are gated behind @pytest.mark.db.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.db.models import Event, UserBelief
from smritikosh.memory.identity import IdentityBuilder, UserIdentity
from smritikosh.memory.semantic import FactRecord, UserProfile
from smritikosh.processing.belief_miner import (
    MIN_CONSOLIDATED_EVENTS,
    BeliefMiner,
    MiningResult,
    _build_belief_prompt,
    _upsert_belief,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(summary: str | None = "User built an AI product") -> Event:
    e = Event(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        raw_text="raw text",
        importance_score=0.8,
        consolidated=True,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )
    e.summary = summary
    return e


def make_fact(category="role", key="current", value="entrepreneur", confidence=0.9):
    return FactRecord(
        category=category, key=key, value=value, confidence=confidence,
        frequency_count=1, first_seen_at="2026-01-01", last_seen_at="2026-03-15",
    )


def make_mock_session(events: list[Event] | None = None) -> AsyncMock:
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = events or []
    session.execute = AsyncMock(return_value=mock_result)
    return session


def make_mock_llm(beliefs: list[dict] | None = None) -> AsyncMock:
    llm = AsyncMock()
    llm.extract_structured = AsyncMock(return_value={
        "beliefs": beliefs if beliefs is not None else [
            {"statement": "believes in iterative development", "category": "value", "confidence": 0.85},
            {"statement": "assumes AI will transform work", "category": "worldview", "confidence": 0.9},
        ]
    })
    return llm


def make_mock_semantic(facts: list[FactRecord] | None = None) -> AsyncMock:
    from smritikosh.memory.semantic import SemanticMemory
    semantic = AsyncMock(spec=SemanticMemory)
    profile = UserProfile(user_id="u1", app_id="default", facts=facts or [])
    semantic.get_user_profile = AsyncMock(return_value=profile)
    return semantic


# ── MiningResult ──────────────────────────────────────────────────────────────


class TestMiningResult:
    def test_defaults(self):
        r = MiningResult(user_id="u1", app_id="default")
        assert r.beliefs_found == 0
        assert r.beliefs_upserted == 0
        assert r.skipped is False
        assert r.skip_reason == ""


# ── _build_belief_prompt ──────────────────────────────────────────────────────


class TestBuildBeliefPrompt:
    def test_contains_instruction(self):
        prompt = _build_belief_prompt([], [make_event()])
        assert "beliefs" in prompt.lower() or "worldview" in prompt.lower()

    def test_includes_facts(self):
        facts = [make_fact("role", "current", "entrepreneur")]
        prompt = _build_belief_prompt(facts, [])
        assert "entrepreneur" in prompt

    def test_includes_event_summaries(self):
        events = [make_event("Started AI startup"), make_event("Shipped first product")]
        prompt = _build_belief_prompt([], events)
        assert "AI startup" in prompt
        assert "first product" in prompt

    def test_truncates_event_text(self):
        long_summary = "x" * 200
        events = [make_event(long_summary)]
        prompt = _build_belief_prompt([], events)
        assert "x" * 200 not in prompt     # truncated to 150
        assert "x" * 150 in prompt

    def test_empty_facts_section_skipped(self):
        prompt = _build_belief_prompt([], [make_event()])
        assert "Known facts" not in prompt

    def test_empty_events_section_skipped(self):
        prompt = _build_belief_prompt([make_fact()], [])
        assert "Recent memory" not in prompt

    def test_uses_summary_over_raw(self):
        event = make_event(summary="Short summary")
        event.raw_text = "Do not use this"
        prompt = _build_belief_prompt([], [event])
        assert "Short summary" in prompt
        assert "Do not use this" not in prompt


# ── BeliefMiner.mine — skip guard ─────────────────────────────────────────────


class TestBeliefMinerSkipGuard:
    @pytest.mark.asyncio
    async def test_skips_when_too_few_events(self):
        events = [make_event() for _ in range(MIN_CONSOLIDATED_EVENTS - 1)]
        session = make_mock_session(events)
        miner = BeliefMiner(llm=make_mock_llm(), semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.skipped is True
        assert str(MIN_CONSOLIDATED_EVENTS - 1) in result.skip_reason

    @pytest.mark.asyncio
    async def test_skips_when_zero_events(self):
        session = make_mock_session([])
        miner = BeliefMiner(llm=make_mock_llm(), semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_runs_when_enough_events(self):
        events = [make_event() for _ in range(MIN_CONSOLIDATED_EVENTS)]
        session = make_mock_session(events)
        miner = BeliefMiner(llm=make_mock_llm(), semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.skipped is False


# ── BeliefMiner.mine — success path ───────────────────────────────────────────


class TestBeliefMinerSuccess:
    @pytest.mark.asyncio
    async def test_beliefs_found_matches_llm_output(self):
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        miner = BeliefMiner(llm=make_mock_llm(), semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.beliefs_found == 2   # LLM returns 2 beliefs

    @pytest.mark.asyncio
    async def test_beliefs_upserted_count(self):
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        miner = BeliefMiner(llm=make_mock_llm(), semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.beliefs_upserted == 2

    @pytest.mark.asyncio
    async def test_llm_called_once(self):
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        llm = make_mock_llm()
        miner = BeliefMiner(llm=llm, semantic=make_mock_semantic())

        await miner.mine(session, AsyncMock(), user_id="u1")

        llm.extract_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_executed_per_belief(self):
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        miner = BeliefMiner(llm=make_mock_llm(), semantic=make_mock_semantic())

        await miner.mine(session, AsyncMock(), user_id="u1")

        # 1 SELECT + 2 upserts (one per belief)
        assert session.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_invalid_category_uses_fallback(self):
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        llm = make_mock_llm(beliefs=[
            {"statement": "values speed", "category": "TOTALLY_INVALID", "confidence": 0.8},
        ])
        miner = BeliefMiner(llm=llm, semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        # Invalid category falls back to "assumption", not skipped
        assert result.beliefs_upserted == 1

    @pytest.mark.asyncio
    async def test_empty_statement_skipped(self):
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        llm = make_mock_llm(beliefs=[
            {"statement": "", "category": "value", "confidence": 0.8},
            {"statement": "values speed", "category": "value", "confidence": 0.8},
        ])
        miner = BeliefMiner(llm=llm, semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.beliefs_upserted == 1   # empty statement skipped

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_range(self):
        """Confidence values outside [0,1] from LLM should be clamped."""
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        llm = make_mock_llm(beliefs=[
            {"statement": "values boldness", "category": "value", "confidence": 1.5},
        ])
        miner = BeliefMiner(llm=llm, semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.beliefs_upserted == 1   # clamped, not rejected


# ── BeliefMiner.mine — LLM failure ────────────────────────────────────────────


class TestBeliefMinerLLMFailure:
    @pytest.mark.asyncio
    async def test_graceful_on_llm_error(self):
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        llm = AsyncMock()
        llm.extract_structured = AsyncMock(side_effect=RuntimeError("LLM down"))
        miner = BeliefMiner(llm=llm, semantic=make_mock_semantic())

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.skipped is True
        assert "LLM" in result.skip_reason

    @pytest.mark.asyncio
    async def test_empty_beliefs_list_no_upserts(self):
        events = [make_event() for _ in range(3)]
        session = make_mock_session(events)
        miner = BeliefMiner(
            llm=make_mock_llm(beliefs=[]),
            semantic=make_mock_semantic(),
        )

        result = await miner.mine(session, AsyncMock(), user_id="u1")

        assert result.beliefs_found == 0
        assert result.beliefs_upserted == 0


# ── BeliefMiner.get_beliefs ───────────────────────────────────────────────────


class TestGetBeliefs:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        belief = UserBelief(
            id=uuid.uuid4(), user_id="u1", app_id="default",
            statement="values speed", category="value", confidence=0.8,
            evidence_count=1,
            first_inferred_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [belief]
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        miner = BeliefMiner(llm=AsyncMock(), semantic=make_mock_semantic())

        beliefs = await miner.get_beliefs(session, "u1")

        assert beliefs == [belief]

    @pytest.mark.asyncio
    async def test_returns_empty_when_none(self):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        miner = BeliefMiner(llm=AsyncMock(), semantic=make_mock_semantic())

        beliefs = await miner.get_beliefs(session, "u1")

        assert beliefs == []


# ── UserIdentity belief integration ───────────────────────────────────────────


class TestUserIdentityBeliefs:
    def _make_belief(self, statement: str, confidence: float = 0.85) -> UserBelief:
        return UserBelief(
            id=uuid.uuid4(), user_id="u1", app_id="default",
            statement=statement, category="value", confidence=confidence,
            evidence_count=2,
            first_inferred_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )

    def test_is_empty_false_with_only_beliefs(self):
        identity = UserIdentity(
            user_id="u1", app_id="default",
            beliefs=[self._make_belief("values speed")],
        )
        assert not identity.is_empty()

    def test_is_empty_true_with_no_data(self):
        identity = UserIdentity(user_id="u1", app_id="default")
        assert identity.is_empty()

    def test_as_prompt_text_includes_beliefs_section(self):
        identity = UserIdentity(
            user_id="u1", app_id="default",
            beliefs=[
                self._make_belief("believes iterative development beats big launches"),
                self._make_belief("values deep user research", confidence=0.7),
            ],
        )
        text = identity.as_prompt_text()
        assert "Core beliefs" in text
        assert "iterative development" in text
        assert "deep user research" in text

    def test_beliefs_sorted_by_confidence_descending(self):
        identity = UserIdentity(
            user_id="u1", app_id="default",
            beliefs=[
                self._make_belief("low confidence belief", confidence=0.5),
                self._make_belief("high confidence belief", confidence=0.95),
            ],
        )
        text = identity.as_prompt_text()
        high_pos = text.index("high confidence")
        low_pos = text.index("low confidence")
        assert high_pos < low_pos   # high confidence rendered first

    def test_beliefs_confidence_shown_in_prompt(self):
        identity = UserIdentity(
            user_id="u1", app_id="default",
            beliefs=[self._make_belief("values speed", confidence=0.85)],
        )
        text = identity.as_prompt_text()
        assert "0.85" in text

    @pytest.mark.asyncio
    async def test_identity_builder_includes_beliefs_when_pg_provided(self):
        """IdentityBuilder.build() fetches beliefs when pg_session is given."""
        from smritikosh.memory.semantic import SemanticMemory
        belief = self._make_belief("values speed")

        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(
            return_value=UserProfile(user_id="u1", app_id="default", facts=[])
        )
        llm = AsyncMock()
        llm.extract_structured = AsyncMock(return_value={"summary": ""})

        # Mock BeliefMiner.get_beliefs to return our belief
        pg_session = AsyncMock()
        with patch(
            "smritikosh.processing.belief_miner.BeliefMiner"
        ) as MockMiner:
            mock_miner_instance = AsyncMock()
            mock_miner_instance.get_beliefs = AsyncMock(return_value=[belief])
            MockMiner.return_value = mock_miner_instance

            builder = IdentityBuilder(llm=llm, semantic=semantic)
            identity = await builder.build(
                AsyncMock(), user_id="u1", pg_session=pg_session
            )

        assert len(identity.beliefs) == 1
        assert identity.beliefs[0].statement == "values speed"

    @pytest.mark.asyncio
    async def test_identity_builder_no_beliefs_without_pg(self):
        """IdentityBuilder.build() returns no beliefs when pg_session is None."""
        from smritikosh.memory.semantic import SemanticMemory
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(
            return_value=UserProfile(user_id="u1", app_id="default", facts=[])
        )
        llm = AsyncMock()
        llm.extract_structured = AsyncMock(return_value={"summary": ""})

        builder = IdentityBuilder(llm=llm, semantic=semantic)
        identity = await builder.build(AsyncMock(), user_id="u1")   # no pg_session

        assert identity.beliefs == []


# ── DB integration tests ──────────────────────────────────────────────────────


@pytest.mark.db
class TestBeliefMinerDB:
    """
    How to run:
        docker compose up -d postgres neo4j
        pytest tests/test_belief_miner.py -v -m db
    """

    @pytest.fixture(autouse=True)
    async def setup_db(self):
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from smritikosh.config import settings
        from smritikosh.db.models import Base
        from smritikosh.llm.adapter import LLMAdapter
        from smritikosh.memory.episodic import EpisodicMemory
        from smritikosh.memory.semantic import SemanticMemory

        engine = create_async_engine(settings.postgres_url)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

        self.SessionFactory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        self.engine = engine
        self.episodic = EpisodicMemory()
        self.miner = BeliefMiner(
            llm=LLMAdapter(),
            semantic=SemanticMemory(),
            min_events=2,
        )

        yield

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    async def _store_consolidated(self, session, summary: str) -> Event:
        event = await self.episodic.store(
            session, user_id="u1", raw_text=summary
        )
        event.consolidated = True
        event.summary = summary
        return event

    async def test_beliefs_written_to_db(self):
        async with self.SessionFactory() as session:
            for s in ["Built AI startup", "Focused on developer market"]:
                await self._store_consolidated(session, s)
            await session.commit()

        async with self.SessionFactory() as session:
            result = await self.miner.mine(
                session, MagicMock(), user_id="u1"
            )
            await session.commit()

        assert not result.skipped
        assert result.beliefs_upserted > 0

    async def test_second_mine_increments_evidence_count(self):
        async with self.SessionFactory() as session:
            for s in ["Built AI startup", "Hired first engineer"]:
                await self._store_consolidated(session, s)
            await session.commit()

        neo_mock = MagicMock()
        neo_mock.get_user_profile = AsyncMock(
            return_value=UserProfile(user_id="u1", app_id="default", facts=[])
        )

        async with self.SessionFactory() as session:
            await self.miner.mine(session, MagicMock(), user_id="u1")
            await session.commit()

        async with self.SessionFactory() as session:
            await self.miner.mine(session, MagicMock(), user_id="u1")
            await session.commit()

        async with self.SessionFactory() as session:
            beliefs = await self.miner.get_beliefs(session, "u1")

        # At least one belief should have evidence_count >= 2
        assert any(b.evidence_count >= 2 for b in beliefs)
