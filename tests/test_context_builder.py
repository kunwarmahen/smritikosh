"""
Tests for ContextBuilder and MemoryContext.

How to run:
    # Unit tests (all dependencies mocked):
    pytest tests/test_context_builder.py -v

    # Integration tests (requires Postgres + Neo4j + LLM):
    pytest tests/test_context_builder.py -v -m db

Test strategy:
    - Unit tests mock LLMAdapter, EpisodicMemory, SemanticMemory so we verify
      orchestration logic, deduplication, graceful degradation, and prompt rendering.
    - MemoryContext rendering tests use real objects (no mocks needed).
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from smritikosh.db.models import Event, MemoryLink, RelationType
from smritikosh.memory.episodic import EpisodicMemory, SearchResult
from smritikosh.memory.narrative import NarrativeMemory
from smritikosh.memory.semantic import FactRecord, SemanticMemory, UserProfile
from smritikosh.retrieval.context_builder import (
    ContextBuilder,
    MemoryContext,
    _format_date,
    _truncate,
)
from smritikosh.retrieval.intent_classifier import IntentClassifier, QueryIntent, _INTENT_WEIGHTS


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_event(user_id="u1", raw_text="test event", created_at=None) -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id=user_id,
        app_id="default",
        raw_text=raw_text,
        importance_score=0.8,
        consolidated=False,
        event_metadata={},
        created_at=created_at or datetime.now(timezone.utc),
    )


def make_search_result(raw_text="similar event", score=0.85) -> SearchResult:
    return SearchResult(
        event=make_event(raw_text=raw_text),
        similarity_score=score,
        recency_score=0.7,
        hybrid_score=score,
    )


def make_profile(facts: list[FactRecord] | None = None) -> UserProfile:
    return UserProfile(
        user_id="u1",
        app_id="default",
        facts=facts or [
            FactRecord("preference", "ui_color", "green", 0.9, 3, "2026-03-01", "2026-03-15"),
            FactRecord("interest", "domain", "AI agents", 0.95, 5, "2026-03-01", "2026-03-15"),
            FactRecord("role", "current", "entrepreneur", 1.0, 1, "2026-03-01", "2026-03-15"),
        ],
    )


def make_builder(**kwargs) -> tuple[ContextBuilder, AsyncMock, AsyncMock, AsyncMock]:
    """Returns (builder, llm_mock, episodic_mock, semantic_mock)."""
    llm = AsyncMock()
    episodic = AsyncMock(spec=EpisodicMemory)
    semantic = AsyncMock(spec=SemanticMemory)

    builder = ContextBuilder(
        llm=llm,
        episodic=episodic,
        semantic=semantic,
        top_k_similar=kwargs.get("top_k_similar", 5),
        recent_limit=kwargs.get("recent_limit", 5),
        min_profile_confidence=kwargs.get("min_profile_confidence", 0.5),
        intent_classifier=kwargs.get("intent_classifier", None),
        narrative=kwargs.get("narrative", None),
        include_chains=kwargs.get("include_chains", False),
        chain_top_k=kwargs.get("chain_top_k", 3),
        chain_boost=kwargs.get("chain_boost", 0.05),
    )
    return builder, llm, episodic, semantic


def make_sessions():
    return AsyncMock(), AsyncMock()


# ── Helper functions ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_truncate_short_text(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncate_long_text(self):
        result = _truncate("a" * 200, 50)
        assert len(result) == 50
        assert result.endswith("…")

    def test_truncate_exact_length(self):
        text = "a" * 50
        assert _truncate(text, 50) == text

    def test_format_date_with_tz(self):
        dt = datetime(2026, 3, 15, 10, 30, tzinfo=timezone.utc)
        assert _format_date(dt) == "2026-03-15"

    def test_format_date_naive(self):
        dt = datetime(2026, 3, 15, 10, 30)
        assert _format_date(dt) == "2026-03-15"

    def test_format_date_none(self):
        assert _format_date(None) == "unknown"


# ── MemoryContext ─────────────────────────────────────────────────────────────


class TestMemoryContext:
    def test_is_empty_with_no_data(self):
        ctx = MemoryContext(user_id="u1", query="test")
        assert ctx.is_empty()

    def test_is_empty_false_with_similar(self):
        ctx = MemoryContext(
            user_id="u1", query="test",
            similar_events=[make_search_result()]
        )
        assert not ctx.is_empty()

    def test_is_empty_false_with_profile(self):
        ctx = MemoryContext(
            user_id="u1", query="test",
            user_profile=make_profile()
        )
        assert not ctx.is_empty()

    def test_is_empty_true_with_empty_profile(self):
        ctx = MemoryContext(
            user_id="u1", query="test",
            user_profile=UserProfile(user_id="u1", app_id="default", facts=[])
        )
        assert ctx.is_empty()

    def test_total_memories_counts_all(self):
        ctx = MemoryContext(
            user_id="u1",
            query="test",
            similar_events=[make_search_result(), make_search_result()],
            user_profile=make_profile(),    # 3 facts
            recent_events=[make_event(), make_event()],
        )
        # 2 similar + 3 facts + 2 recent = 7
        assert ctx.total_memories() == 7

    def test_total_memories_no_profile(self):
        ctx = MemoryContext(user_id="u1", query="test", similar_events=[make_search_result()])
        assert ctx.total_memories() == 1


class TestMemoryContextPromptText:
    def _make_full_ctx(self) -> MemoryContext:
        return MemoryContext(
            user_id="u1",
            query="What should I build next?",
            similar_events=[
                make_search_result("User discussed building AI memory infra", score=0.91),
                make_search_result("User mentioned LangGraph experience", score=0.84),
            ],
            user_profile=make_profile(),
            recent_events=[make_event(raw_text="Asked about RAG pipelines")],
        )

    def test_prompt_contains_header(self):
        ctx = self._make_full_ctx()
        text = ctx.as_prompt_text()
        assert "## User Memory Context" in text

    def test_prompt_contains_profile_section(self):
        ctx = self._make_full_ctx()
        text = ctx.as_prompt_text()
        assert "Who this user is" in text
        assert "green" in text
        assert "AI agents" in text
        assert "entrepreneur" in text

    def test_prompt_contains_recent_section(self):
        ctx = self._make_full_ctx()
        text = ctx.as_prompt_text()
        assert "Recent activity" in text
        assert "Asked about RAG pipelines" in text

    def test_prompt_contains_similar_section(self):
        ctx = self._make_full_ctx()
        text = ctx.as_prompt_text()
        assert "Relevant past memories" in text
        assert "AI memory infra" in text
        assert "0.91" in text

    def test_empty_context_shows_placeholder(self):
        ctx = MemoryContext(user_id="u1", query="test")
        text = ctx.as_prompt_text()
        assert "no memory stored" in text

    def test_long_event_text_is_truncated(self):
        long_text = "x" * 200
        ctx = MemoryContext(
            user_id="u1", query="test",
            similar_events=[make_search_result(raw_text=long_text)],
        )
        text = ctx.as_prompt_text()
        # The truncated text should appear, not the full 200 chars
        assert "x" * 200 not in text
        assert "…" in text

    def test_as_messages_returns_system_role(self):
        ctx = self._make_full_ctx()
        messages = ctx.as_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "## User Memory Context" in messages[0]["content"]


# ── ContextBuilder.build() ────────────────────────────────────────────────────


class TestContextBuilderBuild:
    @pytest.mark.asyncio
    async def test_returns_memory_context(self):
        builder, llm, episodic, semantic = make_builder()
        pg, neo = make_sessions()

        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[make_search_result()])
        episodic.get_recent = AsyncMock(return_value=[make_event()])
        semantic.get_user_profile = AsyncMock(return_value=make_profile())

        ctx = await builder.build(pg, neo, user_id="u1", query="What should I build?")

        assert isinstance(ctx, MemoryContext)
        assert ctx.user_id == "u1"
        assert not ctx.is_empty()

    @pytest.mark.asyncio
    async def test_embed_called_with_query(self):
        builder, llm, episodic, semantic = make_builder()
        pg, neo = make_sessions()

        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        await builder.build(pg, neo, user_id="u1", query="my specific query")

        llm.embed.assert_called_once_with("my specific query")

    @pytest.mark.asyncio
    async def test_three_retrieval_calls_made(self):
        builder, llm, episodic, semantic = make_builder()
        pg, neo = make_sessions()

        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        await builder.build(pg, neo, user_id="u1", query="test")

        episodic.hybrid_search.assert_called_once()
        episodic.get_recent.assert_called_once()
        semantic.get_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_top_k_forwarded_to_hybrid_search(self):
        builder, llm, episodic, semantic = make_builder(top_k_similar=3)
        pg, neo = make_sessions()

        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        await builder.build(pg, neo, user_id="u1", query="test")

        call_kwargs = episodic.hybrid_search.call_args.kwargs
        assert call_kwargs["top_k"] == 3

    @pytest.mark.asyncio
    async def test_recent_limit_forwarded(self):
        builder, llm, episodic, semantic = make_builder(recent_limit=3)
        pg, neo = make_sessions()

        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        await builder.build(pg, neo, user_id="u1", query="test")

        call_kwargs = episodic.get_recent.call_args.kwargs
        assert call_kwargs["limit"] == 3

    @pytest.mark.asyncio
    async def test_deduplication_removes_similar_from_recent(self):
        """Events that appear in similar_events should not also appear in recent_events."""
        shared_event = make_event(raw_text="shared event")
        builder, llm, episodic, semantic = make_builder()
        pg, neo = make_sessions()

        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=shared_event, hybrid_score=0.9)
        ])
        # get_recent returns the same event + another one
        episodic.get_recent = AsyncMock(return_value=[
            shared_event,
            make_event(raw_text="unique recent event"),
        ])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        recent_ids = {e.id for e in ctx.recent_events}
        assert shared_event.id not in recent_ids
        assert len(ctx.recent_events) == 1
        assert ctx.recent_events[0].raw_text == "unique recent event"


# ── Graceful degradation ──────────────────────────────────────────────────────


class TestContextBuilderDegradation:
    @pytest.mark.asyncio
    async def test_embedding_failure_context_still_built(self):
        """Embedding fails → similar_events empty, but profile and recent still returned."""
        builder, llm, episodic, semantic = make_builder()
        pg, neo = make_sessions()

        llm.embed = AsyncMock(side_effect=RuntimeError("embed service down"))
        episodic.get_recent = AsyncMock(return_value=[make_event()])
        semantic.get_user_profile = AsyncMock(return_value=make_profile())

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        assert ctx.embedding_failed is True
        assert ctx.similar_events == []
        assert len(ctx.recent_events) == 1
        assert ctx.user_profile is not None

    @pytest.mark.asyncio
    async def test_profile_failure_context_still_built(self):
        """Neo4j failure → profile is None, episodic results still returned."""
        builder, llm, episodic, semantic = make_builder()
        pg, neo = make_sessions()

        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[make_search_result()])
        episodic.get_recent = AsyncMock(return_value=[make_event()])
        semantic.get_user_profile = AsyncMock(side_effect=RuntimeError("neo4j down"))

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        assert ctx.user_profile is None
        assert len(ctx.similar_events) == 1
        assert len(ctx.recent_events) >= 1

    @pytest.mark.asyncio
    async def test_all_retrieval_fail_returns_empty_context(self):
        builder, llm, episodic, semantic = make_builder()
        pg, neo = make_sessions()

        llm.embed = AsyncMock(side_effect=RuntimeError("down"))
        episodic.get_recent = AsyncMock(side_effect=RuntimeError("down"))
        semantic.get_user_profile = AsyncMock(side_effect=RuntimeError("down"))

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        assert ctx.is_empty()
        assert ctx.embedding_failed is True


# ── Intent-aware retrieval ────────────────────────────────────────────────────


class TestIntentAwareRetrieval:
    def _setup(self, query: str, classifier=None):
        builder, llm, episodic, semantic = make_builder(
            intent_classifier=classifier or IntentClassifier()
        )
        pg, neo = make_sessions()
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))
        return builder, llm, episodic, semantic, pg, neo

    @pytest.mark.asyncio
    async def test_intent_stored_on_memory_context(self):
        builder, llm, episodic, semantic, pg, neo = self._setup("career job role")
        ctx = await builder.build(pg, neo, user_id="u1", query="career job role")
        assert ctx.intent == QueryIntent.CAREER

    @pytest.mark.asyncio
    async def test_general_intent_when_no_classifier(self):
        builder, llm, episodic, semantic = make_builder(intent_classifier=None)
        pg, neo = make_sessions()
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        ctx = await builder.build(pg, neo, user_id="u1", query="career job role")

        assert ctx.intent == QueryIntent.GENERAL

    @pytest.mark.asyncio
    async def test_weights_override_passed_to_hybrid_search(self):
        builder, llm, episodic, semantic, pg, neo = self._setup("career job role")
        await builder.build(pg, neo, user_id="u1", query="career job role")

        call_kwargs = episodic.hybrid_search.call_args.kwargs
        assert call_kwargs["weights_override"] == _INTENT_WEIGHTS[QueryIntent.CAREER]

    @pytest.mark.asyncio
    async def test_no_weights_override_without_classifier(self):
        builder, llm, episodic, semantic = make_builder(intent_classifier=None)
        pg, neo = make_sessions()
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        await builder.build(pg, neo, user_id="u1", query="career job role")

        call_kwargs = episodic.hybrid_search.call_args.kwargs
        assert call_kwargs["weights_override"] is None

    @pytest.mark.asyncio
    async def test_historical_recall_intent_detected(self):
        builder, llm, episodic, semantic, pg, neo = self._setup(
            "do you remember what I said last time"
        )
        ctx = await builder.build(
            pg, neo, user_id="u1", query="do you remember what I said last time"
        )
        assert ctx.intent == QueryIntent.HISTORICAL_RECALL

    @pytest.mark.asyncio
    async def test_intent_default_is_general_on_empty_query(self):
        builder, llm, episodic, semantic, pg, neo = self._setup("hello")
        ctx = await builder.build(pg, neo, user_id="u1", query="hello")
        assert ctx.intent == QueryIntent.GENERAL


# ── Narrative chain traversal ─────────────────────────────────────────────────


def make_memory_link(from_id: uuid.UUID, to_id: uuid.UUID) -> MemoryLink:
    return MemoryLink(
        id=uuid.uuid4(),
        from_event_id=from_id,
        to_event_id=to_id,
        relation_type="preceded",
    )


class TestNarrativeChains:
    def _setup_with_narrative(self, narrative_mock):
        builder, llm, episodic, semantic = make_builder(
            narrative=narrative_mock,
            include_chains=True,
        )
        pg, neo = make_sessions()
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))
        return builder, llm, episodic, semantic, pg, neo

    def _make_pg_with_event(self, event: Event):
        """Return a pg mock whose execute() always returns the given single event."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [event]
        pg = AsyncMock()
        pg.execute = AsyncMock(return_value=mock_result)
        return pg

    @pytest.mark.asyncio
    async def test_narrative_chains_populated_when_links_exist(self):
        anchor_event = make_event(raw_text="anchor")
        linked_event = make_event(raw_text="linked")
        link = make_memory_link(anchor_event.id, linked_event.id)

        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(return_value=[link])
        narrative.get_chain_backward = AsyncMock(return_value=[])

        builder, llm, episodic, semantic, pg, neo = self._setup_with_narrative(narrative)
        pg = self._make_pg_with_event(linked_event)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=anchor_event, hybrid_score=0.9)
        ])
        episodic.increment_recall = AsyncMock()

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        assert len(ctx.narrative_chains) == 1
        assert ctx.narrative_chains[0][0].id == anchor_event.id
        assert ctx.narrative_chains[0][1].id == linked_event.id

    @pytest.mark.asyncio
    async def test_no_chains_when_no_links(self):
        anchor_event = make_event(raw_text="anchor")
        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(return_value=[])
        narrative.get_chain_backward = AsyncMock(return_value=[])

        builder, llm, episodic, semantic, pg, neo = self._setup_with_narrative(narrative)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=anchor_event, hybrid_score=0.9)
        ])

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        assert ctx.narrative_chains == []

    @pytest.mark.asyncio
    async def test_no_chains_when_include_chains_false(self):
        narrative = AsyncMock(spec=NarrativeMemory)
        builder, llm, episodic, semantic = make_builder(
            narrative=narrative,
            include_chains=False,
        )
        pg, neo = make_sessions()
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[make_search_result()])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        narrative.get_chain_forward.assert_not_called()
        assert ctx.narrative_chains == []

    @pytest.mark.asyncio
    async def test_no_chains_when_no_similar_events(self):
        narrative = AsyncMock(spec=NarrativeMemory)
        builder, llm, episodic, semantic, pg, neo = self._setup_with_narrative(narrative)
        episodic.hybrid_search = AsyncMock(return_value=[])

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        narrative.get_chain_forward.assert_not_called()
        assert ctx.narrative_chains == []

    @pytest.mark.asyncio
    async def test_chain_traversal_failure_graceful(self):
        anchor_event = make_event(raw_text="anchor")
        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(side_effect=RuntimeError("db error"))
        narrative.get_chain_backward = AsyncMock(return_value=[])

        builder, llm, episodic, semantic, pg, neo = self._setup_with_narrative(narrative)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=anchor_event, hybrid_score=0.9)
        ])

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        assert ctx.narrative_chains == []

    @pytest.mark.asyncio
    async def test_chain_events_added_to_similar_with_boost(self):
        """Chain-adjacent events should appear in similar_events with boosted score."""
        anchor_event = make_event(raw_text="anchor")
        chain_event = make_event(raw_text="chain event")
        link = make_memory_link(anchor_event.id, chain_event.id)

        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(return_value=[link])
        narrative.get_chain_backward = AsyncMock(return_value=[])

        builder, llm, episodic, semantic, pg, neo = self._setup_with_narrative(narrative)
        pg = self._make_pg_with_event(chain_event)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=anchor_event, hybrid_score=0.80, similarity_score=0.8, recency_score=0.7)
        ])
        episodic.increment_recall = AsyncMock()

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        similar_ids = {sr.event.id for sr in ctx.similar_events}
        assert chain_event.id in similar_ids

    @pytest.mark.asyncio
    async def test_chain_boost_applied_to_score(self):
        """Chain event score = anchor score + chain_boost (capped at 1.0)."""
        anchor_event = make_event(raw_text="anchor")
        chain_event = make_event(raw_text="chain event")
        link = make_memory_link(anchor_event.id, chain_event.id)

        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(return_value=[link])
        narrative.get_chain_backward = AsyncMock(return_value=[])

        builder, llm, episodic, semantic, pg, neo = self._setup_with_narrative(narrative)
        pg = self._make_pg_with_event(chain_event)
        anchor_score = 0.75
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=anchor_event, hybrid_score=anchor_score, similarity_score=0.7, recency_score=0.6)
        ])
        episodic.increment_recall = AsyncMock()

        # Default chain_boost is 0.05
        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        chain_sr = next(sr for sr in ctx.similar_events if sr.event.id == chain_event.id)
        assert abs(chain_sr.hybrid_score - (anchor_score + 0.05)) < 1e-9

    @pytest.mark.asyncio
    async def test_chain_boost_capped_at_one(self):
        """Even if anchor score is very high, boosted score never exceeds 1.0."""
        anchor_event = make_event(raw_text="anchor")
        chain_event = make_event(raw_text="chain event")
        link = make_memory_link(anchor_event.id, chain_event.id)

        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(return_value=[link])
        narrative.get_chain_backward = AsyncMock(return_value=[])

        # custom boost of 0.10, anchor score of 0.95 → would be 1.05 without cap
        builder, llm, episodic, semantic = make_builder(
            narrative=narrative, include_chains=True, chain_boost=0.10
        )
        pg, neo = make_sessions()
        pg = self._make_pg_with_event(chain_event)
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=anchor_event, hybrid_score=0.95, similarity_score=0.9, recency_score=0.8)
        ])
        episodic.get_recent = AsyncMock(return_value=[])
        episodic.increment_recall = AsyncMock()
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        chain_sr = next(sr for sr in ctx.similar_events if sr.event.id == chain_event.id)
        assert chain_sr.hybrid_score <= 1.0

    @pytest.mark.asyncio
    async def test_backward_chain_events_added_to_similar(self):
        """Backward chain (predecessor) events also get added to similar_events."""
        anchor_event = make_event(raw_text="anchor")
        predecessor = make_event(raw_text="predecessor")
        bwd_link = make_memory_link(predecessor.id, anchor_event.id)

        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(return_value=[])
        narrative.get_chain_backward = AsyncMock(return_value=[bwd_link])

        builder, llm, episodic, semantic, pg, neo = self._setup_with_narrative(narrative)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [predecessor]
        pg.execute = AsyncMock(return_value=mock_result)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=anchor_event, hybrid_score=0.80, similarity_score=0.8, recency_score=0.7)
        ])
        episodic.increment_recall = AsyncMock()

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        similar_ids = {sr.event.id for sr in ctx.similar_events}
        assert predecessor.id in similar_ids

    @pytest.mark.asyncio
    async def test_existing_events_not_duplicated_by_chain(self):
        """An event already in similar_events should not be added again via a chain."""
        event_a = make_event(raw_text="event A")
        event_b = make_event(raw_text="event B — already in results")
        link = make_memory_link(event_a.id, event_b.id)

        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(return_value=[link])
        narrative.get_chain_backward = AsyncMock(return_value=[])

        builder, llm, episodic, semantic, pg, neo = self._setup_with_narrative(narrative)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=event_a, hybrid_score=0.9),
            SearchResult(event=event_b, hybrid_score=0.85),
        ])
        episodic.increment_recall = AsyncMock()

        ctx = await builder.build(pg, neo, user_id="u1", query="test")

        b_count = sum(1 for sr in ctx.similar_events if sr.event.id == event_b.id)
        assert b_count == 1

    @pytest.mark.asyncio
    async def test_chain_top_k_limits_anchors_traversed(self):
        """chain_top_k=1 means only the top similar event's chain is traversed."""
        events = [make_event(raw_text=f"event {i}") for i in range(3)]
        narrative = AsyncMock(spec=NarrativeMemory)
        narrative.get_chain_forward = AsyncMock(return_value=[])
        narrative.get_chain_backward = AsyncMock(return_value=[])

        builder, llm, episodic, semantic = make_builder(
            narrative=narrative, include_chains=True, chain_top_k=1
        )
        pg, neo = make_sessions()
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        episodic.hybrid_search = AsyncMock(return_value=[
            SearchResult(event=e, hybrid_score=0.9 - i * 0.1) for i, e in enumerate(events)
        ])
        episodic.get_recent = AsyncMock(return_value=[])
        episodic.increment_recall = AsyncMock()
        semantic.get_user_profile = AsyncMock(return_value=make_profile([]))

        await builder.build(pg, neo, user_id="u1", query="test")

        # With chain_top_k=1, only 1 anchor was traversed → 1 fwd + 1 bwd call
        assert narrative.get_chain_forward.call_count == 1
        assert narrative.get_chain_backward.call_count == 1

    @pytest.mark.asyncio
    async def test_chain_renders_in_prompt_text(self):
        e1 = make_event(raw_text="startup founded")
        e2 = make_event(raw_text="engineers hired")
        ctx = MemoryContext(
            user_id="u1",
            query="test",
            narrative_chains=[[e1, e2]],
        )
        text = ctx.as_prompt_text()
        assert "Memory chains" in text
        assert "startup founded" in text
        assert "engineers hired" in text
        assert "→" in text
