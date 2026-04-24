"""
Tests for CrossSystemSynthesizer.

Unit tests mock Postgres AsyncSession, Neo4j AsyncSession, and LLMAdapter,
so no live databases are required.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.memory.semantic import SemanticMemory, UserProfile
from smritikosh.processing.cross_system_synthesizer import (
    CrossSystemSynthesizer,
    SynthesisResult,
    _build_connector_summaries,
    _build_synthesis_prompt,
)


# ── Fixtures / helpers ─────────────────────────────────────────────────────────


def _make_event(
    raw_text: str = "test content",
    source: str = "calendar",
    created_at: datetime | None = None,
) -> MagicMock:
    ev = MagicMock()
    ev.raw_text = raw_text
    ev.created_at = created_at or datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)
    ev.event_metadata = {"source": source}
    return ev


def _make_pg_session(
    connector_events: list | None = None,
    episodic_texts: list[str] | None = None,
) -> AsyncMock:
    """Return a mock Postgres session that returns connector + episodic rows."""
    connector_events = connector_events or []
    episodic_texts = episodic_texts or []

    # First execute call: _fetch_connector_events → result.scalars().all()
    connector_result = MagicMock()
    connector_result.scalars.return_value.all.return_value = connector_events

    # Second execute call: _fetch_episodic_summaries → result.all()
    episodic_result = MagicMock()
    episodic_result.all.return_value = [(t,) for t in episodic_texts]

    pg = AsyncMock()
    pg.execute = AsyncMock(side_effect=[connector_result, episodic_result])
    return pg


def _make_neo_session() -> AsyncMock:
    return AsyncMock()


def _make_profile(facts: list | None = None) -> UserProfile:
    return UserProfile(user_id="u1", app_id="test", facts=facts or [])


def _make_llm(raw_facts: list | None = None) -> AsyncMock:
    llm = AsyncMock()
    llm.extract_structured = AsyncMock(
        return_value={"facts": raw_facts or []}
    )
    return llm


def _make_synthesizer(
    llm=None,
    semantic=None,
    episodic=None,
) -> CrossSystemSynthesizer:
    llm = llm or _make_llm()
    semantic = semantic or AsyncMock(spec=SemanticMemory)
    episodic = episodic or AsyncMock()
    return CrossSystemSynthesizer(llm=llm, episodic=episodic, semantic=semantic)


# ── _build_connector_summaries ─────────────────────────────────────────────────


class TestBuildConnectorSummaries:
    def test_groups_by_source(self):
        events = [
            _make_event(source="calendar"),
            _make_event(source="calendar"),
            _make_event(source="email"),
        ]
        summaries = _build_connector_summaries(events)
        assert set(summaries) == {"calendar", "email"}
        assert summaries["calendar"]["event_count"] == 2
        assert summaries["email"]["event_count"] == 1

    def test_computes_top_hours(self):
        events = [
            _make_event(source="slack", created_at=datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)),
            _make_event(source="slack", created_at=datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)),
            _make_event(source="slack", created_at=datetime(2026, 4, 10, 14, 0, tzinfo=timezone.utc)),
        ]
        summaries = _build_connector_summaries(events)
        assert 9 in summaries["slack"]["top_active_hours"]

    def test_unknown_sources_excluded(self):
        events = [_make_event(source="unknown_connector")]
        summaries = _build_connector_summaries(events)
        assert summaries == {}

    def test_empty_events_returns_empty(self):
        assert _build_connector_summaries([]) == {}

    def test_sample_topics_capped_at_5(self):
        events = [_make_event(source="email", raw_text=f"email {i}") for i in range(10)]
        summaries = _build_connector_summaries(events)
        assert len(summaries["email"]["sample_topics"]) == 5


# ── _build_synthesis_prompt ────────────────────────────────────────────────────


class TestBuildSynthesisPrompt:
    def test_includes_connector_sections(self):
        summaries = {
            "calendar": {
                "event_count": 5,
                "top_active_hours": [9, 10],
                "active_weekdays": ["Monday", "Tuesday"],
                "sample_topics": ["Calendar event: team standup"],
            }
        }
        prompt = _build_synthesis_prompt(
            connector_summaries=summaries,
            episodic_texts=["I mentioned being overwhelmed"],
            existing_facts_summary="Role: current=engineer",
            lookback_days=30,
        )
        assert "Calendar" in prompt
        assert "09:00" in prompt
        assert "Monday" in prompt
        assert "team standup" in prompt

    def test_includes_episodic_context(self):
        prompt = _build_synthesis_prompt(
            connector_summaries={"email": {"event_count": 1, "top_active_hours": [], "active_weekdays": [], "sample_topics": []}},
            episodic_texts=["I want to set better boundaries"],
            existing_facts_summary="",
            lookback_days=30,
        )
        assert "better boundaries" in prompt

    def test_includes_existing_facts(self):
        prompt = _build_synthesis_prompt(
            connector_summaries={"slack": {"event_count": 2, "top_active_hours": [], "active_weekdays": [], "sample_topics": []}},
            episodic_texts=[],
            existing_facts_summary="Preference: dark_mode=true",
            lookback_days=30,
        )
        assert "dark_mode" in prompt

    def test_instructs_cross_source_requirement(self):
        prompt = _build_synthesis_prompt(
            connector_summaries={},
            episodic_texts=[],
            existing_facts_summary="",
            lookback_days=30,
        )
        assert "TWO OR MORE" in prompt


# ── CrossSystemSynthesizer.run ─────────────────────────────────────────────────


class TestCrossSystemSynthesizerRun:
    @pytest.mark.asyncio
    async def test_skips_when_no_connector_events(self):
        semantic = AsyncMock(spec=SemanticMemory)
        synth = _make_synthesizer(semantic=semantic)

        pg = _make_pg_session(connector_events=[])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert result.skipped is True
        assert "no connector events" in result.skip_reason
        semantic.upsert_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_active_fact_when_confidence_above_threshold(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=_make_profile())
        semantic.upsert_fact = AsyncMock()

        llm = _make_llm(raw_facts=[{
            "category": "habit",
            "key": "morning_meetings",
            "value": "schedules all meetings before noon",
            "confidence": 0.75,
            "rationale": "87% of calendar events before 13:00",
        }])

        synth = _make_synthesizer(llm=llm, semantic=semantic)
        pg = _make_pg_session(connector_events=[_make_event(source="calendar")])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert result.facts_synthesized == 1
        assert result.facts_pending == 0
        assert result.skipped is False
        semantic.upsert_fact.assert_called_once()
        call_kwargs = semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["status"] == "active"
        assert call_kwargs["source_type"] == "cross_system"

    @pytest.mark.asyncio
    async def test_writes_pending_fact_when_confidence_below_threshold(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=_make_profile())
        semantic.upsert_fact = AsyncMock()

        llm = _make_llm(raw_facts=[{
            "category": "lifestyle",
            "key": "evening_disconnect",
            "value": "avoids digital devices after 9pm",
            "confidence": 0.45,
            "rationale": "sparse signals, possible pattern",
        }])

        synth = _make_synthesizer(llm=llm, semantic=semantic)
        pg = _make_pg_session(connector_events=[_make_event(source="email")])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert result.facts_pending == 1
        assert result.facts_synthesized == 0
        call_kwargs = semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["status"] == "pending"

    @pytest.mark.asyncio
    async def test_skips_facts_below_minimum_confidence(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=_make_profile())
        semantic.upsert_fact = AsyncMock()

        llm = _make_llm(raw_facts=[{
            "category": "habit",
            "key": "some_habit",
            "value": "something",
            "confidence": 0.30,
            "rationale": "very weak signal",
        }])

        synth = _make_synthesizer(llm=llm, semantic=semantic)
        pg = _make_pg_session(connector_events=[_make_event(source="calendar")])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert result.facts_skipped == 1
        assert result.facts_synthesized == 0
        semantic.upsert_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_facts_with_invalid_category(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=_make_profile())
        semantic.upsert_fact = AsyncMock()

        llm = _make_llm(raw_facts=[{
            "category": "skill",  # not in _ALLOWED_CATEGORIES for cross_system
            "key": "python",
            "value": "writes Python",
            "confidence": 0.80,
            "rationale": "clear signal",
        }])

        synth = _make_synthesizer(llm=llm, semantic=semantic)
        pg = _make_pg_session(connector_events=[_make_event(source="slack")])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert result.facts_skipped == 1
        semantic.upsert_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_connector_sources_found(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=_make_profile())
        semantic.upsert_fact = AsyncMock()

        llm = _make_llm(raw_facts=[])
        synth = _make_synthesizer(llm=llm, semantic=semantic)
        pg = _make_pg_session(connector_events=[
            _make_event(source="calendar"),
            _make_event(source="email"),
        ])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert "calendar" in result.connector_sources_found
        assert "email" in result.connector_sources_found

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=_make_profile())

        llm = AsyncMock()
        llm.extract_structured = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        synth = _make_synthesizer(llm=llm, semantic=semantic)
        pg = _make_pg_session(connector_events=[_make_event(source="calendar")])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert result.skipped is True
        assert "LLM timeout" in result.skip_reason

    @pytest.mark.asyncio
    async def test_handles_empty_llm_response_gracefully(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=_make_profile())
        semantic.upsert_fact = AsyncMock()

        llm = _make_llm(raw_facts=[])
        synth = _make_synthesizer(llm=llm, semantic=semantic)
        pg = _make_pg_session(connector_events=[_make_event(source="slack")])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert result.skipped is False
        assert result.facts_synthesized == 0
        assert result.facts_pending == 0
        semantic.upsert_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_facts_mixed_confidence(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=_make_profile())
        semantic.upsert_fact = AsyncMock()

        llm = _make_llm(raw_facts=[
            {"category": "habit", "key": "early_riser", "value": "wakes before 7am", "confidence": 0.80, "rationale": "calendar data"},
            {"category": "preference", "key": "async_comms", "value": "prefers async over sync meetings", "confidence": 0.48, "rationale": "email + calendar"},
            {"category": "belief", "key": "some_belief", "value": "something", "confidence": 0.25, "rationale": "weak"},
        ])

        synth = _make_synthesizer(llm=llm, semantic=semantic)
        pg = _make_pg_session(connector_events=[
            _make_event(source="calendar"),
            _make_event(source="email"),
        ])
        neo = _make_neo_session()

        result = await synth.run(pg, neo, user_id="u1", app_id="test")

        assert result.facts_synthesized == 1   # 0.80 → active
        assert result.facts_pending == 1        # 0.48 → pending
        assert result.facts_skipped == 1        # 0.25 → below 0.40


# ── SynthesisResult defaults ───────────────────────────────────────────────────


class TestSynthesisResult:
    def test_defaults(self):
        r = SynthesisResult(user_id="u1", app_id="app")
        assert r.facts_synthesized == 0
        assert r.facts_pending == 0
        assert r.facts_skipped == 0
        assert r.skipped is False
        assert r.skip_reason == ""
        assert r.connector_sources_found == []

    def test_skipped_result(self):
        r = SynthesisResult(user_id="u1", app_id="app", skipped=True, skip_reason="no data")
        assert r.skipped is True
        assert r.skip_reason == "no data"
