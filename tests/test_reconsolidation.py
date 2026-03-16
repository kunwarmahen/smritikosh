"""
Unit tests for ReconsolidationEngine.

Tests cover:
  - Gate conditions (recall_count, importance_score, cooldown)
  - _reconsolidate_one: LLM call, DB update, skip paths
  - reconsolidate_event: UUID validation, event-not-found
  - reconsolidate_after_recall: batch processing, skipped vs updated counts
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.db.models import Event
from smritikosh.memory.episodic import EpisodicMemory, SearchResult
from smritikosh.processing.reconsolidation import (
    ReconsolidationEngine,
    ReconsolidationResult,
    DEFAULT_MIN_RECALL_COUNT,
    DEFAULT_MIN_IMPORTANCE,
    DEFAULT_COOLDOWN_HOURS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(
    recall_count: int = 3,
    importance_score: float = 0.8,
    last_reconsolidated_at: datetime | None = None,
    summary: str | None = "original summary",
) -> Event:
    e = Event(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        raw_text="User discussed building smritikosh",
        summary=summary,
        importance_score=importance_score,
        recall_count=recall_count,
        reconsolidation_count=0,
        last_reconsolidated_at=last_reconsolidated_at,
    )
    return e


def make_engine(
    llm_response: dict | None = None,
    llm_raises: Exception | None = None,
    min_recall_count: int = DEFAULT_MIN_RECALL_COUNT,
    min_importance: float = DEFAULT_MIN_IMPORTANCE,
    cooldown_hours: int = DEFAULT_COOLDOWN_HOURS,
) -> tuple[ReconsolidationEngine, AsyncMock, AsyncMock]:
    llm = AsyncMock()
    if llm_raises:
        llm.extract_structured = AsyncMock(side_effect=llm_raises)
    else:
        llm.extract_structured = AsyncMock(
            return_value=llm_response or {"summary": "refined summary", "changed": True}
        )
    episodic = AsyncMock(spec=EpisodicMemory)
    engine = ReconsolidationEngine(
        llm=llm,
        episodic=episodic,
        min_recall_count=min_recall_count,
        min_importance=min_importance,
        cooldown_hours=cooldown_hours,
    )
    return engine, llm, episodic


# ── Gate conditions ───────────────────────────────────────────────────────────


class TestCheckGate:
    def test_passes_when_all_conditions_met(self):
        engine, _, _ = make_engine()
        event = make_event(recall_count=5, importance_score=0.9)
        assert engine._check_gate(event) == ""

    def test_blocks_low_recall_count(self):
        engine, _, _ = make_engine(min_recall_count=3)
        event = make_event(recall_count=1)
        reason = engine._check_gate(event)
        assert "recall_count" in reason

    def test_blocks_low_importance_score(self):
        engine, _, _ = make_engine(min_importance=0.5)
        event = make_event(importance_score=0.3)
        reason = engine._check_gate(event)
        assert "importance_score" in reason

    def test_blocks_during_cooldown(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=10)
        engine, _, _ = make_engine(cooldown_hours=1)
        event = make_event(last_reconsolidated_at=recent)
        reason = engine._check_gate(event)
        assert "cooldown" in reason

    def test_passes_after_cooldown_expires(self):
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        engine, _, _ = make_engine(cooldown_hours=1)
        event = make_event(last_reconsolidated_at=old)
        assert engine._check_gate(event) == ""

    def test_handles_naive_last_reconsolidated_at(self):
        """Naive datetime (no tzinfo) should be treated as UTC."""
        naive = datetime.utcnow() - timedelta(minutes=5)
        engine, _, _ = make_engine(cooldown_hours=1)
        event = make_event(last_reconsolidated_at=naive)
        reason = engine._check_gate(event)
        assert "cooldown" in reason


# ── _reconsolidate_one ────────────────────────────────────────────────────────


class TestReconsolidateOne:
    @pytest.mark.asyncio
    async def test_updates_event_when_changed(self):
        engine, llm, episodic = make_engine(
            llm_response={"summary": "refined memory", "changed": True}
        )
        session = AsyncMock()
        event = make_event()

        result = await engine._reconsolidate_one(session, event, "what am I building?")

        assert result.updated is True
        assert result.new_summary == "refined memory"
        assert result.old_summary == "original summary"
        episodic.update_summary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_when_llm_reports_unchanged(self):
        engine, _, episodic = make_engine(
            llm_response={"summary": "same summary", "changed": False}
        )
        session = AsyncMock()
        event = make_event()

        result = await engine._reconsolidate_one(session, event, "query")

        assert result.skipped is True
        assert "no meaningful change" in result.skip_reason
        episodic.update_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_gate_fails(self):
        engine, llm, _ = make_engine(min_recall_count=10)
        session = AsyncMock()
        event = make_event(recall_count=1)

        result = await engine._reconsolidate_one(session, event, "query")

        assert result.skipped is True
        llm.extract_structured.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_on_llm_failure(self):
        engine, _, episodic = make_engine(llm_raises=RuntimeError("timeout"))
        session = AsyncMock()
        event = make_event()

        result = await engine._reconsolidate_one(session, event, "query")

        assert result.skipped is True
        assert "LLM call failed" in result.skip_reason
        episodic.update_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_raw_text_when_no_summary(self):
        engine, llm, _ = make_engine()
        session = AsyncMock()
        event = make_event(summary=None)

        result = await engine._reconsolidate_one(session, event, "query")

        # old_summary should fall back to raw_text
        assert result.old_summary == event.raw_text

    @pytest.mark.asyncio
    async def test_skips_when_empty_summary_returned(self):
        engine, _, episodic = make_engine(
            llm_response={"summary": "", "changed": True}
        )
        session = AsyncMock()
        event = make_event()

        result = await engine._reconsolidate_one(session, event, "query")

        assert result.skipped is True
        episodic.update_summary.assert_not_awaited()


# ── reconsolidate_event (admin entry point) ───────────────────────────────────


class TestReconsolidateEvent:
    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_skipped(self):
        engine, _, _ = make_engine()
        result = await engine.reconsolidate_event("not-a-uuid", "query", "u1")

        assert result.skipped is True
        assert "Invalid UUID" in result.skip_reason

    @pytest.mark.asyncio
    async def test_event_not_found_returns_skipped(self):
        engine, _, _ = make_engine()
        event_id = str(uuid.uuid4())

        session_mock = AsyncMock()
        session_mock.get = AsyncMock(return_value=None)
        ctx_mock = MagicMock()
        ctx_mock.__aenter__ = AsyncMock(return_value=session_mock)
        ctx_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("smritikosh.processing.reconsolidation.db_session", return_value=ctx_mock):
            result = await engine.reconsolidate_event(event_id, "query", "u1")

        assert result.skipped is True
        assert "not found" in result.skip_reason.lower()

    @pytest.mark.asyncio
    async def test_successful_reconsolidation(self):
        engine, llm, episodic = make_engine(
            llm_response={"summary": "better summary", "changed": True}
        )
        event = make_event()
        event_id = str(event.id)

        session_mock = AsyncMock()
        session_mock.get = AsyncMock(return_value=event)
        ctx_mock = MagicMock()
        ctx_mock.__aenter__ = AsyncMock(return_value=session_mock)
        ctx_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("smritikosh.processing.reconsolidation.db_session", return_value=ctx_mock):
            result = await engine.reconsolidate_event(event_id, "query", "u1")

        assert result.updated is True
        assert result.new_summary == "better summary"


# ── reconsolidate_after_recall ────────────────────────────────────────────────


class TestReconsolidateAfterRecall:
    def _make_search_result(self, **event_kwargs) -> SearchResult:
        event = make_event(**event_kwargs)
        return SearchResult(event=event, hybrid_score=0.9)

    @pytest.mark.asyncio
    async def test_processes_only_max_events(self):
        engine, llm, _ = make_engine()
        engine.max_events = 1

        results = [self._make_search_result(), self._make_search_result()]

        session_mock = AsyncMock()
        ctx_mock = MagicMock()
        ctx_mock.__aenter__ = AsyncMock(return_value=session_mock)
        ctx_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("smritikosh.processing.reconsolidation.db_session", return_value=ctx_mock):
            batch = await engine.reconsolidate_after_recall(results, "query", "u1")

        assert batch.events_evaluated == 1
        assert llm.extract_structured.await_count <= 1

    @pytest.mark.asyncio
    async def test_counts_updated_and_skipped(self):
        engine, _, _ = make_engine(
            llm_response={"summary": "refined", "changed": True}
        )
        engine.max_events = 3

        sr_good = self._make_search_result(recall_count=5, importance_score=0.9)
        sr_low_recall = self._make_search_result(recall_count=0, importance_score=0.9)
        results = [sr_good, sr_low_recall, sr_low_recall]

        session_mock = AsyncMock()
        ctx_mock = MagicMock()
        ctx_mock.__aenter__ = AsyncMock(return_value=session_mock)
        ctx_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("smritikosh.processing.reconsolidation.db_session", return_value=ctx_mock):
            batch = await engine.reconsolidate_after_recall(results, "query", "u1")

        assert batch.events_updated == 1
        assert batch.events_skipped == 2

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_batch(self):
        engine, _, _ = make_engine()

        session_mock = AsyncMock()
        ctx_mock = MagicMock()
        ctx_mock.__aenter__ = AsyncMock(return_value=session_mock)
        ctx_mock.__aexit__ = AsyncMock(return_value=False)

        with patch("smritikosh.processing.reconsolidation.db_session", return_value=ctx_mock):
            batch = await engine.reconsolidate_after_recall([], "query", "u1")

        assert batch.events_evaluated == 0
        assert batch.events_updated == 0
