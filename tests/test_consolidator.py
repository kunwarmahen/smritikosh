"""
Tests for Consolidator and ConsolidationResult.

Unit tests mock all external dependencies (LLM, EpisodicMemory, SemanticMemory).
DB integration tests are gated behind @pytest.mark.db.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.db.models import RelationType
from smritikosh.memory.narrative import NarrativeMemory
from smritikosh.processing.consolidator import (
    Consolidator,
    ConsolidationResult,
    MIN_EVENTS_TO_CONSOLIDATE,
    BATCH_SIZE,
    _split_batches,
    _build_consolidation_prompt,
    _format_date,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_event(i: int = 0, raw_text: str = "User said something interesting") -> MagicMock:
    event = MagicMock()
    event.id = uuid.uuid4()
    event.raw_text = raw_text
    event.summary = None
    event.created_at = datetime(2024, 1, i + 1, tzinfo=timezone.utc)
    return event


def make_events(n: int) -> list[MagicMock]:
    return [make_event(i, f"Event number {i}") for i in range(n)]


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.extract_structured.return_value = {
        "summary": "User is building a memory system.",
        "facts": [
            {"category": "project", "key": "active", "value": "smritikosh", "confidence": 0.95},
            {"category": "preference", "key": "ui_color", "value": "green", "confidence": 0.9},
        ],
    }
    return llm


@pytest.fixture
def mock_episodic():
    episodic = AsyncMock()
    episodic.get_unconsolidated = AsyncMock(return_value=[])
    episodic.mark_consolidated = AsyncMock()
    return episodic


@pytest.fixture
def mock_semantic():
    semantic = AsyncMock()
    semantic.upsert_fact = AsyncMock(return_value=MagicMock())
    return semantic


@pytest.fixture
def consolidator(mock_llm, mock_episodic, mock_semantic):
    return Consolidator(
        llm=mock_llm,
        episodic=mock_episodic,
        semantic=mock_semantic,
    )


# ── ConsolidationResult ───────────────────────────────────────────────────────

class TestConsolidationResult:
    def test_defaults(self):
        r = ConsolidationResult(user_id="u1", app_id="default")
        assert r.events_processed == 0
        assert r.events_consolidated == 0
        assert r.facts_distilled == 0
        assert r.links_created == 0
        assert r.batches == 0
        assert r.skipped is False
        assert r.skip_reason == ""

    def test_skip_fields(self):
        r = ConsolidationResult(user_id="u1", app_id="app", skipped=True)
        r.skip_reason = "Not enough events."
        assert r.skipped is True
        assert "Not enough events" in r.skip_reason


# ── _split_batches ────────────────────────────────────────────────────────────

class TestSplitBatches:
    def test_exact_batch(self):
        events = make_events(10)
        batches = _split_batches(events, 10)
        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_multiple_batches(self):
        events = make_events(25)
        batches = _split_batches(events, 10)
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

    def test_empty(self):
        assert _split_batches([], 10) == []

    def test_smaller_than_batch(self):
        events = make_events(3)
        batches = _split_batches(events, 10)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_preserves_order(self):
        events = make_events(5)
        batches = _split_batches(events, 3)
        flat = [e for batch in batches for e in batch]
        assert flat == events


# ── _build_consolidation_prompt ───────────────────────────────────────────────

class TestBuildConsolidationPrompt:
    def test_contains_header(self):
        events = make_events(2)
        prompt = _build_consolidation_prompt(events)
        assert "Consolidate" in prompt
        assert "Interactions:" in prompt

    def test_lists_all_events(self):
        events = make_events(3)
        prompt = _build_consolidation_prompt(events)
        assert "1." in prompt
        assert "2." in prompt
        assert "3." in prompt

    def test_truncates_long_text(self):
        event = make_event(0, "x" * 500)
        prompt = _build_consolidation_prompt([event])
        # raw_text is truncated to 300 chars
        assert "x" * 300 in prompt
        assert "x" * 301 not in prompt

    def test_uses_summary_over_raw(self):
        event = make_event(0)
        event.summary = "Short summary"
        event.raw_text = "Very long raw text that should not appear"
        prompt = _build_consolidation_prompt([event])
        assert "Short summary" in prompt

    def test_contains_json_instruction(self):
        events = make_events(1)
        prompt = _build_consolidation_prompt(events)
        assert "JSON" in prompt


# ── _format_date ──────────────────────────────────────────────────────────────

class TestFormatDate:
    def test_formats_utc(self):
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        assert _format_date(dt) == "2024-06-15"

    def test_none_returns_unknown(self):
        assert _format_date(None) == "unknown"

    def test_naive_datetime(self):
        dt = datetime(2024, 3, 1)
        result = _format_date(dt)
        assert result == "2024-03-01"


# ── Consolidator.run — skip guard ─────────────────────────────────────────────

class TestConsolidatorSkipGuard:
    async def test_skips_when_too_few_events(self, consolidator, mock_episodic):
        mock_episodic.get_unconsolidated.return_value = make_events(MIN_EVENTS_TO_CONSOLIDATE - 1)
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        assert result.skipped is True
        assert result.events_consolidated == 0
        assert str(MIN_EVENTS_TO_CONSOLIDATE - 1) in result.skip_reason

    async def test_skips_when_zero_events(self, consolidator, mock_episodic):
        mock_episodic.get_unconsolidated.return_value = []
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        assert result.skipped is True

    async def test_runs_when_enough_events(self, consolidator, mock_episodic):
        mock_episodic.get_unconsolidated.return_value = make_events(MIN_EVENTS_TO_CONSOLIDATE)
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        assert result.skipped is False


# ── Consolidator.run — success path ──────────────────────────────────────────

class TestConsolidatorSuccess:
    async def test_marks_events_consolidated(self, consolidator, mock_episodic, mock_llm):
        events = make_events(5)
        mock_episodic.get_unconsolidated.return_value = events
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        mock_episodic.mark_consolidated.assert_called_once()
        call_args = mock_episodic.mark_consolidated.call_args
        assert len(call_args[0][1]) == 5  # event_ids list

    async def test_upserts_facts(self, consolidator, mock_episodic, mock_semantic):
        mock_episodic.get_unconsolidated.return_value = make_events(5)
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        # LLM returns 2 facts → 2 upsert calls
        assert mock_semantic.upsert_fact.call_count == 2
        assert result.facts_distilled == 2

    async def test_result_counts(self, consolidator, mock_episodic):
        events = make_events(5)
        mock_episodic.get_unconsolidated.return_value = events
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        assert result.events_processed == 5
        assert result.events_consolidated == 5
        assert result.batches == 1

    async def test_multiple_batches(self, consolidator, mock_episodic, mock_llm):
        events = make_events(15)  # batch_size=10 → 2 batches
        mock_episodic.get_unconsolidated.return_value = events
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        assert result.batches == 2
        assert result.events_consolidated == 15
        assert mock_llm.extract_structured.call_count == 2


# ── Consolidator.run — LLM failure ───────────────────────────────────────────

class TestConsolidatorLLMFailure:
    async def test_graceful_on_llm_error(self, consolidator, mock_episodic, mock_llm):
        mock_episodic.get_unconsolidated.return_value = make_events(5)
        mock_llm.extract_structured.side_effect = RuntimeError("API down")
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        # batch was skipped but no exception raised
        assert result.events_consolidated == 0
        assert result.facts_distilled == 0

    async def test_skips_invalid_fact_keys(self, consolidator, mock_episodic, mock_llm, mock_semantic):
        mock_episodic.get_unconsolidated.return_value = make_events(5)
        mock_llm.extract_structured.return_value = {
            "summary": "Test",
            "facts": [
                {"category": "project", "key": "active", "value": "smritikosh", "confidence": 0.9},
                {"category": "project"},  # missing key and value → KeyError
            ],
        }
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        # Only 1 valid fact stored
        assert result.facts_distilled == 1

    async def test_handles_empty_facts_list(self, consolidator, mock_episodic, mock_llm, mock_semantic):
        mock_episodic.get_unconsolidated.return_value = make_events(5)
        mock_llm.extract_structured.return_value = {"summary": "Summary only", "facts": []}
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        assert result.facts_distilled == 0
        assert result.events_consolidated == 5
        mock_semantic.upsert_fact.assert_not_called()

    async def test_missing_summary_key(self, consolidator, mock_episodic, mock_llm):
        mock_episodic.get_unconsolidated.return_value = make_events(5)
        mock_llm.extract_structured.return_value = {"facts": []}  # no "summary"
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1", app_id="default")

        # mark_consolidated called with summary=None (empty string → None)
        mock_episodic.mark_consolidated.assert_called_once()


# ── Consolidator — narrative link extraction ──────────────────────────────────


@pytest.fixture
def mock_narrative():
    narrative = AsyncMock(spec=NarrativeMemory)
    narrative.create_link = AsyncMock(return_value=MagicMock())
    return narrative


@pytest.fixture
def consolidator_with_narrative(mock_llm, mock_episodic, mock_semantic, mock_narrative):
    return Consolidator(
        llm=mock_llm,
        episodic=mock_episodic,
        semantic=mock_semantic,
        narrative=mock_narrative,
    )


class TestConsolidatorLinks:
    async def test_creates_links_from_llm_output(
        self, consolidator_with_narrative, mock_episodic, mock_llm, mock_narrative
    ):
        events = make_events(5)
        mock_episodic.get_unconsolidated.return_value = events
        mock_llm.extract_structured.return_value = {
            "summary": "Summary",
            "facts": [],
            "links": [
                {"from_index": 0, "to_index": 1, "relation_type": "preceded"},
                {"from_index": 1, "to_index": 2, "relation_type": "caused"},
            ],
        }
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator_with_narrative.run(pg, neo, user_id="u1")

        assert mock_narrative.create_link.call_count == 2
        assert result.links_created == 2

    async def test_skips_out_of_bounds_link_indices(
        self, consolidator_with_narrative, mock_episodic, mock_llm, mock_narrative
    ):
        events = make_events(5)
        mock_episodic.get_unconsolidated.return_value = events
        mock_llm.extract_structured.return_value = {
            "summary": "Summary",
            "facts": [],
            "links": [
                {"from_index": 0, "to_index": 99, "relation_type": "preceded"},  # out of bounds
                {"from_index": 0, "to_index": 1, "relation_type": "caused"},     # valid
            ],
        }
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator_with_narrative.run(pg, neo, user_id="u1")

        assert mock_narrative.create_link.call_count == 1
        assert result.links_created == 1

    async def test_skips_self_links(
        self, consolidator_with_narrative, mock_episodic, mock_llm, mock_narrative
    ):
        events = make_events(5)
        mock_episodic.get_unconsolidated.return_value = events
        mock_llm.extract_structured.return_value = {
            "summary": "Summary",
            "facts": [],
            "links": [
                {"from_index": 0, "to_index": 0, "relation_type": "related"},  # self-link
            ],
        }
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator_with_narrative.run(pg, neo, user_id="u1")

        mock_narrative.create_link.assert_not_called()
        assert result.links_created == 0

    async def test_skips_invalid_relation_type(
        self, consolidator_with_narrative, mock_episodic, mock_llm, mock_narrative
    ):
        events = make_events(5)
        mock_episodic.get_unconsolidated.return_value = events
        mock_llm.extract_structured.return_value = {
            "summary": "Summary",
            "facts": [],
            "links": [
                {"from_index": 0, "to_index": 1, "relation_type": "invalid_type"},
            ],
        }
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator_with_narrative.run(pg, neo, user_id="u1")

        mock_narrative.create_link.assert_not_called()
        assert result.links_created == 0

    async def test_no_links_when_narrative_is_none(
        self, mock_llm, mock_episodic, mock_semantic
    ):
        """Consolidator without narrative= set should not create any links."""
        consolidator = Consolidator(
            llm=mock_llm,
            episodic=mock_episodic,
            semantic=mock_semantic,
            narrative=None,
        )
        mock_episodic.get_unconsolidated.return_value = make_events(5)
        mock_llm.extract_structured.return_value = {
            "summary": "Summary",
            "facts": [],
            "links": [{"from_index": 0, "to_index": 1, "relation_type": "preceded"}],
        }
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator.run(pg, neo, user_id="u1")

        assert result.links_created == 0

    async def test_empty_links_list_no_calls(
        self, consolidator_with_narrative, mock_episodic, mock_llm, mock_narrative
    ):
        events = make_events(5)
        mock_episodic.get_unconsolidated.return_value = events
        mock_llm.extract_structured.return_value = {
            "summary": "Summary",
            "facts": [],
            "links": [],
        }
        pg, neo = AsyncMock(), AsyncMock()

        result = await consolidator_with_narrative.run(pg, neo, user_id="u1")

        mock_narrative.create_link.assert_not_called()
        assert result.links_created == 0
