"""
Tests for SynapticPruner, PruningResult, and compute_prune_decision.

Unit tests mock the database session and EpisodicMemory.
DB integration tests are gated behind @pytest.mark.db.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.processing.synaptic_pruner import (
    SynapticPruner,
    PruningResult,
    PruningThresholds,
    compute_prune_decision,
    compute_adaptive_thresholds,
    DEFAULT_IMPORTANCE_THRESHOLD,
    DEFAULT_MIN_RECALL_COUNT,
    DEFAULT_MIN_AGE_DAYS,
    HIGH_VOLUME_EVENT_COUNT,
    LOW_VOLUME_EVENT_COUNT,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_event(
    importance: float = 0.5,
    age_days: int = 10,
    consolidated: bool = True,
    recall_count: int = 0,
) -> MagicMock:
    event = MagicMock()
    event.id = uuid.uuid4()
    event.importance_score = importance
    event.recall_count = recall_count
    event.consolidated = consolidated
    event.created_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return event


def make_events_list(n: int, importance: float, age_days: int, recall_count: int = 0) -> list[MagicMock]:
    return [make_event(importance=importance, age_days=age_days, recall_count=recall_count) for _ in range(n)]


@pytest.fixture
def mock_episodic():
    episodic = AsyncMock()
    episodic.delete = AsyncMock(return_value=True)
    return episodic


@pytest.fixture
def pruner(mock_episodic):
    return SynapticPruner(episodic=mock_episodic)


# ── compute_prune_decision ────────────────────────────────────────────────────

class TestComputePruneDecision:
    def test_all_conditions_met_returns_true(self):
        assert compute_prune_decision(importance=0.1, recall_count=0, age_days=100) is True

    def test_importance_too_high_returns_false(self):
        assert compute_prune_decision(importance=0.5, recall_count=0, age_days=100) is False

    def test_recall_count_too_high_returns_false(self):
        assert compute_prune_decision(importance=0.1, recall_count=3, age_days=100) is False

    def test_too_young_returns_false(self):
        assert compute_prune_decision(importance=0.1, recall_count=0, age_days=10) is False

    def test_boundary_importance_equal_threshold_not_pruned(self):
        # strict less-than: 0.2 is NOT below 0.2
        assert compute_prune_decision(importance=0.2, recall_count=0, age_days=100) is False

    def test_boundary_recall_equal_min_not_pruned(self):
        # strict less-than: 2 is NOT below 2
        assert compute_prune_decision(importance=0.1, recall_count=2, age_days=100) is False

    def test_boundary_age_equal_min_not_pruned(self):
        # strict greater-than: 90 is NOT greater than 90
        assert compute_prune_decision(importance=0.1, recall_count=0, age_days=90) is False

    def test_custom_importance_threshold(self):
        assert compute_prune_decision(0.3, 0, 100, importance_threshold=0.5) is True
        assert compute_prune_decision(0.3, 0, 100, importance_threshold=0.1) is False

    def test_custom_min_recall_count(self):
        assert compute_prune_decision(0.1, 3, 100, min_recall_count=5) is True
        assert compute_prune_decision(0.1, 3, 100, min_recall_count=2) is False

    def test_custom_min_age_days(self):
        assert compute_prune_decision(0.1, 0, 30, min_age_days=20) is True
        assert compute_prune_decision(0.1, 0, 30, min_age_days=60) is False


# ── PruningResult ─────────────────────────────────────────────────────────────

class TestPruningResult:
    def test_defaults(self):
        r = PruningResult(user_id="u1", app_id="default")
        assert r.events_evaluated == 0
        assert r.events_pruned == 0
        assert r.skipped is False

    def test_custom_values(self):
        r = PruningResult(user_id="u1", app_id="app", events_evaluated=10, events_pruned=3)
        assert r.events_evaluated == 10
        assert r.events_pruned == 3


# ── SynapticPruner defaults ───────────────────────────────────────────────────

class TestPrunerDefaults:
    def test_default_importance_threshold(self, pruner):
        assert pruner.importance_threshold == DEFAULT_IMPORTANCE_THRESHOLD

    def test_default_min_recall_count(self, pruner):
        assert pruner.min_recall_count == DEFAULT_MIN_RECALL_COUNT

    def test_default_min_age(self, pruner):
        assert pruner.min_age_days == DEFAULT_MIN_AGE_DAYS

    def test_custom_params(self, mock_episodic):
        p = SynapticPruner(
            episodic=mock_episodic,
            importance_threshold=0.3,
            min_recall_count=5,
            min_age_days=60,
        )
        assert p.importance_threshold == 0.3
        assert p.min_recall_count == 5
        assert p.min_age_days == 60


# ── SynapticPruner._should_prune ─────────────────────────────────────────────

_DEFAULT_THRESHOLDS = PruningThresholds(
    importance_threshold=DEFAULT_IMPORTANCE_THRESHOLD,
    min_recall_count=DEFAULT_MIN_RECALL_COUNT,
    min_age_days=DEFAULT_MIN_AGE_DAYS,
)


class TestShouldPrune:
    def test_prunes_when_all_conditions_met(self, pruner):
        event = make_event(importance=0.1, age_days=100, recall_count=0)
        now = datetime.now(timezone.utc)
        assert pruner._should_prune(event, now, _DEFAULT_THRESHOLDS) is True

    def test_keeps_high_importance(self, pruner):
        event = make_event(importance=0.9, age_days=100, recall_count=0)
        now = datetime.now(timezone.utc)
        assert pruner._should_prune(event, now, _DEFAULT_THRESHOLDS) is False

    def test_keeps_frequently_recalled(self, pruner):
        event = make_event(importance=0.1, age_days=100, recall_count=5)
        now = datetime.now(timezone.utc)
        assert pruner._should_prune(event, now, _DEFAULT_THRESHOLDS) is False

    def test_keeps_young_event(self, pruner):
        event = make_event(importance=0.1, age_days=10, recall_count=0)
        now = datetime.now(timezone.utc)
        assert pruner._should_prune(event, now, _DEFAULT_THRESHOLDS) is False

    def test_none_importance_treated_as_zero(self, pruner):
        event = make_event(importance=0.1, age_days=100, recall_count=0)
        event.importance_score = None
        now = datetime.now(timezone.utc)
        assert pruner._should_prune(event, now, _DEFAULT_THRESHOLDS) is True  # 0.0 < 0.2

    def test_none_recall_count_treated_as_zero(self, pruner):
        event = make_event(importance=0.1, age_days=100, recall_count=0)
        event.recall_count = None
        now = datetime.now(timezone.utc)
        assert pruner._should_prune(event, now, _DEFAULT_THRESHOLDS) is True  # 0 < 2

    def test_none_created_at_returns_false(self, pruner):
        event = make_event()
        event.created_at = None
        now = datetime.now(timezone.utc)
        assert pruner._should_prune(event, now, _DEFAULT_THRESHOLDS) is False

    def test_naive_datetime_handled(self, pruner):
        event = make_event(importance=0.1, age_days=100, recall_count=0)
        event.created_at = datetime.now() - timedelta(days=100)  # naive
        now = datetime.now(timezone.utc)
        result = pruner._should_prune(event, now, _DEFAULT_THRESHOLDS)
        assert isinstance(result, bool)


# ── SynapticPruner.prune — no candidates ─────────────────────────────────────

def _patch_prune(pruner, candidates, event_count=500):
    """Patch both internal methods so prune() doesn't need a real DB session."""
    return (
        patch.object(pruner, "_count_user_events", AsyncMock(return_value=event_count)),
        patch.object(pruner, "_get_prune_candidates", AsyncMock(return_value=candidates)),
    )


class TestPruneNoCandidates:
    async def test_skips_when_no_candidates(self, pruner, mock_episodic):
        session = AsyncMock()
        count_p, cands_p = _patch_prune(pruner, [])
        with count_p, cands_p:
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.skipped is True
        assert result.events_evaluated == 0
        assert result.events_pruned == 0
        mock_episodic.delete.assert_not_called()

    async def test_result_contains_thresholds(self, pruner):
        session = AsyncMock()
        count_p, cands_p = _patch_prune(pruner, [])
        with count_p, cands_p:
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.thresholds is not None
        assert isinstance(result.thresholds, PruningThresholds)


# ── SynapticPruner.prune — pruning logic ─────────────────────────────────────

class TestPruneLogic:
    async def test_prunes_event_meeting_all_conditions(self, pruner, mock_episodic):
        # Low importance, never recalled, old → should be pruned
        low_event = make_event(importance=0.05, age_days=100, recall_count=0)
        session = AsyncMock()
        count_p, cands_p = _patch_prune(pruner, [low_event])
        with count_p, cands_p:
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 1
        mock_episodic.delete.assert_called_once_with(session, low_event.id)

    async def test_keeps_high_importance_event(self, pruner, mock_episodic):
        high_event = make_event(importance=0.9, age_days=100, recall_count=0)
        session = AsyncMock()
        count_p, cands_p = _patch_prune(pruner, [high_event])
        with count_p, cands_p:
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 0
        mock_episodic.delete.assert_not_called()

    async def test_keeps_recalled_event(self, pruner, mock_episodic):
        recalled_event = make_event(importance=0.05, age_days=100, recall_count=3)
        session = AsyncMock()
        count_p, cands_p = _patch_prune(pruner, [recalled_event])
        with count_p, cands_p:
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 0
        mock_episodic.delete.assert_not_called()

    async def test_mixed_events(self, pruner, mock_episodic):
        events = [
            make_event(importance=0.05, age_days=100, recall_count=0),  # pruned
            make_event(importance=0.9,  age_days=100, recall_count=0),  # kept: high importance
            make_event(importance=0.03, age_days=100, recall_count=0),  # pruned
        ]
        session = AsyncMock()
        count_p, cands_p = _patch_prune(pruner, events)
        with count_p, cands_p:
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_evaluated == 3
        assert result.events_pruned == 2
        assert mock_episodic.delete.call_count == 2

    async def test_delete_returning_false_not_counted(self, pruner, mock_episodic):
        mock_episodic.delete.return_value = False
        low_event = make_event(importance=0.01, age_days=100, recall_count=0)
        session = AsyncMock()
        count_p, cands_p = _patch_prune(pruner, [low_event])
        with count_p, cands_p:
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 0

    async def test_evaluates_all_candidates(self, pruner, mock_episodic):
        events = make_events_list(10, importance=0.9, age_days=100)  # all kept
        session = AsyncMock()
        count_p, cands_p = _patch_prune(pruner, events)
        with count_p, cands_p:
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_evaluated == 10
        assert result.events_pruned == 0


# ── Custom threshold ──────────────────────────────────────────────────────────

class TestCustomThreshold:
    async def test_higher_threshold_prunes_more(self, mock_episodic):
        aggressive = SynapticPruner(
            episodic=mock_episodic,
            importance_threshold=0.8,  # importance=0.5 < 0.8 → pruned
        )
        events = [make_event(importance=0.5, age_days=100, recall_count=0)]
        session = AsyncMock()
        count_p, cands_p = _patch_prune(aggressive, events)
        with count_p, cands_p:
            result = await aggressive.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 1

    async def test_lower_threshold_prunes_less(self, mock_episodic):
        conservative = SynapticPruner(
            episodic=mock_episodic,
            importance_threshold=0.01,  # importance=0.5 >= 0.01 → kept
        )
        events = [make_event(importance=0.5, age_days=100, recall_count=0)]
        session = AsyncMock()
        count_p, cands_p = _patch_prune(conservative, events)
        with count_p, cands_p:
            result = await conservative.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 0

    async def test_higher_min_recall_prunes_more(self, mock_episodic):
        aggressive = SynapticPruner(
            episodic=mock_episodic,
            min_recall_count=10,  # recall=5 < 10 → pruned
        )
        events = [make_event(importance=0.1, age_days=100, recall_count=5)]
        session = AsyncMock()
        count_p, cands_p = _patch_prune(aggressive, events)
        with count_p, cands_p:
            result = await aggressive.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 1


# ── compute_adaptive_thresholds ───────────────────────────────────────────────

class TestComputeAdaptiveThresholds:
    def test_normal_volume_returns_base_defaults(self):
        t = compute_adaptive_thresholds(500)
        assert t.importance_threshold == DEFAULT_IMPORTANCE_THRESHOLD
        assert t.min_recall_count == DEFAULT_MIN_RECALL_COUNT
        assert t.min_age_days == DEFAULT_MIN_AGE_DAYS

    def test_high_volume_raises_importance_threshold(self):
        t = compute_adaptive_thresholds(HIGH_VOLUME_EVENT_COUNT + 1)
        assert t.importance_threshold > DEFAULT_IMPORTANCE_THRESHOLD

    def test_high_volume_lowers_min_age(self):
        t = compute_adaptive_thresholds(HIGH_VOLUME_EVENT_COUNT + 1)
        assert t.min_age_days < DEFAULT_MIN_AGE_DAYS

    def test_low_volume_lowers_importance_threshold(self):
        t = compute_adaptive_thresholds(LOW_VOLUME_EVENT_COUNT - 1)
        assert t.importance_threshold < DEFAULT_IMPORTANCE_THRESHOLD

    def test_low_volume_raises_min_age(self):
        t = compute_adaptive_thresholds(LOW_VOLUME_EVENT_COUNT - 1)
        assert t.min_age_days > DEFAULT_MIN_AGE_DAYS

    def test_boundary_high_volume_is_high_tier(self):
        t = compute_adaptive_thresholds(HIGH_VOLUME_EVENT_COUNT + 1)
        t_normal = compute_adaptive_thresholds(HIGH_VOLUME_EVENT_COUNT)
        # Exactly at the boundary is normal tier; one above is high
        assert t.importance_threshold > t_normal.importance_threshold

    def test_boundary_low_volume_is_low_tier(self):
        t = compute_adaptive_thresholds(LOW_VOLUME_EVENT_COUNT - 1)
        t_normal = compute_adaptive_thresholds(LOW_VOLUME_EVENT_COUNT)
        assert t.importance_threshold < t_normal.importance_threshold

    def test_custom_base_values_scaled_correctly(self):
        t = compute_adaptive_thresholds(
            HIGH_VOLUME_EVENT_COUNT + 1,
            base_importance=0.4,
            base_min_recall=3,
            base_min_age=120,
        )
        assert t.importance_threshold == pytest.approx(0.4 * 1.5, abs=1e-4)
        assert t.min_recall_count == 3
        assert t.min_age_days == max(1, round(120 * 0.67))

    def test_returns_pruning_thresholds_type(self):
        t = compute_adaptive_thresholds(100)
        assert isinstance(t, PruningThresholds)

    def test_prune_uses_adaptive_thresholds_for_high_volume_user(self, mock_episodic):
        """High-volume user gets a tighter threshold → event that normal tier keeps is pruned."""
        pruner = SynapticPruner(episodic=mock_episodic)
        # importance=0.25 is above the default 0.2 threshold but below 0.3 (high-volume tier)
        event = make_event(importance=0.25, age_days=70, recall_count=0)
        now = datetime.now(timezone.utc)
        high_thresholds = compute_adaptive_thresholds(HIGH_VOLUME_EVENT_COUNT + 1)
        normal_thresholds = compute_adaptive_thresholds(500)
        assert pruner._should_prune(event, now, high_thresholds) is True
        assert pruner._should_prune(event, now, normal_thresholds) is False
