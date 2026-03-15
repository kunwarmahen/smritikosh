"""
Tests for SynapticPruner, PruningResult, and compute_prune_score.

Unit tests mock the database session and EpisodicMemory.
DB integration tests are gated behind @pytest.mark.db.
"""

import math
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.processing.synaptic_pruner import (
    SynapticPruner,
    PruningResult,
    compute_prune_score,
    DEFAULT_PRUNE_THRESHOLD,
    DEFAULT_MIN_AGE_DAYS,
    DEFAULT_DECAY_DAYS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_event(
    importance: float = 0.5,
    age_days: int = 10,
    consolidated: bool = True,
) -> MagicMock:
    event = MagicMock()
    event.id = uuid.uuid4()
    event.importance_score = importance
    event.consolidated = consolidated
    event.created_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    return event


@pytest.fixture
def mock_episodic():
    episodic = AsyncMock()
    episodic.delete = AsyncMock(return_value=True)
    return episodic


@pytest.fixture
def pruner(mock_episodic):
    return SynapticPruner(episodic=mock_episodic)


# ── compute_prune_score ───────────────────────────────────────────────────────

class TestComputePruneScore:
    def test_zero_age(self):
        score = compute_prune_score(importance=1.0, age_days=0)
        assert score == pytest.approx(1.0)

    def test_full_decay(self):
        # At age == decay_days, score = importance * exp(-1) ≈ 0.368
        score = compute_prune_score(importance=1.0, age_days=DEFAULT_DECAY_DAYS)
        assert score == pytest.approx(math.exp(-1), rel=1e-5)

    def test_zero_importance(self):
        score = compute_prune_score(importance=0.0, age_days=10)
        assert score == 0.0

    def test_high_importance_survives_long(self):
        score = compute_prune_score(importance=1.0, age_days=30)
        assert score > DEFAULT_PRUNE_THRESHOLD

    def test_low_importance_pruned_quickly(self):
        score = compute_prune_score(importance=0.1, age_days=15)
        assert score < DEFAULT_PRUNE_THRESHOLD

    def test_custom_decay_days(self):
        # Short decay → faster drop
        fast = compute_prune_score(importance=1.0, age_days=10, decay_days=5.0)
        slow = compute_prune_score(importance=1.0, age_days=10, decay_days=60.0)
        assert fast < slow

    def test_monotonically_decreasing_with_age(self):
        scores = [compute_prune_score(1.0, age) for age in range(0, 100, 10)]
        for i in range(1, len(scores)):
            assert scores[i] < scores[i - 1]


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
    def test_default_threshold(self, pruner):
        assert pruner.prune_threshold == DEFAULT_PRUNE_THRESHOLD

    def test_default_min_age(self, pruner):
        assert pruner.min_age_days == DEFAULT_MIN_AGE_DAYS

    def test_default_decay_days(self, pruner):
        assert pruner.decay_days == DEFAULT_DECAY_DAYS

    def test_custom_params(self, mock_episodic):
        p = SynapticPruner(
            episodic=mock_episodic,
            prune_threshold=0.3,
            min_age_days=14,
            decay_days=60.0,
        )
        assert p.prune_threshold == 0.3
        assert p.min_age_days == 14
        assert p.decay_days == 60.0


# ── SynapticPruner._prune_score ───────────────────────────────────────────────

class TestPrunerScore:
    def test_matches_compute_fn(self, pruner):
        event = make_event(importance=0.8, age_days=20)
        now = datetime.now(timezone.utc)
        score = pruner._prune_score(event, now)
        expected = compute_prune_score(0.8, 20.0, DEFAULT_DECAY_DAYS)
        assert score == pytest.approx(expected, rel=0.01)

    def test_none_importance_treated_as_zero(self, pruner):
        event = make_event()
        event.importance_score = None
        now = datetime.now(timezone.utc)
        assert pruner._prune_score(event, now) == 0.0

    def test_none_created_at_returns_zero(self, pruner):
        event = make_event()
        event.created_at = None
        now = datetime.now(timezone.utc)
        assert pruner._prune_score(event, now) == 0.0

    def test_naive_datetime_handled(self, pruner):
        event = make_event()
        event.created_at = datetime.now() - timedelta(days=10)  # naive
        now = datetime.now(timezone.utc)
        score = pruner._prune_score(event, now)
        assert score >= 0.0


# ── SynapticPruner.prune — no candidates ─────────────────────────────────────

class TestPruneNoCandidates:
    async def test_skips_when_no_candidates(self, pruner, mock_episodic):
        session = AsyncMock()
        with patch.object(pruner, "_get_prune_candidates", AsyncMock(return_value=[])):
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.skipped is True
        assert result.events_evaluated == 0
        assert result.events_pruned == 0
        mock_episodic.delete.assert_not_called()


# ── SynapticPruner.prune — pruning logic ─────────────────────────────────────

class TestPruneLogic:
    async def test_prunes_low_score_event(self, pruner, mock_episodic):
        # Very old, very low importance → should be pruned
        low_event = make_event(importance=0.05, age_days=60)
        session = AsyncMock()
        with patch.object(pruner, "_get_prune_candidates", AsyncMock(return_value=[low_event])):
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 1
        mock_episodic.delete.assert_called_once_with(session, low_event.id)

    async def test_keeps_high_score_event(self, pruner, mock_episodic):
        # High importance, recent → score above threshold
        high_event = make_event(importance=0.9, age_days=8)
        session = AsyncMock()
        with patch.object(pruner, "_get_prune_candidates", AsyncMock(return_value=[high_event])):
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 0
        mock_episodic.delete.assert_not_called()

    async def test_mixed_events(self, pruner, mock_episodic):
        events = [
            make_event(importance=0.05, age_days=90),  # pruned
            make_event(importance=0.9, age_days=8),    # kept
            make_event(importance=0.03, age_days=60),  # pruned
        ]
        session = AsyncMock()
        with patch.object(pruner, "_get_prune_candidates", AsyncMock(return_value=events)):
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_evaluated == 3
        assert result.events_pruned == 2
        assert mock_episodic.delete.call_count == 2

    async def test_delete_returning_false_not_counted(self, pruner, mock_episodic):
        mock_episodic.delete.return_value = False
        low_event = make_event(importance=0.01, age_days=90)
        session = AsyncMock()
        with patch.object(pruner, "_get_prune_candidates", AsyncMock(return_value=[low_event])):
            result = await pruner.prune(session, user_id="u1", app_id="default")

        # delete returned False (already gone) → don't count as pruned
        assert result.events_pruned == 0

    async def test_evaluates_all_candidates(self, pruner, mock_episodic):
        events = make_events_list(10, importance=0.9, age_days=8)  # all kept
        session = AsyncMock()
        with patch.object(pruner, "_get_prune_candidates", AsyncMock(return_value=events)):
            result = await pruner.prune(session, user_id="u1", app_id="default")

        assert result.events_evaluated == 10
        assert result.events_pruned == 0


def make_events_list(n: int, importance: float, age_days: int) -> list[MagicMock]:
    return [make_event(importance=importance, age_days=age_days) for _ in range(n)]


# ── Custom threshold ──────────────────────────────────────────────────────────

class TestCustomThreshold:
    async def test_higher_threshold_prunes_more(self, mock_episodic):
        aggressive = SynapticPruner(
            episodic=mock_episodic,
            prune_threshold=0.8,  # very aggressive
        )
        events = [make_event(importance=0.5, age_days=10)]
        session = AsyncMock()
        with patch.object(aggressive, "_get_prune_candidates", AsyncMock(return_value=events)):
            result = await aggressive.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 1

    async def test_lower_threshold_prunes_less(self, mock_episodic):
        conservative = SynapticPruner(
            episodic=mock_episodic,
            prune_threshold=0.01,  # very conservative
        )
        events = [make_event(importance=0.5, age_days=10)]
        session = AsyncMock()
        with patch.object(conservative, "_get_prune_candidates", AsyncMock(return_value=events)):
            result = await conservative.prune(session, user_id="u1", app_id="default")

        assert result.events_pruned == 0
