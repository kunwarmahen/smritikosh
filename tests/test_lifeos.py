"""
Tests for the Proactive Life OS (E4, FUTURE.md #5):
    LifeOSAgent — nudge composition from reflections, cooldown, severity
    threshold, at-most-once per insight, webhook delivery + feed fallback,
    acknowledge scoping.

All tests run offline with mocked sessions; no LLM is involved (the agent
is deliberately LLM-free).
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.cognition.lifeos import LifeOSAgent, NudgeResult
from smritikosh.db.models import Nudge, Reflection


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_reflection(
    severity: str = "notice",
    kind: str = "drift",
    insight: str = "You stated a goal but logged nothing recently.",
) -> Reflection:
    return Reflection(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        kind=kind,
        insight=insight,
        severity=severity,
        evidence={},
        acknowledged=False,
        created_at=datetime.now(timezone.utc),
    )


def make_session(
    last_nudge_at: datetime | None = None,
    nudged_reflection_ids: list[list[str]] | None = None,
    reflections: list[Reflection] | None = None,
) -> AsyncMock:
    """Session scripted for the cycle's three queries, in order."""
    session = AsyncMock()

    cooldown_result = MagicMock()
    cooldown_result.scalar_one_or_none.return_value = last_nudge_at

    prior_ids_result = MagicMock()
    prior_ids_result.__iter__ = MagicMock(
        return_value=iter([(ids,) for ids in (nudged_reflection_ids or [])])
    )

    reflections_result = MagicMock()
    reflections_result.scalars.return_value.all.return_value = reflections or []

    session.execute = AsyncMock(
        side_effect=[cooldown_result, prior_ids_result, reflections_result]
    )
    session.add = MagicMock()
    return session


# ── nudge_cycle ───────────────────────────────────────────────────────────────


class TestNudgeCycle:
    @pytest.mark.asyncio
    async def test_composes_and_persists_nudge(self):
        r1 = make_reflection(severity="warning")
        r2 = make_reflection(severity="notice", kind="contradiction", insight="Conflict.")
        agent = LifeOSAgent()
        session = make_session(reflections=[r1, r2])

        result = await agent.nudge_cycle(session, user_id="u1")

        assert result.skipped is False
        assert result.insights_included == 2
        assert result.channel == "feed"
        assert result.delivered is True
        nudge = session.add.call_args.args[0]
        assert isinstance(nudge, Nudge)
        # warning outranks notice → digest leads with it, severity is the max
        assert nudge.severity == "warning"
        assert nudge.digest.startswith("Your memory noticed")
        assert "[Drift]" in nudge.digest and "[Contradiction]" in nudge.digest
        assert set(nudge.reflection_ids) == {str(r1.id), str(r2.id)}

    @pytest.mark.asyncio
    async def test_cooldown_skips(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=2)
        agent = LifeOSAgent(cooldown_hours=24)
        session = make_session(last_nudge_at=recent)

        result = await agent.nudge_cycle(session, user_id="u1")

        assert result.skipped is True
        assert "Cooldown" in result.skip_reason
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_cooldown_expired_proceeds(self):
        old = datetime.now(timezone.utc) - timedelta(hours=30)
        agent = LifeOSAgent(cooldown_hours=24)
        session = make_session(
            last_nudge_at=old, reflections=[make_reflection()]
        )

        result = await agent.nudge_cycle(session, user_id="u1")

        assert result.skipped is False

    @pytest.mark.asyncio
    async def test_below_severity_threshold_skips(self):
        agent = LifeOSAgent(min_severity="notice")
        session = make_session(reflections=[make_reflection(severity="info")])

        result = await agent.nudge_cycle(session, user_id="u1")

        assert result.skipped is True
        assert "No fresh insights" in result.skip_reason

    @pytest.mark.asyncio
    async def test_already_nudged_insight_not_repeated(self):
        r = make_reflection()
        agent = LifeOSAgent()
        session = make_session(
            nudged_reflection_ids=[[str(r.id)]], reflections=[r]
        )

        result = await agent.nudge_cycle(session, user_id="u1")

        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_no_reflections_skips(self):
        agent = LifeOSAgent()
        session = make_session(reflections=[])

        result = await agent.nudge_cycle(session, user_id="u1")

        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_webhook_delivery_success(self):
        agent = LifeOSAgent(webhook_url="https://hooks.example.com/nudge")
        session = make_session(reflections=[make_reflection()])

        with patch.object(
            agent, "_deliver_webhook", new=AsyncMock(return_value=True)
        ) as deliver:
            result = await agent.nudge_cycle(session, user_id="u1")

        deliver.assert_awaited_once()
        assert result.channel == "webhook"
        assert result.delivered is True
        nudge = session.add.call_args.args[0]
        assert nudge.status == "delivered"
        assert nudge.delivered_at is not None

    @pytest.mark.asyncio
    async def test_webhook_failure_keeps_nudge_in_feed(self):
        agent = LifeOSAgent(webhook_url="https://hooks.example.com/nudge")
        session = make_session(reflections=[make_reflection()])

        with patch.object(
            agent, "_deliver_webhook", new=AsyncMock(return_value=False)
        ):
            result = await agent.nudge_cycle(session, user_id="u1")

        assert result.delivered is False
        nudge = session.add.call_args.args[0]
        assert nudge.status == "failed"
        # the row persists regardless — the feed still shows it
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_insight_cap(self):
        reflections = [make_reflection() for _ in range(8)]
        agent = LifeOSAgent()
        session = make_session(reflections=reflections)

        result = await agent.nudge_cycle(session, user_id="u1")

        assert result.insights_included == 5   # MAX_INSIGHTS_PER_NUDGE

    @pytest.mark.asyncio
    async def test_invalid_min_severity_falls_back_to_notice(self):
        agent = LifeOSAgent(min_severity="catastrophic")
        assert agent.min_severity == "notice"


# ── webhook delivery ──────────────────────────────────────────────────────────


class TestWebhookDelivery:
    @pytest.mark.asyncio
    async def test_payload_and_success(self):
        agent = LifeOSAgent(webhook_url="https://hooks.example.com/nudge")
        nudge = Nudge(
            id=uuid.uuid4(), user_id="u1", app_id="default",
            digest="d", reflection_ids=["r1"], severity="notice",
        )

        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=MagicMock(raise_for_status=MagicMock()))

        with patch("httpx.AsyncClient", return_value=client):
            ok = await agent._deliver_webhook(nudge)

        assert ok is True
        url, = client.post.await_args.args
        payload = client.post.await_args.kwargs["json"]
        assert url == "https://hooks.example.com/nudge"
        assert payload["type"] == "smritikosh.nudge"
        assert payload["user_id"] == "u1"
        assert payload["reflection_ids"] == ["r1"]

    @pytest.mark.asyncio
    async def test_http_error_returns_false(self):
        agent = LifeOSAgent(webhook_url="https://hooks.example.com/nudge")
        nudge = Nudge(
            id=uuid.uuid4(), user_id="u1", app_id="default",
            digest="d", reflection_ids=[], severity="notice",
        )

        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(side_effect=RuntimeError("connect timeout"))

        with patch("httpx.AsyncClient", return_value=client):
            ok = await agent._deliver_webhook(nudge)

        assert ok is False


# ── feed / acknowledge ────────────────────────────────────────────────────────


class TestAcknowledge:
    @pytest.mark.asyncio
    async def test_acknowledge_scopes_to_user(self):
        agent = LifeOSAgent()
        row = Nudge(
            id=uuid.uuid4(), user_id="u1", app_id="default",
            digest="d", reflection_ids=[], severity="notice",
        )
        session = AsyncMock()
        session.get = AsyncMock(return_value=row)

        assert await agent.acknowledge(session, "u1", row.id) is True
        assert row.acknowledged is True
        row.acknowledged = False
        assert await agent.acknowledge(session, "intruder", row.id) is False
        assert row.acknowledged is False

    @pytest.mark.asyncio
    async def test_acknowledge_missing_returns_false(self):
        agent = LifeOSAgent()
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)

        assert await agent.acknowledge(session, "u1", uuid.uuid4()) is False


class TestNudgeResult:
    def test_defaults(self):
        r = NudgeResult(user_id="u1", app_id="default")
        assert r.insights_included == 0
        assert r.delivered is False
        assert r.skipped is False
