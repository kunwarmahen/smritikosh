"""Tests for per-tenant usage quotas (item D2).

Covers:
- get_effective_quota — override/default/unlimited resolution
- enforce_event_quota / enforce_token_quota — pass, 429, window selection
- window helpers — UTC day/month starts
- enforcement wiring — POST /memory/event returns 429 when over quota
- admin endpoints — GET/PUT/DELETE /admin/quotas/{user_id}
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from smritikosh.api.quotas import (
    EffectiveQuota,
    _day_start,
    _month_start,
    enforce_event_quota,
    enforce_token_quota,
    get_effective_quota,
    quota_usage_snapshot,
)
from smritikosh.db.models import UserQuota


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_pg(quota_row=None, count=0, token_sum=0):
    """Mock session: first execute returns the quota row, later ones counters."""
    pg = AsyncMock()
    row_result = MagicMock()
    row_result.scalar_one_or_none.return_value = quota_row
    count_result = MagicMock()
    count_result.scalar.return_value = count
    token_result = MagicMock()
    token_result.scalar.return_value = token_sum

    async def _execute(stmt, *a, **kw):
        text = str(stmt).lower()
        if "user_quotas" in text:
            return row_result
        if "llm_usage" in text:
            return token_result
        return count_result

    pg.execute = AsyncMock(side_effect=_execute)
    return pg


def make_quota_row(**overrides) -> UserQuota:
    return UserQuota(user_id="u1", app_id="default", **overrides)


# ── Effective quota resolution ────────────────────────────────────────────────


class TestEffectiveQuota:
    @pytest.mark.asyncio
    async def test_no_row_no_defaults_is_unlimited(self):
        quota = await get_effective_quota(make_pg(), "u1")
        assert quota == EffectiveQuota()
        assert not quota.has_event_limits
        assert not quota.has_token_limits

    @pytest.mark.asyncio
    async def test_row_overrides_default(self):
        pg = make_pg(quota_row=make_quota_row(daily_event_limit=100))
        quota = await get_effective_quota(pg, "u1")
        assert quota.daily_events == 100
        assert quota.monthly_events == 0  # no override, default 0 = unlimited

    @pytest.mark.asyncio
    async def test_null_override_falls_back_to_config_default(self, monkeypatch):
        from smritikosh.config import settings

        monkeypatch.setattr(settings, "quota_default_daily_events", 50)
        pg = make_pg(quota_row=make_quota_row(daily_event_limit=None))
        quota = await get_effective_quota(pg, "u1")
        assert quota.daily_events == 50

    @pytest.mark.asyncio
    async def test_config_default_applies_without_row(self, monkeypatch):
        from smritikosh.config import settings

        monkeypatch.setattr(settings, "quota_default_monthly_tokens", 1_000_000)
        quota = await get_effective_quota(make_pg(), "u1")
        assert quota.monthly_tokens == 1_000_000
        assert quota.has_token_limits

    @pytest.mark.asyncio
    async def test_non_quota_row_treated_as_no_override(self):
        # A session mocked for another query must not crash quota resolution.
        pg = make_pg(quota_row=MagicMock())
        quota = await get_effective_quota(pg, "u1")
        assert quota == EffectiveQuota()


# ── Window helpers ────────────────────────────────────────────────────────────


class TestWindows:
    def test_day_start(self):
        now = datetime(2026, 6, 11, 15, 30, 45, tzinfo=timezone.utc)
        assert _day_start(now) == datetime(2026, 6, 11, tzinfo=timezone.utc)

    def test_month_start(self):
        now = datetime(2026, 6, 11, 15, 30, 45, tzinfo=timezone.utc)
        assert _month_start(now) == datetime(2026, 6, 1, tzinfo=timezone.utc)


# ── Enforcement ───────────────────────────────────────────────────────────────


class TestEnforceEventQuota:
    @pytest.mark.asyncio
    async def test_unlimited_makes_no_count_queries(self):
        pg = make_pg()
        await enforce_event_quota(pg, "u1")
        # only the quota-row lookup ran
        assert pg.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_under_limit_passes(self):
        pg = make_pg(quota_row=make_quota_row(daily_event_limit=100), count=99)
        await enforce_event_quota(pg, "u1")

    @pytest.mark.asyncio
    async def test_at_limit_raises_429(self):
        pg = make_pg(quota_row=make_quota_row(daily_event_limit=100), count=100)
        with pytest.raises(HTTPException) as exc_info:
            await enforce_event_quota(pg, "u1")
        assert exc_info.value.status_code == 429
        assert "Quota exceeded" in exc_info.value.detail
        assert "day" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_monthly_limit_enforced(self):
        pg = make_pg(quota_row=make_quota_row(monthly_event_limit=1000), count=1000)
        with pytest.raises(HTTPException) as exc_info:
            await enforce_event_quota(pg, "u1")
        assert exc_info.value.status_code == 429
        assert "month" in exc_info.value.detail


class TestEnforceTokenQuota:
    @pytest.mark.asyncio
    async def test_under_budget_passes(self):
        pg = make_pg(quota_row=make_quota_row(daily_token_limit=10_000), token_sum=9_999)
        await enforce_token_quota(pg, "u1")

    @pytest.mark.asyncio
    async def test_over_budget_raises_429(self):
        pg = make_pg(quota_row=make_quota_row(daily_token_limit=10_000), token_sum=10_000)
        with pytest.raises(HTTPException) as exc_info:
            await enforce_token_quota(pg, "u1")
        assert exc_info.value.status_code == 429
        assert "LLM tokens" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_retry_after_header_set(self):
        pg = make_pg(quota_row=make_quota_row(daily_token_limit=1), token_sum=5)
        with pytest.raises(HTTPException) as exc_info:
            await enforce_token_quota(pg, "u1")
        assert "Retry-After" in exc_info.value.headers


# ── Usage snapshot ────────────────────────────────────────────────────────────


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_shape(self):
        pg = make_pg(
            quota_row=make_quota_row(daily_event_limit=100, monthly_token_limit=50_000),
            count=7,
            token_sum=1234,
        )
        snap = await quota_usage_snapshot(pg, "u1")
        assert snap["limits"]["daily_events"] == 100
        assert snap["limits"]["monthly_tokens"] == 50_000
        assert snap["limits"]["daily_tokens"] is None  # unlimited → null
        assert snap["used"]["daily_events"] == 7
        assert snap["used"]["daily_tokens"] == 1234


# ── Route enforcement wiring ──────────────────────────────────────────────────


_USER_PAYLOAD = {"sub": "u1", "role": "user", "app_ids": ["default"]}


@pytest.fixture
def quota_app():
    from smritikosh.api.main import app
    from smritikosh.auth.deps import get_current_user, require_write_scope
    from smritikosh.db.postgres import get_session

    pg = make_pg(quota_row=make_quota_row(daily_event_limit=10), count=10)
    app.dependency_overrides[get_session] = lambda: pg
    app.dependency_overrides[get_current_user] = lambda: _USER_PAYLOAD
    app.dependency_overrides[require_write_scope] = lambda: _USER_PAYLOAD
    yield app
    app.dependency_overrides.clear()


class TestRouteEnforcement:
    @pytest.mark.asyncio
    async def test_encode_returns_429_when_over_quota(self, quota_app):
        async with AsyncClient(
            transport=ASGITransport(app=quota_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/memory/event",
                json={"user_id": "u1", "content": "hello world"},
            )
        assert resp.status_code == 429
        assert "Quota exceeded" in resp.json()["detail"]


# ── Admin endpoints ───────────────────────────────────────────────────────────


_ADMIN_PAYLOAD = {"sub": "admin", "role": "admin", "app_ids": ["default"]}


@pytest.fixture
def admin_app():
    from smritikosh.api.main import app
    from smritikosh.auth.deps import get_current_user, require_admin
    from smritikosh.db.postgres import get_session

    pg = make_pg(count=3, token_sum=500)
    pg.add = MagicMock()
    pg.flush = AsyncMock()
    pg.delete = AsyncMock()
    app.dependency_overrides[get_session] = lambda: pg
    app.dependency_overrides[get_current_user] = lambda: _ADMIN_PAYLOAD
    app.dependency_overrides[require_admin] = lambda: _ADMIN_PAYLOAD
    yield app, pg
    app.dependency_overrides.clear()


class TestAdminQuotaRoutes:
    @pytest.mark.asyncio
    async def test_get_returns_snapshot(self, admin_app):
        app, _ = admin_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/admin/quotas/u1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "u1"
        assert body["used"]["daily_events"] == 3
        assert body["used"]["daily_tokens"] == 500

    @pytest.mark.asyncio
    async def test_put_creates_override(self, admin_app):
        app, pg = admin_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.put(
                "/admin/quotas/u1",
                json={"daily_event_limit": 200, "note": "trial tenant"},
            )
        assert resp.status_code == 200
        pg.add.assert_called_once()
        added: UserQuota = pg.add.call_args.args[0]
        assert added.user_id == "u1"
        assert added.daily_event_limit == 200
        assert added.note == "trial tenant"

    @pytest.mark.asyncio
    async def test_put_rejects_negative_limit(self, admin_app):
        app, _ = admin_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.put(
                "/admin/quotas/u1", json={"daily_event_limit": -5}
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_when_no_row(self, admin_app):
        app, _ = admin_app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/admin/quotas/u1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is False

    @pytest.mark.asyncio
    async def test_delete_removes_existing_row(self, admin_app):
        app, pg = admin_app
        row = make_quota_row(daily_event_limit=10)
        # Re-route the quota-row lookup to return an existing row
        row_result = MagicMock()
        row_result.scalar_one_or_none.return_value = row
        original_execute = pg.execute.side_effect

        async def _execute(stmt, *a, **kw):
            if "user_quotas" in str(stmt).lower():
                return row_result
            return await original_execute(stmt, *a, **kw)

        pg.execute.side_effect = _execute

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/admin/quotas/u1")
        assert resp.json()["deleted"] is True
        pg.delete.assert_awaited_once_with(row)
