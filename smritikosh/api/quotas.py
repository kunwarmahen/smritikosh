"""
Per-tenant usage quota enforcement (item D2).

Rate limiting (api/ratelimit.py) throttles burst; quotas cap volume — a
tenant inside the rate limit can still drive unbounded LLM spend without
them. Two dimensions, two UTC-calendar windows each:

    events  — rows stored in `events`           (checked on write paths)
    tokens  — prompt+completion from `llm_usage` (checked on every LLM path)

Resolution order per (user_id, app_id):
    user_quotas row value (NULL column = no override)
      → QUOTA_DEFAULT_* config value
        → 0 = unlimited

Token enforcement is post-hoc by design: a request that *crosses* the budget
completes, and subsequent requests are rejected until the window rolls over.
Exact pre-metering would require knowing token counts before the call.

Enforcement raises HTTP 429 with a quota-specific detail, mirroring the rate
limiter's status code so client retry logic can treat both uniformly.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.config import settings
from smritikosh.db.models import Event, LlmUsage, UserQuota

logger = logging.getLogger(__name__)


# ── Effective quota resolution ────────────────────────────────────────────────


@dataclass(frozen=True)
class EffectiveQuota:
    """Resolved limits for one (user_id, app_id). 0 = unlimited."""

    daily_events: int = 0
    monthly_events: int = 0
    daily_tokens: int = 0
    monthly_tokens: int = 0

    @property
    def has_event_limits(self) -> bool:
        return bool(self.daily_events or self.monthly_events)

    @property
    def has_token_limits(self) -> bool:
        return bool(self.daily_tokens or self.monthly_tokens)


def _resolve(override, default: int) -> int:
    """Row value wins over the config default; anything non-int (None, or a
    mock in unit tests) falls back to the default."""
    if isinstance(override, int) and not isinstance(override, bool):
        return override
    return default


async def get_effective_quota(
    pg: AsyncSession, user_id: str, app_id: str = "default"
) -> EffectiveQuota:
    """Merge the tenant's user_quotas row (if any) over the config defaults."""
    result = await pg.execute(
        select(UserQuota).where(
            UserQuota.user_id == user_id, UserQuota.app_id == app_id
        )
    )
    row = result.scalar_one_or_none()
    # Anything that isn't a real UserQuota row (None in prod; arbitrary mocks
    # in unit tests that stub the whole session) means "no override".
    if not isinstance(row, UserQuota):
        return EffectiveQuota(
            daily_events=settings.quota_default_daily_events,
            monthly_events=settings.quota_default_monthly_events,
            daily_tokens=settings.quota_default_daily_tokens,
            monthly_tokens=settings.quota_default_monthly_tokens,
        )
    return EffectiveQuota(
        daily_events=_resolve(row.daily_event_limit, settings.quota_default_daily_events),
        monthly_events=_resolve(row.monthly_event_limit, settings.quota_default_monthly_events),
        daily_tokens=_resolve(row.daily_token_limit, settings.quota_default_daily_tokens),
        monthly_tokens=_resolve(row.monthly_token_limit, settings.quota_default_monthly_tokens),
    )


# ── Window starts ─────────────────────────────────────────────────────────────


def _day_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# ── Usage counters ────────────────────────────────────────────────────────────


async def _count_events_since(
    pg: AsyncSession, user_id: str, app_id: str, since: datetime
) -> int:
    result = await pg.execute(
        select(func.count())
        .select_from(Event)
        .where(
            Event.user_id == user_id,
            Event.app_id == app_id,
            Event.created_at >= since,
        )
    )
    return int(result.scalar() or 0)


async def _sum_tokens_since(
    pg: AsyncSession, user_id: str, app_id: str, since: datetime
) -> int:
    result = await pg.execute(
        select(
            func.coalesce(
                func.sum(LlmUsage.prompt_tokens + LlmUsage.completion_tokens), 0
            )
        ).where(
            LlmUsage.user_id == user_id,
            LlmUsage.app_id == app_id,
            LlmUsage.created_at >= since,
        )
    )
    return int(result.scalar() or 0)


# ── Enforcement ───────────────────────────────────────────────────────────────


def _quota_exceeded(dimension: str, window: str, used: int, limit: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=(
            f"Quota exceeded: {used} of {limit} {dimension} used this {window}. "
            f"The window resets at UTC {'midnight' if window == 'day' else 'month start'}; "
            "contact your administrator to raise the limit."
        ),
        headers={"Retry-After": "3600"},
    )


async def enforce_event_quota(
    pg: AsyncSession, user_id: str, app_id: str = "default"
) -> None:
    """Raise 429 if storing one more event would exceed the tenant's caps.

    Call before encode/ingest writes. No-op (zero queries beyond the quota
    lookup) when the tenant has no event limits configured.
    """
    quota = await get_effective_quota(pg, user_id, app_id)
    if not quota.has_event_limits:
        return
    if quota.daily_events:
        used = await _count_events_since(pg, user_id, app_id, _day_start())
        if used >= quota.daily_events:
            raise _quota_exceeded("events", "day", used, quota.daily_events)
    if quota.monthly_events:
        used = await _count_events_since(pg, user_id, app_id, _month_start())
        if used >= quota.monthly_events:
            raise _quota_exceeded("events", "month", used, quota.monthly_events)


async def enforce_token_quota(
    pg: AsyncSession, user_id: str, app_id: str = "default"
) -> None:
    """Raise 429 if the tenant has already spent its LLM token budget.

    Call before any LLM-consuming entry point (encode, ingest, context).
    Post-hoc: the request that crosses the budget completes; later ones 429.
    """
    quota = await get_effective_quota(pg, user_id, app_id)
    if not quota.has_token_limits:
        return
    if quota.daily_tokens:
        used = await _sum_tokens_since(pg, user_id, app_id, _day_start())
        if used >= quota.daily_tokens:
            raise _quota_exceeded("LLM tokens", "day", used, quota.daily_tokens)
    if quota.monthly_tokens:
        used = await _sum_tokens_since(pg, user_id, app_id, _month_start())
        if used >= quota.monthly_tokens:
            raise _quota_exceeded("LLM tokens", "month", used, quota.monthly_tokens)


async def quota_usage_snapshot(
    pg: AsyncSession, user_id: str, app_id: str = "default"
) -> dict:
    """Current consumption vs effective limits — for the admin quota endpoint."""
    quota = await get_effective_quota(pg, user_id, app_id)
    day, month = _day_start(), _month_start()
    return {
        "user_id": user_id,
        "app_id": app_id,
        "limits": {
            "daily_events": quota.daily_events or None,
            "monthly_events": quota.monthly_events or None,
            "daily_tokens": quota.daily_tokens or None,
            "monthly_tokens": quota.monthly_tokens or None,
        },
        "used": {
            "daily_events": await _count_events_since(pg, user_id, app_id, day),
            "monthly_events": await _count_events_since(pg, user_id, app_id, month),
            "daily_tokens": await _sum_tokens_since(pg, user_id, app_id, day),
            "monthly_tokens": await _sum_tokens_since(pg, user_id, app_id, month),
        },
    }
