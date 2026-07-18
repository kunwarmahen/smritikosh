"""
LifeOSAgent — proactive nudges from reflection insights (E4, FUTURE.md #5).

The ReflectionAgent (#9) detects drift, contradictions, and stale beliefs —
but its insights sit in the `reflections` table until someone opens the
dashboard. Life OS is the delivery layer that makes them proactive:

    reflection cycle (05:00) → insights persist → nudge cycle (07:00) →
    fresh, unacknowledged insights at/above LIFEOS_MIN_SEVERITY are composed
    into ONE digest per user and delivered.

Channels:
    feed     — always: the nudge persists in the `nudges` table, surfaced by
               GET /cognition/nudges/{user_id} (in-app feed).
    webhook  — when LIFEOS_WEBHOOK_URL is set: the digest is POSTed as JSON
               (payload carries user_id/app_id so the receiving app can
               route). Webhook failure downgrades to the feed, never loses
               the nudge.

Design decisions:
    - Deliberately LLM-free. Reflection insights are already user-addressed
      sentences ("You stated a goal of…"); composition is a template, so a
      nudge cycle across every tenant costs zero tokens.
    - One digest per user per cycle, with a cooldown (LIFEOS_COOLDOWN_HOURS)
      and a per-insight guarantee: an insight is included in at most one
      nudge (tracked via nudged reflection ids), so acknowledging nothing
      still never produces repeat nags for the same finding.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh import metrics
from smritikosh.db.models import Nudge, Reflection

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"info": 0, "notice": 1, "warning": 2}
MAX_INSIGHTS_PER_NUDGE = 5
WEBHOOK_TIMEOUT_SECONDS = 10.0

_KIND_LABEL = {
    "drift": "Drift",
    "contradiction": "Contradiction",
    "stale_belief": "Stale belief",
    "observation": "Observation",
}


@dataclass
class NudgeResult:
    user_id: str
    app_id: str
    nudge_id: str | None = None
    insights_included: int = 0
    channel: str = ""          # feed | webhook
    delivered: bool = False
    skipped: bool = False
    skip_reason: str = ""


class LifeOSAgent:
    """
    Composes and delivers proactive nudges from unacknowledged reflections.

    Usage:
        agent = LifeOSAgent(webhook_url=settings.lifeos_webhook_url)

        async with db_session() as pg:
            result = await agent.nudge_cycle(pg, user_id="u1")
    """

    def __init__(
        self,
        *,
        webhook_url: str | None = None,
        min_severity: str = "notice",
        cooldown_hours: int = 24,
        audit=None,   # AuditLogger | None
    ) -> None:
        self.webhook_url = webhook_url or None
        self.min_severity = min_severity if min_severity in _SEVERITY_RANK else "notice"
        self.cooldown_hours = cooldown_hours
        self.audit = audit

    async def nudge_cycle(
        self,
        pg_session: AsyncSession,
        *,
        user_id: str,
        app_id: str = "default",
    ) -> NudgeResult:
        """Compose + deliver at most one nudge for one user."""
        result = NudgeResult(user_id=user_id, app_id=app_id)
        now = datetime.now(timezone.utc)

        # ── 1. Cooldown: at most one nudge per user per window ─────────────
        last_nudge = (await pg_session.execute(
            select(Nudge.created_at)
            .where(Nudge.user_id == user_id, Nudge.app_id == app_id)
            .order_by(Nudge.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if last_nudge is not None:
            if last_nudge.tzinfo is None:
                last_nudge = last_nudge.replace(tzinfo=timezone.utc)
            if now - last_nudge < timedelta(hours=self.cooldown_hours):
                result.skipped = True
                result.skip_reason = (
                    f"Cooldown: last nudge at {last_nudge.isoformat()} is within "
                    f"{self.cooldown_hours}h."
                )
                return result

        # ── 2. Fresh insights: unacknowledged, severe enough, never nudged ─
        already_nudged: set[str] = set()
        for (ids,) in await pg_session.execute(
            select(Nudge.reflection_ids)
            .where(Nudge.user_id == user_id, Nudge.app_id == app_id)
        ):
            already_nudged.update(str(i) for i in (ids or []))

        rows = (await pg_session.execute(
            select(Reflection)
            .where(
                Reflection.user_id == user_id,
                Reflection.app_id == app_id,
                Reflection.acknowledged.is_(False),
            )
            .order_by(Reflection.created_at.desc())
            .limit(50)
        )).scalars().all()
        threshold = _SEVERITY_RANK[self.min_severity]
        fresh = [
            r for r in rows
            if str(r.id) not in already_nudged
            and _SEVERITY_RANK.get(r.severity, 0) >= threshold
        ]
        if not fresh:
            result.skipped = True
            result.skip_reason = "No fresh insights at or above the severity threshold."
            return result

        fresh.sort(key=lambda r: -_SEVERITY_RANK.get(r.severity, 0))
        fresh = fresh[:MAX_INSIGHTS_PER_NUDGE]
        result.insights_included = len(fresh)

        # ── 3. Compose the digest (template — no LLM) ──────────────────────
        digest = self._compose_digest(fresh)
        severity = fresh[0].severity   # max, thanks to the sort

        # ── 4. Persist, then deliver ───────────────────────────────────────
        nudge = Nudge(
            user_id=user_id,
            app_id=app_id,
            digest=digest,
            reflection_ids=[str(r.id) for r in fresh],
            severity=severity,
            channel="feed",
            status="delivered",   # the feed IS delivery; webhook may upgrade
        )
        pg_session.add(nudge)
        await pg_session.flush()
        result.nudge_id = str(nudge.id)

        if self.webhook_url:
            delivered = await self._deliver_webhook(nudge)
            nudge.channel = "webhook"
            nudge.status = "delivered" if delivered else "failed"
            if delivered:
                nudge.delivered_at = datetime.now(timezone.utc)
        else:
            nudge.delivered_at = now
        result.channel = nudge.channel
        result.delivered = nudge.status == "delivered"

        if self.audit:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.AGENT_NUDGE,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "nudge_id": result.nudge_id,
                    "severity": severity,
                    "channel": nudge.channel,
                    "status": nudge.status,
                    "insights_included": result.insights_included,
                    "reflection_ids": nudge.reflection_ids,
                },
            ))

        metrics.AGENT_RUNS.labels(agent="lifeos", outcome="success").inc()
        logger.info(
            "Nudge composed",
            extra={
                "user_id": user_id,
                "insights": result.insights_included,
                "channel": nudge.channel,
                "status": nudge.status,
            },
        )
        return result

    def _compose_digest(self, reflections: list[Reflection]) -> str:
        lines = ["Your memory noticed a few things worth your attention:"]
        for r in reflections:
            label = _KIND_LABEL.get(r.kind, "Observation")
            lines.append(f"• [{label}] {r.insight}")
        return "\n".join(lines)

    async def _deliver_webhook(self, nudge: Nudge) -> bool:
        """POST the digest; False on any failure (the feed still has it)."""
        import httpx

        payload = {
            "type": "smritikosh.nudge",
            "nudge_id": str(nudge.id),
            "user_id": nudge.user_id,
            "app_id": nudge.app_id,
            "severity": nudge.severity,
            "digest": nudge.digest,
            "reflection_ids": nudge.reflection_ids,
        }
        try:
            async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
                resp = await client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning(
                "Nudge webhook delivery failed: %s", exc,
                extra={"user_id": nudge.user_id, "nudge_id": str(nudge.id)},
            )
            return False

    # ── Read / acknowledge (the in-app feed) ───────────────────────────────

    async def list_nudges(
        self,
        pg_session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        include_acknowledged: bool = False,
        limit: int = 50,
    ) -> list[Nudge]:
        q = (
            select(Nudge)
            .where(Nudge.user_id == user_id, Nudge.app_id == app_id)
            .order_by(Nudge.created_at.desc())
            .limit(limit)
        )
        if not include_acknowledged:
            q = q.where(Nudge.acknowledged.is_(False))
        result = await pg_session.execute(q)
        return list(result.scalars().all())

    async def acknowledge(
        self,
        pg_session: AsyncSession,
        user_id: str,
        nudge_id,
    ) -> bool:
        """Dismiss a nudge from the feed (scoped to its owner)."""
        row = await pg_session.get(Nudge, nudge_id)
        if row is None or row.user_id != user_id:
            return False
        row.acknowledged = True
        return True
