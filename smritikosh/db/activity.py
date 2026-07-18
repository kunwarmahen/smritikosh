"""
user_activity maintenance (item A5) — indexed user discovery for background jobs.

touch_user_activity()  — upsert last_event_at; called from EpisodicMemory.store
                         so every event write, regardless of entry point,
                         keeps the tenant discoverable.
mark_job_done()        — stamp a per-job watermark after a job cycle completes
                         for one tenant; scheduler orders work by these and
                         consolidation skips tenants with nothing new.

Both are single-row upserts on a unique (user_id, app_id) key — O(1) per call,
versus the SELECT DISTINCT full scan of `events` they replace at discovery time.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import UserActivity

logger = logging.getLogger(__name__)

# Watermark columns settable via mark_job_done — guards against typos burning
# a silent no-op upsert.
JOB_WATERMARKS = {
    "consolidated": "last_consolidated_at",
    "pruned": "last_pruned_at",
    "clustered": "last_clustered_at",
    "belief_mined": "last_belief_mined_at",
    "synthesized": "last_synthesized_at",
    "reflected": "last_reflected_at",
    "nudged": "last_nudged_at",
}


async def touch_user_activity(
    session: AsyncSession, user_id: str, app_id: str = "default"
) -> None:
    """Record that this tenant just stored an event (upsert last_event_at)."""
    now = datetime.now(timezone.utc)
    stmt = pg_insert(UserActivity).values(
        user_id=user_id, app_id=app_id, last_event_at=now
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_user_activity",
        set_={"last_event_at": now},
    )
    await session.execute(stmt)


async def mark_job_done(
    session: AsyncSession, user_id: str, app_id: str, job: str
) -> None:
    """Stamp the job's watermark for one tenant (upsert; row may not exist yet
    for tenants created before the activity table was introduced)."""
    column = JOB_WATERMARKS.get(job)
    if column is None:
        raise ValueError(f"Unknown job watermark {job!r}; one of {sorted(JOB_WATERMARKS)}")
    now = datetime.now(timezone.utc)
    stmt = pg_insert(UserActivity).values(
        user_id=user_id, app_id=app_id, last_event_at=now, **{column: now}
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_user_activity",
        set_={column: now},
    )
    await session.execute(stmt)
