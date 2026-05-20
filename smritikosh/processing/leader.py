"""Postgres advisory-lock leader election for the background scheduler.

Only one process should run the memory-maintenance jobs. The dedicated worker
(`python -m smritikosh.worker.main`) is the intended job runner, but an API
process may also have RUN_SCHEDULER=true. To make that safe, every process that
wants to run the scheduler first acquires a Postgres *session-level advisory
lock*. `pg_try_advisory_lock` is non-blocking — the first caller wins, the rest
get False and stand by.

The lock lives on a dedicated connection held for the process lifetime. If the
leader process dies, its connection drops and Postgres frees the lock
automatically, so a standby acquires it on its next retry — automatic failover.

Known limitation: if the dedicated connection is severed *without* the process
dying (e.g. a network blip), this process keeps `is_leader == True` until it
calls `try_acquire()` again. The primary safeguard remains "run exactly one
worker"; the advisory lock is defence-in-depth against accidental double-runs.
"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from smritikosh.db.postgres import engine

logger = logging.getLogger(__name__)

# Fixed 64-bit key identifying the scheduler advisory lock. Arbitrary, but it
# must stay stable across releases or old and new processes would not contend
# for the same lock.
SCHEDULER_LOCK_KEY = 0x536D72744D656D31  # "SmrtMem1" as bytes → bigint


class LeaderLock:
    """Holds a Postgres session-level advisory lock to elect a single leader."""

    def __init__(self, lock_key: int = SCHEDULER_LOCK_KEY) -> None:
        self._lock_key = lock_key
        self._conn: AsyncConnection | None = None
        self._is_leader = False

    @property
    def is_leader(self) -> bool:
        """True if this process currently holds the scheduler lock."""
        return self._is_leader

    async def try_acquire(self) -> bool:
        """Attempt to become leader. Returns True if this process holds the lock.

        Idempotent: if already leader, returns True without re-locking.
        """
        if self._is_leader:
            return True
        conn: AsyncConnection | None = None
        try:
            conn = await engine.connect()
            result = await conn.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": self._lock_key}
            )
            acquired = bool(result.scalar())
            # Commit so the connection does not sit in an open transaction —
            # a session-level advisory lock survives commits on the same conn.
            await conn.commit()
            if acquired:
                self._conn = conn  # keep the connection alive — the lock lives on it
                self._is_leader = True
                logger.info("Acquired scheduler leader lock — this process runs the jobs.")
            else:
                await conn.close()
            return acquired
        except Exception as exc:
            logger.warning("Leader election attempt failed: %s", exc)
            if conn is not None:
                try:
                    await conn.close()
                except Exception:
                    pass
            return False

    async def release(self) -> None:
        """Release the advisory lock and close the dedicated connection."""
        if self._conn is not None:
            try:
                await self._conn.execute(
                    text("SELECT pg_advisory_unlock(:k)"), {"k": self._lock_key}
                )
                await self._conn.commit()
            except Exception as exc:
                logger.debug("Advisory unlock failed (connection may be gone): %s", exc)
            try:
                await self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._is_leader = False
