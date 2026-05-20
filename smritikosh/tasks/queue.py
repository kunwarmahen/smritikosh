"""ARQ connection + enqueue helpers for the durable task queue (item A3)."""

import logging

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from smritikosh.config import settings

logger = logging.getLogger(__name__)

# Cached enqueue-side pool. The ARQ worker process manages its own pool.
_pool: ArqRedis | None = None


def queue_enabled() -> bool:
    """True when REDIS_URL is configured — the durable task queue is usable.

    When False, callers fall back to in-process execution (BackgroundTask).
    """
    return bool(settings.redis_url)


def redis_settings() -> RedisSettings:
    """Build ARQ RedisSettings from REDIS_URL. Raises if Redis is not configured."""
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is not configured — the task queue is unavailable.")
    return RedisSettings.from_dsn(settings.redis_url)


async def get_pool() -> ArqRedis:
    """Return the cached enqueue-side ARQ pool, creating it on first use."""
    global _pool
    if _pool is None:
        _pool = await create_pool(redis_settings())
    return _pool


async def close_pool() -> None:
    """Close the enqueue-side pool. Called on API shutdown."""
    global _pool
    if _pool is not None:
        try:
            await _pool.aclose()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.debug("ARQ pool close failed: %s", exc)
        _pool = None


async def enqueue(func_name: str, *args, **kwargs):
    """Enqueue a task on the durable queue.

    Returns the ARQ Job on success, or None when the queue is unavailable
    (REDIS_URL unset) or enqueueing failed — the caller then runs the work
    in-process so nothing is silently dropped.
    """
    if not queue_enabled():
        return None
    try:
        pool = await get_pool()
        return await pool.enqueue_job(func_name, *args, **kwargs)
    except Exception as exc:
        logger.warning(
            "Task enqueue failed for %r: %s — caller will fall back in-process",
            func_name,
            exc,
        )
        return None
