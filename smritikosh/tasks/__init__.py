"""Durable background task queue (item A3).

Redis-backed (ARQ) queue for on-demand work that must survive an API/worker
restart — media processing and bulk re-embedding. Cron-style maintenance jobs
stay on APScheduler (see processing/scheduler.py); this queue is only for
one-off, retryable tasks.

The ARQ worker runs as a separate `taskworker` service:
    arq smritikosh.tasks.jobs.WorkerSettings

If REDIS_URL is unset the queue is disabled and callers fall back to running
the work in-process (FastAPI BackgroundTask) — so single-process development
keeps working with no Redis.
"""

from smritikosh.tasks.queue import close_pool, enqueue, queue_enabled

__all__ = ["close_pool", "enqueue", "queue_enabled"]
