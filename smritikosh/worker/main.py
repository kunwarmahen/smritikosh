"""Standalone background-job worker.

Runs the memory-maintenance scheduler — consolidation, pruning, clustering,
belief mining, fact decay, cross-system synthesis — in its own process. This
lets the API tier scale horizontally without every replica also running the
jobs (which would double LLM cost and race on the same rows).

A Postgres advisory lock elects a single leader: if more than one worker (or an
API process with RUN_SCHEDULER=true) is running, only the lock holder executes
jobs; the rest stand by and take over automatically if the leader dies.

Run:
    python -m smritikosh.worker.main

In Docker, run this as a separate service alongside the API (see
docker-compose.prod.yml).
"""

import asyncio
import logging
import signal

from smritikosh.audit.mongodb import close_audit, init_audit_indexes
from smritikosh.config import enforce_runtime_security, settings
from smritikosh.db.neo4j import close_neo4j, init_neo4j
from smritikosh.db.postgres import close_db, init_db
from smritikosh.processing.leader import LeaderLock
from smritikosh.processing.scheduler import build_scheduler, elect_and_start_scheduler

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger("sqlalchemy").setLevel(settings.sqlalchemy_log_level)
logging.getLogger("apscheduler.executors").setLevel(logging.WARNING)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    """Initialise infrastructure, win leader election, run jobs until signalled."""
    enforce_runtime_security(settings)
    if settings.worker_metrics_port:
        from prometheus_client import start_http_server

        start_http_server(settings.worker_metrics_port)
        logger.info(
            "Worker metrics exposed on :%d/metrics", settings.worker_metrics_port
        )
    logger.info("Smritikosh worker starting — initialising databases …")
    await init_db()
    await init_neo4j()
    await init_audit_indexes()   # no-op if MONGODB_URL is not set

    scheduler = build_scheduler()
    leader_lock = LeaderLock()

    # Stop on SIGTERM (container stop) and SIGINT (Ctrl-C).
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:  # add_signal_handler is unsupported on Windows
            signal.signal(sig, lambda *_: stop_event.set())

    election_task = asyncio.create_task(elect_and_start_scheduler(scheduler, leader_lock))
    logger.info("Smritikosh worker ready — waiting for shutdown signal.")

    await stop_event.wait()

    logger.info("Smritikosh worker shutting down …")
    if not election_task.done():
        election_task.cancel()
        try:
            await election_task
        except asyncio.CancelledError:
            pass
    scheduler.shutdown()                 # idempotent — no-op if never started
    await leader_lock.release()
    await close_db()
    await close_neo4j()
    await close_audit()
    logger.info("Smritikosh worker stopped.")


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
