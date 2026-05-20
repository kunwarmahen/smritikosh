"""
Smritikosh FastAPI application.

Startup sequence (lifespan):
    1. Enable pgvector extension + create Postgres tables.
    2. Apply Neo4j schema constraints and indexes.
    3. Start background scheduler (consolidation + pruning jobs).

Shutdown sequence:
    1. Stop the scheduler.
    2. Close Postgres connection pool.
    3. Close Neo4j driver.

Run locally:
    uvicorn smritikosh.api.main:app --reload --port 8080

API docs:
    http://localhost:8080/docs
"""

import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from smritikosh.api.ratelimit import limiter, using_persistent_storage
from smritikosh.api.routes import admin, audit, auth, context, facts, feedback, graph, health, identity, ingest, keys, memory, procedures
from smritikosh.api.routes import session_ingest, media_ingest, voice_enrollment, connectors
from smritikosh.audit.mongodb import close_audit, init_audit_indexes
from smritikosh.db.neo4j import close_neo4j, init_neo4j
from smritikosh.db.postgres import close_db, init_db
from smritikosh.config import enforce_runtime_security, is_production, settings
from smritikosh.processing.leader import LeaderLock
from smritikosh.processing.scheduler import build_scheduler, elect_and_start_scheduler
from smritikosh.tasks import close_pool as close_task_pool

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # override any handlers uvicorn set before importing this module
)

logger = logging.getLogger(__name__)

logging.getLogger("sqlalchemy").setLevel(settings.sqlalchemy_log_level)
logging.getLogger("apscheduler.executors").setLevel(logging.WARNING)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


def _enforce_runtime_security() -> None:
    """Refuse to boot a production deployment with insecure secrets.

    Thin wrapper over config.enforce_runtime_security so the API process and the
    standalone worker share one implementation. Reads the module-level settings.
    """
    enforce_runtime_security(settings)


def _warn_runtime_topology() -> None:
    """Warn about configuration that is unsafe for a multi-replica deployment."""
    if not using_persistent_storage():
        msg = (
            "Rate limiter is using per-process in-memory storage. Limits are NOT "
            "enforced correctly across multiple API replicas — set REDIS_URL to a "
            "shared Redis instance before scaling out."
        )
        if is_production(settings):
            logger.warning(msg)
        else:
            logger.info("%s (fine for single-instance development.)", msg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all infrastructure on startup, cleanly close on shutdown."""
    _enforce_runtime_security()
    _warn_runtime_topology()
    logger.info("Smritikosh starting — initialising databases …")
    await init_db()
    await init_neo4j()
    await init_audit_indexes()   # no-op if MONGODB_URL is not set

    # Background scheduler.
    #   RUN_SCHEDULER=true  — this API process runs the maintenance jobs (after
    #                         winning leader election; safe even with >1 replica).
    #   RUN_SCHEDULER=false — a dedicated worker (smritikosh.worker.main) runs them.
    app.state.scheduler = None
    app.state.leader_lock = None
    app.state.election_task = None
    if settings.run_scheduler:
        scheduler = build_scheduler()
        leader_lock = LeaderLock()
        app.state.scheduler = scheduler
        app.state.leader_lock = leader_lock
        # Election may have to wait for another process — run it in the
        # background so startup is not blocked.
        app.state.election_task = asyncio.create_task(
            elect_and_start_scheduler(scheduler, leader_lock)
        )
    else:
        logger.info(
            "RUN_SCHEDULER is false — background jobs are not run by this API "
            "process (expecting a dedicated worker: python -m smritikosh.worker.main)."
        )

    # Store async_sessionmaker for background tasks (media processing, etc.)
    from smritikosh.db.postgres import get_async_sessionmaker
    app.state.async_sessionmaker = get_async_sessionmaker()

    logger.info("Smritikosh ready.")

    yield

    logger.info("Smritikosh shutting down …")
    election_task = app.state.election_task
    if election_task is not None and not election_task.done():
        election_task.cancel()
        try:
            await election_task
        except asyncio.CancelledError:
            pass
    if app.state.scheduler is not None:
        app.state.scheduler.shutdown()        # idempotent — no-op if never started
    if app.state.leader_lock is not None:
        await app.state.leader_lock.release()
    await close_task_pool()
    await close_db()
    await close_neo4j()
    await close_audit()


app = FastAPI(
    title="Smritikosh",
    description="Universal memory layer for LLM applications — a hippocampus for AI.",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiter — attach state and register the 429 handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus metrics — exposes GET /metrics with per-route latency, throughput, error rates.
# Disable by setting ENABLE_METRICS=false in your environment.
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,   # respects ENABLE_METRICS env var
    should_instrument_requests_inprogress=True,
    inprogress_name="smritikosh_requests_inprogress",
    inprogress_labels=True,
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True, tags=["system"])

app.include_router(auth.router)
app.include_router(health.router)
app.include_router(keys.router)
app.include_router(memory.router)
app.include_router(context.router)
app.include_router(identity.router)
app.include_router(feedback.router)
app.include_router(procedures.router)
app.include_router(admin.router)
app.include_router(ingest.router)
app.include_router(session_ingest.router)
app.include_router(media_ingest.router)
app.include_router(facts.router)
app.include_router(audit.router)
app.include_router(graph.router)
app.include_router(voice_enrollment.router)
app.include_router(connectors.router)
