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

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from smritikosh.api.deps import (
    get_belief_miner,
    get_clusterer,
    get_consolidator,
    get_episodic,
    get_fact_decayer,
    get_pruner,
    get_synthesizer,
)
from smritikosh.api.ratelimit import limiter
from smritikosh.api.routes import admin, audit, auth, context, facts, feedback, graph, health, identity, ingest, keys, memory, procedures
from smritikosh.api.routes import session_ingest, media_ingest, voice_enrollment, connectors
from smritikosh.audit.mongodb import close_audit, init_audit_indexes
from smritikosh.db.neo4j import close_neo4j, init_neo4j
from smritikosh.db.postgres import close_db, init_db
from smritikosh.config import settings
from smritikosh.processing.scheduler import MemoryScheduler

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all infrastructure on startup, cleanly close on shutdown."""
    logger.info("Smritikosh starting — initialising databases …")
    await init_db()
    await init_neo4j()
    await init_audit_indexes()   # no-op if MONGODB_URL is not set

    scheduler = MemoryScheduler(
        consolidator=get_consolidator(),
        pruner=get_pruner(),
        episodic=get_episodic(),
        clusterer=get_clusterer(),
        belief_miner=get_belief_miner(),
        fact_decayer=get_fact_decayer(),
        synthesizer=get_synthesizer(),
        consolidation_cron=settings.scheduler_consolidation_cron,
        pruning_cron=settings.scheduler_pruning_cron,
        clustering_cron=settings.scheduler_clustering_cron,
        belief_mining_cron=settings.scheduler_belief_mining_cron,
        fact_decay_cron=settings.scheduler_fact_decay_cron,
    )
    scheduler.start()
    app.state.scheduler = scheduler

    # Store async_sessionmaker for background tasks (media processing, etc.)
    from smritikosh.db.postgres import get_async_sessionmaker
    app.state.async_sessionmaker = get_async_sessionmaker()

    logger.info("Smritikosh ready.")

    yield

    logger.info("Smritikosh shutting down …")
    scheduler.shutdown()
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
