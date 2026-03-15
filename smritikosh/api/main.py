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
from contextlib import asynccontextmanager

from fastapi import FastAPI

from smritikosh.api.deps import (
    get_consolidator,
    get_episodic,
    get_pruner,
)
from smritikosh.api.routes import context, health, memory
from smritikosh.db.neo4j import close_neo4j, init_neo4j
from smritikosh.db.postgres import close_db, init_db
from smritikosh.processing.scheduler import MemoryScheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all infrastructure on startup, cleanly close on shutdown."""
    logger.info("Smritikosh starting — initialising databases …")
    await init_db()
    await init_neo4j()

    scheduler = MemoryScheduler(
        consolidator=get_consolidator(),
        pruner=get_pruner(),
        episodic=get_episodic(),
    )
    scheduler.start()
    logger.info("Smritikosh ready.")

    yield

    logger.info("Smritikosh shutting down …")
    scheduler.shutdown()
    await close_db()
    await close_neo4j()


app = FastAPI(
    title="Smritikosh",
    description="Universal memory layer for LLM applications — a hippocampus for AI.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(memory.router)
app.include_router(context.router)
