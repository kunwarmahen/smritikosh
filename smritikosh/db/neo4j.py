"""
Neo4j connection — identity and semantic graph for SemanticMemory.

Graph schema:
    Nodes:
        (:User  {user_id, app_id, created_at})
        (:Fact  {category, key, value, confidence, updated_at})

    Relationships:
        (:User)-[:HAS_PREFERENCE]->(:Fact)   e.g. prefers green UI
        (:User)-[:HAS_INTEREST]->(:Fact)     e.g. interested in AI agents
        (:User)-[:HAS_ROLE]->(:Fact)         e.g. role: entrepreneur
        (:User)-[:WORKS_ON]->(:Fact)         e.g. project: smritikosh
        (:User)-[:HAS_SKILL]->(:Fact)        e.g. skill: RAG
        (:User)-[:HAS_GOAL]->(:Fact)         e.g. goal: ship MVP
        (:Fact)-[:RELATED_TO]->(:Fact)       cross-fact associations

Usage in FastAPI routes:
    @router.post("/memory/event")
    async def capture(neo: AsyncDriver = Depends(get_driver)):
        async with neo.session() as s:
            await s.run(...)

Usage in background jobs:
    async with neo4j_session() as session:
        await session.run(...)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from smritikosh.config import settings

# ── Driver (singleton) ────────────────────────────────────────────────────────

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    """Return the shared Neo4j async driver. Initialised lazily on first call."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )
    return _driver


# ── FastAPI dependency ─────────────────────────────────────────────────────────


async def get_neo4j_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a Neo4j session for use as a FastAPI dependency."""
    async with get_driver().session() as session:
        yield session


# ── Context manager for scripts / background jobs ─────────────────────────────


@asynccontextmanager
async def neo4j_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for Neo4j outside FastAPI.

    Example:
        async with neo4j_session() as session:
            await session.run("MERGE (:User {user_id: $uid})", uid="123")
    """
    async with get_driver().session() as session:
        yield session


# ── Schema initialisation ─────────────────────────────────────────────────────

# Cypher queries that create uniqueness constraints and indexes.
# Run once on first startup via init_neo4j().
_SCHEMA_QUERIES = [
    # Unique user node per (user_id, app_id) combination
    """
    CREATE CONSTRAINT user_unique IF NOT EXISTS
    FOR (u:User) REQUIRE (u.user_id, u.app_id) IS UNIQUE
    """,
    # Unique fact node per (category, key, value) — facts are shared across users
    # if they express the same concept; the relationship carries user context.
    """
    CREATE CONSTRAINT fact_unique IF NOT EXISTS
    FOR (f:Fact) REQUIRE (f.category, f.key, f.value) IS UNIQUE
    """,
    # Index for fast user lookups
    """
    CREATE INDEX user_id_idx IF NOT EXISTS
    FOR (u:User) ON (u.user_id)
    """,
    # Index for fact lookups by category
    """
    CREATE INDEX fact_category_idx IF NOT EXISTS
    FOR (f:Fact) ON (f.category)
    """,
]


async def init_neo4j() -> None:
    """
    Apply schema constraints and indexes to Neo4j.
    Idempotent — safe to call on every startup.
    """
    async with neo4j_session() as session:
        for query in _SCHEMA_QUERIES:
            await session.run(query)


async def close_neo4j() -> None:
    """Close the Neo4j driver. Call on app shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
