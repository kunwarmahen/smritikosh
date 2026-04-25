"""
PostgreSQL connection — async SQLAlchemy engine and session factory.

Usage in FastAPI routes:
    @router.post("/memory/event")
    async def capture(session: AsyncSession = Depends(get_session)):
        ...

Usage in background jobs or scripts:
    async with db_session() as session:
        ...
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from smritikosh.config import settings

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.postgres_url,
    echo=False,
    pool_pre_ping=True,   # detect stale connections
    pool_size=10,
    max_overflow=20,
)

# ── Session factory ───────────────────────────────────────────────────────────

_SessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # keep objects usable after commit
)


# ── Session factory accessor ──────────────────────────────────────────────────


def get_async_sessionmaker() -> async_sessionmaker:
    return _SessionFactory


# ── FastAPI dependency ─────────────────────────────────────────────────────────


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a session, commit on success, rollback on error."""
    async with _SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            logger.warning("DB session rolled back: %s", exc)
            await session.rollback()
            raise


# ── Context manager for scripts / background jobs ─────────────────────────────


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for use outside FastAPI (jobs, CLI, tests).

    Example:
        async with db_session() as session:
            session.add(Event(...))
    """
    async with _SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            logger.warning("DB session rolled back: %s", exc)
            await session.rollback()
            raise


# ── Database lifecycle ────────────────────────────────────────────────────────


async def init_db() -> None:
    """
    Enable pgvector extension and create all tables.
    Only used for development / testing. Production uses Alembic migrations.
    """
    from sqlalchemy import text

    from smritikosh.db.models import Base

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose the connection pool. Call on app shutdown."""
    await engine.dispose()
