import logging

from fastapi import APIRouter
from sqlalchemy import text

from smritikosh.api.schemas import HealthResponse
from smritikosh.db.neo4j import get_driver
from smritikosh.db.postgres import engine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """
    Check server health including database connectivity.

    Returns ``status="ok"`` only when both databases are reachable.
    Returns ``status="degraded"`` when one or both databases are unavailable
    but the server is still running.
    """
    pg_status = "ok"
    neo_status = "ok"

    # Ping PostgreSQL
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("PostgreSQL health check failed: %s", exc)
        pg_status = "error"

    # Ping Neo4j
    try:
        async with get_driver().session() as session:
            await session.run("RETURN 1")
    except Exception as exc:
        logger.warning("Neo4j health check failed: %s", exc)
        neo_status = "error"

    overall = "ok" if pg_status == "ok" and neo_status == "ok" else "degraded"
    return HealthResponse(status=overall, postgres=pg_status, neo4j=neo_status)
