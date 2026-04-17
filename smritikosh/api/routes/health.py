import logging

from fastapi import APIRouter
from sqlalchemy import text

from smritikosh.api.schemas import HealthResponse
from smritikosh.config import settings
from smritikosh.db.neo4j import get_driver
from smritikosh.db.postgres import engine
from smritikosh.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)
router = APIRouter()

_CLOUD_PROVIDERS = {"claude", "openai", "gemini"}


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """
    Check server health including database connectivity.

    Returns ``status="ok"`` only when both required services are reachable.
    Returns ``status="degraded"`` when one or more services are unavailable
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

    # MongoDB (optional — not_configured if MONGODB_URL is unset)
    if not settings.mongodb_url:
        mongo_status = "not_configured"
    else:
        try:
            from smritikosh.audit.mongodb import get_audit_collection
            col = get_audit_collection()
            if col is None:
                mongo_status = "error"
            else:
                await col.database.client.admin.command("ping")
                mongo_status = "ok"
        except Exception as exc:
            logger.warning("MongoDB health check failed: %s", exc)
            mongo_status = "error"

    # LLM — verify API key is present for cloud providers; local providers assumed ok
    adapter = LLMAdapter()
    llm_model = adapter._chat_model
    provider = settings.llm_provider.lower()
    if provider in _CLOUD_PROVIDERS:
        llm_status = "ok" if settings.llm_api_key else "error"
    else:
        # Local providers (ollama, vllm) — assume reachable if base URL is set
        llm_status = "ok" if settings.llm_base_url else "ok"

    critical_ok = pg_status == "ok" and neo_status == "ok" and llm_status == "ok"
    overall = "ok" if critical_ok else "degraded"
    return HealthResponse(
        status=overall,
        postgres=pg_status,
        neo4j=neo_status,
        mongodb=mongo_status,
        llm_model=llm_model,
        llm_status=llm_status,
    )
