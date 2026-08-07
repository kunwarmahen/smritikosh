import logging

from fastapi import APIRouter
from sqlalchemy import text

from smritikosh import subsystems
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

    Status contract (item B3 — see ``smritikosh/subsystems.py``):

    ``ok``        every configured subsystem answered.
    ``degraded``  an OPTIONAL subsystem (Neo4j, Mongo) is unreachable, or the
                  LLM is misconfigured. The API still serves: Neo4j down costs
                  semantic facts, Mongo down costs the audit trail.
    ``error``     a REQUIRED subsystem (Postgres) is unreachable — this
                  instance cannot serve meaningfully; pull it from the pool.

    Each probe also refreshes the process's degraded registry, so recovery is
    picked up automatically without a restart.
    """
    pg_status = "ok"
    neo_status = "ok"

    # Ping PostgreSQL (required)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        subsystems.mark_healthy("postgres")
    except Exception as exc:
        logger.warning("PostgreSQL health check failed: %s", exc)
        pg_status = "error"
        subsystems.mark_degraded("postgres", f"unreachable: {exc}")

    # Ping Neo4j (optional — a failure degrades, it does not take the API down)
    try:
        async with get_driver().session() as session:
            await session.run("RETURN 1")
        neo_status = "ok"
        subsystems.mark_healthy("neo4j")
    except Exception as exc:
        logger.warning("Neo4j health check failed: %s", exc)
        neo_status = "error"
        subsystems.mark_degraded("neo4j", f"unreachable: {exc}")

    # MongoDB (optional — not_configured if MONGODB_URL is unset)
    if not settings.mongodb_url:
        mongo_status = "not_configured"
    else:
        try:
            from smritikosh.audit.mongodb import get_audit_collection
            col = get_audit_collection()
            if col is None:
                mongo_status = "error"
                subsystems.mark_degraded("mongodb", "audit collection unavailable")
            else:
                await col.database.client.admin.command("ping")
                mongo_status = "ok"
                subsystems.mark_healthy("mongodb")
        except Exception as exc:
            logger.warning("MongoDB health check failed: %s", exc)
            mongo_status = "error"
            subsystems.mark_degraded("mongodb", f"unreachable: {exc}")

    # LLM — verify API key is present for cloud providers; local providers assumed ok
    adapter = LLMAdapter()
    llm_model = adapter._chat_model
    provider = settings.llm_provider.lower()
    if provider in _CLOUD_PROVIDERS:
        llm_status = "ok" if settings.llm_api_key else "error"
    else:
        # Local providers (ollama, vllm) — assume reachable if base URL is set
        llm_status = "ok" if settings.llm_base_url else "ok"

    # Required subsystem down → error; optional down → degraded (B3).
    if pg_status != "ok":
        overall = "error"
    elif neo_status != "ok" or mongo_status == "error" or llm_status != "ok":
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        postgres=pg_status,
        neo4j=neo_status,
        mongodb=mongo_status,
        llm_model=llm_model,
        llm_status=llm_status,
        pg_pool=_pg_pool_status(),
        degraded_subsystems=subsystems.degraded_subsystems(),
    )


def _pg_pool_status() -> dict:
    """Live utilisation of this process's Postgres pool (item A4).

    checked_out near max means new requests will block for PG_POOL_TIMEOUT
    seconds and then error — the earliest warning that the connection budget
    (replicas × (pool_size + max_overflow)) is undersized.
    """
    try:
        pool = engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "max": settings.pg_pool_size + settings.pg_max_overflow,
        }
    except Exception as exc:  # pragma: no cover - depends on pool implementation
        logger.debug("Pool status unavailable: %s", exc)
        return {}
