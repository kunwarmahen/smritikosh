"""
MongoDB connection for the audit trail.

Uses motor (async MongoDB driver). The connection is initialised lazily on
first use and closed on app shutdown.

If MONGODB_URL is not set in config, get_audit_collection() returns None and
the AuditLogger is never created — the system runs without audit persistence.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None     # motor.motor_asyncio.AsyncIOMotorClient
_db: Any = None         # AsyncIOMotorDatabase


def get_audit_collection() -> Any | None:
    """
    Return the motor collection for audit events, or None if MongoDB is not
    configured. Initialised lazily on first call.
    """
    global _client, _db

    from smritikosh.config import settings
    if not settings.mongodb_url:
        return None

    if _client is None:
        try:
            import motor.motor_asyncio as motor  # type: ignore[import]
            _client = motor.AsyncIOMotorClient(settings.mongodb_url)
            _db = _client[settings.mongodb_db_name]
            logger.info("MongoDB audit client initialised.")
        except ImportError:
            logger.warning(
                "motor package not installed — audit trail disabled. "
                "Install with: pip install motor"
            )
            return None
        except Exception as exc:
            logger.warning("MongoDB connection failed — audit trail disabled: %s", exc)
            return None

    return _db["audit_events"]


async def init_audit_indexes() -> None:
    """
    Create MongoDB indexes for common query patterns.
    Idempotent — safe to call on every startup.
    """
    col = get_audit_collection()
    if col is None:
        return
    try:
        from pymongo import ASCENDING, DESCENDING  # type: ignore[import]
        await col.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
        await col.create_index([("event_id", ASCENDING), ("timestamp", ASCENDING)])
        await col.create_index([("event_type", ASCENDING), ("timestamp", DESCENDING)])
        await col.create_index([("user_id", ASCENDING), ("app_id", ASCENDING), ("event_type", ASCENDING)])
        logger.info("MongoDB audit indexes ensured.")
    except Exception as exc:
        logger.warning("MongoDB index creation failed (non-fatal): %s", exc)


async def close_audit() -> None:
    """Close the MongoDB client. Call on app shutdown."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
