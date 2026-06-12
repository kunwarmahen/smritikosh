"""Tests for connection-pool governance (item A4).

Covers:
- Postgres engine is sized from the PG_POOL_* settings
- Neo4j driver is sized from NEO4J_MAX_POOL_SIZE
- GET /health reports live pool utilisation (pg_pool sub-check)
"""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from smritikosh.config import settings


class TestEngineSizing:
    def test_engine_pool_matches_settings(self):
        from smritikosh.db.postgres import engine

        assert engine.pool.size() == settings.pg_pool_size
        # max_overflow is not exposed as a method; check the configured attr
        assert engine.pool._max_overflow == settings.pg_max_overflow

    def test_neo4j_driver_sized_from_settings(self, monkeypatch):
        import smritikosh.db.neo4j as neo_mod

        monkeypatch.setattr(neo_mod, "_driver", None)  # force re-creation
        with patch.object(
            neo_mod.AsyncGraphDatabase, "driver", return_value=MagicMock()
        ) as mock_driver:
            neo_mod.get_driver()

        kwargs = mock_driver.call_args.kwargs
        assert kwargs["max_connection_pool_size"] == settings.neo4j_max_pool_size
        monkeypatch.setattr(neo_mod, "_driver", None)  # don't leak the mock


class TestHealthPoolStatus:
    def test_pool_status_shape(self):
        from smritikosh.api.routes.health import _pg_pool_status

        status = _pg_pool_status()
        assert set(status) == {"size", "checked_in", "checked_out", "overflow", "max"}
        assert status["max"] == settings.pg_pool_size + settings.pg_max_overflow

    @pytest.mark.asyncio
    async def test_health_response_includes_pool(self):
        from smritikosh.api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        pool = resp.json()["pg_pool"]
        assert pool["max"] == settings.pg_pool_size + settings.pg_max_overflow
        assert "checked_out" in pool
