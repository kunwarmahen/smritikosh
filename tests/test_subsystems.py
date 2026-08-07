"""
Tests for graceful degradation of optional subsystems (item B3):
    subsystems registry — mark/clear, reason tracking, Prometheus gauge
    lifespan            — Neo4j failure is non-fatal, Postgres failure is not
    /health             — status contract (ok | degraded | error) + reporting

The core guarantee: an unreachable Neo4j must not stop the API from booting,
because everything Postgres backs still works.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh import subsystems


@pytest.fixture(autouse=True)
def clean_registry():
    subsystems.reset()
    yield
    subsystems.reset()


# ── registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_starts_empty(self):
        assert subsystems.degraded_subsystems() == {}

    def test_mark_degraded_records_reason(self):
        subsystems.mark_degraded("neo4j", "connection refused")

        assert subsystems.is_degraded("neo4j") is True
        assert subsystems.degraded_subsystems() == {"neo4j": "connection refused"}

    def test_mark_healthy_clears(self):
        subsystems.mark_degraded("neo4j", "down")
        subsystems.mark_healthy("neo4j")

        assert subsystems.is_degraded("neo4j") is False
        assert subsystems.degraded_subsystems() == {}

    def test_mark_healthy_is_safe_when_never_degraded(self):
        subsystems.mark_healthy("neo4j")
        assert subsystems.degraded_subsystems() == {}

    def test_latest_reason_wins(self):
        subsystems.mark_degraded("neo4j", "first")
        subsystems.mark_degraded("neo4j", "second")

        assert subsystems.degraded_subsystems()["neo4j"] == "second"

    def test_tracks_subsystems_independently(self):
        subsystems.mark_degraded("neo4j", "a")
        subsystems.mark_degraded("mongodb", "b")
        subsystems.mark_healthy("neo4j")

        assert subsystems.degraded_subsystems() == {"mongodb": "b"}

    def test_returned_mapping_is_a_copy(self):
        subsystems.mark_degraded("neo4j", "down")
        snapshot = subsystems.degraded_subsystems()
        snapshot["neo4j"] = "mutated"

        assert subsystems.degraded_subsystems()["neo4j"] == "down"

    def test_gauge_follows_state(self):
        subsystems.mark_degraded("neo4j", "down")
        gauge = subsystems.SUBSYSTEM_DEGRADED.labels(subsystem="neo4j")
        assert gauge._value.get() == 1

        subsystems.mark_healthy("neo4j")
        assert gauge._value.get() == 0

    def test_postgres_is_the_only_required_subsystem(self):
        # The whole point of B3: only Postgres may abort startup.
        assert subsystems.REQUIRED_SUBSYSTEMS == ("postgres",)
        assert "neo4j" in subsystems.OPTIONAL_SUBSYSTEMS
        assert "mongodb" in subsystems.OPTIONAL_SUBSYSTEMS


# ── startup behaviour ─────────────────────────────────────────────────────────


class TestStartupDegradation:
    @pytest.mark.asyncio
    async def test_neo4j_failure_does_not_abort_startup(self):
        """The B3 headline: Neo4j down must not stop the API from booting."""
        from smritikosh.api.main import lifespan

        app = MagicMock()
        app.state = MagicMock()

        with (
            patch("smritikosh.api.main._enforce_runtime_security"),
            patch("smritikosh.api.main._warn_runtime_topology"),
            patch("smritikosh.api.main.init_db", new=AsyncMock()),
            patch(
                "smritikosh.api.main.init_neo4j",
                new=AsyncMock(side_effect=OSError("connection refused")),
            ),
            patch("smritikosh.api.main.init_audit_indexes", new=AsyncMock()),
            patch("smritikosh.api.main.settings") as mock_settings,
            patch("smritikosh.db.postgres.get_async_sessionmaker"),
            patch("smritikosh.api.main.close_db", new=AsyncMock()),
            patch("smritikosh.api.main.close_neo4j", new=AsyncMock()),
            patch("smritikosh.api.main.close_audit", new=AsyncMock()),
            patch("smritikosh.api.main.close_task_pool", new=AsyncMock()),
        ):
            mock_settings.run_scheduler = False

            async with lifespan(app):
                # Startup completed despite Neo4j being unreachable.
                assert subsystems.is_degraded("neo4j") is True
                assert "connection refused" in subsystems.degraded_subsystems()["neo4j"]
                assert subsystems.is_degraded("postgres") is False

    @pytest.mark.asyncio
    async def test_postgres_failure_still_aborts_startup(self):
        """Postgres is required — a failure there must remain fatal."""
        from smritikosh.api.main import lifespan

        app = MagicMock()
        app.state = MagicMock()

        with (
            patch("smritikosh.api.main._enforce_runtime_security"),
            patch("smritikosh.api.main._warn_runtime_topology"),
            patch(
                "smritikosh.api.main.init_db",
                new=AsyncMock(side_effect=OSError("pg down")),
            ),
            patch("smritikosh.api.main.init_neo4j", new=AsyncMock()),
            patch("smritikosh.api.main.init_audit_indexes", new=AsyncMock()),
        ):
            with pytest.raises(OSError, match="pg down"):
                async with lifespan(app):
                    pass

    @pytest.mark.asyncio
    async def test_healthy_neo4j_leaves_registry_clean(self):
        from smritikosh.api.main import lifespan

        app = MagicMock()
        app.state = MagicMock()

        with (
            patch("smritikosh.api.main._enforce_runtime_security"),
            patch("smritikosh.api.main._warn_runtime_topology"),
            patch("smritikosh.api.main.init_db", new=AsyncMock()),
            patch("smritikosh.api.main.init_neo4j", new=AsyncMock()),
            patch("smritikosh.api.main.init_audit_indexes", new=AsyncMock()),
            patch("smritikosh.api.main.settings") as mock_settings,
            patch("smritikosh.db.postgres.get_async_sessionmaker"),
            patch("smritikosh.api.main.close_db", new=AsyncMock()),
            patch("smritikosh.api.main.close_neo4j", new=AsyncMock()),
            patch("smritikosh.api.main.close_audit", new=AsyncMock()),
            patch("smritikosh.api.main.close_task_pool", new=AsyncMock()),
        ):
            mock_settings.run_scheduler = False

            async with lifespan(app):
                assert subsystems.degraded_subsystems() == {}


# ── encode path degradation ───────────────────────────────────────────────────


def _hippocampus_with_broken_neo4j(fail_profile: bool, fail_upsert: bool):
    """Hippocampus whose semantic store raises like an unreachable Neo4j."""
    from smritikosh.memory.episodic import EpisodicMemory
    from smritikosh.memory.hippocampus import Hippocampus
    from smritikosh.memory.semantic import SemanticMemory, UserProfile
    from smritikosh.processing.amygdala import Amygdala

    llm = AsyncMock()
    llm.embed = AsyncMock(return_value=[0.1] * 8)
    llm.extract_structured = AsyncMock(
        return_value={"facts": [
            {"category": "tool", "key": "language", "value": "rust", "confidence": 0.9},
        ]}
    )

    episodic = AsyncMock(spec=EpisodicMemory)
    stored_event = MagicMock()
    stored_event.id = "evt-1"
    episodic.store = AsyncMock(return_value=stored_event)

    semantic = AsyncMock(spec=SemanticMemory)
    down = OSError("Couldn't connect to localhost:7687")
    semantic.get_user_profile = AsyncMock(
        side_effect=down if fail_profile
        else None,
        return_value=None if fail_profile else UserProfile(
            user_id="u1", app_id="default", facts=[]
        ),
    )
    semantic.check_fact_conflict = AsyncMock(
        side_effect=down if fail_upsert else None, return_value=None
    )
    semantic.upsert_fact = AsyncMock(side_effect=down if fail_upsert else None)

    amygdala = MagicMock(spec=Amygdala)
    amygdala.score = MagicMock(return_value=0.6)

    return Hippocampus(llm=llm, episodic=episodic, semantic=semantic, amygdala=amygdala), episodic


class TestEncodeDegradation:
    @pytest.mark.asyncio
    async def test_encode_survives_unreachable_profile_fetch(self):
        """The profile is only a naming hint — losing it must not fail encode."""
        hippo, episodic = _hippocampus_with_broken_neo4j(
            fail_profile=True, fail_upsert=True
        )

        result = await hippo.encode(
            AsyncMock(), AsyncMock(), user_id="u1", raw_text="I now write Rust.",
        )

        # The episodic event is what matters — it was still stored.
        episodic.store.assert_awaited_once()
        assert result.event is not None
        assert result.facts == []
        assert subsystems.is_degraded("neo4j") is True

    @pytest.mark.asyncio
    async def test_fact_upsert_failure_keeps_the_event(self):
        """Neo4j dying mid-encode drops facts, never the event."""
        hippo, episodic = _hippocampus_with_broken_neo4j(
            fail_profile=False, fail_upsert=True
        )

        result = await hippo.encode(
            AsyncMock(), AsyncMock(), user_id="u1", raw_text="I now write Rust.",
        )

        episodic.store.assert_awaited_once()
        assert result.facts == []
        assert "neo4j" in subsystems.degraded_subsystems()

    @pytest.mark.asyncio
    async def test_upsert_stops_after_first_connection_failure(self):
        """One timeout, not N — the loop breaks instead of retrying each fact."""
        from smritikosh.memory.semantic import SemanticMemory

        hippo, _ = _hippocampus_with_broken_neo4j(fail_profile=False, fail_upsert=True)
        semantic: SemanticMemory = hippo.semantic

        stored = await hippo._upsert_facts(
            AsyncMock(), "u1", "default",
            [
                {"category": "tool", "key": "a", "value": "1", "confidence": 0.9},
                {"category": "tool", "key": "b", "value": "2", "confidence": 0.9},
                {"category": "tool", "key": "c", "value": "3", "confidence": 0.9},
            ],
        )

        assert stored == []
        # Broke out on the first failure rather than trying all three.
        assert semantic.check_fact_conflict.await_count == 1

    @pytest.mark.asyncio
    async def test_invalid_fact_still_skipped_individually(self):
        """A malformed fact must not be mistaken for an outage — keep going."""
        from smritikosh.memory.semantic import FactRecord

        hippo, _ = _hippocampus_with_broken_neo4j(fail_profile=False, fail_upsert=False)
        hippo.semantic.upsert_fact = AsyncMock(
            return_value=FactRecord(
                category="tool", key="language", value="rust", confidence=0.9,
                frequency_count=1, first_seen_at="", last_seen_at="",
            )
        )

        stored = await hippo._upsert_facts(
            AsyncMock(), "u1", "default",
            [
                {"category": "tool"},   # missing key/value → KeyError → skipped
                {"category": "tool", "key": "language", "value": "rust", "confidence": 0.9},
            ],
        )

        assert len(stored) == 1
        assert subsystems.is_degraded("neo4j") is False


# ── /health status contract ───────────────────────────────────────────────────


def _patch_health(pg_ok: bool, neo_ok: bool, mongo_url: str | None = None):
    """Context managers driving the health probes to the requested outcome."""
    pg_conn = MagicMock()
    pg_conn.__aenter__ = AsyncMock(
        return_value=AsyncMock(execute=AsyncMock())
        if pg_ok
        else AsyncMock(execute=AsyncMock(side_effect=OSError("pg down")))
    )
    pg_conn.__aexit__ = AsyncMock(return_value=False)
    if not pg_ok:
        pg_conn.__aenter__ = AsyncMock(side_effect=OSError("pg down"))

    neo_session = MagicMock()
    if neo_ok:
        neo_session.__aenter__ = AsyncMock(return_value=AsyncMock(run=AsyncMock()))
        neo_session.__aexit__ = AsyncMock(return_value=False)
    else:
        neo_session.__aenter__ = AsyncMock(side_effect=OSError("neo down"))
        neo_session.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session = MagicMock(return_value=neo_session)

    engine_mock = MagicMock()
    engine_mock.connect = MagicMock(return_value=pg_conn)

    settings_mock = MagicMock()
    settings_mock.mongodb_url = mongo_url
    settings_mock.llm_provider = "ollama"
    settings_mock.llm_base_url = "http://localhost:11434"
    settings_mock.pg_pool_size = 5
    settings_mock.pg_max_overflow = 10

    return (
        patch("smritikosh.api.routes.health.engine", engine_mock),
        patch("smritikosh.api.routes.health.get_driver", return_value=driver),
        patch("smritikosh.api.routes.health.settings", settings_mock),
    )


class TestHealthContract:
    @pytest.mark.asyncio
    async def test_all_healthy_is_ok(self):
        from smritikosh.api.routes.health import health

        patches = _patch_health(pg_ok=True, neo_ok=True)
        with patches[0], patches[1], patches[2]:
            result = await health()

        assert result.status == "ok"
        assert result.degraded_subsystems == {}

    @pytest.mark.asyncio
    async def test_neo4j_down_is_degraded_not_error(self):
        """Optional subsystem down → still serving."""
        from smritikosh.api.routes.health import health

        patches = _patch_health(pg_ok=True, neo_ok=False)
        with patches[0], patches[1], patches[2]:
            result = await health()

        assert result.status == "degraded"
        assert result.postgres == "ok"
        assert result.neo4j == "error"
        assert "neo4j" in result.degraded_subsystems

    @pytest.mark.asyncio
    async def test_postgres_down_is_error(self):
        """Required subsystem down → pull this instance from the pool."""
        from smritikosh.api.routes.health import health

        patches = _patch_health(pg_ok=False, neo_ok=True)
        with patches[0], patches[1], patches[2]:
            result = await health()

        assert result.status == "error"
        assert result.postgres == "error"

    @pytest.mark.asyncio
    async def test_unconfigured_mongo_is_not_degraded(self):
        from smritikosh.api.routes.health import health

        patches = _patch_health(pg_ok=True, neo_ok=True, mongo_url=None)
        with patches[0], patches[1], patches[2]:
            result = await health()

        assert result.mongodb == "not_configured"
        assert "mongodb" not in result.degraded_subsystems
        assert result.status == "ok"

    @pytest.mark.asyncio
    async def test_recovery_clears_degraded_state(self):
        """A probe that succeeds again must clear the flag without a restart."""
        from smritikosh.api.routes.health import health

        subsystems.mark_degraded("neo4j", "stale failure")

        patches = _patch_health(pg_ok=True, neo_ok=True)
        with patches[0], patches[1], patches[2]:
            result = await health()

        assert result.status == "ok"
        assert result.degraded_subsystems == {}
