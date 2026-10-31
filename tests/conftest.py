"""
pytest configuration for Smritikosh tests.

Marks:
    live   — requires real API keys (ANTHROPIC / OPENAI / GEMINI). Skipped by default.
             Run with: pytest -m live
    ollama — requires local Ollama server running. Skipped by default.
             Run with: pytest -m ollama
    db     — requires running PostgreSQL + Neo4j (docker compose up -d). Skipped by default.
             Run with: pytest -m db
             THESE TESTS DROP TABLES AND DELETE GRAPH NODES. They run only when
             SMRITIKOSH_TEST_DATABASES=1 is set and the Postgres database's name
             ends in "_test" (POSTGRES_URL); otherwise they are skipped, so a
             -m db run can never wipe a database that holds real memories.

All other tests run offline with mocked LLM calls and in-memory mocks.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: integration test requiring real API keys")
    config.addinivalue_line("markers", "ollama: integration test requiring local Ollama server")
    config.addinivalue_line("markers", "db: integration test requiring running Postgres + Neo4j")


def pytest_collection_modifyitems(config, items):
    """Auto-skip live and ollama tests unless explicitly selected via -m."""
    # Determine which marks the user explicitly requested
    markexpr = config.option.markexpr if hasattr(config.option, "markexpr") else ""

    skip_live = pytest.mark.skip(reason="Live test — run with: pytest -m live")
    skip_ollama = pytest.mark.skip(reason="Ollama test — run with: pytest -m ollama")
    skip_db = pytest.mark.skip(reason="DB test — run with: pytest -m db (needs docker compose up -d)")

    for item in items:
        if "live" in item.keywords and "live" not in markexpr:
            item.add_marker(skip_live)
        if "ollama" in item.keywords and "ollama" not in markexpr:
            item.add_marker(skip_ollama)
        if "db" in item.keywords and "db" not in markexpr:
            item.add_marker(skip_db)


def _databases_are_for_tests() -> str | None:
    """None when the db tests may run; else why not."""
    from smritikosh.config import settings
    if os.environ.get("SMRITIKOSH_TEST_DATABASES") != "1":
        return ("db tests drop tables: set SMRITIKOSH_TEST_DATABASES=1 and point "
                "POSTGRES_URL / NEO4J_URI at throwaway databases")
    name = settings.postgres_url.rsplit("/", 1)[-1].split("?", 1)[0]
    if not name.endswith("_test"):
        return f"db tests drop tables: POSTGRES_URL's database {name!r} must end in _test"
    return None


@pytest.fixture(autouse=True)
def _refuse_real_databases(request):
    """A db test against a database holding real memories once dropped all its
    tables at teardown. Refuse before any fixture opens a connection."""
    if request.node.get_closest_marker("db") is not None:
        why = _databases_are_for_tests()
        if why:
            pytest.skip(why)

