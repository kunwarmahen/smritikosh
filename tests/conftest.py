"""
pytest configuration for Smritikosh tests.

Marks:
    live   — requires real API keys (ANTHROPIC / OPENAI / GEMINI). Skipped by default.
             Run with: pytest -m live
    ollama — requires local Ollama server running. Skipped by default.
             Run with: pytest -m ollama
    db     — requires running PostgreSQL + Neo4j (docker compose up -d). Skipped by default.
             Run with: pytest -m db

All other tests run offline with mocked LLM calls and in-memory mocks.
"""

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
