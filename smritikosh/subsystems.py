"""
Subsystem criticality and degraded-state registry (item B3).

Smritikosh talks to four stores, and they are **not** equally essential:

    postgres  — REQUIRED. Episodic memory, auth, quotas, jobs. Without it
                there is no product, so a failure here is fatal at startup.
    neo4j     — optional. Semantic facts/beliefs. Retrieval degrades to
                episodic-only; encode still stores events and skips fact
                upserts.
    mongodb   — optional. Audit trail. Already a no-op when unset.
    redis     — optional. Rate-limit store + durable task queue; both have
                documented in-process fallbacks.

Before B3, `init_neo4j()` ran unguarded in the startup path, so an unreachable
Neo4j stopped the whole API from booting — even though everything Postgres
backs would have worked fine. Optional subsystems now record their failure
here instead of raising, and `GET /health` reports what is degraded and why.

This is process-local state: each replica reports its own view, which is what
you want — one replica losing its Neo4j connection is exactly the thing a
per-instance health check should surface.

Note on Redis: it is listed as optional for documentation, but nothing probes
it here. Its outage signal already exists and is better — every task that falls
back to in-process execution increments
``smritikosh_tasks_total{path="inline"}``, which the shipped Grafana rules alert
on. A synthetic ping would only duplicate that.
"""

import logging

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# Failure here is fatal — the API refuses to start.
REQUIRED_SUBSYSTEMS: tuple[str, ...] = ("postgres",)

# Failure here degrades functionality but the API still serves.
OPTIONAL_SUBSYSTEMS: tuple[str, ...] = ("neo4j", "mongodb", "redis")

SUBSYSTEM_DEGRADED = Gauge(
    "smritikosh_subsystem_degraded",
    "1 when an optional subsystem is unreachable from this process, 0 when "
    "healthy. Alert on a sustained 1: the API stays up but the corresponding "
    "capability (semantic facts, audit trail, shared rate limits) is missing.",
    ["subsystem"],
)

# subsystem name → reason it is degraded
_degraded: dict[str, str] = {}


def mark_degraded(name: str, reason: str) -> None:
    """Record that an optional subsystem is unreachable from this process."""
    previously = _degraded.get(name)
    _degraded[name] = reason
    SUBSYSTEM_DEGRADED.labels(subsystem=name).set(1)
    # Only log the transition, not every repeated health check.
    if previously is None:
        logger.warning(
            "Subsystem degraded: %s — %s", name, reason, extra={"subsystem": name}
        )


def mark_healthy(name: str) -> None:
    """Clear a subsystem's degraded state (it answered again)."""
    recovered = _degraded.pop(name, None) is not None
    SUBSYSTEM_DEGRADED.labels(subsystem=name).set(0)
    if recovered:
        logger.info("Subsystem recovered: %s", name, extra={"subsystem": name})


def is_degraded(name: str) -> bool:
    return name in _degraded


def degraded_subsystems() -> dict[str, str]:
    """Currently-degraded subsystems mapped to the reason, for `/health`."""
    return dict(_degraded)


def reset() -> None:
    """Clear all degraded state (tests)."""
    for name in list(_degraded):
        mark_healthy(name)
