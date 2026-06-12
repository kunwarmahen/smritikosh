"""
Central Prometheus collectors for Smritikosh internals (item G1).

The HTTP layer is already instrumented by prometheus-fastapi-instrumentator
(per-route latency / throughput / errors on GET /metrics). This module covers
what that cannot see:

    Background jobs (B1)   smritikosh_job_runs_total{job,outcome}
                           smritikosh_job_duration_seconds{job}
                           smritikosh_job_last_success_timestamp_seconds{job}
                           smritikosh_job_user_errors_total{job}
    LLM usage (D1)         smritikosh_llm_calls_total{model,kind,outcome}
                           smritikosh_llm_tokens_total{model,kind,token_type}
                           smritikosh_llm_cost_usd_total{model,kind}
                           smritikosh_llm_latency_seconds{model,kind}
    Task queue (A3)        smritikosh_tasks_total{task,path}

All collectors live in the default prometheus_client registry, so the API
process exposes them automatically on GET /metrics. The standalone worker
serves the same registry on WORKER_METRICS_PORT (see worker/main.py).

Use track_job() around each scheduler job so outcome, duration, and freshness
are recorded consistently:

    async def run_consolidation_for_all_users(self):
        with track_job("consolidation"):
            ...
"""

import time
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

# ── Background jobs (B1) ──────────────────────────────────────────────────────

JOB_RUNS = Counter(
    "smritikosh_job_runs_total",
    "Background job cycles, by job name and outcome (success | error).",
    ["job", "outcome"],
)

JOB_DURATION = Histogram(
    "smritikosh_job_duration_seconds",
    "Wall-clock duration of one background job cycle.",
    ["job"],
    buckets=(0.1, 0.5, 1, 5, 15, 60, 300, 900, 3600),
)

JOB_LAST_SUCCESS = Gauge(
    "smritikosh_job_last_success_timestamp_seconds",
    "Unix timestamp of the last successful cycle, per job. Alert on staleness: "
    "time() - this > 2x the job's cron interval means the job is not running.",
    ["job"],
)

JOB_USER_ERRORS = Counter(
    "smritikosh_job_user_errors_total",
    "Per-user job failures that were swallowed for resilience (the cycle "
    "continues with the next user). A rising rate means memory quality is "
    "degrading for some users even though the job itself reports success.",
    ["job"],
)


@contextmanager
def track_job(job: str):
    """Record outcome, duration, and last-success freshness for one job cycle."""
    start = time.monotonic()
    try:
        yield
    except Exception:
        JOB_RUNS.labels(job=job, outcome="error").inc()
        raise
    else:
        JOB_RUNS.labels(job=job, outcome="success").inc()
        JOB_LAST_SUCCESS.labels(job=job).set_to_current_time()
    finally:
        JOB_DURATION.labels(job=job).observe(time.monotonic() - start)


# ── LLM usage (D1) ────────────────────────────────────────────────────────────

LLM_CALLS = Counter(
    "smritikosh_llm_calls_total",
    "LLM API calls, by resolved model string, kind (chat | embedding | vision), "
    "and outcome (success | error). Each retry attempt counts — it is billed.",
    ["model", "kind", "outcome"],
)

LLM_TOKENS = Counter(
    "smritikosh_llm_tokens_total",
    "Tokens consumed, by model, kind, and token_type (prompt | completion).",
    ["model", "kind", "token_type"],
)

LLM_COST = Counter(
    "smritikosh_llm_cost_usd_total",
    "Estimated spend in USD (litellm pricing table; 0 for local models).",
    ["model", "kind"],
)

LLM_LATENCY = Histogram(
    "smritikosh_llm_latency_seconds",
    "Latency of one LLM API call (per attempt, not per logical request).",
    ["model", "kind"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)


# ── Task queue (A3) ───────────────────────────────────────────────────────────

TASKS = Counter(
    "smritikosh_tasks_total",
    "Background tasks dispatched, by task name and path "
    "(queued = durable ARQ queue | inline = in-process fallback).",
    ["task", "path"],
)
