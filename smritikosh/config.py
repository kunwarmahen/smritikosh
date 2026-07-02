import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

_logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM (chat / extraction) ────────────────────────────────────────────
    # Supported providers: claude | openai | gemini | ollama | vllm
    llm_provider: str = "claude"
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key: str | None = None
    # For ollama (http://localhost:11434) or vllm (http://localhost:8000/v1)
    llm_base_url: str | None = None
    # Set to an integer to cap token output; leave as None for no limit
    llm_max_tokens: int | None = None
    # Seconds before a single LLM/embedding call is abandoned (prevents hung bg tasks)
    llm_timeout: int = 120

    # ── LLM fallback provider ──────────────────────────────────────────────
    # When the primary LLM is down or rate-limited, requests are retried against
    # the fallback before raising an error.  Leave unset to disable fallback.
    llm_fallback_provider: str | None = None   # e.g. "openai"
    llm_fallback_model: str | None = None      # e.g. "gpt-4o-mini"
    llm_fallback_api_key: str | None = None    # if the fallback uses a different key
    llm_fallback_base_url: str | None = None   # for local fallback providers

    # ── Embeddings ─────────────────────────────────────────────────────────
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    # Dimension of the embedding model (text-embedding-3-small = 1536)
    embedding_dimensions: int = 1536

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    postgres_url: str = "postgresql+asyncpg://smritikosh:smritikosh@localhost:5432/smritikosh"

    # ── Connection pools (item A4) ─────────────────────────────────────────
    # Budget: every process holds its own pool — each API replica, the
    # scheduler worker, and each ARQ taskworker. Total Postgres connections:
    #   processes × (PG_POOL_SIZE + PG_MAX_OVERFLOW)
    # Keep that under Postgres max_connections (default 100) with headroom
    # for migrations and ad-hoc psql. The defaults (5 + 10) budget ~6
    # processes against a stock Postgres; for larger fleets raise
    # max_connections or put PgBouncer in front of the API tier.
    pg_pool_size: int = 5         # persistent connections per process
    pg_max_overflow: int = 10     # extra burst connections (closed when idle)
    pg_pool_timeout: int = 30     # seconds to wait for a free connection before erroring
    pg_pool_recycle: int = 1800   # recycle connections older than this (seconds)

    # ── Neo4j ──────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "smritikosh"
    # Per-process cap on Neo4j driver connections (driver default is 100).
    neo4j_max_pool_size: int = 50

    # ── App ────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    sqlalchemy_log_level: str = "WARNING"

    # ── Connectors ─────────────────────────────────────────────────────────
    # Set SLACK_SIGNING_SECRET in .env to enable Slack event verification
    slack_signing_secret: str | None = None

    # Google OAuth2 connectors (Gmail + Google Calendar)
    # Register a new OAuth2 application at https://console.cloud.google.com/
    # Required scopes: gmail.readonly, calendar.readonly
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str = "http://localhost:8080/connectors/google/callback"

    # ── Audit trail (MongoDB) ───────────────────────────────────────────────
    # Leave unset to disable audit trail (system works without it).
    mongodb_url: str | None = None
    mongodb_db_name: str = "smritikosh_audit"

    # ── Auth (UI login) ─────────────────────────────────────────────────────
    # Generate a strong secret: python -c "import secrets; print(secrets.token_hex(32))"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 30

    # ── Connector token encryption ──────────────────────────────────────────
    # Key used to encrypt OAuth tokens stored in `user_connectors` at rest.
    # If unset, the key is derived from jwt_secret (keeps existing deployments
    # working). Set this so the JWT secret and the connector-token key can be
    # rotated independently. Generate:
    #   python -c "import secrets; print(secrets.token_hex(32))"
    connector_encryption_key: str | None = None

    # Set BOOTSTRAP_ADMIN=1 temporarily to allow the first admin registration
    # without a token. Remove it immediately after creating the first account.
    bootstrap_admin: bool = False

    # Reconsolidate the top recalled event after each /context call. The task
    # runs an LLM call per request; on slow/local models it can saturate the
    # provider and stall subsequent requests (see A3-followup: move it onto
    # the ARQ queue). Set RECONSOLIDATION_ON_RECALL=0 for benchmark runs.
    reconsolidation_on_recall: bool = True

    # ── CORS ────────────────────────────────────────────────────────────────
    # Comma-separated list of origins allowed to call the API from a browser,
    # e.g. "https://app.example.com,https://admin.example.com".
    # Empty (default) — no CORS headers are sent, so browsers block cross-origin
    # calls. This is the intended posture when only server-side clients talk to
    # the API (the SDKs and the Next.js dashboard both call it server-side).
    # Set "*" only for a credential-less public API: when "*" is present,
    # credentials (cookies / Authorization via fetch credentials mode) are
    # disabled, as the CORS spec forbids combining them with a wildcard.
    cors_allowed_origins: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS_ALLOWED_ORIGINS parsed into a list; empty list = CORS disabled."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # ── Redis ───────────────────────────────────────────────────────────────
    # Shared store for the rate limiter (and, later, the task queue + caches).
    # If unset, the rate limiter falls back to per-process in-memory storage —
    # fine for a single instance, but limits are NOT enforced correctly across
    # multiple API replicas. Set this whenever you run more than one instance.
    #   Example: redis://localhost:6379/0
    redis_url: str | None = None

    # ── Rate limiting ───────────────────────────────────────────────────────
    # Limits applied per authenticated user (user_id extracted from JWT/API key).
    # Format: "<count>/<period>" — e.g. "60/minute", "1000/hour"
    # Set to "" to disable rate limiting entirely.
    rate_limit_encode: str = "60/minute"     # POST /memory/event
    rate_limit_context: str = "60/minute"    # POST /context
    rate_limit_search: str = "120/minute"    # POST /memory/search

    # ── Usage quotas (per-tenant caps) ──────────────────────────────────────
    # Defaults applied to every (user_id, app_id) without a user_quotas row.
    # 0 = unlimited. Event limits count stored events; token limits sum LLM
    # prompt+completion tokens from the llm_usage accounting table. Windows
    # are UTC-calendar (daily = since midnight, monthly = since the 1st).
    # Per-tenant overrides: PUT /admin/quotas/{user_id}.
    quota_default_daily_events: int = 0
    quota_default_monthly_events: int = 0
    quota_default_daily_tokens: int = 0
    quota_default_monthly_tokens: int = 0

    # ── Semantic fact decay ─────────────────────────────────────────────────
    # Confidence halves every N days without reinforcement (exponential decay).
    # Relationships that fall below the floor are deleted; orphaned Fact nodes
    # are cleaned up automatically on the same run.
    fact_decay_half_life_days: float = 60.0  # days until confidence halves
    fact_decay_floor: float = 0.1            # delete relationships below this confidence

    # ── Background scheduler ────────────────────────────────────────────────
    # Whether THIS process runs the in-process memory-maintenance scheduler
    # (consolidation, pruning, clustering, belief mining, fact decay, synthesis).
    #   true  — single-process / development: the API process runs the jobs.
    #   false — multi-replica: run a dedicated worker (python -m smritikosh.worker.main)
    #           and set this to false on the API replicas.
    # A Postgres advisory lock elects a single leader regardless of this flag, so
    # leaving it true on several replicas is safe — only one actually runs jobs.
    run_scheduler: bool = True

    # ── Worker metrics ──────────────────────────────────────────────────────
    # Port for the standalone worker's Prometheus /metrics endpoint (job runs,
    # durations, LLM tokens/cost from background jobs). 0 (default) = disabled.
    # The API process always exposes /metrics regardless of this setting.
    worker_metrics_port: int = 0

    # ── Scheduler (cron expressions, UTC) ──────────────────────────────────
    # Standard 5-field cron: minute hour day-of-month month day-of-week
    # Examples:
    #   "0 * * * *"   — every hour on the hour
    #   "0 2 * * *"   — daily at 02:00 UTC
    #   "0 */6 * * *" — every 6 hours
    #   "0 3 * * 0"   — every Sunday at 03:00 UTC
    # How many tenants a background job processes concurrently (item A5).
    # Each concurrent slot holds its own Postgres + Neo4j session and may make
    # LLM calls, so size against pool limits and provider rate limits.
    scheduler_job_concurrency: int = 4

    scheduler_consolidation_cron: str = "0 * * * *"    # hourly
    scheduler_pruning_cron: str = "0 2 * * *"          # daily at 02:00 UTC
    scheduler_clustering_cron: str = "0 */6 * * *"     # every 6 hours
    scheduler_belief_mining_cron: str = "0 */12 * * *"  # every 12 hours
    scheduler_fact_decay_cron: str = "0 3 * * 0"       # weekly, Sunday 03:00 UTC

    # ── Media ingestion (Whisper transcription) ────────────────────────────
    # Provider: "openai" (cloud) or "local" (self-hosted, via ollama/vllm/llamacpp)
    whisper_provider: str = "openai"
    whisper_model: str = "whisper-1"
    whisper_api_key: str | None = None  # for OpenAI; defaults to embedding_api_key if unset
    whisper_base_url: str | None = None  # for local providers (ollama, vllm, etc.)

    # ── Media file size limits ────────────────────────────────────────────
    media_max_audio_mb: int = 25  # Whisper API limit
    media_max_document_mb: int = 10
    media_max_document_pages: int = 50  # PDF page cap
    media_max_image_mb: int = 20

    # ── Vision model (image description) ─────────────────────────────────
    # Provider: "openai" (cloud, gpt-4o-mini) or "claude" (anthropic) or "ollama" (local)
    # The vision model must support multimodal (image) inputs.
    vision_provider: str = "openai"
    vision_model: str = "gpt-4o-mini"
    vision_api_key: str | None = None   # defaults to llm_api_key if unset
    vision_base_url: str | None = None  # for local providers

    # ── Speaker diarization (Phase 12 — meeting recordings) ───────────────
    # Provider: "none" (default, no diarization — first-person filter only) | "pyannote"
    # "pyannote" requires: pip install pyannote.audio torch  +  HF_TOKEN set below.
    diarization_provider: str = "none"
    # HuggingFace token — required for pyannote/speaker-diarization-3.1 model download.
    # Generate at https://huggingface.co/settings/tokens (role: read).
    hf_token: str | None = None
    # Cosine similarity threshold for voice matching (0–1). Higher = stricter.
    speaker_similarity_threshold: float = 0.75
    # Max meeting recording file size (MB)
    media_max_meeting_mb: int = 500


# ── Runtime security validation ────────────────────────────────────────────────

# Values of APP_ENV treated as non-production — security checks are relaxed to
# warnings so local development is never blocked.
_NON_PROD_ENVS = {"development", "dev", "local", "test", "testing", "ci"}

# The secret shipped in config defaults and .env.example. Must never reach prod.
_DEFAULT_JWT_SECRET = "change-me-in-production"

# Minimum acceptable length for any at-rest secret.
_MIN_SECRET_LEN = 32


def is_production(s: "Settings | None" = None) -> bool:
    """True when APP_ENV is a production-like environment.

    Anything not explicitly listed in _NON_PROD_ENVS is treated as production —
    fail closed, so a typo in APP_ENV does not silently relax security checks.
    """
    s = s or settings
    return s.app_env.strip().lower() not in _NON_PROD_ENVS


def security_warnings(s: "Settings | None" = None) -> list[str]:
    """Return fatal security misconfigurations for the given settings.

    An empty list means the configuration is safe to boot. The caller decides
    severity: in production a non-empty list should refuse startup; in a
    non-production env the same items are surfaced as warnings only.
    """
    s = s or settings
    problems: list[str] = []

    if s.jwt_secret == _DEFAULT_JWT_SECRET:
        problems.append(
            "JWT_SECRET is still the shipped default 'change-me-in-production'. "
            'Generate one: python -c "import secrets; print(secrets.token_hex(32))"'
        )
    elif len(s.jwt_secret) < _MIN_SECRET_LEN:
        problems.append(
            f"JWT_SECRET is too short ({len(s.jwt_secret)} chars); "
            f"use at least {_MIN_SECRET_LEN} characters."
        )

    if s.connector_encryption_key is not None and len(s.connector_encryption_key) < _MIN_SECRET_LEN:
        problems.append(
            f"CONNECTOR_ENCRYPTION_KEY is too short ({len(s.connector_encryption_key)} chars); "
            f"use at least {_MIN_SECRET_LEN} characters."
        )

    return problems


def enforce_runtime_security(s: "Settings | None" = None) -> None:
    """Refuse to boot a production deployment with insecure secrets.

    In a production-like APP_ENV any fatal misconfiguration raises RuntimeError;
    in a non-production env the same problems are logged as a warning so local
    development is never blocked. Shared by the API process and the worker.
    """
    s = s or settings
    problems = security_warnings(s)
    if not problems:
        return
    formatted = "\n".join(f"  - {p}" for p in problems)
    if is_production(s):
        raise RuntimeError(
            f"Refusing to start: insecure configuration for APP_ENV={s.app_env!r}.\n"
            f"{formatted}\n"
            "Fix these before deploying. (Setting APP_ENV to a development value bypasses "
            "this check — never do that in production.)"
        )
    _logger.warning(
        "Insecure configuration detected — allowed because APP_ENV=%r is non-production:\n%s",
        s.app_env,
        formatted,
    )


# Single shared instance — import this everywhere
settings = Settings()
