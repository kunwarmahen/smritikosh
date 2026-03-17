from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM (chat / extraction) ────────────────────────────────────────────
    # Supported providers: claude | openai | gemini | ollama | vllm
    llm_provider: str = "claude"
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key: str | None = None
    # For ollama (http://localhost:11434) or vllm (http://localhost:8000/v1)
    llm_base_url: str | None = None

    # ── Embeddings ─────────────────────────────────────────────────────────
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    # Dimension of the embedding model (text-embedding-3-small = 1536)
    embedding_dimensions: int = 1536

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    postgres_url: str = "postgresql+asyncpg://smritikosh:smritikosh@localhost:5432/smritikosh"

    # ── Neo4j ──────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "smritikosh"

    # ── App ────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Connectors ─────────────────────────────────────────────────────────
    # Set SLACK_SIGNING_SECRET in .env to enable Slack event verification
    slack_signing_secret: str | None = None

    # ── Audit trail (MongoDB) ───────────────────────────────────────────────
    # Leave unset to disable audit trail (system works without it).
    mongodb_url: str | None = None
    mongodb_db_name: str = "smritikosh_audit"

    # ── Auth (UI login) ─────────────────────────────────────────────────────
    # Generate a strong secret: python -c "import secrets; print(secrets.token_hex(32))"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 30


# Single shared instance — import this everywhere
settings = Settings()
