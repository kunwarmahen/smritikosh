# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — build
#   Install all Python dependencies into a clean prefix so that Stage 2 can
#   copy only the installed packages, not the full build toolchain.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps (needed by some native extensions, e.g. asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed to install the package
COPY pyproject.toml README.md ./
COPY smritikosh/ ./smritikosh/

# Install the package and all runtime deps into /install
RUN pip install --no-cache-dir --prefix=/install .


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
#   Minimal image: no compiler, no build tools, runs as a non-root user.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps (asyncpg needs libpq at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Create a non-root user
RUN groupadd --gid 1001 smriti \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /sbin/nologin smriti

WORKDIR /app

# Copy the package source and Alembic config so migrations can run at boot
COPY smritikosh/ ./smritikosh/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# All files belong to the non-root user
RUN chown -R smriti:smriti /app

USER smriti

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# Liveness probe: hits the /health endpoint (no auth required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run Alembic migrations then start the server.
# Using a shell entrypoint so that the PORT env var is expanded at runtime.
CMD alembic upgrade head \
    && uvicorn smritikosh.api.main:app \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --workers 2 \
        --log-level info
