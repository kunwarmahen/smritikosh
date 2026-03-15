# Smritikosh

**स्मृतिकोश** *(Sanskrit: "memory treasury")*

A universal memory layer for LLM applications — a hippocampus for AI.

Smritikosh gives any LLM application persistent, user-specific memory modelled on how the human brain actually stores and retrieves information: episodic events encoded as vectors in PostgreSQL, semantic facts distilled into a Neo4j knowledge graph, background consolidation that compresses raw events into durable knowledge, and synaptic pruning that discards low-value memories over time.

---

## Table of Contents

- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the server](#running-the-server)
- [API reference](#api-reference)
- [Python SDK](#python-sdk)
- [Testing](#testing)
- [LLM provider guide](#llm-provider-guide)
- [Background jobs](#background-jobs)

---

## How it works

```
User message
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Hippocampus  (intake coordinator)              │
│                                                 │
│  1. Amygdala scores emotional importance        │
│  2. Embed text + extract facts  (parallel)      │
│  3. Store event → PostgreSQL + pgvector         │
│  4. Upsert facts → Neo4j knowledge graph        │
└─────────────────────────────────────────────────┘
     │                         │
     ▼                         ▼
EpisodicMemory            SemanticMemory
(raw events +             (stable facts:
 vectors)                  preferences,
                           skills, goals…)
                                │
          ┌─────────────────────┤
          │  Background jobs    │
          │  (every hour/day)   │
          ▼                     ▼
    Consolidator          SynapticPruner
    raw → summary         deletes low-value
    + distilled facts     memories
```

When your app needs context before an LLM call:

```
Query → ContextBuilder
          │
          ├── hybrid_search()   (vector + recency + importance)
          ├── get_user_profile() (Neo4j semantic facts)
          └── get_recent()      (last N raw events)
                │
                ▼
        MemoryContext.messages  →  prepend to LLM messages
```

---

## Architecture

| Component | Role | Storage |
|---|---|---|
| **Amygdala** | Scores importance of each event (0.1 – 1.0) | — |
| **EpisodicMemory** | Stores raw events; hybrid search over vectors | PostgreSQL + pgvector |
| **SemanticMemory** | Distilled facts organised in a knowledge graph | Neo4j |
| **Hippocampus** | Orchestrates intake: score → embed → extract → store | — |
| **ContextBuilder** | Retrieves relevant context before an LLM call | — |
| **Consolidator** | Background: compresses events into facts via LLM | — |
| **SynapticPruner** | Background: deletes old low-scoring events | — |
| **MemoryScheduler** | Runs Consolidator + Pruner on a timer (APScheduler) | — |
| **LLMAdapter** | Unified interface to Claude, OpenAI, Gemini, Ollama, vLLM | — |
| **SmritikoshClient** | Python SDK wrapping the REST API | — |

---

## Project structure

```
smritikosh/
├── api/
│   ├── deps.py           # FastAPI dependency injection (@lru_cache singletons)
│   ├── main.py           # App factory + lifespan (startup/shutdown)
│   ├── schemas.py        # Pydantic request/response models
│   └── routes/
│       ├── health.py     # GET /health
│       ├── memory.py     # POST /memory/event, GET /memory/{user_id}
│       └── context.py    # POST /context
├── config.py             # Pydantic Settings (reads .env)
├── db/
│   ├── models.py         # SQLAlchemy 2.0 ORM: Event, UserFact, MemoryLink
│   ├── postgres.py       # Async engine, session helpers
│   └── neo4j.py          # Driver singleton, session helpers, schema init
├── llm/
│   └── adapter.py        # LLMAdapter: complete(), embed(), extract_structured()
├── memory/
│   ├── episodic.py       # EpisodicMemory: store, search, hybrid_search
│   ├── semantic.py       # SemanticMemory: upsert_fact, get_user_profile
│   └── hippocampus.py    # Hippocampus: encode()
├── processing/
│   ├── amygdala.py       # Importance scoring (pure, no LLM)
│   ├── consolidator.py   # Batch compress events → facts
│   ├── synaptic_pruner.py# Delete low-value memories
│   └── scheduler.py      # APScheduler background jobs
├── retrieval/
│   └── context_builder.py# Build memory context for LLM calls
└── sdk/
    ├── client.py         # SmritikoshClient (async HTTP)
    └── types.py          # EncodedEvent, MemoryContext, RecentEvent, HealthStatus

tests/
├── conftest.py           # pytest marks: live, ollama, db
├── test_llm_adapter.py
├── test_db_models.py
├── test_episodic_memory.py
├── test_semantic_memory.py
├── test_amygdala.py
├── test_hippocampus.py
├── test_context_builder.py
├── test_api.py
├── test_consolidator.py
├── test_synaptic_pruner.py
├── test_scheduler.py
└── test_sdk_client.py

alembic/
└── versions/
    └── 0001_initial_schema.py  # events, user_facts, memory_links tables + IVFFlat index
```

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | StrEnum, `match`, type syntax |
| Docker + Compose | any recent | PostgreSQL + Neo4j |
| An LLM API key | — | Claude / OpenAI / Gemini (or Ollama locally) |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-org/smritikosh.git
cd smritikosh

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install the package

```bash
# Runtime + dev dependencies
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```dotenv
LLM_PROVIDER=claude
LLM_MODEL=claude-haiku-4-5-20251001
LLM_API_KEY=sk-ant-...

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...
```

See [Configuration](#configuration) for all options and [LLM provider guide](#llm-provider-guide) for provider-specific examples.

### 4. Start the databases

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 17** with the `pgvector` extension on port `5432`
- **Neo4j 5.26** on ports `7474` (browser) and `7687` (bolt)

Wait for both to be healthy:

```bash
docker compose ps   # both should show "healthy"
```

### 5. Run database migrations

```bash
alembic upgrade head
```

This creates the `events`, `user_facts`, and `memory_links` tables, enables the `vector` extension, and adds an IVFFlat index for fast similarity search.

---

## Configuration

All settings are read from the environment (or `.env`). Every field has a default so only the keys you need to change are required.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `claude` | `claude` / `openai` / `gemini` / `ollama` / `vllm` |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Model name for chat/extraction |
| `LLM_API_KEY` | — | API key for the chat provider |
| `LLM_BASE_URL` | — | Custom base URL (Ollama / vLLM only) |
| `EMBEDDING_PROVIDER` | `openai` | `openai` / `ollama` / `vllm` / `gemini` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_API_KEY` | — | API key for the embedding provider |
| `EMBEDDING_BASE_URL` | — | Custom base URL for embeddings |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector size — must match your model |
| `POSTGRES_URL` | `postgresql+asyncpg://smritikosh:smritikosh@localhost:5432/smritikosh` | Async PostgreSQL connection string |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `smritikosh` | Neo4j password |
| `LOG_LEVEL` | `INFO` | Python log level |

---

## Running the server

```bash
uvicorn smritikosh.api.main:app --reload --port 8080
```

On startup the server will:
1. Enable the `pgvector` extension and create tables (if not already present via Alembic)
2. Apply Neo4j schema constraints and indexes
3. Start background scheduler (consolidation every hour, pruning every 24 hours)

Interactive API docs are available at `http://localhost:8080/docs`.

---

## API reference

### `GET /health`

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok", "version": "0.1.0"}
```

---

### `POST /memory/event`

Store a user interaction in episodic memory. Runs the full Hippocampus pipeline: importance scoring → embedding → fact extraction → PostgreSQL + Neo4j writes.

```bash
curl -X POST http://localhost:8080/memory/event \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "content": "I prefer dark mode and use Neovim as my editor.",
    "app_id": "myapp",
    "metadata": {"source": "chat"}
  }'
```

```json
{
  "event_id": "3f7a1b2c-...",
  "user_id": "alice",
  "importance_score": 0.72,
  "facts_extracted": 2,
  "extraction_failed": false
}
```

| Field | Type | Description |
|---|---|---|
| `user_id` | string | **Required.** Unique user identifier |
| `content` | string | **Required.** Raw text to encode |
| `app_id` | string | Application namespace (default: `"default"`) |
| `metadata` | object | Optional extra context |

---

### `POST /context`

Retrieve a memory context block for a user query. Uses hybrid search (vector similarity + recency decay + importance score) and the user's semantic profile from Neo4j.

```bash
curl -X POST http://localhost:8080/context \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "query": "What editor and theme does Alice prefer?",
    "app_id": "myapp"
  }'
```

```json
{
  "user_id": "alice",
  "query": "What editor and theme does Alice prefer?",
  "context_text": "## User Memory Context\n### Who this user is:\n...",
  "messages": [{"role": "system", "content": "## User Memory Context\n..."}],
  "total_memories": 4,
  "embedding_failed": false
}
```

Inject `messages` directly before your LLM call:

```python
memory_messages = context.messages          # [{"role": "system", ...}]
user_messages   = [{"role": "user", "content": "What editor should I use?"}]
response = await llm.complete(memory_messages + user_messages)
```

---

### `GET /memory/{user_id}`

Browse recent events for a user.

```bash
curl "http://localhost:8080/memory/alice?app_id=myapp&limit=5"
```

```json
{
  "user_id": "alice",
  "app_id": "myapp",
  "events": [
    {
      "event_id": "3f7a1b2c-...",
      "raw_text": "I prefer dark mode and use Neovim.",
      "importance_score": 0.72,
      "consolidated": false,
      "created_at": "2024-06-01T12:00:00+00:00"
    }
  ]
}
```

| Query param | Default | Description |
|---|---|---|
| `app_id` | `"default"` | Application namespace |
| `limit` | `10` | Events to return (1–50) |

---

## Python SDK

Install the package (the SDK is included):

```bash
pip install smritikosh          # or pip install -e . from the repo
```

### Basic usage

```python
import asyncio
from smritikosh.sdk import SmritikoshClient

async def main():
    async with SmritikoshClient(base_url="http://localhost:8080", app_id="myapp") as client:

        # 1. Store a memory
        event = await client.encode(
            user_id="alice",
            content="I'm building an AI startup and prefer concise answers.",
            metadata={"source": "onboarding"},
        )
        print(f"Stored event {event.event_id}, importance={event.importance_score:.2f}")

        # 2. Build context before an LLM call
        ctx = await client.build_context(
            user_id="alice",
            query="What does Alice prefer?",
        )
        if not ctx.is_empty():
            # ctx.messages is OpenAI-style — prepend to your LLM call
            print(ctx.context_text)

        # 3. Browse recent events
        events = await client.get_recent(user_id="alice", limit=5)
        for e in events:
            print(f"[{e.created_at}] {e.raw_text[:60]}")

        # 4. Check server health
        status = await client.health()
        print(f"Server status: {status.status}")

asyncio.run(main())
```

### Multi-tenant / multi-app isolation

Use `app_id` to isolate memory between different applications or tenants sharing one server:

```python
# Two apps, same user — memories are fully isolated
chat_client = SmritikoshClient(base_url="...", app_id="chat-app")
docs_client = SmritikoshClient(base_url="...", app_id="docs-app")

# Override per-call
await client.encode(user_id="alice", content="...", app_id="special-context")
```

### Error handling

```python
from smritikosh.sdk import SmritikoshClient
from smritikosh.sdk.client import SmritikoshError

async with SmritikoshClient(base_url="http://localhost:8080") as client:
    try:
        ctx = await client.build_context(user_id="alice", query="...")
    except SmritikoshError as e:
        print(f"API error {e.status_code}: {e.detail}")
```

### SDK reference

| Method | Description |
|---|---|
| `encode(user_id, content, *, app_id, metadata)` | Store a memory event → `EncodedEvent` |
| `build_context(user_id, query, *, app_id)` | Retrieve LLM-ready context → `MemoryContext` |
| `get_recent(user_id, *, app_id, limit)` | List recent events → `list[RecentEvent]` |
| `health()` | Server liveness check → `HealthStatus` |

---

## Testing

### Run all unit tests (no external dependencies)

```bash
pytest
```

The default run executes **256 tests** in about 6 seconds. All tests that require real API keys, a local Ollama server, or running databases are automatically skipped.

```
256 passed, 28 skipped in 5.95s
```

### Run with coverage report

Coverage is on by default (`--cov=smritikosh` in `pyproject.toml`):

```bash
pytest                              # shows term-missing coverage table
pytest --cov-report=html            # generates htmlcov/index.html
```

### Test marks

Tests are organised into three opt-in groups:

| Mark | Requires | Run with |
|---|---|---|
| `live` | Real API keys in `.env` (Anthropic / OpenAI / Gemini) | `pytest -m live` |
| `ollama` | Local Ollama server (`ollama serve`) | `pytest -m ollama` |
| `db` | Running Docker databases (`docker compose up -d`) | `pytest -m db` |

Examples:

```bash
# Only unit tests (default)
pytest

# Unit tests + DB integration tests
pytest -m db

# Everything including live LLM calls
pytest -m "live or ollama or db"

# One specific file
pytest tests/test_hippocampus.py -v

# One specific test
pytest tests/test_amygdala.py::TestAmygdala::test_scores_decision_text -v
```

### Test suite overview

| File | Tests | What it covers |
|---|---|---|
| `test_llm_adapter.py` | 22 | Model resolution, complete(), embed(), extract_structured(), retry logic |
| `test_db_models.py` | 18 | ORM field types, StrEnum, cascade delete, vector roundtrip |
| `test_episodic_memory.py` | 28 | store, search, hybrid_search, HybridWeights validation |
| `test_semantic_memory.py` | 37 | upsert_fact, get_user_profile, UserProfile.as_text_summary() |
| `test_amygdala.py` | 19 | All scoring rules, boosts, penalties, clamp behaviour |
| `test_hippocampus.py` | 16 | Parallel LLM calls, embedding failure, extraction failure |
| `test_context_builder.py` | 28 | Deduplication, degraded-mode fallbacks, prompt rendering |
| `test_api.py` | 24 | All HTTP routes via httpx test client + dependency overrides |
| `test_consolidator.py` | 20 | Batch splitting, LLM failures, fact upserts, skip guard |
| `test_synaptic_pruner.py` | 22 | Score formula, pruning logic, threshold sensitivity |
| `test_scheduler.py` | 14 | Job registration, manual triggers, error recovery |
| `test_sdk_client.py` | 28 | HTTP mocking via respx, error handling, type checks |

---

## LLM provider guide

Smritikosh uses [LiteLLM](https://docs.litellm.ai) under the hood, so switching providers is a `.env` change.

### Claude (Anthropic)

```dotenv
LLM_PROVIDER=claude
LLM_MODEL=claude-haiku-4-5-20251001
LLM_API_KEY=sk-ant-...

EMBEDDING_PROVIDER=openai          # Anthropic has no embedding API — use OpenAI
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...
```

### OpenAI

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...
```

### Gemini (Google)

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
LLM_API_KEY=AIza...

EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_API_KEY=AIza...
EMBEDDING_DIMENSIONS=768           # Gemini embeddings are 768-dimensional
```

> **Important:** if you change `EMBEDDING_DIMENSIONS`, run `alembic downgrade base && alembic upgrade head` to recreate the vector column at the new size.

### Ollama (local)

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama serve
```

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_DIMENSIONS=768
```

Run Ollama-specific tests:

```bash
pytest -m ollama
```

### vLLM

```dotenv
LLM_PROVIDER=vllm
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=none                   # vLLM doesn't require a key

EMBEDDING_PROVIDER=vllm
EMBEDDING_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_BASE_URL=http://localhost:8000/v1
EMBEDDING_DIMENSIONS=3584          # match your model's output dimension
```

---

## Background jobs

The `MemoryScheduler` runs two jobs inside the FastAPI process using APScheduler:

### Consolidation (every hour)

Finds users with ≥ 5 unconsolidated events from the last 24 hours and compresses them:

```
10 raw events  →  1 consolidated event  +  N distilled Neo4j facts
```

The LLM extracts a summary and structured facts (`category`, `key`, `value`, `confidence`). Raw events are marked `consolidated=True` in Postgres; facts are upserted into Neo4j (incrementing `frequency_count` on each re-encounter).

### Synaptic pruning (every 24 hours)

Scores consolidated events older than 7 days:

```
prune_score = importance_score × exp(−age_days / 30)
```

Events scoring below `0.15` are deleted. High-importance or recently-accessed memories are preserved.

### Manual triggers (admin / testing)

```python
from smritikosh.processing.scheduler import MemoryScheduler

# Trigger immediately for one user
await scheduler.run_consolidation_now(user_id="alice", app_id="myapp")
await scheduler.run_pruning_now(user_id="alice", app_id="myapp")

# Run batch across all users
await scheduler.run_consolidation_for_all_users()
await scheduler.run_pruning_for_all_users()
```

### Tune the schedule

Pass custom intervals when constructing the scheduler (or subclass `MemoryScheduler`):

```python
MemoryScheduler(
    consolidator=..., pruner=..., episodic=...,
    consolidation_hours=2,   # consolidate every 2 hours
    pruning_hours=48,        # prune every 2 days
)
```
