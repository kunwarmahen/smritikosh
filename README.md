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
- [Database setup](#database-setup)
  - [PostgreSQL + pgvector](#postgresql--pgvector)
  - [Neo4j](#neo4j)
  - [MongoDB (audit trail)](#mongodb-audit-trail)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the server](#running-the-server)
- [API reference](#api-reference)
  - [Memory](#memory)
  - [Context](#context)
  - [Identity](#identity)
  - [Feedback](#feedback)
  - [Procedural memory](#procedural-memory-api)
  - [Admin jobs](#admin-jobs)
  - [External ingest](#external-ingest)
  - [Audit trail](#audit-trail-api)
- [Audit trail](#audit-trail)
- [Python SDK](#python-sdk)
- [Node.js SDK](#nodejs-sdk)
- [Testing](#testing)
- [LLM provider guide](#llm-provider-guide)
- [Background jobs](#background-jobs)

---

## How it works

### Intake pipeline

```
External sources                    Direct API call
──────────────────────────────────  ─────────────────
  File / Webhook / Slack            POST /memory/event
  IMAP email / iCalendar                   │
        │                                  │
        ▼                                  │
  SourceConnector                          │
  (normalise → ConnectorEvent)             │
        │                                  │
        └─────────────┬────────────────────┘
                      ▼
        ┌─────────────────────────────────────────────┐
        │  Hippocampus  (intake coordinator)          │
        │                                             │
        │  1. Amygdala  — scores emotional importance │
        │  2. Embed text + extract facts  (parallel)  │
        │  3. Store event  → PostgreSQL + pgvector    │
        │  4. Upsert facts → Neo4j knowledge graph    │
        └─────────────────────────────────────────────┘
                  │                    │
                  ▼                    ▼
          EpisodicMemory          SemanticMemory
          (raw events +           (stable facts:
           vectors)                preferences,
                                   skills, goals…)
```

### Background jobs

```
EpisodicMemory
      │
      │  (scheduled / POST /admin/*)
      ├──────────────────────────────────────────────────┐
      ▼                                                  │
Consolidator          MemoryClusterer    BeliefMiner     │
raw → summary         groups similar     infers values   │
+ Neo4j facts         events by topic    & beliefs       │
      │                                                  │
      ▼                                                  │
SynapticPruner        ReconsolidationEngine ◄────────────┘
deletes low-value     re-summarises events
memories              after recall
```

### Context retrieval

```
Query + ProceduralMemory lookup
      │
      ▼
ContextBuilder
      │
      ├── hybrid_search()      (vector + recency + importance)
      ├── get_user_profile()   (Neo4j semantic facts)
      ├── get_recent()         (last N raw events)
      └── search_by_query()    (trigger→instruction rules)
                │
                ▼
        MemoryContext.messages  →  prepend to LLM messages
```

### Identity model

```
GET /identity/{user_id}
      │
      ├── IdentityBuilder  →  groups facts into dimensions
      ├── BeliefMiner      →  fetches inferred beliefs
      └── LLM              →  generates narrative summary
                │
                ▼
        IdentityProfile  (dimensions + beliefs + summary)
```

### SDK surface

```
Your application
      │
      ├── SmritikoshClient (Python)   smritikosh.sdk
      └── SmritikoshClient (Node.js)  sdk-node/
                │
                ▼  REST API (FastAPI)
        ┌───────────────────────────────┐
        │  /memory   /context           │
        │  /identity /feedback          │
        │  /procedures                  │
        │  /ingest/{push,file,slack,…}  │
        │  /admin/{consolidate,prune,…} │
        └───────────────────────────────┘
```

---

## Architecture

| Component | Role | Storage |
|---|---|---|
| **Amygdala** | Scores importance of each event (0.1 – 1.0) | — |
| **EpisodicMemory** | Stores raw events; hybrid search over vectors | PostgreSQL + pgvector |
| **SemanticMemory** | Distilled facts organised in a knowledge graph | Neo4j |
| **Hippocampus** | Orchestrates intake: score → embed → extract → store | — |
| **NarrativeMemory** | Tracks causal/temporal links between events (memory chains) | PostgreSQL |
| **ContextBuilder** | Retrieves relevant context before an LLM call | — |
| **Consolidator** | Background: compresses events into summaries + Neo4j facts | — |
| **SynapticPruner** | Background: deletes old low-scoring events | — |
| **MemoryClusterer** | Background: groups similar events by topic using embeddings | PostgreSQL |
| **BeliefMiner** | Background: infers durable beliefs and values from event patterns | PostgreSQL |
| **IdentityBuilder** | Synthesises semantic facts + beliefs into a user identity model | — |
| **ReinforcementLoop** | Adjusts event importance scores based on user feedback signals | PostgreSQL |
| **ProceduralMemory** | Stores trigger→instruction rules; fuzzy-matched against each query | PostgreSQL |
| **ReconsolidationEngine** | Re-summarises events after recall to incorporate new context | PostgreSQL |
| **SourceConnector** | Normalises external sources (file, webhook, Slack, email, calendar) into events | — |
| **MemoryScheduler** | Runs all background jobs on configurable timers (APScheduler) | — |
| **LLMAdapter** | Unified interface to Claude, OpenAI, Gemini, Ollama, vLLM | — |
| **SmritikoshClient (Python)** | Python SDK wrapping the REST API | — |
| **SmritikoshClient (Node.js)** | TypeScript/ESM SDK with identical surface to the Python SDK | — |

---

## Project structure

```
smritikosh/
├── api/
│   ├── deps.py              # FastAPI dependency injection (@lru_cache singletons)
│   ├── main.py              # App factory + lifespan (startup/shutdown)
│   ├── schemas.py           # Pydantic request/response models
│   └── routes/
│       ├── health.py        # GET /health
│       ├── memory.py        # POST /memory/event, GET /memory/{user_id},
│       │                    #   DELETE /memory/event/{id}, DELETE /memory/user/{id}
│       ├── context.py       # POST /context
│       ├── identity.py      # GET /identity/{user_id}
│       ├── feedback.py      # POST /feedback
│       ├── procedures.py    # CRUD /procedures + DELETE /procedures/user/{id}
│       ├── admin.py         # POST /admin/{consolidate,prune,cluster,mine-beliefs,reconsolidate}
│       └── ingest.py        # POST /ingest/{push,file,slack/events,email/sync,calendar}
├── config.py                # Pydantic Settings (reads .env)
├── connectors/
│   ├── __init__.py          # Re-exports ConnectorEvent, SourceConnector
│   ├── base.py              # ConnectorEvent dataclass + SourceConnector ABC
│   ├── file.py              # FileConnector: txt/md/csv/json → events
│   ├── webhook.py           # WebhookConnector: arbitrary JSON payload → events
│   ├── slack.py             # SlackConnector: Events API + HMAC verification
│   ├── email.py             # EmailConnector: IMAP fetch (runs in thread executor)
│   └── calendar.py          # CalendarConnector: RFC 5545 iCal stdlib parser
├── db/
│   ├── models.py            # SQLAlchemy 2.0 ORM: Event, UserFact, MemoryLink,
│   │                        #   MemoryFeedback, UserBelief, UserProcedure
│   ├── postgres.py          # Async engine, session helpers
│   └── neo4j.py             # Driver singleton, session helpers, schema init
├── llm/
│   └── adapter.py           # LLMAdapter: complete(), embed(), extract_structured()
├── memory/
│   ├── episodic.py          # EpisodicMemory: store, search, hybrid_search
│   ├── semantic.py          # SemanticMemory: upsert_fact, get_user_profile
│   ├── narrative.py         # NarrativeMemory: memory link chains
│   ├── identity.py          # IdentityBuilder: dimensions + beliefs + summary
│   ├── procedural.py        # ProceduralMemory: store, search_by_query (3-strategy fuzzy match)
│   └── hippocampus.py       # Hippocampus: encode()
├── processing/
│   ├── amygdala.py          # Importance scoring (pure, no LLM)
│   ├── consolidator.py      # Batch compress events → summaries + Neo4j facts
│   ├── synaptic_pruner.py   # Delete low-value memories
│   ├── memory_clusterer.py  # Cluster events by topic using embeddings
│   ├── belief_miner.py      # Infer beliefs/values from consolidated events
│   ├── reinforcement.py     # Adjust importance scores from user feedback
│   ├── reconsolidation.py   # Re-summarise events after recall
│   └── scheduler.py         # APScheduler background jobs
├── retrieval/
│   └── context_builder.py   # Build memory context for LLM calls
├── audit/
│   ├── __init__.py          # Re-exports AuditLogger, AuditEvent, EventType
│   ├── logger.py            # AuditLogger: emit(), get_timeline(), get_event_lineage(), get_stats()
│   └── mongodb.py           # Motor connection, lazy init, index creation
└── sdk/
    ├── client.py            # SmritikoshClient (async HTTP)
    └── types.py             # EncodedEvent, MemoryContext, RecentEvent,
                             #   IdentityProfile, FeedbackRecord, HealthStatus

sdk-node/                    # TypeScript / Node.js SDK
├── src/
│   ├── client.ts            # SmritikoshClient (native fetch, ESM)
│   ├── types.ts             # Branded types, request/response shapes
│   ├── errors.ts            # SmritikoshError
│   └── client.test.ts       # 41 Vitest tests (all methods, error paths)
├── package.json
└── tsconfig*.json

tests/
├── conftest.py              # pytest marks: live, ollama, db
├── test_llm_adapter.py
├── test_db_models.py
├── test_episodic_memory.py
├── test_semantic_memory.py
├── test_amygdala.py
├── test_hippocampus.py
├── test_narrative_memory.py
├── test_context_builder.py
├── test_consolidator.py
├── test_synaptic_pruner.py
├── test_scheduler.py
├── test_identity.py
├── test_memory_clusterer.py
├── test_reinforcement.py
├── test_belief_miner.py
├── test_procedural_memory.py
├── test_reconsolidation.py
├── test_connectors.py
├── test_api.py
├── test_api_procedures.py
├── test_api_admin.py
└── test_sdk_client.py

alembic/
└── versions/
    ├── 0001_initial_schema.py        # events, user_facts, memory_links + IVFFlat index
    ├── 0002_narrative_links.py       # memory_links relation types
    ├── 0003_add_cluster_fields.py    # cluster_id, cluster_label on events
    ├── 0004_add_memory_feedback.py   # memory_feedback table
    ├── 0005_add_user_beliefs.py      # user_beliefs table
    ├── 0006_add_user_procedures.py   # user_procedures table + priority/active indexes
    └── 0007_add_reconsolidation_fields.py  # reconsolidation_count, last_reconsolidated_at
```

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | StrEnum, `match`, type syntax |
| Docker + Compose | any recent | PostgreSQL + Neo4j + MongoDB (recommended) |
| An LLM API key | — | Claude / OpenAI / Gemini (or Ollama locally) |
| MongoDB 6+ | optional | Audit trail / provenance log (disabled if `MONGODB_URL` unset) |

---

## Database setup

Smritikosh uses three databases:

| Database | Purpose | Why |
|---|---|---|
| **PostgreSQL + pgvector** | Episodic memory store (events + vector embeddings) | ACID guarantees, hybrid SQL + vector search in one query, no extra infrastructure |
| **Neo4j** | Semantic memory (knowledge graph of user facts) | Native graph traversal for fact relationships, Cypher MERGE for upsert-with-reinforcement |
| **MongoDB** *(optional)* | Audit trail (provenance log of every pipeline step) | Schema-flexible documents, independent I/O path so audit never blocks the main pipeline |

### PostgreSQL + pgvector

#### Option A — Docker (recommended)

The `docker-compose.yml` uses the official `pgvector/pgvector:pg17` image which ships with the extension pre-installed. No manual extension setup needed.

```bash
docker compose up -d postgres
```

Verify it is healthy:

```bash
docker compose ps postgres
# postgres   running (healthy)
```

#### Option B — Existing PostgreSQL instance

You need PostgreSQL ≥ 13 and the `pgvector` extension.

**Install the extension on the server:**

```bash
# Ubuntu / Debian
sudo apt install postgresql-17-pgvector

# macOS (Homebrew)
brew install pgvector

# From source (any platform)
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

**Create the database and user:**

```sql
-- run as a superuser (e.g. psql -U postgres)
CREATE USER smritikosh WITH PASSWORD 'smritikosh';
CREATE DATABASE smritikosh OWNER smritikosh;

-- connect to the new database and enable the extension
\c smritikosh
CREATE EXTENSION IF NOT EXISTS vector;
```

**Update `.env`:**

```dotenv
POSTGRES_URL=postgresql+asyncpg://smritikosh:smritikosh@localhost:5432/smritikosh
```

#### Apply the schema

Regardless of which option you chose, run Alembic migrations to create tables and the IVFFlat vector index:

```bash
alembic upgrade head
```

What the migration does:
- Enables the `pgvector` extension (`CREATE EXTENSION IF NOT EXISTS vector`)
- Creates the `events` table with a `vector(1536)` embedding column
- Adds an IVFFlat index on `embedding` for fast cosine-distance search
- Creates `user_facts` and `memory_links` tables

> **Changing embedding dimensions?** If you switch to a model with different output dimensions (e.g. Gemini's 768), set `EMBEDDING_DIMENSIONS=768` in `.env`, then re-run migrations: `alembic downgrade base && alembic upgrade head`.

#### Verify

```bash
psql postgresql://smritikosh:smritikosh@localhost:5432/smritikosh \
  -c "\dx vector"           # should show pgvector version
  -c "\d events"            # should show the embedding column
```

---

### Neo4j

#### Option A — Docker (recommended)

```bash
docker compose up -d neo4j
```

Neo4j takes ~15 seconds to initialise. Check it is ready:

```bash
docker compose ps neo4j
# neo4j   running (healthy)

# Or open the browser interface:
open http://localhost:7474   # login: neo4j / smritikosh
```

#### Option B — Neo4j Desktop

1. Download [Neo4j Desktop](https://neo4j.com/download/) and install it.
2. Create a new project → **Add** → **Local DBMS**.
3. Set the password to `smritikosh` (or update `NEO4J_PASSWORD` in `.env`).
4. Start the DBMS.
5. Install the **APOC** plugin from the **Plugins** tab (optional but recommended).

#### Option C — Neo4j AuraDB (cloud)

1. Create a free instance at [console.neo4j.io](https://console.neo4j.io).
2. Copy the connection URI and credentials into `.env`:

```dotenv
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-aura-password>
```

#### Schema initialisation

Smritikosh automatically applies Neo4j constraints and indexes on startup — no manual Cypher needed. On first boot the server runs:

```cypher
-- Uniqueness constraints
CREATE CONSTRAINT user_unique IF NOT EXISTS
  FOR (u:User) REQUIRE (u.user_id, u.app_id) IS UNIQUE;

CREATE CONSTRAINT fact_unique IF NOT EXISTS
  FOR (f:Fact) REQUIRE (f.category, f.key, f.value) IS UNIQUE;

-- Lookup indexes
CREATE INDEX user_lookup IF NOT EXISTS FOR (u:User) ON (u.user_id);
CREATE INDEX fact_category IF NOT EXISTS FOR (f:Fact) ON (f.category);
```

#### Verify

Open the Neo4j browser at `http://localhost:7474` and run:

```cypher
SHOW CONSTRAINTS;
SHOW INDEXES;
```

Both should list the Smritikosh constraints after the server has started at least once.

---

### MongoDB (audit trail)

MongoDB is **fully optional**. If `MONGODB_URL` is not set, the audit system is disabled and all pipeline components operate identically — the only difference is that no provenance records are written.

#### Option A — Docker (recommended)

The `docker-compose.yml` includes a MongoDB 7 service with a `mongosh`-based healthcheck:

```bash
docker compose up -d mongo
```

Verify it is healthy:

```bash
docker compose ps mongo
# mongo   running (healthy)
```

The container uses the default port `27017` with no authentication (development only). For production, add auth via `MONGO_INITDB_ROOT_USERNAME` / `MONGO_INITDB_ROOT_PASSWORD` and update the connection string.

#### Option B — MongoDB Atlas (cloud)

1. Create a free cluster at [cloud.mongodb.com](https://cloud.mongodb.com).
2. Create a database user and allowlist your IP.
3. Copy the connection string into `.env`:

```dotenv
MONGODB_URL=mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=smritikosh_audit
```

#### Option C — Existing MongoDB instance

```dotenv
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=smritikosh_audit
```

#### Schema initialisation

Smritikosh automatically creates the `audit_events` collection and its indexes on startup — no manual setup needed. The indexes created are:

| Index | Purpose |
|---|---|
| `user_id + app_id + timestamp` (compound) | Timeline queries per user |
| `event_type + timestamp` | Filter by pipeline stage |
| `event_id` | Provenance chain lookups |
| `session_id` | Group all records from one pipeline run |

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
- **MongoDB 7** on port `27017` (audit trail — optional, safe to omit)

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
| `SLACK_SIGNING_SECRET` | — | Signing secret for Slack Events API signature verification (required only for `POST /ingest/slack/events`) |
| `MONGODB_URL` | — | MongoDB connection string. If unset, audit trail is disabled (no-op) |
| `MONGODB_DB_NAME` | `smritikosh_audit` | MongoDB database to store audit events in |

---

## Running the server

```bash
uvicorn smritikosh.api.main:app --reload --port 8080
```

On startup the server will:
1. Enable the `pgvector` extension and create tables (if not already present via Alembic)
2. Apply Neo4j schema constraints and indexes
3. Create MongoDB `audit_events` collection and indexes (if `MONGODB_URL` is configured)
4. Start background scheduler (consolidation every hour, pruning every 24 hours)

Interactive API docs are available at `http://localhost:8080/docs`.

---

## API reference

### `GET /health`

Checks server liveness **and** database connectivity. Useful for container readiness probes.

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "postgres": "ok",
  "neo4j": "ok"
}
```

| `status` value | Meaning |
|---|---|
| `"ok"` | Server running, both databases reachable |
| `"degraded"` | Server running, but one or both databases unreachable |
| `"error"` | Server internal error |

The `postgres` and `neo4j` fields each carry `"ok"` or `"error"` independently so you can tell which dependency is down.

---

### `POST /memory/search`

Hybrid search over a user's episodic memory. Returns raw scored events with score breakdown — useful for building custom memory UIs or your own ranking logic.

Unlike `/context`, this endpoint does not inject semantic facts from Neo4j or procedural rules; it returns event rows with their full score breakdown.

```bash
curl -X POST http://localhost:8080/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "query": "What editor does Alice use?",
    "app_id": "myapp",
    "limit": 5
  }'
```

```json
{
  "user_id": "alice",
  "query": "What editor does Alice use?",
  "results": [
    {
      "event_id": "3f7a1b2c-...",
      "raw_text": "I prefer dark mode and use Neovim as my editor.",
      "importance_score": 0.72,
      "hybrid_score": 0.8341,
      "similarity_score": 0.9102,
      "recency_score": 0.6712,
      "consolidated": false,
      "created_at": "2024-06-01T12:00:00+00:00"
    }
  ],
  "total": 1,
  "embedding_failed": false
}
```

| Field | Type | Description |
|---|---|---|
| `user_id` | string | **Required.** User to search |
| `query` | string | **Required.** Natural-language search query |
| `app_id` | string | Application namespace (default: `"default"`) |
| `limit` | int | Maximum results (1–50, default: 10) |
| `from_date` | datetime | Only include events on or after this datetime (ISO 8601) |
| `to_date` | datetime | Only include events on or before this datetime (ISO 8601) |

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

### `GET /identity/{user_id}`

Retrieve the synthesized identity model for a user. Aggregates semantic facts from Neo4j into per-category dimensions, generates a narrative summary via LLM, and includes any inferred beliefs from the `user_beliefs` table.

```bash
curl "http://localhost:8080/identity/alice?app_id=myapp"
```

```json
{
  "user_id": "alice",
  "app_id": "myapp",
  "summary": "Alice is an AI entrepreneur who values speed and iterative development...",
  "dimensions": [
    {
      "category": "role",
      "dominant_value": "founder",
      "confidence": 0.95,
      "fact_count": 3
    }
  ],
  "beliefs": [
    {
      "statement": "values iterative development over big-bang launches",
      "category": "value",
      "confidence": 0.88,
      "evidence_count": 4
    }
  ],
  "total_facts": 12,
  "computed_at": "2026-03-15T10:00:00+00:00",
  "is_empty": false
}
```

| Query param | Default | Description |
|---|---|---|
| `app_id` | `"default"` | Application namespace |

---

### `POST /feedback`

Submit feedback on a recalled memory event. Immediately adjusts the event's `importance_score`, influencing future hybrid search rankings.

```bash
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "3f7a1b2c-...",
    "user_id": "alice",
    "feedback_type": "positive",
    "comment": "Exactly what I was looking for"
  }'
```

```json
{
  "feedback_id": "9a2c4e1f-...",
  "event_id": "3f7a1b2c-...",
  "new_importance_score": 0.82
}
```

| Field | Type | Description |
|---|---|---|
| `event_id` | string | **Required.** UUID of the recalled event |
| `user_id` | string | **Required.** User submitting the feedback |
| `feedback_type` | string | **Required.** `"positive"`, `"negative"`, or `"neutral"` |
| `app_id` | string | Application namespace (default: `"default"`) |
| `comment` | string | Optional free-text note |

Score adjustments: `positive` → +0.10, `negative` → −0.10, `neutral` → no change (signal is stored but score is unchanged). Score is always clamped to [0.0, 1.0].

---

### `DELETE /memory/event/{event_id}`

Delete a single event from episodic memory (PostgreSQL only — Neo4j facts distilled from this event are retained).

```bash
curl -X DELETE "http://localhost:8080/memory/event/3f7a1b2c-..."
```

```json
{"deleted": true}
```

---

### `DELETE /memory/user/{user_id}`

Delete **all** episodic events for a user. Useful for GDPR/right-to-erasure workflows.

```bash
curl -X DELETE "http://localhost:8080/memory/user/alice?app_id=myapp"
```

```json
{"deleted_count": 42}
```

| Query param | Default | Description |
|---|---|---|
| `app_id` | `"default"` | Scope deletion to one application namespace |

---

## Procedural memory API

Procedural memories are persistent **trigger → instruction** rules attached to a user. On every `/context` call they are fuzzy-matched against the current query and injected into the context block, so the LLM automatically adjusts its behaviour without the application needing to track this separately.

Matching uses three escalating strategies:
1. **Substring match** — trigger phrase appears inside the query → score 1.0
2. **Token match** — any query token (> 3 chars) appears inside the trigger → score 0.5
3. **Jaccard overlap** — token overlap between query and trigger ≥ threshold → score varies

### `POST /procedures`

```bash
curl -X POST http://localhost:8080/procedures \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "trigger": "LLM deployment",
    "instruction": "Always mention GPU memory requirements and batching strategies.",
    "priority": 8,
    "category": "topic_response"
  }'
```

```json
{
  "procedure_id": "7c3d9f...",
  "user_id": "alice",
  "trigger": "LLM deployment",
  "instruction": "Always mention GPU memory requirements...",
  "priority": 8,
  "category": "topic_response",
  "is_active": true,
  "hit_count": 0
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `user_id` | string | **Required** | User this rule applies to |
| `trigger` | string | **Required** | Topic/phrase that activates this rule |
| `instruction` | string | **Required** | Instruction injected into the system prompt |
| `app_id` | string | `"default"` | Application namespace |
| `priority` | int | `5` | 1–10, higher = appears first in context |
| `category` | string | `"topic_response"` | Free-form label (e.g. `communication`, `format`) |

### `GET /procedures/{user_id}`

```bash
curl "http://localhost:8080/procedures/alice?app_id=myapp&active_only=true"
```

Returns a JSON array of the user's procedures ordered by priority descending.

### `PATCH /procedures/{procedure_id}`

Update any field on an existing procedure. Useful for deactivating (`is_active: false`) or adjusting priority.

```bash
curl -X PATCH "http://localhost:8080/procedures/7c3d9f..." \
  -H "Content-Type: application/json" \
  -d '{"priority": 10, "is_active": true}'
```

### `DELETE /procedures/{procedure_id}`

```bash
curl -X DELETE "http://localhost:8080/procedures/7c3d9f..."
```

```json
{"deleted": true}
```

### `DELETE /procedures/user/{user_id}`

Delete all procedures for a user.

```bash
curl -X DELETE "http://localhost:8080/procedures/user/alice?app_id=myapp"
```

```json
{"deleted_count": 5}
```

---

## Admin jobs

Manual triggers for background jobs. Useful during development, testing, or forced re-processing.

### `POST /admin/consolidate`

Trigger consolidation immediately for one user.

```bash
curl -X POST "http://localhost:8080/admin/consolidate?user_id=alice&app_id=myapp"
```

### `POST /admin/prune`

Trigger synaptic pruning immediately.

```bash
curl -X POST "http://localhost:8080/admin/prune?user_id=alice&app_id=myapp"
```

### `POST /admin/cluster`

Trigger memory clustering immediately.

```bash
curl -X POST "http://localhost:8080/admin/cluster?user_id=alice&app_id=myapp"
```

### `POST /admin/mine-beliefs`

Trigger belief mining immediately.

```bash
curl -X POST "http://localhost:8080/admin/mine-beliefs?user_id=alice&app_id=myapp"
```

### `POST /admin/reconsolidate`

Re-summarise a single event after it has been recalled, blending its existing summary with new context.

```bash
curl -X POST http://localhost:8080/admin/reconsolidate \
  -H "Content-Type: application/json" \
  -d '{"event_id": "3f7a1b2c-...", "new_context": "Alice now works at a larger team."}'
```

```json
{
  "event_id": "3f7a1b2c-...",
  "updated": true,
  "new_summary": "Alice is a founder who recently expanded her team...",
  "reconsolidation_count": 1
}
```

All admin routes return `503 Service Unavailable` if the background scheduler has not been started (e.g. bare ASGI without `lifespan`).

---

## External ingest

Smritikosh can ingest memories from external sources. All five endpoints share the same pipeline: `ConnectorEvent` objects are normalised → run through Hippocampus → stored in PostgreSQL/Neo4j.

### `POST /ingest/push`

Push a single event programmatically (webhook or backend service).

```bash
curl -X POST http://localhost:8080/ingest/push \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "content": "Alice merged PR #412 fixing the vector index.",
    "source": "github",
    "source_id": "pr-412",
    "metadata": {"repo": "smritikosh"}
  }'
```

```json
{"source": "github", "events_ingested": 1, "events_failed": 0, "event_ids": ["3f7a..."]}
```

### `POST /ingest/file`

Upload a file. Supported formats: `.txt`, `.md` (paragraph chunks), `.csv` (one event per row), `.json` (array of strings or objects).

```bash
curl -X POST http://localhost:8080/ingest/file \
  -F "user_id=alice" \
  -F "file=@notes.md"
```

```json
{"source": "file:notes.md", "events_ingested": 7, "events_failed": 0, "event_ids": [...]}
```

### `POST /ingest/slack/events`

Receive events from the [Slack Events API](https://api.slack.com/apis/events-api). Register this URL as your Slack app's event subscription endpoint.

Smritikosh handles the `url_verification` challenge automatically. For production, set `SLACK_SIGNING_SECRET` to verify request signatures.

```dotenv
SLACK_SIGNING_SECRET=your_slack_signing_secret_here
```

Supported event types: `message`, `app_mention`, `message.im`. Bot messages are filtered by default.

The `user_id` and `app_id` to store events under are passed as **query parameters** (since Slack controls the request body):

```
POST /ingest/slack/events?user_id=alice&app_id=myapp
```

### `POST /ingest/email/sync`

Fetch unread emails from an IMAP mailbox and ingest them as episodic memories. Credentials are used per-request and never stored.

```bash
curl -X POST http://localhost:8080/ingest/email/sync \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "host": "imap.gmail.com",
    "port": 993,
    "username": "alice@example.com",
    "password": "app-password",
    "mailbox": "INBOX",
    "limit": 20,
    "unseen_only": true
  }'
```

```json
{"source": "email:imap.gmail.com", "events_ingested": 5, "events_failed": 0, "event_ids": [...]}
```

### `POST /ingest/calendar`

Upload an `.ics` file (RFC 5545 iCalendar). Each `VEVENT` becomes one memory event containing the summary, description, location, and time range. Parsed using the stdlib — no extra dependencies.

```bash
curl -X POST http://localhost:8080/ingest/calendar \
  -F "user_id=alice" \
  -F "file=@calendar.ics"
```

```json
{"source": "calendar:calendar.ics", "events_ingested": 12, "events_failed": 0, "event_ids": [...]}
```

---

## Audit trail API

These endpoints are available only when `MONGODB_URL` is configured. All return `503 Service Unavailable` if MongoDB is not set up.

### `GET /audit/{user_id}`

Returns the full chronological audit timeline for a user — every pipeline step that touched their data, across all event types.

```bash
curl "http://localhost:8080/audit/alice?app_id=myapp&limit=20"
```

```json
[
  {
    "id": "a1b2c3d4-...",
    "event_type": "memory.encoded",
    "user_id": "alice",
    "app_id": "myapp",
    "event_id": "3f7a1b2c-...",
    "session_id": "9e8d7c6b-...",
    "timestamp": "2026-03-16T10:00:00+00:00",
    "payload": {
      "raw_text_preview": "I prefer dark mode and use Neovim.",
      "importance_score": 0.72,
      "embedding_success": true,
      "extraction_failed": false
    }
  },
  {
    "id": "b2c3d4e5-...",
    "event_type": "memory.facts_extracted",
    "user_id": "alice",
    "app_id": "myapp",
    "event_id": "3f7a1b2c-...",
    "session_id": "9e8d7c6b-...",
    "timestamp": "2026-03-16T10:00:00+00:00",
    "payload": {
      "facts_count": 2,
      "facts": [
        {"category": "preference", "key": "theme", "value": "dark mode", "confidence": 0.9},
        {"category": "skill",      "key": "editor", "value": "Neovim",    "confidence": 0.95}
      ]
    }
  }
]
```

| Query param | Default | Description |
|---|---|---|
| `app_id` | `"default"` | Application namespace |
| `event_type` | — | Filter to one event type (e.g. `memory.encoded`) |
| `limit` | `50` | Maximum records to return |
| `offset` | `0` | Pagination offset |
| `from_ts` | — | Only events on or after this ISO 8601 timestamp |
| `to_ts` | — | Only events on or before this ISO 8601 timestamp |

---

### `GET /audit/event/{event_id}/lineage`

Returns the complete provenance chain for a single episodic event — every audit record associated with that event's UUID, in chronological order.

```bash
curl "http://localhost:8080/audit/event/3f7a1b2c-.../lineage"
```

```json
[
  {"event_type": "memory.encoded",         "timestamp": "2026-03-16T10:00:00+00:00", "payload": {...}},
  {"event_type": "memory.facts_extracted", "timestamp": "2026-03-16T10:00:00+00:00", "payload": {...}},
  {"event_type": "memory.consolidated",    "timestamp": "2026-03-16T11:00:00+00:00", "payload": {...}},
  {"event_type": "memory.reconsolidated",  "timestamp": "2026-03-16T14:23:00+00:00", "payload": {...}}
]
```

This is the primary "why was this memory stored / how did it change" endpoint — useful for debugging and building provenance UIs.

---

### `GET /audit/stats/{user_id}`

Returns per-event-type counts for a user. Useful for dashboards and monitoring.

```bash
curl "http://localhost:8080/audit/stats/alice?app_id=myapp"
```

```json
{
  "memory.encoded":         42,
  "memory.facts_extracted": 38,
  "memory.consolidated":    12,
  "memory.reconsolidated":   5,
  "memory.pruned":           3,
  "memory.clustered":        2,
  "belief.mined":            4,
  "feedback.submitted":      9,
  "context.built":          87,
  "search.performed":       31
}
```

---

## Audit trail

The audit trail captures a complete, immutable provenance record of every step the memory pipeline takes. Each record answers: *what happened, to which event, for which user, at what time, and what data was involved*.

### How it works

Every pipeline component emits a structured `AuditEvent` document to MongoDB after completing its work. Writes are **fire-and-forget** — they use `asyncio.create_task()` so MongoDB I/O never adds latency to the API response. Audit failures are logged as warnings and never raise exceptions to the caller.

```
POST /memory/event
      │
      ▼
  Hippocampus.encode()
      ├─► emit: memory.encoded          (importance score, embedding success, metadata)
      └─► emit: memory.facts_extracted  (facts list with categories + confidence)
                                                │
                                                ▼ (background scheduler)
                                     Consolidator._consolidate_batch()
                                             └─► emit: memory.consolidated
                                                        (event IDs, summary, facts distilled)

                                     ReconsolidationEngine._reconsolidate_one()
                                             └─► emit: memory.reconsolidated
                                                        (old summary, new summary, recall context)

                                     SynapticPruner.prune()
                                             └─► emit: memory.pruned
                                                        (importance, recall count, age, thresholds)

                                     MemoryClusterer.run()
                                             └─► emit: memory.clustered
                                                        (cluster labels, event counts per cluster)

                                     BeliefMiner.mine()
                                             └─► emit: belief.mined
                                                        (belief statements, categories, confidence)

POST /feedback
      └─► emit: feedback.submitted      (feedback type, new importance score)

POST /context
      └─► emit: context.built           (intent, memory counts, embedding status)

POST /memory/search
      └─► emit: search.performed        (query preview, results count, embedding status)
```

### Event types

| Event type | Emitted by | Key payload fields |
|---|---|---|
| `memory.encoded` | `Hippocampus.encode()` | `raw_text_preview`, `importance_score`, `embedding_success`, `extraction_failed` |
| `memory.facts_extracted` | `Hippocampus.encode()` | `facts` (list with category/key/value/confidence), `facts_count` |
| `memory.consolidated` | `Consolidator._consolidate_batch()` | `event_ids`, `summary`, `facts_distilled`, `links_created`, `facts` |
| `memory.reconsolidated` | `ReconsolidationEngine._reconsolidate_one()` | `old_summary`, `new_summary`, `recall_context`, `reconsolidation_count` |
| `memory.pruned` | `SynapticPruner.prune()` | `importance_score`, `recall_count`, `age_days`, `thresholds`, `raw_text_preview` |
| `memory.clustered` | `MemoryClusterer.run()` | `clusters_found`, `events_clustered`, `clusters` (label + event count per cluster) |
| `belief.mined` | `BeliefMiner.mine()` | `beliefs_found`, `beliefs_upserted`, `beliefs` (statement/category/confidence list) |
| `feedback.submitted` | `POST /feedback` route | `feedback_type`, `comment`, `new_importance_score` |
| `context.built` | `ContextBuilder.build()` | `query_preview`, `intent`, `similar_events_count`, `recent_events_count`, `facts_count` |
| `search.performed` | `POST /memory/search` route | `query_preview`, `results_count`, `embedding_failed`, `limit` |

### Session grouping

All audit records from a single `POST /memory/event` request share a `session_id` UUID. This lets you reconstruct the full intake run — embedding, extraction, and storage — from a single ID:

```bash
# Find all records from one intake session
curl "http://localhost:8080/audit/alice?session_id=9e8d7c6b-..."
```

### Enabling audit

1. Start MongoDB (Docker or external — see [MongoDB setup](#mongodb-audit-trail) above).
2. Add to `.env`:

```dotenv
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=smritikosh_audit   # optional, this is the default
```

3. Restart the server. On startup it logs:

```
INFO  smritikosh.audit.mongodb — audit indexes created on smritikosh_audit.audit_events
```

To **disable** the audit trail, remove `MONGODB_URL` from `.env`. All pipeline components fall back to no-ops — zero performance impact.

### Running MongoDB with Docker Compose

```bash
# Start MongoDB only
docker compose up -d mongo

# Start all services (Postgres + Neo4j + MongoDB)
docker compose up -d

# Check status
docker compose ps mongo
# mongo   running (healthy)

# Connect with mongosh (for inspection)
docker compose exec mongo mongosh smritikosh_audit
```

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

        # 4. Submit feedback on a recalled event
        record = await client.submit_feedback(
            event_id=events[0].event_id,
            user_id="alice",
            feedback_type="positive",
            comment="Very relevant",
        )
        print(f"New importance score: {record.new_importance_score:.2f}")

        # 5. Fetch the user's identity model
        identity = await client.get_identity(user_id="alice")
        print(identity.summary)
        for belief in identity.beliefs:
            print(f"  [{belief.category}] {belief.statement} ({belief.confidence:.0%})")

        # 6. Check server health
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
| `search(user_id, query, *, app_id, limit, from_date, to_date)` | Hybrid search → `SearchResult` with scored `SearchResultItem` list |
| `get_recent(user_id, *, app_id, limit)` | List recent events → `list[RecentEvent]` |
| `submit_feedback(event_id, user_id, feedback_type, *, app_id, comment)` | Rate a recalled event → `FeedbackRecord` |
| `get_identity(user_id, *, app_id)` | Fetch synthesized identity model → `IdentityProfile` |
| `delete_event(event_id)` | Delete a single episodic event |
| `delete_user_memory(user_id, *, app_id)` | Delete all events for a user |
| `store_procedure(user_id, trigger, instruction, *, ...)` | Create a procedural memory rule |
| `list_procedures(user_id, *, app_id, active_only)` | List a user's procedures |
| `delete_procedure(procedure_id)` | Delete a single procedure |
| `delete_user_procedures(user_id, *, app_id)` | Delete all procedures for a user |
| `reconsolidate(event_id, new_context)` | Re-summarise an event with new context |
| `ingest_push(user_id, content, *, source, source_id, app_id, metadata)` | Push a single event from an external source → `IngestResult` |
| `ingest_file(user_id, file_content, filename, *, app_id)` | Upload a file (txt/md/csv/json) → `IngestResult` |
| `ingest_email(user_id, host, username, password, *, ...)` | Fetch IMAP emails → `IngestResult` |
| `ingest_calendar(user_id, file_content, *, filename, app_id)` | Upload an `.ics` file → `IngestResult` |
| `health()` | Server + DB liveness check → `HealthStatus` |

---

## Node.js SDK

A native TypeScript SDK is available in `sdk-node/`. It targets Node.js ≥ 18 and uses the built-in `fetch` — no extra HTTP dependencies.

### Installation

```bash
cd sdk-node
npm install
npm run build          # emits dist/esm/ + dist/cjs/ + dist/types/
```

### Basic usage

```typescript
import { SmritikoshClient } from 'smritikosh';

const client = new SmritikoshClient({
  baseUrl: 'http://localhost:8080',
  appId: 'myapp',
});

// Store a memory
const event = await client.encode({
  userId: 'alice',
  content: "I prefer TypeScript over plain JavaScript for large projects.",
});
console.log(event.eventId, event.importanceScore);

// Build context before an LLM call
const ctx = await client.buildContext({
  userId: 'alice',
  query: 'What language does Alice prefer?',
});
if (!ctx.isEmpty()) {
  // ctx.messages is OpenAI-style — prepend to your messages array
  console.log(ctx.contextText);
}

// Browse recent events
const events = await client.getRecent({ userId: 'alice', limit: 5 });

// Submit feedback
await client.submitFeedback({
  eventId: events[0].eventId,
  userId: 'alice',
  feedbackType: 'positive',
});

// Procedural memory
await client.storeProcedure({
  userId: 'alice',
  trigger: 'code review',
  instruction: 'Always suggest adding tests first.',
  priority: 8,
});

// Admin
await client.adminConsolidate({ userId: 'alice' });
```

### Error handling

```typescript
import { SmritikoshError } from 'smritikosh';

try {
  await client.encode({ userId: 'alice', content: '...' });
} catch (err) {
  if (err instanceof SmritikoshError) {
    console.error(`API error ${err.status}: ${err.message}`);
  }
}
```

### Node.js SDK reference

| Method | Description |
|---|---|
| `encode(params)` | Store a memory → `EncodedEvent` |
| `buildContext(params)` | Retrieve LLM-ready context → `MemoryContext` |
| `search(params)` | Hybrid search → `SearchResult` with scored items |
| `getRecent(params)` | List recent events → `RecentEvent[]` |
| `submitFeedback(params)` | Rate a recalled event → `FeedbackRecord` |
| `deleteEvent(params)` | Delete a single episodic event |
| `deleteUserMemory(params)` | Delete all events for a user |
| `storeProcedure(params)` | Create a procedural memory rule |
| `listProcedures(params)` | List procedures → `ProcedureRecord[]` |
| `deleteProcedure(params)` | Delete a single procedure |
| `deleteUserProcedures(params)` | Delete all procedures for a user |
| `reconsolidate(params)` | Re-summarise an event with new context |
| `ingestPush(params)` | Push a single event from an external source → `IngestResult` |
| `ingestFile(params)` | Upload a file (txt/md/csv/json) → `IngestResult` |
| `ingestEmail(params)` | Fetch IMAP emails → `IngestResult` |
| `ingestCalendar(params)` | Upload an `.ics` file → `IngestResult` |
| `adminConsolidate(params)` | Trigger consolidation for a user |
| `adminPrune(params)` | Trigger synaptic pruning |
| `adminCluster(params)` | Trigger memory clustering |
| `adminMineBeliefs(params)` | Trigger belief mining |
| `health()` | Server + DB liveness check → `HealthStatus` |

### Running Node.js tests

```bash
cd sdk-node
npm test          # vitest run — 41 tests, ~300ms
npm run test:watch
```

---

## Testing

### Run all unit tests (no external dependencies)

```bash
pytest
```

The default run executes **~600 tests** in about 8 seconds. All tests that require real API keys, a local Ollama server, or running databases are automatically skipped.

```
601 passed, 42 skipped in 7.9s
```

Run the Node.js SDK tests separately:

```bash
cd sdk-node && npm test    # 41 tests, ~300ms
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

#### Python (`pytest`)

| File | Tests | What it covers |
|---|---|---|
| `test_llm_adapter.py` | 22 | Model resolution, complete(), embed(), extract_structured(), retry logic |
| `test_db_models.py` | 18 | ORM field types, StrEnum, cascade delete, vector roundtrip |
| `test_episodic_memory.py` | 28 | store, search, hybrid_search, HybridWeights validation |
| `test_semantic_memory.py` | 37 | upsert_fact, get_user_profile, UserProfile.as_text_summary() |
| `test_amygdala.py` | 19 | All scoring rules, boosts, penalties, clamp behaviour |
| `test_hippocampus.py` | 16 | Parallel LLM calls, embedding failure, extraction failure |
| `test_narrative_memory.py` | 18 | Memory link creation, chain traversal, relation types |
| `test_context_builder.py` | 34 | Deduplication, degraded-mode fallbacks, prompt rendering, narrative chains |
| `test_consolidator.py` | 26 | Batch splitting, LLM failures, fact upserts, narrative link creation |
| `test_synaptic_pruner.py` | 22 | Score formula, pruning logic, threshold sensitivity |
| `test_scheduler.py` | 14 | Job registration, manual triggers, error recovery |
| `test_identity.py` | 26 | Dimension grouping, dominant value, LLM summary, empty profile |
| `test_memory_clusterer.py` | 29 | Cosine sim, greedy clustering, LLM labelling, skip guards |
| `test_reinforcement.py` | 23 | apply_delta clamping, submit(), score update, neutral no-op |
| `test_belief_miner.py` | 29 | Prompt building, skip guards, upsert logic, LLM failure, identity integration |
| `test_procedural_memory.py` | 25 | _tokenise, _jaccard, store/update/delete/search (all 3 strategies), priority ranking |
| `test_reconsolidation.py` | 22 | Gate conditions, _reconsolidate_one, reconsolidate_event, reconsolidate_after_recall |
| `test_connectors.py` | 30 | All 5 connector types, to_metadata(), Slack signature verification, ICS parsing |
| `test_api.py` | 43 | Core HTTP routes via httpx test client + dependency overrides; health DB fields |
| `test_api_procedures.py` | 18 | Procedure CRUD routes + delete_all_for_user, delete event/user memory |
| `test_api_admin.py` | 22 | Admin job endpoints, ingest routes (push/file/slack/calendar), 503 on missing scheduler |
| `test_sdk_client.py` | 40 | HTTP mocking via respx, error handling, type checks |

#### Node.js (`vitest`)

| File | Tests | What it covers |
|---|---|---|
| `src/client.test.ts` | 41 | All client methods, snake↔camelCase mapping, error handling, baseUrl normalisation |

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

The `MemoryScheduler` runs four jobs inside the FastAPI process using APScheduler:

| Job | Default interval | What it does |
|---|---|---|
| **Consolidation** | every 1 hour | Compresses raw events → summaries + Neo4j facts |
| **Synaptic pruning** | every 24 hours | Deletes old low-scoring events |
| **Memory clustering** | every 6 hours | Groups similar events by topic using embeddings |
| **Belief mining** | every 12 hours | Infers durable beliefs and values from event patterns |

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

### Memory clustering (every 6 hours)

Groups events with embeddings into topical clusters using a greedy centroid algorithm (cosine similarity ≥ 0.75). Each cluster is labelled by the LLM (`cluster_label`) and stored on the event rows. Requires at least 5 events with embeddings to run.

### Belief mining (every 12 hours)

Reads consolidated events (minimum 3) and semantic facts, then prompts the LLM to infer higher-order beliefs and values. Results are upserted into `user_beliefs` — `evidence_count` increments each time the same belief is independently inferred, reinforcing confidence over time.

### Manual triggers (admin / testing)

```python
from smritikosh.processing.scheduler import MemoryScheduler

# Trigger immediately for one user
await scheduler.run_consolidation_now(user_id="alice", app_id="myapp")
await scheduler.run_pruning_now(user_id="alice", app_id="myapp")
await scheduler.run_clustering_now(user_id="alice", app_id="myapp")
await scheduler.run_belief_mining_now(user_id="alice", app_id="myapp")

# Run batch across all users
await scheduler.run_consolidation_for_all_users()
await scheduler.run_pruning_for_all_users()
await scheduler.run_clustering_for_all_users()
await scheduler.run_belief_mining_for_all_users()
```

### Tune the schedule

Pass custom intervals when constructing the scheduler (or subclass `MemoryScheduler`):

```python
MemoryScheduler(
    consolidator=..., pruner=..., episodic=...,
    clusterer=..., belief_miner=...,
    consolidation_hours=2,    # consolidate every 2 hours
    pruning_hours=48,         # prune every 2 days
    clustering_hours=12,      # cluster every 12 hours
    belief_mining_hours=24,   # mine beliefs once a day
)
```
