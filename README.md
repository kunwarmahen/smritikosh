# Smritikosh

**स्मृतिकोश** *(Sanskrit: "memory bank/treasury")*

A universal memory layer for LLM applications — a hippocampus for AI.

Smritikosh gives any LLM application persistent, user-specific memory modelled on how the human brain actually stores and retrieves information: episodic events encoded as vectors in PostgreSQL, semantic facts distilled into a Neo4j knowledge graph, background consolidation that compresses raw events into durable knowledge, and synaptic pruning that discards low-value memories over time.

---

## Guides

| Document | What it covers |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Step-by-step setup from zero to a running server |
| [FLOW.md](FLOW.md) | End-to-end walkthrough — how memory flows through the system with real examples |
| [sample/README.md](sample/README.md) | Running the demo chatbot against a live server |

---

## Table of Contents

- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Quick start (from scratch)](#quick-start-from-scratch)
- [Sample project](#sample-project)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Database setup](#database-setup)
  - [PostgreSQL + pgvector](#postgresql--pgvector)
  - [Neo4j](#neo4j)
  - [MongoDB (audit trail)](#mongodb-audit-trail)
- [Setup](#setup)
  - [Backend setup](#backend-setup)
  - [Dashboard UI setup](#dashboard-ui-setup)
- [Configuration](#configuration)
  - [Backend environment variables](#backend-environment-variables)
  - [UI environment variables](#ui-environment-variables)
- [Running the server](#running-the-server)
- [Dashboard UI](#dashboard-ui)
- [API reference](#api-reference)
  - [Authentication](#authentication-api)
  - [Memory](#memory)
  - [Graph](#graph-api)
  - [Context](#context)
  - [Identity](#identity)
  - [Feedback](#feedback)
  - [Procedural memory](#procedural-memory-api)
  - [Admin jobs](#admin-jobs)
  - [Admin users](#admin-users-api)
  - [External ingest](#external-ingest)
  - [Passive Memory Extraction](#passive-memory-extraction)
  - [Facts QC](#facts-qc-api)
  - [Audit trail](#audit-trail-api)
- [Audit trail](#audit-trail)
- [Authentication & API keys](#authentication--api-keys)
- [Python SDK](#python-sdk)
- [Node.js SDK](#nodejs-sdk)
- [Testing](#testing)
- [LLM provider guide](#llm-provider-guide)
- [Background jobs](#background-jobs)
- [Production deployment](#production-deployment)
- [Data reset script](#data-reset-script)

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
      ├── SmritikoshMiddleware (Python)  ← wraps OpenAI / Anthropic sync client
      │         │  intercepts create() calls, buffers turns,
      │         │  injects remember() tool, fires POST /ingest/session
      │         ▼
      ├── LiteLLMMiddleware (Python)   ← wraps litellm.completion()
      │         │  covers Gemini, Ollama, vLLM, llama.cpp via LiteLLM
      │         ▼
      ├── SmritikoshClient (Python)   smritikosh.sdk   ← async API client
      └── SmritikoshClient (Node.js)  sdk-node/
                │
                ▼  REST API (FastAPI)
        ┌───────────────────────────────────────┐
        │  /memory   /context                   │
        │  /identity /feedback                  │
        │  /facts    (QC: pending, contradictions)│
        │  /procedures                          │
        │  /ingest/{push,file,slack,session}    │
        │  /admin/{consolidate,prune,…}         │
        └───────────────────────────────────────┘
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
| **ContextBuilder** | Retrieves relevant context before an LLM call; boosts chain-adjacent events | — |
| **Consolidator** | Background: compresses events into summaries + Neo4j facts + re-embeds summaries | — |
| **SynapticPruner** | Background: deletes old low-scoring events | — |
| **MemoryClusterer** | Background: groups similar events by topic using embeddings | PostgreSQL |
| **BeliefMiner** | Background: infers durable beliefs and values from event patterns; tracks evidence event IDs | PostgreSQL |
| **FactDecayer** | Background (weekly): exponential confidence decay on Neo4j facts; skips `ui_manual` facts; `cross_system` decays 2× faster; promotes stale facts (confidence < 0.20) to `pending` before deletion | Neo4j |
| **IdentityBuilder** | Synthesises semantic facts + beliefs into a user identity model | — |
| **ReinforcementLoop** | Adjusts event importance scores based on user feedback signals | PostgreSQL |
| **ProceduralMemory** | Stores trigger→instruction rules; fuzzy-matched against each query | PostgreSQL |
| **ReconsolidationEngine** | Re-summarises events after recall to incorporate new context | PostgreSQL |
| **SourceConnector** | Normalises external sources (file, webhook, Slack, email, calendar) into events | — |
| **MemoryScheduler** | Runs all background jobs on configurable timers (APScheduler) | — |
| **IntentClassifier** | Two-tier intent classification: keyword heuristic + LLM fallback for ambiguous queries | — |
| **LLMAdapter** | Unified interface to Claude, OpenAI, Gemini, Ollama, vLLM, llama.cpp; optional fallback provider (tries secondary after primary exhausts retries); logs resolved provider + model + fallback at startup | — |
| **SmritikoshClient (Python)** | Async Python SDK wrapping the REST API | — |
| **SmritikoshClient (Node.js)** | TypeScript/ESM SDK with identical surface to the Python SDK | — |
| **SmritikoshMiddleware** | Sync wrapper for OpenAI/Anthropic clients; auto-injects `remember()` tool, intercepts tool calls transparently, buffers turns, fires `POST /ingest/session` in background, optionally auto-injects context | — |
| **LiteLLMMiddleware** | Subclass of SmritikoshMiddleware wrapping `litellm.completion()`; covers Gemini, Ollama, vLLM, llama.cpp, OpenAI, Claude through a single interface | — |
| **TriggerDetector** | Regex pre-filter (30 patterns) — skips LLM extraction when no high-signal phrases detected | — |
| **QualityControlLayer** | Confidence threshold gate (active/pending/rejected); contradiction detection on fact upsert; auto-promotes or flags conflicts for user review | Neo4j + PostgreSQL |

---

## Quick start (from scratch)

This section walks a complete beginner through every step — from a blank machine to a running Smritikosh server with the dashboard. Nothing is assumed except that you have a terminal.

> **What you will have at the end:** a local Smritikosh server, a working dashboard at `http://localhost:3000`, and your first admin account.

---

### Step 1 — Install system requirements

You need three things on your machine before anything else.

#### Python 3.11+

```bash
# Check if you already have it:
python3 --version    # should print 3.11 or higher

# macOS (Homebrew):
brew install python@3.11

# Ubuntu / Debian:
sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip

# Windows:
# Download the installer from https://www.python.org/downloads/
# Make sure to tick "Add python.exe to PATH" during installation.
```

#### Node.js 18+

```bash
# Check if you already have it:
node --version    # should print v18 or higher

# macOS (Homebrew):
brew install node

# Ubuntu / Debian:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Windows:
# Download the LTS installer from https://nodejs.org/
```

#### Docker Desktop

Docker is used to run PostgreSQL, Neo4j, and MongoDB locally. You do not need to know how Docker works — just install it and leave it running.

- **macOS / Windows:** Download from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/) and install. Open it once so the Docker engine starts.
- **Ubuntu:** Follow the [official guide](https://docs.docker.com/engine/install/ubuntu/) then run `sudo usermod -aG docker $USER` and log out/in.

Verify Docker is running:

```bash
docker --version       # Docker version 24.x.x or similar
docker compose version # Docker Compose version v2.x.x
```

---

### Step 2 — Get an LLM API key

Smritikosh uses an LLM for:
- extracting semantic facts from memories
- generating narrative summaries
- scoring event importance

You need **at least one** of these keys:

| Provider | Where to get it | Cheapest model to start with |
|---|---|---|
| **Anthropic (Claude)** | [console.anthropic.com](https://console.anthropic.com) | `claude-haiku-4-5-20251001` |
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `gpt-4o-mini` |
| **Gemini** | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | `gemini-1.5-flash` |
| **Ollama (free, local)** | [ollama.com](https://ollama.com) | `llama3.2` |
| **llama.cpp (free, local)** | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | any GGUF model |

You also need an **embedding model**. The easiest option is to reuse OpenAI's `text-embedding-3-small` (it is very cheap — a few cents per million tokens). If you prefer to run fully offline, see the [LLM provider guide](#llm-provider-guide) for Ollama or llama.cpp embeddings.

---

### Step 3 — Clone the repository

```bash
git clone https://github.com/your-org/smritikosh.git
cd smritikosh
```

---

### Step 4 — Create a Python virtual environment

A virtual environment keeps Smritikosh's dependencies isolated from the rest of your system.

```bash
python3 -m venv .venv

# Activate it — you must do this every time you open a new terminal:
source .venv/bin/activate          # macOS / Linux
# or on Windows:
# .venv\Scripts\activate
```

When the environment is active your prompt will show `(.venv)`.

```bash
# Install Smritikosh and all its dependencies:
pip install -e ".[dev]"
```

This will take a minute or two on first run.

---

### Step 5 — Configure the backend

```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in **at minimum** these four lines (everything else can stay as the default for local development):

```dotenv
# Which LLM to use for fact extraction and summarisation
LLM_PROVIDER=claude                          # or: openai / gemini / ollama / vllm / llamacpp
LLM_MODEL=claude-haiku-4-5-20251001          # or your chosen model
LLM_API_KEY=sk-ant-...                       # paste your API key here

# Which model to use for turning text into vectors
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...                     # your OpenAI key (even if you use Claude above)

# Secret key for signing login tokens — change this to any long random string
JWT_SECRET=replace-this-with-something-random-and-long
```

> **Tip:** If you want to run 100% locally with no API costs, see the [Ollama section](#ollama--local-models) in the LLM provider guide. You can use Ollama for both the LLM and the embeddings.

---

### Step 6 — Start the databases

```bash
docker compose up -d
```

This starts three containers in the background:
- **PostgreSQL 17** (stores memory events and vectors) — port `5432`
- **Neo4j 5.26** (stores the knowledge graph) — port `7687`, browser at `http://localhost:7474`
- **MongoDB 7** (stores the audit trail) — port `27017`

Wait about 15–20 seconds for all three to become healthy, then check:

```bash
docker compose ps
```

All three services should show `running (healthy)`. If any shows `starting`, wait a few more seconds and re-run the command.

---

### Step 7 — Create the database tables

```bash
alembic upgrade head
```

This runs all the database migrations — it creates every table, enables the pgvector extension, and sets up the vector index. You should see output ending with something like:

```
INFO  [alembic.runtime.migration] Running upgrade 0016 -> 0017, add media_ingests table
INFO  [alembic.runtime.migration] Running upgrade 0017 -> 0018, add user_voice_profiles table
```

---

### Step 8 — Create your first admin account

On a fresh install there are no users, so you need to bootstrap the first admin account. Set `BOOTSTRAP_ADMIN=1` in your `.env` to allow the first registration without needing an existing token:

```bash
# Add this line to .env temporarily:
echo "BOOTSTRAP_ADMIN=1" >> .env
```

Now start the server (in a separate terminal, with your venv active):

```bash
uvicorn smritikosh.api.main:app --reload --port 8080
```

You should see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

Register the admin account:

```bash
curl -s -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "changeme123", "role": "admin"}' | python3 -m json.tool
```

Expected response:

```json
{
  "user_id": "admin",
  "username": "admin",
  "role": "admin",
  "app_ids": ["default"],
  "is_active": true,
  "created_at": "..."
}
```

**Important:** remove `BOOTSTRAP_ADMIN=1` from `.env` now, then restart the server.

```bash
# Remove the line from .env (Linux/macOS):
sed -i '/BOOTSTRAP_ADMIN/d' .env

# Or just open .env and delete that line manually, then Ctrl+C the server and restart:
uvicorn smritikosh.api.main:app --reload --port 8080
```

Verify everything is working:

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok", "postgres": "ok", "neo4j": "ok", "mongodb": "not_configured", "llm_model": "...", "llm_status": "ok"}
```

---

### Step 9 — Set up the dashboard UI

Open a **new terminal** (keep the server running in the first one).

```bash
cd ui
npm install          # downloads all frontend dependencies (~1–2 minutes)
```

Create the UI config file:

```bash
cp .env.local.example .env.local 2>/dev/null || \
  printf "SMRITIKOSH_API_URL=http://localhost:8080\nAUTH_SECRET=$(openssl rand -hex 32)\n" > .env.local
```

Start the dashboard:

```bash
npm run dev
```

Open **http://localhost:3000** in your browser. You should see the Smritikosh login page. Sign in with `admin` / `changeme123`.

---

### Step 10 — Create a regular user for testing

From the dashboard, go to **Admin → Users → New user** and create a test account (e.g. `alice`, role: `user`).

Or via the API:

```bash
# First, get an admin token:
TOKEN=$(curl -s -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "changeme123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Create alice:
curl -s -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"username": "alice", "password": "alicepass", "role": "user"}' | python3 -m json.tool
```

You are now fully set up. Continue to the [Sample project](#sample-project) to see Smritikosh in action.

---

## Sample project

This section builds a small but complete **memory-aware chatbot** in Python — step by step. The bot remembers what users tell it across separate sessions and uses those memories to give personalised answers.

By the end you will have seen:
1. How to store memories (the intake pipeline)
2. How to search memories
3. How to retrieve context and inject it into an LLM call
4. How those memories turn into a knowledge graph over time

> **Prerequisite:** the Smritikosh server must be running (`uvicorn smritikosh.api.main:app --reload --port 8080`). If you followed the Quick Start, it already is.

---

### Project layout

Create a new directory for the sample:

```bash
mkdir smritikosh-demo
cd smritikosh-demo
```

We will create three files:

```
smritikosh-demo/
├── client.py      # thin wrapper around the Smritikosh REST API
├── chatbot.py     # the memory-aware chatbot loop
└── seed.py        # script to pre-load some memories so the demo is interesting
```

---

### File 1 — `client.py`

This is a minimal API client. It supports two authentication modes and exposes the three methods the chatbot needs: store a memory, get context, and search.

```python
# client.py
import httpx
import os

BASE_URL = os.getenv("SMRITIKOSH_URL", "http://localhost:8080")


class SmritikoshClient:
    """
    Minimal sync client for the Smritikosh REST API.

    Two authentication modes:

    1. Username + password (exchanges credentials for a short-lived JWT):
        client = SmritikoshClient(username="alice", password="alicepass")

    2. API key (no login round-trip, never expires unless revoked):
        client = SmritikoshClient(api_key="sk-smriti-...")
        # or set SMRITIKOSH_API_KEY in your environment and call SmritikoshClient()
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        api_key: str | None = None,
        app_id: str = "default",
    ):
        self.app_id = app_id
        resolved_key = api_key or os.getenv("SMRITIKOSH_API_KEY")
        if resolved_key:
            token = resolved_key
        elif username and password:
            token = self._login(username, password)
        else:
            raise ValueError(
                "Provide (username + password) or an api_key "
                "(or set SMRITIKOSH_API_KEY in your environment)."
            )
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _login(self, username: str, password: str) -> str:
        resp = httpx.post(
            f"{BASE_URL}/auth/token",
            json={"username": username, "password": password},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    # ── Core API ──────────────────────────────────────────────────────────────

    def remember(self, user_id: str, text: str) -> dict:
        """Store a piece of text as a memory event."""
        resp = httpx.post(
            f"{BASE_URL}/memory/event",
            headers=self._headers,
            json={"user_id": user_id, "content": text, "app_id": self.app_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_context(self, user_id: str, query: str) -> str:
        """Retrieve the memory context block for an LLM call."""
        resp = httpx.post(
            f"{BASE_URL}/context",
            headers=self._headers,
            json={"user_id": user_id, "query": query, "app_ids": [self.app_id]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("context_text", "")

    def search(self, user_id: str, query: str, limit: int = 5) -> list[dict]:
        """Search a user's memories and return scored results."""
        resp = httpx.post(
            f"{BASE_URL}/memory/search",
            headers=self._headers,
            json={
                "user_id": user_id,
                "query": query,
                "app_ids": [self.app_id],
                "limit": limit,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
```

---

### File 2 — `seed.py`

Before running the chatbot we will seed Alice's memory with a few facts so the demo shows personalised answers straight away.

```python
# seed.py
"""
Pre-load some memories for 'alice' so the chatbot has something to recall.
Run once:  python seed.py
"""

from client import SmritikoshClient

# Sign in as the admin account (which can store memories for any user)
client = SmritikoshClient(username="admin", password="changeme123")

USER = "alice"

memories = [
    "My name is Alice and I work as a machine learning engineer at a Series B startup.",
    "I prefer Python over other languages, especially for data pipelines and ML work.",
    "My favourite editor is Neovim with the lazy.nvim plugin manager.",
    "I am learning Rust in my spare time. I find the borrow checker confusing but rewarding.",
    "I use a MacBook Pro M3 Max for local development.",
    "My team is migrating from PyTorch to JAX for our training infrastructure.",
    "I dislike meetings before 10 am. My most productive hours are 9 pm to midnight.",
    "I recently read 'The Pragmatic Programmer' and found the chapter on orthogonality very useful.",
    "I deployed a RAG pipeline last week using pgvector and LangChain. Latency was higher than expected.",
    "My manager asked me to evaluate Smritikosh as a memory layer for our internal LLM assistant.",
]

print(f"Seeding {len(memories)} memories for user '{USER}'...\n")

for i, text in enumerate(memories, 1):
    result = client.remember(USER, text)
    importance = result.get("importance_score", 0)
    facts = result.get("facts_extracted", 0)
    print(f"  [{i:2d}] importance={importance:.2f}  facts={facts}  "{text[:60]}..."")

print("\nDone. Run 'python chatbot.py' to start the chatbot.")
```

Run it:

```bash
python seed.py
```

You should see output like:

```
Seeding 10 memories for user 'alice'...

  [ 1] importance=0.68  facts=3  "My name is Alice and I work as a machine learning engine..."
  [ 2] importance=0.61  facts=2  "I prefer Python over other languages, especially for dat..."
  ...

Done. Run 'python chatbot.py' to start the chatbot.
```

Each line shows the importance score Smritikosh assigned to that memory and how many semantic facts it extracted into the Neo4j knowledge graph.

---

### File 3 — `chatbot.py`

This is the chatbot itself. For simplicity it uses the Anthropic SDK directly — swap in OpenAI or any other provider if you prefer.

```bash
pip install anthropic   # or: pip install openai
```

```python
# chatbot.py
"""
Memory-aware chatbot using Smritikosh for persistent user memory.

Usage:
    python chatbot.py

Commands during the chat:
    /remember <text>    — Manually store something as a memory
    /search <query>     — Search Alice's memories and show scored results
    /quit               — Exit
"""

import os
import anthropic
from client import SmritikoshClient

# ── Configuration ────────────────────────────────────────────────────────────

SMRITIKOSH_USER = "alice"           # the user whose memories we read/write
SMRITIKOSH_USER_PASS = "alicepass"  # alice's login password

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")   # or hardcode for local testing
LLM_MODEL = "claude-haiku-4-5-20251001"              # fast and cheap for demos

# ── Clients ───────────────────────────────────────────────────────────────────

memory = SmritikoshClient(username=SMRITIKOSH_USER, password=SMRITIKOSH_USER_PASS)
llm = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Conversation history (kept in RAM — resets when the script restarts,
# but *memories* persist across restarts because they live in Smritikosh)
conversation: list[dict] = []

# ── Helpers ───────────────────────────────────────────────────────────────────

def chat(user_message: str) -> str:
    """
    Send a message to the LLM, injecting the user's memory context first.
    The LLM response is also stored back into Smritikosh so it can be
    recalled in future sessions.
    """
    # 1. Retrieve the memory context relevant to this message
    context = memory.get_context(SMRITIKOSH_USER, user_message)

    # 2. Build the system prompt — memory context goes first
    system_prompt = (
        "You are a helpful personal assistant. "
        "Use the memory context below to give personalised, accurate answers. "
        "If the context contains relevant information, use it naturally — "
        "do not say 'according to your memory'. Just answer as if you know the user well.\n\n"
        + context
    )

    # 3. Add the new user message to the conversation history
    conversation.append({"role": "user", "content": user_message})

    # 4. Call the LLM
    response = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=conversation,
    )
    assistant_message = response.content[0].text

    # 5. Add the assistant reply to conversation history
    conversation.append({"role": "assistant", "content": assistant_message})

    # 6. Store the exchange as a new memory so future sessions can recall it
    memory.remember(
        SMRITIKOSH_USER,
        f"User asked: {user_message}\nAssistant replied: {assistant_message}",
    )

    return assistant_message


def show_search(query: str) -> None:
    """Print scored memory search results for a query."""
    results = memory.search(SMRITIKOSH_USER, query)
    if not results:
        print("  (no results)")
        return
    for r in results:
        score = r.get("hybrid_score", 0)
        text = r.get("raw_text", "")[:90]
        consolidated = "✓" if r.get("consolidated") else "·"
        print(f"  [{score:.3f}] {consolidated} {text}...")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Smritikosh demo chatbot  (user: alice)")
    print("  Commands: /remember <text>  /search <query>  /quit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────────────

        if user_input.startswith("/quit"):
            print("Goodbye!")
            break

        if user_input.startswith("/remember "):
            text = user_input[len("/remember "):]
            result = memory.remember(SMRITIKOSH_USER, text)
            print(f"  Stored. importance={result['importance_score']:.2f}  "
                  f"facts_extracted={result['facts_extracted']}")
            continue

        if user_input.startswith("/search "):
            query = user_input[len("/search "):]
            print(f"  Search results for: '{query}'")
            show_search(query)
            continue

        # ── Normal chat ───────────────────────────────────────────────────────

        reply = chat(user_input)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
```

---

### Running the chatbot

Make sure the Smritikosh server is still running in another terminal, then:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or set OPENAI_API_KEY if you adapted chatbot.py
python chatbot.py
```

#### Example session

```
============================================================
  Smritikosh demo chatbot  (user: alice)
  Commands: /remember <text>  /search <query>  /quit
============================================================

You: What editor do I use?

Assistant: You use Neovim with the lazy.nvim plugin manager — it's what you've set up for your development environment.

You: Do I know Rust?

Assistant: You're learning Rust in your spare time. You mentioned finding the borrow checker confusing but also rewarding — sounds like it's starting to click.

You: /search JAX

  Search results for: 'JAX'
  [0.834] · Your team is migrating from PyTorch to JAX for your training infrastructure...
  [0.612] · I deployed a RAG pipeline last week using pgvector and LangChain. Latency was...

You: /remember I finished the Rust ownership chapter today — it finally clicked!

  Stored. importance=0.72  facts_extracted=1

You: What am I learning these days?

Assistant: You're learning Rust in your spare time, and you just told me the ownership chapter finally clicked today. On the work side, your team is also migrating from PyTorch to JAX for your ML training infrastructure.

/quit
Goodbye!
```

Notice that on the last message, the bot knows about the Rust milestone stored via `/remember` in the same session. Each time you use `/remember` or have a normal chat exchange, the content is persisted — so the next time you start `chatbot.py`, Alice's bot will still know everything from this session.

### What just happened under the hood

| Step | What Smritikosh did |
|---|---|
| `seed.py` ran | 10 texts → Hippocampus → importance scored → embedded → stored in PostgreSQL → facts extracted → written to Neo4j |
| `chat()` called | `GET /context` retrieved the 3 most relevant memories + Alice's Neo4j profile → injected as system prompt |
| LLM responded | Claude answered using the injected context |
| Exchange stored | The full Q&A was stored as a new memory event for future sessions |
| `/remember` ran | `POST /memory/event` stored the Rust milestone immediately |

### Where to look next

- **Dashboard** (`http://localhost:3000`) — log in as `alice` and browse the memory timeline, see the fact graph, check the audit trail
- **Identity page** — the 3D force graph shows all 23 fact categories; click any fact orb to see which memories contributed to it
- **Memory detail** — click any event in the timeline to see its narrative links graph (orb nodes, gold RELATED_TO edges)
- **Admin panel** — log in as `admin` to see all users, trigger consolidation (`/admin/jobs`), or check system health
- **Run consolidation** — `curl -X POST "http://localhost:8080/admin/consolidate?user_id=alice"` to compress Alice's memories into summaries and extract more facts into Neo4j

---

## Demo scripts (passive extraction + SDK middleware + media ingestion)

All scripts live in `sample/` and authenticate as `admin` — no API key setup required.
Run every command from the **repo root** with the virtualenv active.

```bash
source .venv/bin/activate
docker compose up -d    # server must be running
```

### Script 1 — `seed_priya.py` (one-time setup)

Pre-loads 15 rich memories for user `priya` — a homemaker who loves fashion, luxury travel, and books. All the other demo scripts use `priya` as their subject.

```bash
python sample/seed_priya.py
```

Expected output:
```
Seeding 15 memories for user 'priya'...

  [ 1] importance=0.74  facts=3  "My name is Priya and I'm a homemaker who loves..."
  [ 2] importance=0.81  facts=4  "I am passionate about fashion and shopping. I f..."
  ...
Done. Run 'python chatbot.py' to chat with Priya's memory.
```

---

### Script 2 — `passive_extraction_demo.py`

Demonstrates passive memory extraction from a conversation transcript — no per-turn developer work required.

```bash
python sample/passive_extraction_demo.py
```

**What it does (5 steps):**

| Step | What happens |
|---|---|
| 1. Session ingest | Posts a 7-turn Priya conversation to `POST /ingest/session`; trigger phrases (`I always`, `I prefer`, `My goal is`, `I believe`, `I never`) fire the LLM extraction |
| 2. Idempotency | Re-posts the same `session_id` — server returns `already_processed=True`, nothing re-extracted |
| 3. Manual facts | Calls `store_fact()` four times — stores fashion/habit/goal facts at `confidence=1.0`, `source_type=ui_manual`, bypassing the LLM entirely |
| 4. Streaming windows | Posts 3 partial windows with `partial=True`; each window only processes new turns via `last_turn_index` tracking |
| 5. Verification | Calls `get_context()` to confirm extracted facts appear in context retrieval |

Expected output (abbreviated):
```
── Step 1 — POST /ingest/session ────────────────────────────
  turns_processed:    4
  facts_extracted:    5
  extraction_skipped: False
  already_processed:  False

── Step 2 — Idempotency check ───────────────────────────────
  already_processed: True (should be True)
  ✓ Idempotency working correctly

── Step 3 — store_fact() ────────────────────────────────────
  Stored: preference/fashion_brand = 'Bottega Veneta'
    confidence:  1.00  (source: ui_manual)
    status:      active
  ...
  ✓ All manual facts stored with confidence=1.0, status=active
```

---

### Script 3 — `middleware_demo.py`

Demonstrates `SmritikoshMiddleware` — wraps a fake OpenAI-style client so no real API key is needed. Shows how memory extraction becomes transparent.

```bash
python sample/middleware_demo.py
```

**What it does (4 steps):**

| Step | What happens |
|---|---|
| 1. Wrap the client | Shows the one-line change — `SmritikoshMiddleware(FakeOpenAI(), ...)` instead of `FakeOpenAI()` |
| 2. 5-turn conversation | Runs 5 turns through the middleware with `extract_every_n_turns=3`; partial flush fires after turn 3 in a background thread, `close()` waits for it then flushes turns 4–5 |
| 3. `auto_inject=True` | Makes one call with context injection — middleware fetches Priya's memory context and prepends it as a sentinel-wrapped system message |
| 4. Verification | Calls `get_context()` to confirm extracted facts appeared |

Expected output (abbreviated):
```
── Step 1 — Wrap the LLM client ─────────────────────────────
  # Before: llm = FakeOpenAI()
  # After:  llm = SmritikoshMiddleware(FakeOpenAI(), ...)

── Step 2 — Simulating a 5-turn conversation ─────────────────
  User:      I always have Rohan pick the wine...
  Assistant: [fake LLM] Got: I always have Rohan pick the wine...
  ...
  Buffered turns: 5 user turns across 10 total
  Calling close() → flushes remaining turns as final ingest …
  ✓ Session closed and flushed

── Step 3 — auto_inject=True ────────────────────────────────
  Context fetched and injected for user='priya'
  ✓ The fake LLM received a system message with Priya's memory context
```

> **Note:** Step 4 (context verification) calls `GET /context` immediately after two back-to-back LLM ingest calls. If the API is slow, it may timeout — the memories are still extracted. Run `python sample/chatbot.py` and ask about Priya's travel plans to verify.

---

### Script 4 — `media_ingest_demo.py`

Demonstrates media ingestion — uploading a personal notes document and having facts extracted automatically. **No Whisper or Vision API key required** — document extraction runs with your core LLM.

```bash
python sample/media_ingest_demo.py
```

**What it does (5 steps):**

| Step | What happens |
|---|---|
| 1. Upload | Posts a personal `.md` notes document to `POST /ingest/media`; server returns `status=processing` immediately — no blocking wait |
| 2. Poll | Calls `GET /ingest/media/{id}/status` every 2 seconds until extraction is done |
| 3. Review | Shows facts saved automatically (relevance > 0.75) and facts pending review (0.60–0.75) |
| 4. Confirm | Calls `POST /ingest/media/{id}/confirm` to move pending facts to active status — mirrors the UI review modal |
| 5. Verify | Calls `GET /context` to confirm the extracted facts appear in context retrieval |

The script's summary also shows the equivalent code snippets for **voice notes** (requires `WHISPER_PROVIDER`), **image uploads** (requires `VISION_PROVIDER`), and **meeting recordings** — so you can extend to those once those optional providers are configured.

Expected output (abbreviated):
```
── Step 1 — Upload a personal notes document ────────────────
  media_id: 550e8400-e29b-41d4-a716-446655440000
  status:   processing  ← processing runs in the background

── Step 2 — Wait for extraction to complete ─────────────────
  [1/30] status='processing' … waiting 2s
  status:               complete
  facts_extracted:      6
  facts_pending_review: 2

── Step 3 — Extracted facts ──────────────────────────────────
  ✓ 6 fact(s) saved automatically
  2 fact(s) are waiting for your review:
    [0] shops at Whole Foods every Saturday  confidence=0.71
    [1] follows mostly plant-based diet      confidence=0.68

── Step 4 — Confirm 2 pending fact(s) ───────────────────────
  Server response: Confirmed 2 facts
  facts_pending_review remaining: 0
  ✓ All pending facts moved to active status
```

---

### Script 5 — `chatbot.py` (interactive)

A live chat loop against Priya's memories. After running the scripts above, Priya's memory will include extracted facts from all demos.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # required — chatbot makes direct LLM calls
python sample/chatbot.py
```

Try asking:
- `What are Priya's travel plans?`
- `What luxury brands does she follow?`
- `What does she like to read?`
- `/search Japan`

---

### Recommended run order

```bash
# 1. One-time setup
python sample/seed_priya.py

# 2. Session ingest, streaming, manual facts (no extra API key)
python sample/passive_extraction_demo.py

# 3. SDK middleware + remember() tool (no OpenAI key — uses fake client)
python sample/middleware_demo.py

# 4. Media ingestion from a document (no Whisper/Vision key needed)
python sample/media_ingest_demo.py

# 5. Chat with the fully enriched memory (requires LLM API key)
export ANTHROPIC_API_KEY=sk-ant-...
python sample/chatbot.py
```

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
│       ├── memory.py        # POST /memory/event, GET /memory/event/{id},
│       │                    #   GET /memory/event/{id}/links, GET /memory/{user_id},
│       │                    #   DELETE /memory/event/{id}, DELETE /memory/user/{id}
│       ├── graph.py         # GET /graph/facts/{user_id}
│       ├── context.py       # POST /context
│       ├── identity.py      # GET /identity/{user_id}
│       ├── feedback.py      # POST /feedback
│       ├── procedures.py    # CRUD /procedures + DELETE /procedures/user/{id}
│       ├── admin.py         # POST /admin/{consolidate,prune,cluster,mine-beliefs,reconsolidate,synthesize}
│       │                    #   GET /admin/embedding-health, POST /admin/re-embed
│       │                    #   GET /admin/users, GET /admin/users/{username},
│       │                    #   PATCH /admin/users/{username}
│       ├── ingest.py        # POST /ingest/{push,file,slack/events,email/sync,calendar}
│       ├── session_ingest.py # POST /ingest/session, POST /ingest/transcript
│       ├── facts.py         # GET /facts/{user_id}, PATCH /facts/.../status,
│       │                    #   GET /facts/contradictions/{user_id}, PATCH /facts/contradictions/{id}
│       ├── media_ingest.py  # POST /ingest/media, GET /ingest/media/{id}/status,
│       │                    #   POST /ingest/media/{id}/confirm
│       └── voice_enrollment.py # POST/GET/DELETE /user/{user_id}/voice-enrollment
├── auth/
│   ├── __init__.py          # Re-exports router, require_admin, require_auth
│   ├── deps.py              # require_auth / require_admin FastAPI dependencies
│   ├── models.py            # AppUser ORM model, UserRole StrEnum
│   ├── routes.py            # POST /auth/token, POST /auth/register, GET /auth/me
│   └── utils.py             # hash_password, verify_password, create_access_token
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
│   │                        #   MemoryFeedback, UserBelief, UserProcedure, AppUser
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
│   ├── fact_decayer.py      # Weekly Neo4j confidence decay; staleness→pending promotion
│   ├── trigger_detector.py  # Regex pre-filter: detects high-signal phrases before LLM
│   ├── transcript_utils.py  # sentinel-strip, user_turns_only, delta prompt builder
│   ├── cross_system_synthesizer.py  # Daily job: correlates connector signals → cross_system facts
│   ├── media_processor.py   # MediaProcessor: transcription, text extraction, vision, relevance routing
│   └── scheduler.py         # APScheduler background jobs (consolidation, pruning, clustering,
│                             #   belief mining, fact decay, cross-system synthesis)
├── retrieval/
│   └── context_builder.py   # Build memory context for LLM calls
├── audit/
│   ├── __init__.py          # Re-exports AuditLogger, AuditEvent, EventType
│   ├── logger.py            # AuditLogger: emit(), get_timeline(), get_event_lineage(), get_stats()
│   └── mongodb.py           # Motor connection, lazy init, index creation
└── sdk/
    ├── client.py            # SmritikoshClient (async HTTP)
    ├── middleware.py        # SmritikoshMiddleware (OpenAI/Anthropic sync wrapper);
    │                        #   LiteLLMMiddleware; remember() tool injection + interception
    ├── __init__.py          # Exports SmritikoshClient, SmritikoshMiddleware, LiteLLMMiddleware,
    │                        #   SessionIngestResult
    └── types.py             # EncodedEvent, MemoryContext, RecentEvent,
                             #   IdentityProfile, FeedbackRecord, HealthStatus, SessionIngestResult

sdk-node/                    # TypeScript / Node.js SDK
├── src/
│   ├── client.ts            # SmritikoshClient (native fetch, ESM)
│   ├── types.ts             # Branded types, request/response shapes
│   ├── errors.ts            # SmritikoshError
│   └── client.test.ts       # 41 Vitest tests (all methods, error paths)
├── package.json
└── tsconfig*.json

ui/                          # Next.js 16 dashboard (App Router)
├── auth.ts                  # NextAuth v5 config (CredentialsProvider → /auth/token)
├── middleware.ts             # Route protection: auth → /dashboard, admin → /admin
├── next.config.ts
├── tailwind.config.ts
├── src/
│   ├── app/
│   │   ├── (auth)/login/    # Sign-in page with error handling
│   │   ├── (dashboard)/dashboard/
│   │   │   ├── page.tsx            # Redirect → /dashboard/memories
│   │   │   ├── memories/           # Memory timeline list (+ Add Memory + Upload buttons)
│   │   │   │   └── [id]/           # Memory detail: importance card + narrative graph
│   │   │   ├── review/             # Auto-extracted memory review queue (approve / remove)
│   │   │   ├── search/             # Hybrid search with score breakdown
│   │   │   ├── identity/           # Identity profile + React Flow fact graph toggle
│   │   │   ├── clusters/           # Events grouped by topic cluster
│   │   │   ├── audit/              # Personal audit timeline + stats
│   │   │   ├── procedures/         # Procedural rules CRUD
│   │   │   └── settings/
│   │   │       ├── api-keys/       # Generate and revoke API keys
│   │   │       └── voice-enrollment/  # 30-sec voice sample recording, waveform, enrollment status
│   │   └── (admin)/admin/
│   │       ├── page.tsx            # Redirect → /admin/users
│   │       ├── users/              # Paginated user list
│   │       │   └── [userId]/       # User detail: toggle active/role, danger zone
│   │       ├── jobs/               # Trigger background jobs per user
│   │       ├── health/             # System health panel
│   │       └── audit/              # Global audit log (all users)
│   ├── components/
│   │   ├── memory/
│   │   │   ├── MemoryTimeline.tsx  # Event list with importance badges + Add Memory + Upload buttons
│   │   │   ├── MemoryCard.tsx      # Memory card with source badge
│   │   │   ├── MemoryGraphView.tsx # React Flow: narrative links (caused/preceded/…)
│   │   │   ├── SourceBadge.tsx     # Reusable source-type badge (13 types, icon + colour)
│   │   │   ├── AddMemoryForm.tsx   # Modal: manual fact entry → POST /memory/fact
│   │   │   └── UploadMediaForm.tsx # Multi-step upload modal (voice / document / image / meeting);
│   │   │                           #   upload → processing → review → success/nothing_found
│   │   ├── identity/
│   │   │   ├── IdentityProfile.tsx # Dimension grid + confidence bars + beliefs
│   │   │   └── IdentityFactGraph.tsx # React Flow: radial fact knowledge graph
│   │   ├── search/SearchPanel.tsx
│   │   ├── audit/
│   │   │   ├── AuditTimeline.tsx   # Filterable audit log with payload expand
│   │   │   └── AuditStatsBar.tsx   # Event-type count cards
│   │   ├── procedures/
│   │   │   ├── ProcedureTable.tsx
│   │   │   └── NewProcedureDrawer.tsx
│   │   └── admin/
│   │       ├── UserTable.tsx       # Paginated table, inline active toggle
│   │       ├── NewUserDrawer.tsx   # Create user modal
│   │       ├── JobTriggerPanel.tsx # Trigger jobs by user ID
│   │       └── HealthPanel.tsx
│   ├── hooks/
│   │   ├── useMemoryGraph.ts       # useMemoryEvent, useMemoryLinks
│   │   ├── useFactGraph.ts         # useFactGraph
│   │   ├── useAdmin.ts             # useAdminUsers, useAdminUser, useAdminPatchUser
│   │   └── useProcedures.ts        # CRUD hooks for procedural rules
│   ├── lib/api-client.ts           # Typed API client (wraps fetch with auth token)
│   └── types/index.ts              # Shared TypeScript types (MemoryEvent, FactGraph, …)
└── package.json

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
    ├── 0002_add_recall_count.py      # recall_count on events
    ├── 0003_add_cluster_fields.py    # cluster_id, cluster_label on events
    ├── 0004_add_memory_feedback.py   # memory_feedback table
    ├── 0005_add_user_beliefs.py      # user_beliefs table
    ├── 0006_add_user_procedures.py   # user_procedures table + priority/active indexes
    ├── 0007_add_reconsolidation_fields.py  # reconsolidation_count, last_reconsolidated_at
    ├── 0008_add_app_users.py         # app_users table (username, role, is_active, app_id)
    ├── 0009_multi_app_ids.py         # app_ids TEXT[] on app_users and api_keys; create api_keys table
    ├── 0010_add_belief_evidence_ids.py  # evidence_event_ids JSONB on user_beliefs
    ├── 0011_hnsw_index.py            # replace IVFFlat with HNSW index on events.embedding
    ├── 0012_resize_embedding_dims.py # resize embedding column to configured dimension
    ├── 0013_dynamic_embedding_dims.py # dynamic embedding dimension support
    ├── 0014_add_source_type_and_fact_status.py  # source_type + source_meta on events + user_facts; FactStatus
    ├── 0015_add_processed_sessions.py           # processed_sessions table (idempotency + last_turn_index)
    ├── 0016_add_fact_contradictions.py          # fact_contradictions table (QC contradiction log)
    ├── 0017_add_media_ingests.py                # media_ingests table (async processing + status)
    └── 0018_add_user_voice_profiles.py          # user_voice_profiles table (speaker d-vector enrollment)
```

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | StrEnum, `match`, type syntax |
| Node.js | ≥ 18 | Dashboard UI (Next.js 16) |
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

> **Changing embedding dimensions?** If you switch to a model with different output dimensions (e.g. Gemini's 768), follow these steps:
> 1. Update `.env`: set `EMBEDDING_DIMENSIONS=768` (or your model's dimension) and update `EMBEDDING_MODEL`.
> 2. Re-run migrations to resize the vector column: `alembic downgrade base && alembic upgrade head`.
> 3. Re-embed all existing events so they match the new dimension: `POST /admin/re-embed` (runs in the background; check progress with `GET /admin/embedding-health`).
>
> Smritikosh validates the embedding dimension on every insert — if a generated vector doesn't match `EMBEDDING_DIMENSIONS` it raises an error immediately rather than silently storing an incompatible vector.

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

### Backend setup

#### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-org/smritikosh.git
cd smritikosh

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

#### 2. Install the package

```bash
# Runtime + dev dependencies
pip install -e ".[dev]"
```

#### 3. Configure environment

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

#### 4. Start the databases

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 17** with the `pgvector` extension on port `5432`
- **Neo4j 5.26** on ports `7474` (browser) and `7687` (bolt)
- **MongoDB 7** on port `27017` (audit trail — optional, safe to omit)

Wait for all services to be healthy:

```bash
docker compose ps   # all should show "healthy"
```

#### 5. Run database migrations

```bash
alembic upgrade head
```

This creates all tables (`events`, `user_facts`, `memory_links`, `app_users`, `procedures`, `user_beliefs`, `feedbacks`), enables the `vector` extension, and adds an IVFFlat index for fast similarity search.

#### 6. Create the first admin user

```bash
curl -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-token>" \
  -d '{"username": "admin", "password": "changeme123", "role": "admin"}'
```

> **Bootstrapping:** On a fresh install with no users, register the first admin account by temporarily setting `BOOTSTRAP_ADMIN=1` in `.env`. This disables the auth requirement on `POST /auth/register` for one call. Remove it afterwards.

---

### Dashboard UI setup

The UI is a standalone Next.js 16 application in the `ui/` directory.

#### 1. Install dependencies

```bash
cd ui
npm install
```

#### 2. Configure environment

```bash
cp .env.local.example .env.local
```

Edit `.env.local`:

```dotenv
# URL of the Smritikosh FastAPI backend (server-side only, never exposed to browser)
SMRITIKOSH_API_URL=http://localhost:8080

# NextAuth.js secret — generate a strong value:
# node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
AUTH_SECRET=your-secret-here

# Optional: required in production for CSRF protection
# AUTH_URL=https://your-domain.com
```

#### 3. Start the dev server

```bash
npm run dev
```

The dashboard is available at **http://localhost:3000**.

> The backend must be running before logging in. Start it with `uvicorn smritikosh.api.main:app --reload --port 8080`.

---

## Configuration

### Backend environment variables

All backend settings are read from the environment (or `.env`). Every field has a default so only the keys you need to change are required.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `claude` | `claude` / `openai` / `gemini` / `ollama` / `vllm` / `llamacpp` |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Model name for chat/extraction |
| `LLM_API_KEY` | — | API key for the chat provider |
| `LLM_BASE_URL` | — | Custom base URL (Ollama / vLLM / llama.cpp only) |
| `LLM_MAX_TOKENS` | *(unset)* | Max tokens for LLM responses. Leave unset for no limit. |
| `LLM_FALLBACK_PROVIDER` | *(unset)* | Secondary LLM provider used when the primary exhausts all retries. Leave unset to disable fallback. |
| `LLM_FALLBACK_MODEL` | *(unset)* | Secondary LLM model name (required if `LLM_FALLBACK_PROVIDER` is set). |
| `LLM_FALLBACK_API_KEY` | *(unset)* | API key for the fallback provider; defaults to `LLM_API_KEY` if unset. |
| `LLM_FALLBACK_BASE_URL` | *(unset)* | Custom base URL for local fallback providers (Ollama / vLLM / llama.cpp). |
| `EMBEDDING_PROVIDER` | `openai` | `openai` / `ollama` / `vllm` / `gemini` / `llamacpp` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_API_KEY` | — | API key for the embedding provider |
| `EMBEDDING_BASE_URL` | — | Custom base URL for embeddings |
| `EMBEDDING_DIMENSIONS` | *(unset)* | Vector size — must match your model's output dimension. Leave unset if your model's dimension is detected automatically. |
| `SQLALCHEMY_LOG_LEVEL` | `WARNING` | SQLAlchemy engine log verbosity. `INFO` shows all SQL queries; `ERROR` for errors only. |
| `POSTGRES_URL` | `postgresql+asyncpg://smritikosh:smritikosh@localhost:5432/smritikosh` | Async PostgreSQL connection string |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `smritikosh` | Neo4j password |
| `LOG_LEVEL` | `INFO` | Python log level |
| `SLACK_SIGNING_SECRET` | — | Signing secret for Slack Events API signature verification (required only for `POST /ingest/slack/events`) |
| `MONGODB_URL` | — | MongoDB connection string. If unset, audit trail is disabled (no-op) |
| `MONGODB_DB_NAME` | `smritikosh_audit` | MongoDB database to store audit events in |
| `JWT_SECRET` | `change-me-in-production` | Secret key for signing JWT tokens — **change this in production** |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_EXPIRE_DAYS` | `30` | Token lifetime in days |
| `RATE_LIMIT_ENCODE` | `60/minute` | Per-user rate limit for `POST /memory/event`. Set to `""` to disable. |
| `RATE_LIMIT_CONTEXT` | `60/minute` | Per-user rate limit for `POST /context`. Set to `""` to disable. |
| `RATE_LIMIT_SEARCH` | `120/minute` | Per-user rate limit for `POST /memory/search`. Set to `""` to disable. |
| `FACT_DECAY_HALF_LIFE_DAYS` | `60.0` | Days for Neo4j fact confidence to halve without reinforcement. |
| `FACT_DECAY_FLOOR` | `0.1` | Facts whose confidence falls below this threshold are deleted. |
| **Whisper (audio transcription)** | | |
| `WHISPER_PROVIDER` | `openai` | `openai` (cloud) or `local` (self-hosted via Ollama / vLLM / llama.cpp) |
| `WHISPER_MODEL` | `whisper-1` | Whisper model name |
| `WHISPER_API_KEY` | — | API key for cloud Whisper; falls back to `EMBEDDING_API_KEY` if unset |
| `WHISPER_BASE_URL` | — | Base URL for local Whisper provider (e.g. `http://localhost:11434`) |
| **Vision (image description)** | | |
| `VISION_PROVIDER` | `openai` | `openai` / `claude` / `gemini` / `ollama` / `vllm` / `llamacpp` |
| `VISION_MODEL` | `gpt-4o-mini` | Multimodal model for image description |
| `VISION_API_KEY` | — | API key for cloud vision provider; falls back to `LLM_API_KEY` if unset |
| `VISION_BASE_URL` | — | Base URL for local vision provider |
| **Diarization (speaker identification)** | | |
| `DIARIZATION_PROVIDER` | `none` | `none` (first-person filter only) or `pyannote` (full speaker diarization) |
| `HF_TOKEN` | — | Hugging Face read token — required for `pyannote` diarization |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity threshold for matching a diarized speaker to enrolled voice (0–1) |
| **Media size limits** | | |
| `MEDIA_MAX_AUDIO_MB` | `25` | Max size for voice note uploads (Whisper API limit) |
| `MEDIA_MAX_DOCUMENT_MB` | `10` | Max size for document uploads |
| `MEDIA_MAX_DOCUMENT_PAGES` | `50` | Max PDF page count |
| `MEDIA_MAX_IMAGE_MB` | `20` | Max size for image uploads |
| `MEDIA_MAX_MEETING_MB` | `500` | Max size for meeting recording uploads |

### UI environment variables

These go in `ui/.env.local` (never committed to git):

| Variable | Required | Description |
|---|---|---|
| `AUTH_SECRET` | ✅ | NextAuth.js secret — generate with `openssl rand -hex 32` |
| `SMRITIKOSH_API_URL` | ✅ | Backend URL used server-side only (e.g. `http://localhost:8080`) |
| `AUTH_URL` | Production only | Public base URL of the UI (e.g. `https://app.example.com`) — required for CSRF in production |

---

## Running the server

```bash
uvicorn smritikosh.api.main:app --reload --port 8080
```

On startup the server will:
1. Enable the `pgvector` extension and create tables (if not already present via Alembic)
2. Apply Neo4j schema constraints and indexes
3. Create MongoDB `audit_events` collection and indexes (if `MONGODB_URL` is configured)
4. Start background scheduler (consolidation every hour, pruning every 24 hours, fact decay weekly)

Interactive API docs are available at `http://localhost:8080/docs`.

---

## Dashboard UI

The dashboard is a standalone Next.js 16 application that connects to the FastAPI backend.

```bash
cd ui
npm run dev      # development (http://localhost:3000)
npm run build    # production build
npm start        # serve the production build
```

### Pages

| Route | Role | Description |
|---|---|---|
| `/login` | All | Sign in with username + password |
| `/dashboard/memories` | User | Memory timeline — search, feedback, delete; **+ Add Memory** button opens manual entry form |
| `/dashboard/memories/[id]` | User | Narrative link graph + audit lineage for one event |
| `/dashboard/review` | User | Review queue for auto-extracted memories — approve or remove, filterable by source type |
| `/dashboard/identity` | User | Identity model: summary, dimensions, inferred beliefs, fact graph |
| `/dashboard/clusters` | User | Memories grouped by topic cluster |
| `/dashboard/audit` | User | Personal audit trail with event-type filter |
| `/dashboard/procedures` | User | Procedural rules — create, toggle, delete |
| `/admin/health` | Admin | Live status of all backend services |
| `/admin/jobs` | Admin | Manually trigger pipeline jobs (consolidate / prune / cluster / mine) for any user |
| `/admin/audit` | Admin | System-wide audit log |
| `/admin/users` | Admin | Paginated user list — create, activate/deactivate, change role |
| `/admin/users/[userId]` | Admin | Per-user detail, role toggle, memory wipe |
| `/dashboard/settings/api-keys` | User | Generate and revoke long-lived API keys |
| `/dashboard/settings/voice-enrollment` | User | Record a 30-second voice sample for speaker diarization; waveform visualiser, re-record, delete enrollment |

### Authentication

The UI uses **NextAuth.js v5** with a Credentials provider that exchanges a username + password for a JWT issued by `POST /auth/token`. The token is stored in a server-side session cookie and forwarded as a `Bearer` header to all API calls.

- Regular users can only access `/dashboard/**`
- Admin users also have access to `/admin/**`
- Middleware redirects unauthenticated requests to `/login`

### Fact graph (Identity page)

The Identity page includes an interactive **3D force-directed graph** (powered by `react-force-graph-3d`) visualising the Neo4j fact graph, with a toggle to switch to a 2D layout:

- The logged-in user appears as the central orb
- Facts are grouped radially by category — 23 categories spanning identity, location, role, skill, education, project, goal, interest, hobby, habit, preference, personality, relationship, pet, health, diet, belief, value, religion, finance, lifestyle, event, and tool
- Each category has its own colour; a legend is shown in the bottom-left corner
- Clicking a fact node opens a side panel showing the **contributing memories** — the specific events that caused that fact to be extracted
- The sidebar is collapsed by default; click the expand handle to open it

### Source badges

Every memory card in the timeline and review queue displays a **source badge** showing how the memory entered the system. Badges are colour-coded by provenance:

| Badge | Colour | `source_type` |
|---|---|---|
| API | zinc | `api_explicit` |
| Manual | blue | `ui_manual` |
| Distilled | amber | `passive_distillation` |
| Streaming | orange | `passive_streaming` |
| Triggered | yellow | `trigger_word` |
| SDK | sky | `sdk_middleware` |
| Webhook | indigo | `webhook_ingest` |
| Tool | purple | `tool_use` |
| Synthesized | teal | `cross_system` |
| Voice Note | rose | `media_voice` |
| Document | slate | `media_document` |
| Image | cyan | `media_image` |

`api_explicit` is the default and shows no badge (to avoid visual noise for the common case). All other sources display an icon + label.

### Add Memory form

The **+ button** in the memory timeline toolbar opens a modal for manually recording a structured fact. Fields: category (all 23 fact categories), key, value, optional note. The fact is stored via `POST /memory/fact` with `source_type="ui_manual"` and confidence 1.0, and the identity graph + fact graph refresh automatically.

### Review page (`/dashboard/review`)

Auto-extracted memories (source types other than `api_explicit` and `ui_manual`) surface here for human review before they influence the knowledge graph. Features:

- Filterable by source type; counts shown per filter chip
- **Approve** (thumbs-up feedback) marks a memory as verified without deleting it
- **Remove** (trash) deletes the event entirely
- Approved items fade out of the queue immediately
- Empty state when nothing is pending review

### Memory graph (Memory detail page)

Each memory event has a dedicated narrative-links graph at `/dashboard/memories/[id]`:

- Nodes are rendered as orbs using `react-force-graph-2d`
- The focal event is centred; predecessor and successor events radiate outward
- `RELATED_TO` edges are shown in gold; other narrative links use type-coded colours (rose=caused, amber=preceded, violet=related, cyan=contradicts)
- Clicking any node navigates to that event's own graph

---

## API reference

### Authentication API

Smritikosh has a built-in user system. Most write endpoints require a Bearer JWT; the auth endpoints themselves are public.

#### `POST /auth/token`

Exchange a username + password for a JWT access token.

```bash
curl -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret123"}'
```

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "alice",
  "role": "user",
  "app_ids": ["default"]
}
```

Use the token in subsequent requests:

```bash
curl http://localhost:8080/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

Returns `401 Unauthorized` if credentials are wrong or the account is inactive.

#### `POST /auth/register`

Register a new user. **Requires admin JWT.**

```bash
curl -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-token>" \
  -d '{
    "username": "bob",
    "password": "securepass",
    "role": "user",
    "app_ids": ["default"],
    "email": "bob@example.com"
  }'
```

```json
{
  "user_id": "bob",
  "username": "bob",
  "role": "user",
  "app_ids": ["default"],
  "email": "bob@example.com",
  "is_active": true,
  "created_at": "2026-01-01T00:00:00+00:00"
}
```

Returns `409 Conflict` if the username is already taken. Returns `422` if `role` is invalid or password is shorter than 8 characters.

#### `GET /auth/me`

Return the currently authenticated user's profile.

```bash
curl http://localhost:8080/auth/me \
  -H "Authorization: Bearer <token>"
```

---

### `GET /health`

Checks server liveness **and** database connectivity. Useful for container readiness probes.

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "postgres": "ok",
  "neo4j": "ok",
  "mongodb": "ok",
  "llm_model": "claude-haiku-4-5-20251001",
  "llm_status": "ok"
}
```

| `status` value | Meaning |
|---|---|
| `"ok"` | Server running, all required services reachable |
| `"degraded"` | Server running, but one or more services are unavailable |
| `"error"` | Server internal error |

- `postgres` and `neo4j` are required — either `"ok"` or `"error"`.
- `mongodb` is optional — `"ok"`, `"error"`, or `"not_configured"` (when `MONGODB_URL` is unset).
- `llm_status` is `"ok"` when an API key is present (cloud providers) or a base URL is set (local providers). `llm_model` shows the resolved model name.

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

### `GET /memory/export`

Export all memory events for a user as NDJSON (newline-delimited JSON). Each line is a self-contained JSON object. Safe for large memory sets — the response streams incrementally.

```bash
curl -o memories_alice.ndjson \
  "http://localhost:8080/memory/export?user_id=alice&app_ids=myapp" \
  -H "Authorization: Bearer $TOKEN"
```

Each line:

```json
{"event_id":"a3f1c2d4-...","raw_text":"I switched to rocks.nvim today","summary":null,"importance_score":0.61,"consolidated":false,"recall_count":0,"cluster_label":"Editor & tooling preferences","created_at":"2026-04-11T09:14:22+00:00"}
```

| Field | Type | Description |
|---|---|---|
| `user_id` | string | **Required.** Query parameter — user whose memories to export |
| `app_ids` | string (repeatable) | Query parameter — filter by app namespace |

---

### `POST /memory/event`

Encode a raw interaction into memory. The full Hippocampus pipeline runs: importance scoring, embedding, fact extraction, episodic store, knowledge graph upsert.

```bash
curl -X POST http://localhost:8080/memory/event \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "content": "I decided to use PostgreSQL over MySQL.", "app_id": "myapp"}'
```

The optional `source_type` field defaults to `"api_explicit"`. You can pass any valid `SourceType` value.

---

### `POST /memory/fact`

Directly store a structured fact — bypasses LLM extraction entirely.

Use this for:
- **Manual entry** from the UI dashboard (`source_type="ui_manual"`, confidence=1.0)
- **SDK tool-use** intercepts when the LLM calls `remember()`
- **Programmatic seeding** of known facts at onboarding

```bash
curl -X POST http://localhost:8080/memory/fact \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "app_id": "myapp",
    "category": "preference",
    "key": "editor",
    "value": "neovim",
    "note": "user mentioned in onboarding survey"
  }'
```

```json
{
  "user_id": "alice",
  "app_id": "myapp",
  "category": "preference",
  "key": "editor",
  "value": "neovim",
  "confidence": 1.0,
  "frequency_count": 1,
  "source_type": "ui_manual",
  "status": "active",
  "first_seen_at": "2026-04-21T10:00:00+00:00",
  "last_seen_at": "2026-04-21T10:00:00+00:00"
}
```

Facts are upserted — calling with the same `(user_id, app_id, category, key)` increments `frequency_count` rather than creating a duplicate.

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

### `POST /admin/synthesize`

Trigger cross-system synthesis immediately for one user. Correlates signals across all connector sources (calendar, email, Slack, webhook) to infer behavioral patterns.

```bash
curl -X POST "http://localhost:8080/admin/synthesize?user_id=alice&app_id=myapp"
```

```json
{
  "user_id": "alice",
  "app_id": "myapp",
  "facts_written": 3,
  "facts_pending": 1,
  "facts_skipped": 2
}
```

### `GET /admin/embedding-health`

Report how many stored event embeddings match the currently configured dimension. Use this to detect stale embeddings after switching models.

```bash
curl http://localhost:8080/admin/embedding-health \
  -H "Authorization: Bearer <admin-token>"
```

```json
{
  "configured_dim": 768,
  "total_embedded": 1240,
  "stale_events": 0,
  "null_embeddings": 3,
  "healthy": true
}
```

`stale_events` counts vectors whose `vector_dims()` ≠ `EMBEDDING_DIMENSIONS`. Any non-zero value means hybrid search is comparing vectors of different sizes — results will be incorrect. Fix with `POST /admin/re-embed`.

### `POST /admin/re-embed`

Re-embed all events whose stored vector dimension doesn't match `EMBEDDING_DIMENSIONS`, plus any events with a missing embedding. Runs as a background task and returns immediately.

```bash
curl -X POST http://localhost:8080/admin/re-embed \
  -H "Authorization: Bearer <admin-token>"
```

```json
{"status": "started", "queued": 3}
```

When `queued` is 0, all embeddings are already healthy: `{"status": "ok", "queued": 0}`. Check `GET /admin/embedding-health` afterward to confirm.

All admin job routes return `503 Service Unavailable` if the background scheduler has not been started (e.g. bare ASGI without `lifespan`).

---

## Admin users API

Manage registered users. All endpoints require an **admin JWT**.

### `GET /admin/users`

Return a paginated list of all registered accounts.

```bash
curl "http://localhost:8080/admin/users?limit=20&offset=0" \
  -H "Authorization: Bearer <admin-token>"
```

```json
{
  "users": [
    {
      "username": "alice",
      "email": "alice@example.com",
      "role": "user",
      "app_ids": ["default"],
      "is_active": true,
      "created_at": "2026-01-01T00:00:00+00:00",
      "updated_at": "2026-01-01T00:00:00+00:00"
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

| Query param | Default | Description |
|---|---|---|
| `limit` | `50` | Results per page (1–200) |
| `offset` | `0` | Pagination offset |
| `role` | — | Filter by role (`user` or `admin`) |

### `GET /admin/users/{username}`

Fetch a single user by username.

```bash
curl http://localhost:8080/admin/users/alice \
  -H "Authorization: Bearer <admin-token>"
```

Returns `404` if the username does not exist.

### `PATCH /admin/users/{username}`

Update a user's `is_active` flag or `role`. Only the fields you send are changed.

```bash
# Deactivate a user
curl -X PATCH http://localhost:8080/admin/users/alice \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-token>" \
  -d '{"is_active": false}'

# Promote to admin
curl -X PATCH http://localhost:8080/admin/users/alice \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-token>" \
  -d '{"role": "admin"}'
```

Returns the updated user object. Returns `422` if `role` is not `"user"` or `"admin"`.

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

### Google OAuth Connectors (Gmail + Google Calendar)

Smritikosh can connect to Gmail and Google Calendar via OAuth2. Once authorized, you can sync emails and events on demand, and they flow into the daily synthesis job.

**Setup:**

1. Create a new OAuth2 application at [console.cloud.google.com](https://console.cloud.google.com).
2. Grant permissions: Gmail API (`gmail.readonly`) and Calendar API (`calendar.readonly`).
3. Create an OAuth2 credential (type: Desktop app). Download the client ID and secret.
4. Set environment variables:
   ```dotenv
   GOOGLE_CLIENT_ID=your_client_id_here
   GOOGLE_CLIENT_SECRET=your_client_secret_here
   GOOGLE_REDIRECT_URI=http://localhost:8080/connectors/google/callback
   ```

**Authorization:**

Get an authorization URL with a 1-hour expiry state token:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  'http://localhost:8080/connectors/google/authorize?user_id=alice&app_id=default'
```

```json
{"authorize_url": "https://accounts.google.com/o/oauth2/v2/auth?..."}
```

Visit that URL, grant Smritikosh access to your Gmail and Calendar, and Google redirects back to the callback. Tokens are encrypted and stored automatically.

**Sync Gmail:**

Fetch unread emails and ingest them as episodic memories:

```bash
curl -X POST http://localhost:8080/connectors/gmail/sync \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "limit": 20,
    "query": "is:unread"
  }'
```

```json
{"source": "gmail", "events_ingested": 5, "events_failed": 0, "event_ids": [...]}
```

**Sync Google Calendar:**

Fetch calendar events from the past N days:

```bash
curl -X POST http://localhost:8080/connectors/gcal/sync \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "days_back": 7,
    "max_results": 50
  }'
```

```json
{"source": "gcal", "events_ingested": 12, "events_failed": 0, "event_ids": [...]}
```

**Auto-refresh:**

Access tokens expire in ~1 hour. Smritikosh automatically refreshes them if they're about to expire — no action needed.

**Disconnect:**

Revoke access and delete stored tokens:

```bash
curl -X DELETE http://localhost:8080/connectors/alice/gmail \
  -H "Authorization: Bearer $TOKEN"
```

**Note:** If `GOOGLE_CLIENT_ID` is not set, all `/connectors/google/*` routes return `501 Not Configured`. This feature is entirely optional.

### `POST /ingest/session` — Passive extraction from conversation transcript

Post a full or partial conversation transcript. Smritikosh automatically:
1. Filters to **user turns only** (assistant turns discarded — anti-contamination)
2. Strips injected context sentinel blocks (`<!-- smritikosh:context-start/end -->`)
3. Runs the **trigger-word pre-filter** to skip the LLM on low-signal windows (cost saver)
4. Calls the LLM with a **delta-extraction prompt** that only asks for NEW or CONTRADICTING facts
5. Upserts surviving facts to the knowledge graph with `source_type="passive_distillation"` or `"trigger_word"`
6. Stores one episodic event summarising the session

The endpoint is **idempotent**: re-posting the same `session_id` is a safe no-op.

```bash
curl -X POST http://localhost:8080/ingest/session \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "app_id": "my-chatbot",
    "session_id": "sess-2026-04-21-abc",
    "turns": [
      {"role": "user",      "content": "I always prefer dark mode and I use neovim."},
      {"role": "assistant", "content": "Got it, I will keep that in mind."},
      {"role": "user",      "content": "My goal is to launch by Q3 this year."}
    ],
    "partial": false,
    "use_trigger_filter": true
  }'
```

```json
{
  "session_id": "sess-2026-04-21-abc",
  "user_id": "alice",
  "app_id": "my-chatbot",
  "turns_processed": 2,
  "facts_extracted": 3,
  "skipped_duplicates": 0,
  "extraction_skipped": false,
  "already_processed": false,
  "partial": false
}
```

**Streaming (mid-session) usage** — set `partial: true` for windows; the server tracks `last_turn_index` so each POST only processes new turns:

```bash
# Window 1 (partial)
curl -X POST http://localhost:8080/ingest/session \
  -d '{"session_id": "sess-xyz", "turns": [...first 10 turns...], "partial": true, ...}'

# Window 2 (partial)
curl -X POST http://localhost:8080/ingest/session \
  -d '{"session_id": "sess-xyz", "turns": [...next 10 turns...], "partial": true, ...}'

# Final close
curl -X POST http://localhost:8080/ingest/session \
  -d '{"session_id": "sess-xyz", "turns": [...last turns...], "partial": false, ...}'
```

**Alias**: `POST /ingest/transcript` — identical behaviour, kept for backwards compatibility.

---

## Media Ingestion — Voice Notes, Documents & Images (Phase 10 + 11)

Users can upload audio files, documents, and images to have facts automatically extracted. The pipeline transcribes audio, extracts or describes file content, applies first-person filtering to focus on the user, scores relevance, and routes high-confidence facts to memory while presenting borderline cases for user review.

### Supported media types

| Type | Formats | Extraction method |
|---|---|---|
| **Voice note** | MP3, WAV, M4A, WebM, OGG | Transcribed via Whisper (OpenAI or local) |
| **Meeting recording** | MP3, WAV, M4A, WebM, MP4 | Transcribed → diarized → user segments extracted |
| **Document** | PDF, TXT, MD, CSV | Text extracted; first-person filtered |
| **Receipt** | JPG, PNG, WebP, GIF | Vision model → purchase/lifestyle signals |
| **Screenshot** | JPG, PNG, WebP, GIF | Vision model → tool/tech/workflow signals |
| **Whiteboard** | JPG, PNG, WebP, GIF | Vision model → project/goal/decision signals |

### `POST /ingest/media` — upload media for extraction (202 async)

Upload a file and begin the extraction pipeline. Processing is asynchronous; the response includes a `media_id` for status polling.

```bash
# Upload a voice note
curl -X POST http://localhost:8080/ingest/media \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@note.mp3" \
  -F "user_id=alice" \
  -F "app_id=my-app" \
  -F "content_type=voice_note" \
  -F "context_note=thoughts on our upcoming launch"

# Upload a receipt image
curl -X POST http://localhost:8080/ingest/media \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@receipt.jpg" \
  -F "user_id=alice" \
  -F "app_id=my-app" \
  -F "content_type=receipt"
```

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | ✓ | The media file to upload |
| `user_id` | string | ✓ | User ID (must be self or admin) |
| `app_id` | string | — | App namespace; defaults to "default" |
| `content_type` | enum | ✓ | `voice_note` \| `meeting_recording` \| `document` \| `receipt` \| `screenshot` \| `whiteboard` |
| `context_note` | string | — | Optional context (e.g. "what should I know about this?") |
| `idempotency_key` | string | — | Optional; if provided, re-posting same key returns cached result |

**Response (202 Accepted):**

```json
{
  "media_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "alice",
  "app_id": "my-app",
  "content_type": "voice_note",
  "status": "processing",
  "facts_extracted": 0,
  "facts_pending_review": 0,
  "message": "Analysing your file…"
}
```

### `GET /ingest/media/{media_id}/status` — poll processing status

Poll the status of an in-progress or completed upload. Useful for updating UI progress indicators.

```bash
curl http://localhost:8080/ingest/media/550e8400-e29b-41d4-a716-446655440000/status \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**

```json
{
  "media_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "complete",
  "facts_extracted": 3,
  "facts_pending_review": 1,
  "pending_facts": [
    {
      "content": "User prefers asynchronous meetings",
      "category": "preference",
      "key": "meetings",
      "value": "async",
      "relevance_score": 0.68,
      "confidence": 0.68
    }
  ],
  "message": "Processing complete. 3 high-confidence facts saved; 1 pending your review."
}
```

**Status values:**
- `processing` — extraction in progress
- `complete` — extraction finished; facts may be saved or pending review
- `nothing_found` — no extractable facts found (likely third-person or irrelevant content)
- `failed` — extraction failed due to file corruption, unsupported format, or system error

### `POST /ingest/media/{media_id}/confirm` — save pending facts

When `facts_pending_review > 0`, confirm which ambiguous facts to save. Facts with relevance scores in the 0.60–0.75 range surface here before being written to memory.

```bash
curl -X POST http://localhost:8080/ingest/media/550e8400-e29b-41d4-a716-446655440000/confirm \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "app_id": "my-app",
    "confirmed_indices": [0]
  }'
```

**Body:**

| Field | Type | Description |
|---|---|---|
| `user_id` | string | User ID (for auth verification) |
| `app_id` | string | App namespace |
| `confirmed_indices` | int[] | Indices into `pending_facts` to save; empty array = dismiss all |

**Response (200 OK):**

```json
{
  "media_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "complete",
  "facts_extracted": 4,
  "facts_pending_review": 0,
  "message": "1 fact saved; 0 dismissed."
}
```

### Processing pipeline

```
1. Validate file extension + size
2. Route by content_type:
   a. voice_note  → Whisper transcription
   b. document    → PDF/text extraction
   c. receipt     → Vision model: "what does this receipt reveal about the user's lifestyle?"
      screenshot  → Vision model: "what tools/stack/workflow is visible?"
      whiteboard  → Vision model: "what project or goal is being planned?"
3. Apply first-person filter (documents + images) → keeps only "I/my/we" sentences
4. LLM extracts candidate facts from filtered content (delta-aware: skips already-known facts)
5. LLM scores each fact for relevance (0–1): "does this tell us something durable about the user?"
6. Route by score:
   - > 0.75 relevance → save immediately to memory (active status)
   - 0.60–0.75 relevance → queue for user confirmation modal
   - < 0.60 relevance → discard
7. Hippocampus registers the media upload as an episodic event
   (source_type = media_voice | media_document | media_image)
```

### Routing thresholds

| Relevance Score | Action |
|---|---|
| > 0.75 | ✅ Save immediately (high confidence) |
| 0.60–0.75 | 📋 Pending review — show in confirmation modal |
| < 0.60 | ❌ Discard (low relevance) |

### Whisper provider configuration

Transcription is powered by Whisper. Configure the provider in `.env`:

**Option 1: OpenAI Whisper (cloud)**
```bash
WHISPER_PROVIDER=openai
WHISPER_API_KEY=sk-...  # or falls back to EMBEDDING_API_KEY
WHISPER_MODEL=whisper-1
```

**Option 2: Local Whisper (self-hosted via ollama/vllm/llamacpp)**
```bash
WHISPER_PROVIDER=local
WHISPER_BASE_URL=http://localhost:8000/v1  # or http://localhost:11434 for ollama
WHISPER_MODEL=whisper-1  # or model name deployed at base URL
```

### Vision model configuration

Image description is powered by a multimodal vision model. Configure it in `.env`:

**Option 1: OpenAI (cloud, recommended)**
```bash
VISION_PROVIDER=openai
VISION_MODEL=gpt-4o-mini
VISION_API_KEY=sk-...  # or falls back to LLM_API_KEY
```

**Option 2: Anthropic Claude**
```bash
VISION_PROVIDER=claude
VISION_MODEL=claude-haiku-4-5-20251001
# uses LLM_API_KEY
```

**Option 3: Local (ollama)**
```bash
VISION_PROVIDER=ollama
VISION_MODEL=llava:13b
VISION_BASE_URL=http://localhost:11434
```

The model must support multimodal (image) inputs. The vision call is separate from the main LLM chat model — you can use a cheaper/faster vision model while keeping a more capable model for reasoning.

### Content-type extraction prompts

Each image subtype receives a targeted prompt to guide the vision model:

| `content_type` | Extraction focus |
|---|---|
| `receipt` | Items purchased, store, date → lifestyle, dietary, shopping preferences |
| `screenshot` | App name, tools, code, workflows → tech stack, expertise, work patterns |
| `whiteboard` | Projects, goals, decisions, diagrams → planning style, current initiatives |

### Size limits (configurable)

| Type | Default | Env variable |
|---|---|---|
| Audio file | 25 MB | `MEDIA_MAX_AUDIO_MB` |
| Document file | 10 MB | `MEDIA_MAX_DOCUMENT_MB` |
| PDF page count | 50 pages | `MEDIA_MAX_DOCUMENT_PAGES` |
| Image file | 20 MB | `MEDIA_MAX_IMAGE_MB` |
| Meeting recording | 500 MB | `MEDIA_MAX_MEETING_MB` |

### Source badges

Media facts appear in the dashboard with source badges:

| source_type | Badge | Color | Icon |
|---|---|---|---|
| `media_voice` | 🎙 Voice Note | Rose | Microphone |
| `media_document` | 📄 Document | Slate | Document |
| `media_image` | 🖼 Image | Cyan | Image |
| `media_audio` | 🎧 Meeting | Pink | Headphones |

---

## Meeting Recordings + Voice Enrollment (Phase 12)

Smritikosh can extract memories from meeting and call recordings. Because multiple people speak, the system identifies the user's voice segments before extraction — only what **you** said is analysed.

### How it works

```
Meeting recording (MP3, WAV, M4A, WebM, MP4)
        │
        ▼
  Whisper transcription (full audio)
        │
        ▼
  Has enrolled voice + diarization enabled?
     │                          │
    YES                         NO
     │                          │
     ▼                          ▼
  Diarize → find user speaker   First-person filter
  → extract user segments       (I / my / we / our)
        │
        ▼
  Fact extraction → relevance scoring → write/pending/discard
  (source_type = media_audio)
```

### Voice enrollment

Enroll once from the dashboard (Settings → Voice) or via API:

```bash
# Upload a 30-second voice sample
curl -X POST http://localhost:8080/user/alice/voice-enrollment \
  -H "Authorization: Bearer $TOKEN" \
  -F "app_id=default" \
  -F "file=@sample.wav"
```

```json
{
  "user_id": "alice",
  "enrolled": true,
  "has_embedding": true,
  "embedding_dim": 256,
  "enrolled_at": "2026-04-25T10:00:00+00:00",
  "message": "Voice enrolled successfully with speaker embedding."
}
```

Without resemblyzer installed (`pip install resemblyzer`), enrollment is recorded but speaker matching falls back to the first-person filter — still useful, just less precise.

### Voice enrollment endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/user/{user_id}/voice-enrollment` | Upload 30-sec sample, compute d-vector, store profile |
| `GET` | `/user/{user_id}/voice-enrollment` | Check enrollment status |
| `DELETE` | `/user/{user_id}/voice-enrollment` | Remove voice profile |

### Meeting recording upload

```bash
curl -X POST http://localhost:8080/ingest/media \
  -H "Authorization: Bearer $TOKEN" \
  -F "user_id=alice" \
  -F "app_id=default" \
  -F "content_type=meeting_recording" \
  -F "context_note=Weekly engineering standup" \
  -F "file=@standup_2026-04-25.mp4"
```

Returns `media_id` immediately; poll `/ingest/media/{id}/status` for results. Same two-tier routing as other media types (auto-save > 0.75, pending review 0.60–0.75).

### Diarization configuration

| `DIARIZATION_PROVIDER` | Behaviour | Requirements |
|---|---|---|
| `none` (default) | No diarization — first-person filter on full transcript | None |
| `pyannote` | Full speaker diarization + voice matching | `pip install pyannote.audio torch` + `HF_TOKEN` |

```bash
# .env — to enable pyannote diarization:
DIARIZATION_PROVIDER=pyannote
HF_TOKEN=hf_your_read_token_here      # https://huggingface.co/settings/tokens
SPEAKER_SIMILARITY_THRESHOLD=0.75     # 0–1; lower = more permissive matching
```

For speaker d-vector embedding (voice enrollment matching):
```bash
pip install resemblyzer          # lightweight, no HF auth required
# or install the optional extra:
pip install smritikosh[diarization]  # includes resemblyzer + pyannote.audio + torch
```

---

## Passive Memory Extraction

Smritikosh learns from conversations automatically. No per-turn developer work required.

### How it works

```
Conversation transcript
        │
        ▼
  user_turns_only()       ← strips assistant turns (anti-contamination)
        │
        ▼
  strip_sentinels()       ← removes injected context blocks
        │
        ▼
  TriggerDetector         ← regex pre-filter (skips LLM if no high-signal phrases)
        │
        ▼
  delta extraction LLM    ← "extract only NEW facts not already known"
        │
        ▼
  SemanticMemory.upsert() ← source_type="passive_distillation" | "trigger_word"
```

### Source types

Every memory now carries a `source_type` tracking how it entered the system:

| `source_type` | Confidence | Description |
|---|---|---|
| `ui_manual` | 1.00 | User typed it in the dashboard |
| `api_explicit` | 0.90 | App called `POST /memory/event` directly |
| `tool_use` | 0.90 | LLM called the `remember()` tool |
| `trigger_word` | 0.85 | Trigger phrase detected; LLM confirmed |
| `media_voice` | 0.85 | Extracted from uploaded voice note |
| `passive_distillation` | 0.75 | Post-session extraction from transcript |
| `media_document` | 0.75 | Extracted from uploaded document |
| `passive_streaming` | 0.70 | Mid-session rolling-window extraction |
| `sdk_middleware` | 0.70 | SDK wrapper intercepted transparently |
| `webhook_ingest` | 0.70 | Structured transcript via `/ingest/transcript` |
| `media_image` | 0.70 | Extracted from uploaded image (receipt/screenshot/whiteboard) |
| `cross_system` | 0.65 | Synthesized from cross-integration signals |

### `remember()` tool — LLM-curated memory (Phase 5)

The middleware automatically injects a `remember()` tool into every LLM call. When the LLM decides something is worth remembering, it calls the tool — the middleware intercepts it, calls `POST /memory/fact` with `source_type="tool_use"`, and continues the conversation transparently. The app developer sees no tool calls.

```python
# The LLM may call remember() during reasoning:
# {"name": "remember", "arguments": {"content": "User prefers neovim", "category": "preference", "key": "editor", "value": "neovim"}}
# → middleware saves the fact, returns synthetic tool_result, calls LLM again
# → app receives the continuation response as if no tool call happened
```

**Two cases handled transparently:**
- **All calls are `remember()`** → facts saved, follow-up LLM call made, continuation returned to app
- **Mixed tool calls** → `remember()` facts saved in background, other tool calls passed through to app

**Disable** with `enable_remember_tool=False` on the middleware constructor.

### Fact status

Facts have a lifecycle — the QC layer gates what enters context assembly:

| Status | Meaning |
|---|---|
| `active` | Included in context assembly |
| `pending` | Below confidence threshold (0.60) or awaiting user review |
| `rejected` | User dismissed or system discarded |

### Quality Control Layer (Phase 6)

All passive extraction paths pass through a shared control layer before facts are durably written:

**1. Confidence threshold gate** — facts below 0.60 confidence are written as `pending` and excluded from context assembly until the user approves them.

**2. Contradiction detection** — before every fact write, Smritikosh checks whether the same `(user, app, category, key)` already has a *different* value:
- Confidence delta > 0.15 → overwrite automatically (logs superseded value to `source_meta.superseded`)
- Confidence delta ≤ 0.15 → create a `fact_contradictions` record and skip the write; surface to user for resolution via `/facts/contradictions/{user_id}`
- `ui_manual` source → always overwrites (explicit user intent wins)

**3. Decay rules** (weekly FactDecayer job):
- `ui_manual` facts are **never decayed** — user explicitly stated them
- `cross_system` facts decay **2× faster** — behavioral patterns shift quickly
- Facts falling below confidence 0.20 are promoted to `pending` before eventual deletion

### Anti-contamination rules

1. **User-turns only** — assistant messages are discarded before extraction
2. **Sentinel stripping** — `<!-- smritikosh:context-start/end -->` blocks removed
3. **Delta prompt** — LLM told to extract only NEW or CONTRADICTING facts
4. **Trigger pre-filter** — skip LLM entirely if no high-signal phrases found (cost saver)

### Sample demo scripts

#### `passive_extraction_demo.py` — session ingest + manual facts

```bash
# Seed Priya's base memories first (one-time)
python sample/seed_priya.py

# Run the end-to-end passive extraction demo
python sample/passive_extraction_demo.py
```

Posts a realistic 7-turn conversation, verifies trigger detection, tests idempotency, demonstrates streaming windows, stores manual facts via `store_fact()`, and checks that all extracted facts appear in context retrieval.

#### `middleware_demo.py` — transparent LLM wrapper (Phase 4)

```bash
python sample/middleware_demo.py
```

Wraps a fake OpenAI-style client with `SmritikoshMiddleware` — no real API keys required. Shows turn buffering, windowed partial flushing every N turns, `auto_inject=True` context prepending, and final flush on `close()`. Swap `FakeOpenAI()` for `openai.OpenAI()` or `anthropic.Anthropic()` in production.

### Dashboard integration (Phase 7)

All memory source types are surfaced in the dashboard UI:

- **Source badges** — every memory card shows a colour-coded badge (amber = Distilled, sky = SDK, blue = Manual, etc.). `api_explicit` shows no badge to avoid noise.
- **Review queue** (`/dashboard/review`) — auto-extracted memories appear here for human review. Approve (thumbs-up) or remove (trash) each one; filter by source type.
- **Add Memory form** — the `+` button in the memory timeline opens a modal to manually record a structured fact (category → key → value). Stored as `ui_manual` with confidence 1.0; the identity graph refreshes automatically.

---

## Facts QC API

These endpoints manage the Quality Control layer for semantic facts — the review queue, status changes, and contradiction resolution.

### `GET /facts/{user_id}`

List semantic facts for a user. Use `status=pending` to get the review queue.

```bash
# Review queue — all pending facts awaiting approval
curl "http://localhost:8080/facts/alice?status=pending&app_id=myapp" \
  -H "Authorization: Bearer $TOKEN"

# All facts regardless of status
curl "http://localhost:8080/facts/alice?app_id=myapp" \
  -H "Authorization: Bearer $TOKEN"

# Filter by category
curl "http://localhost:8080/facts/alice?category=preference&app_id=myapp" \
  -H "Authorization: Bearer $TOKEN"
```

```json
{
  "user_id": "alice",
  "app_id": "myapp",
  "facts": [
    {
      "category": "preference",
      "key": "editor",
      "value": "neovim",
      "confidence": 0.54,
      "frequency_count": 1,
      "status": "pending",
      "source_type": "passive_distillation",
      "first_seen_at": "2026-04-23T10:00:00+00:00",
      "last_seen_at": "2026-04-23T10:00:00+00:00"
    }
  ],
  "total": 1
}
```

### `PATCH /facts/{user_id}/{category}/{key}/status`

Approve (`active`) or reject a pending fact.

```bash
# Approve a pending fact
curl -X PATCH "http://localhost:8080/facts/alice/preference/editor/status?app_id=myapp" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "active"}'

# Reject a pending fact
curl -X PATCH "http://localhost:8080/facts/alice/preference/editor/status?app_id=myapp" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "rejected"}'
```

### `GET /facts/contradictions/{user_id}`

List unresolved contradictions — cases where a new extraction proposed a different value for an existing fact and the confidence delta was too small to auto-overwrite.

```bash
curl "http://localhost:8080/facts/contradictions/alice?app_id=myapp" \
  -H "Authorization: Bearer $TOKEN"
```

```json
{
  "user_id": "alice",
  "app_id": "myapp",
  "contradictions": [
    {
      "id": "b2c3d4e5-...",
      "category": "preference",
      "key": "editor",
      "existing_value": "neovim",
      "existing_confidence": 0.90,
      "candidate_value": "emacs",
      "candidate_source": "passive_distillation",
      "candidate_confidence": 0.75,
      "created_at": "2026-04-23T12:00:00+00:00"
    }
  ],
  "total": 1
}
```

### `PATCH /facts/contradictions/{contradiction_id}`

Resolve a contradiction. `keep=existing` dismisses the candidate; `keep=candidate` overwrites the fact with the candidate value.

```bash
# Keep the existing value — dismiss the candidate
curl -X PATCH "http://localhost:8080/facts/contradictions/b2c3d4e5-..." \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"keep": "existing"}'

# Take the candidate — overwrite the fact
curl -X PATCH "http://localhost:8080/facts/contradictions/b2c3d4e5-..." \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"keep": "candidate"}'
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

## Authentication & API keys

Every API endpoint (except `/auth/token`, `/auth/register`, and `/health`) requires a Bearer token.  Two token formats are accepted:

| Format | When to use |
|--------|-------------|
| **JWT** | UI sessions and short-lived programmatic access. Obtained via `POST /auth/token`. Expires after `JWT_EXPIRE_DAYS` (default 30 days). |
| **API key** (`sk-smriti-…`) | SDK integrations, CI pipelines, external tools. Never expires unless revoked. |

### Generating an API key

**Via the dashboard** (recommended): sign in → **API Keys** in the left sidebar → **New key** → copy the key immediately (shown once only).

**Via the API:**

```bash
# 1. Get a JWT first
TOKEN=$(curl -s -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "alicepass"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Generate a key
curl -s -X POST http://localhost:8080/keys \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "My integration", "app_ids": ["default"]}' \
  | python3 -m json.tool
# Returns: { "id": "...", "key": "sk-smriti-abc123...", "key_prefix": "abc123ab", ... }
# The full key is returned ONCE — store it immediately.
```

### Using an API key

Pass it as a Bearer token, exactly like a JWT:

```bash
curl -s http://localhost:8080/memory/alice \
  -H "Authorization: Bearer sk-smriti-your-key-here"
```

In the Python sample client:

```python
# Via constructor
client = SmritikoshClient(api_key="sk-smriti-your-key-here")

# Via environment variable (recommended for CI / production)
# export SMRITIKOSH_API_KEY=sk-smriti-your-key-here
client = SmritikoshClient()
```

### Listing and revoking keys

```bash
# List active keys
curl -s http://localhost:8080/keys \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

# Revoke a key
curl -s -X DELETE http://localhost:8080/keys/<key-id> \
  -H "Authorization: Bearer $TOKEN"
```

### Access control

All memory endpoints enforce ownership — a user can only read and write their own data:

| Caller | Access |
|--------|--------|
| Regular user token / API key | Own data only |
| Admin token / API key | Any user's data |
| No token | 401 Unauthorized |

Background jobs and admin job triggers bypass the HTTP layer entirely — they call Python functions directly and are unaffected by API auth.

---

## Python SDK

Install the package (the SDK is included):

```bash
pip install smritikosh          # or pip install -e . from the repo
```

### Basic usage

The SDK accepts either an API key or username/password:

```python
# API key (recommended for integrations)
client = SmritikoshClient(
    base_url="http://localhost:8080",
    app_id="myapp",
    headers={"Authorization": "Bearer sk-smriti-your-key-here"},
)

# Username/password (obtains a JWT automatically)
# Use the sample client.py for this pattern — the async SDK accepts pre-built headers.
```

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

Each memory event is stored under a single `app_id` namespace. Tokens (JWTs and API keys) carry an `app_ids` list that controls which namespaces can be read. This means:

- **Write**: always to one `app_id` (where the memory lives)
- **Read**: across any subset of `app_ids` your token has access to

```python
# Store memories in two separate namespaces
chat_client  = SmritikoshClient(base_url="...", app_id="chat-app")
docs_client  = SmritikoshClient(base_url="...", app_id="docs-app")

# An API key scoped to both namespaces can read across them in a single context call
curl -X POST /context -d '{"user_id": "alice", "query": "...", "app_ids": ["chat-app", "docs-app"]}'
```

Generate API keys scoped to specific app namespaces from the dashboard under **Settings → API Keys**.

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
| `build_context(user_id, query, *, app_id)` | Retrieve LLM-ready context → `MemoryContext` with similar events, user profile, procedures, and narrative chains |
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
| `ingest_session(user_id, turns, *, session_id, partial, use_trigger_filter, metadata)` | Submit a conversation transcript for passive extraction → `SessionIngestResult` |
| `ingest_push(user_id, content, *, source, source_id, app_id, metadata)` | Push a single event from an external source → `IngestResult` |
| `ingest_file(user_id, file_content, filename, *, app_id)` | Upload a file (txt/md/csv/json) → `IngestResult` |
| `ingest_email(user_id, host, username, password, *, ...)` | Fetch IMAP emails → `IngestResult` |
| `ingest_calendar(user_id, file_content, *, filename, app_id)` | Upload an `.ics` file → `IngestResult` |
| `health()` | Server + DB liveness check → `HealthStatus` |

### SmritikoshMiddleware — transparent LLM wrapper

`SmritikoshMiddleware` wraps any OpenAI or Anthropic **sync** client. The developer changes one line; memory extraction and the `remember()` tool happen invisibly in the background.

```python
from smritikosh.sdk import SmritikoshMiddleware
import openai  # or: import anthropic

# OpenAI
with SmritikoshMiddleware(
    openai.OpenAI(),
    smritikosh_url="http://localhost:8080",
    smritikosh_api_key="sk-smriti-...",   # JWT or API key
    user_id="alice",
    app_id="my-app",
    extract_every_n_turns=10,      # partial flush every N user turns (0 = only on close)
    use_trigger_filter=True,       # skip LLM when no trigger phrases detected
    auto_inject=False,             # True → prepend memory context before each call
    enable_remember_tool=True,     # True → auto-inject remember() tool + intercept calls
) as llm:
    response = llm.chat.completions.create(model="gpt-4o", messages=[...])
    # ... more turns ...
# close() / __exit__ flushes remaining turns as the final ingest
```

```python
# Anthropic — identical, just swap the client
with SmritikoshMiddleware(
    anthropic.Anthropic(),
    smritikosh_url="http://localhost:8080",
    smritikosh_api_key="sk-smriti-...",
    user_id="alice",
) as llm:
    response = llm.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=1024, messages=[...]
    )
```

**How it works:**

1. All attribute access is proxied to the underlying LLM client unchanged
2. `chat.completions.create()` / `messages.create()` are intercepted
3. The `remember()` tool definition is auto-injected into every `tools` list
4. When the LLM returns a `remember()` tool call, the middleware saves the fact via `POST /memory/fact` and makes a transparent follow-up call — the app never sees the `remember()` call
5. User turns are buffered in memory for this session
6. Every N user turns a `POST /ingest/session` (partial) fires in a background thread — sends only the **new** turns since the last partial flush (streaming window); server uses `last_turn_index` to resume correctly
7. `close()` or `__exit__` sends the final non-partial ingest containing only turns not yet covered by a partial flush
8. If `auto_inject=True`, `GET /context` is called before each LLM call and the result is prepended as a sentinel-wrapped system message (which the extraction pass strips automatically)
9. All network errors are swallowed — the LLM call is never blocked or interrupted

**Run the demo:**

```bash
python sample/middleware_demo.py   # no OpenAI key required — uses a fake client
```

### LiteLLMMiddleware — multi-provider wrapper

`LiteLLMMiddleware` is a subclass of `SmritikoshMiddleware` that wraps `litellm.completion()` instead of a provider-specific client. Use this when your app targets multiple providers (Gemini, Ollama, vLLM, llama.cpp) through LiteLLM's unified interface.

```python
import litellm
from smritikosh.sdk import LiteLLMMiddleware

with LiteLLMMiddleware(
    litellm,
    smritikosh_url="http://localhost:8080",
    smritikosh_api_key="sk-smriti-...",
    user_id="alice",
    app_id="my-app",
) as mw:
    # Gemini
    response = mw.completion(model="gemini/gemini-1.5-pro", messages=[...])

    # Ollama (local)
    response = mw.completion(model="ollama_chat/llama3", messages=[...])

    # vLLM / llama.cpp
    response = mw.completion(
        model="openai/my-model",
        api_base="http://localhost:8000/v1",
        messages=[...],
    )
```

**Provider coverage:**

| Provider | Config (`LLM_PROVIDER`) | Middleware |
|---|---|---|
| OpenAI / Azure | `openai` | `SmritikoshMiddleware(openai.OpenAI(...))` |
| Anthropic (Claude) | `claude` | `SmritikoshMiddleware(anthropic.Anthropic())` |
| Google Gemini | `gemini` | `LiteLLMMiddleware(litellm, model="gemini/...")` |
| Ollama (local) | `ollama` | `LiteLLMMiddleware(litellm, model="ollama_chat/...")` |
| vLLM (local) | `vllm` | `LiteLLMMiddleware(litellm, model="openai/...", api_base=...)` |
| llama.cpp (local) | `llamacpp` | `LiteLLMMiddleware(litellm, model="openai/...", api_base=...)` |

All features (turn buffering, `remember()` tool, context injection, session ingest) work identically across providers because LiteLLM responses follow the OpenAI schema.

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

The default run executes **~830 tests** in about 10–15 seconds. All tests that require real API keys, a local Ollama server, or running databases are automatically skipped.

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
| `test_context_builder.py` | 42 | Deduplication, degraded-mode fallbacks, prompt rendering, narrative chain boost, chain_top_k |
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
| `test_intent_classifier.py` | 32 | Weight table, keyword detection, confidence, two-tier classify_async, LLM fallback |
| `test_fact_decayer.py` | 11 | Decay Cypher execution, count forwarding, config defaults, error skip |
| `test_e2e_pipeline.py` | 17 | Full encode → consolidate → context pipeline; embedding failure survival |
| `test_trigger_detector.py` | 33 | TriggerDetector patterns, filter_turns, any_triggered, collect_all_phrases; transcript_utils sentinel stripping and delta prompt |
| `test_session_ingest.py` | 21 | POST /ingest/session (201, shape, idempotency, trigger filter, partial flag, assistant-only → 0 turns); POST /memory/fact (ui_manual defaults, confidence, status, low-confidence → pending) |
| `test_sdk_middleware.py` | 55 | SmritikoshMiddleware proxy, buffering, partial flush threshold, close() idempotency, auto_inject (OpenAI + Anthropic), remember() tool injection + interception, LiteLLMMiddleware, windowed streaming, error resilience, thread safety |
| `test_cross_system_synthesizer.py` | 20 | CrossSystemSynthesizer prompt builder, connector summary builder, active/pending/skipped confidence routing, LLM failure handling, empty response handling |
| `test_media_processor.py` | 31 | MediaProcessor transcription, PDF/text extraction, vision model description, first-person filter, relevance scoring, content-type routing, error handling; image subtypes (receipt/screenshot/whiteboard) |
| `test_media_ingest.py` | 12 | POST /ingest/media (upload, status polling, confirm), async processing, idempotency, size gate enforcement |
| `test_voice_enrollment.py` | 15 | Voice enrollment API (enroll/status/delete/re-enroll), meeting recording processor (validation, first-person fallback, diarization path, no-embedding fallback, size limit) |

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

### Provider fallback chain

Configure a secondary LLM so Smritikosh keeps working when the primary provider is down or rate-limited. After exhausting three retries on the primary, all chat/extraction calls are automatically retried against the fallback before raising an error.

```dotenv
# Primary
LLM_PROVIDER=claude
LLM_MODEL=claude-haiku-4-5-20251001
LLM_API_KEY=sk-ant-...

# Fallback — used when primary exhausts retries
LLM_FALLBACK_PROVIDER=openai
LLM_FALLBACK_MODEL=gpt-4o-mini
LLM_FALLBACK_API_KEY=sk-...   # optional — omit if same as LLM_API_KEY
```

The fallback covers `complete()` and `extract_structured()` calls (fact extraction, importance scoring, consolidation, belief mining). Embedding calls are not covered — they have separate retry logic.

When the fallback fires, a `WARNING` log line is emitted:

```
Primary LLM exhausted retries (model=claude-haiku-4-5-20251001): ... — trying fallback=gpt-4o-mini
```

### llama.cpp (local)

`llama-server` exposes an OpenAI-compatible API, so Smritikosh treats it as a local OpenAI endpoint. Download and build [llama.cpp](https://github.com/ggml-org/llama.cpp), then start the server with your GGUF model:

```bash
llama-server -m /path/to/model.gguf --port 8081 --embedding
```

The `--embedding` flag enables the native `/embedding` endpoint that Smritikosh uses for vector generation.

```dotenv
LLM_PROVIDER=llamacpp
LLM_MODEL=my-model                 # name is passed through; llama-server ignores it
LLM_BASE_URL=http://localhost:8081/v1

EMBEDDING_PROVIDER=llamacpp
EMBEDDING_MODEL=my-model
EMBEDDING_BASE_URL=http://localhost:8081  # note: no /v1 — uses native /embedding path
EMBEDDING_DIMENSIONS=4096          # match your model's output dimension
```

> **Tip:** llama.cpp does not require an API key. You do not need to set `LLM_API_KEY`.

---

## Background jobs

The `MemoryScheduler` runs five jobs inside the FastAPI process using APScheduler:

| Job | Default interval | What it does |
|---|---|---|
| **Consolidation** | every 1 hour | Compresses raw events → summaries + Neo4j facts |
| **Synaptic pruning** | every 24 hours | Deletes old low-scoring events |
| **Memory clustering** | every 6 hours | Groups similar events by topic using embeddings |
| **Belief mining** | every 12 hours | Infers durable beliefs and values from event patterns |
| **Semantic fact decay** | every 1 week | Decays Neo4j fact confidence over time; deletes facts below threshold |
| **Cross-system synthesis** | daily at 01:00 UTC | Correlates calendar/email/Slack/webhook behavioral signals → `cross_system` facts |

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

After each pruning run, **fact garbage collection** removes Neo4j facts for that user whose `last_seen_at` timestamp falls within the same age window as the deleted events. These facts were only reinforced by the pruned events and will never be re-confirmed by future consolidation — removing them immediately avoids waiting for the weekly confidence-decay cycle.

### Memory clustering (every 6 hours)

Groups events with embeddings into topical clusters using a **two-phase algorithm**:

1. **Greedy centroid pass** — each event is assigned to the nearest existing cluster centroid if cosine similarity ≥ 0.75, otherwise starts a new cluster.
2. **K-means refinement** (3 iterations) — recomputes true mean centroids from current assignments and reassigns every point to the nearest centroid. Removes order-dependence and moves borderline points to their best cluster.

Each cluster is labelled by the LLM (`cluster_label`) and stored on the event rows. Requires at least 5 events with embeddings to run.

### Belief mining (every 12 hours)

Reads consolidated events (minimum 3) and semantic facts, then prompts the LLM to infer higher-order beliefs and values. Results are upserted into `user_beliefs` — `evidence_count` increments each time the same belief is independently inferred, reinforcing confidence over time.

### Semantic fact decay (every week)

Facts in Neo4j persist indefinitely by default. The `FactDecayer` applies exponential confidence decay to every fact whose `last_seen_at` timestamp is older than a full epoch:

```
new_confidence = confidence × exp(−ln2 × age_days / half_life_days)
```

The default half-life is **60 days** (`FACT_DECAY_HALF_LIFE_DAYS`). Facts whose confidence drops below `0.1` (`FACT_DECAY_FLOOR`) are deleted. Orphaned `Fact` nodes with no remaining user relationship are also cleaned up in a third pass.

### Cross-system synthesis (daily at 01:00 UTC)

The `CrossSystemSynthesizer` queries connector-originated events (calendar, email, Slack, webhook) for the last 30 days, computes per-connector behavioral summaries (send-time distributions, active weekdays, topic samples), and prompts the LLM to infer durable patterns that no single source could surface alone.

Examples of what it finds:
- "User rescheduled 3 meetings this week" + "mentioned being overwhelmed in chat" → stress/capacity preference
- "No emails after 6pm for 30 days" + "mentioned work-life balance" → boundary preference
- "Slack messages spike on Tuesdays" + "mentioned standup prep" → weekly routine

Confidence routing: ≥ 0.50 → `active`, 0.40–0.49 → `pending` for user review, < 0.40 → skipped.

Facts are tagged `source_type="cross_system"` and decay **2× faster** than other sources in the weekly decay job — behavioral patterns shift quickly.

Trigger via `POST /admin/synthesize` or from Python:

```python
await scheduler.run_synthesis_now(user_id="alice", app_id="myapp")
await scheduler.run_synthesis_for_all_users()
```

### Manual triggers (admin / testing)

```python
from smritikosh.processing.scheduler import MemoryScheduler

# Trigger immediately for one user
await scheduler.run_consolidation_now(user_id="alice", app_id="myapp")
await scheduler.run_pruning_now(user_id="alice", app_id="myapp")
await scheduler.run_clustering_now(user_id="alice", app_id="myapp")
await scheduler.run_belief_mining_now(user_id="alice", app_id="myapp")
await scheduler.run_synthesis_now(user_id="alice", app_id="myapp")

# Run batch across all users
await scheduler.run_consolidation_for_all_users()
await scheduler.run_pruning_for_all_users()
await scheduler.run_clustering_for_all_users()
await scheduler.run_belief_mining_for_all_users()
await scheduler.run_fact_decay()
await scheduler.run_synthesis_for_all_users()
```

### Tune the schedule

Pass custom intervals when constructing the scheduler (or subclass `MemoryScheduler`):

```python
MemoryScheduler(
    consolidator=..., pruner=..., episodic=...,
    clusterer=..., belief_miner=..., fact_decayer=...,
    consolidation_hours=2,    # consolidate every 2 hours
    pruning_hours=48,         # prune every 2 days
    clustering_hours=12,      # cluster every 12 hours
    belief_mining_hours=24,   # mine beliefs once a day
    fact_decay_weeks=2,       # decay facts every 2 weeks
)
```

---

## Production deployment

### Build and run with Docker

```bash
# Build the image
docker build -t smritikosh-api:latest .

# Run a single container (databases must already be accessible)
docker run -d \
  --name smritikosh \
  -p 8080:8080 \
  --env-file .env.prod \
  smritikosh-api:latest
```

The image is a two-stage build:
- **Stage 1 (builder):** installs all Python deps into `/install` using a full build toolchain.
- **Stage 2 (runtime):** copies only `/install` — no compiler, no build tools. Runs as non-root user `smriti` (uid 1001).

The container runs `alembic upgrade head` before starting uvicorn, so migrations are always applied on deploy.

A `HEALTHCHECK` polls `GET /health` every 30 seconds. Container orchestrators (ECS, Kubernetes, Fly.io) use this to determine readiness.

### Full stack with Docker Compose

`docker-compose.prod.yml` wires up the API, PostgreSQL, Neo4j, and MongoDB with production-safe defaults:
- Databases are not exposed to the host (internal network only)
- `restart: unless-stopped` on all services
- `depends_on` with `condition: service_healthy` so the API only starts after all databases pass their health checks
- All secrets come from environment variables (never hardcoded)

```bash
# Create your production .env (never commit this)
cp .env.example .env.prod
# Fill in POSTGRES_PASSWORD, NEO4J_PASSWORD, JWT_SECRET, LLM_*, EMBEDDING_*

# Start everything
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f api

# Tear down (data volumes are preserved)
docker compose -f docker-compose.prod.yml down
```

### Deploying without Docker

If you deploy directly to a VM or PaaS (e.g. Railway, Render, Fly.io):

```bash
pip install .
alembic upgrade head
uvicorn smritikosh.api.main:app --host 0.0.0.0 --port 8080 --workers 2
```

Set all environment variables from `.env.example` in your platform's config panel.

---

## Data reset script

`scripts/reset_data.py` wipes user data across all three databases (PostgreSQL,
Neo4j, MongoDB) in a single command. Useful for development, testing, or
clearing a demo environment.

### Usage

```bash
# Preview what would be deleted — no changes made
python scripts/reset_data.py --dry-run

# Wipe all data for one user (keeps their login account)
python scripts/reset_data.py --user alice

# Wipe all user data across all DBs (keeps app_users / login accounts)
python scripts/reset_data.py

# Full factory reset — wipes everything including user accounts
python scripts/reset_data.py --include-users
```

All destructive modes prompt `Type 'yes' to confirm` before executing.

### What gets cleared

| Database | Tables / collections cleared |
|---|---|
| **PostgreSQL** | `events`, `memory_links`, `memory_feedback`, `user_facts`, `user_beliefs`, `user_procedures` |
| **Neo4j** | All nodes and relationships (or only those belonging to `--user`) |
| **MongoDB** | All collections in `smritikosh_audit` |

`--user` mode filters PostgreSQL tables by `user_id` and Neo4j/MongoDB by
`user_id` field. `memory_links` has no `user_id` column so it is cleaned via a
subquery on the `events` table.

`--include-users` additionally truncates `app_users` (PostgreSQL) — use this
to reset a demo environment back to a completely blank state.
