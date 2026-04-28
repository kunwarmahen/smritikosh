# Smritikosh — Sample Project

A minimal memory-aware chatbot that demonstrates the full Smritikosh flow:
store memories → search → inject context → LLM call.

## Prerequisites

- Smritikosh server running on `http://localhost:8080`
- Demo users created (see [QUICKSTART.md](../QUICKSTART.md) Step 10)

## Setup

```bash
pip install httpx openai python-dotenv   # openai SDK works with ollama, openai, and gemini
```

Copy the local env template and configure your active user:

```bash
cp .env.example .env
# then edit .env — at minimum set SMRITIKOSH_USER and SMRITIKOSH_USER_PASS
```

## Files

| File | Purpose |
|---|---|
| `client.py` | Thin wrapper around the Smritikosh REST API |
| `seed.py` | Pre-loads 10 memories for `alice` — run once |
| `seed_priya.py` | Pre-loads 15 memories for `priya` — run once |
| `chatbot.py` | Interactive memory-aware chatbot loop |
| `passive_extraction_demo.py` | End-to-end passive extraction demo — session ingest, idempotency, streaming windows, manual facts |
| `middleware_demo.py` | SmritikoshMiddleware demo — transparent LLM wrapper with turn buffering and `remember()` tool; uses a fake client (no API key needed) |
| `litellm_middleware_notebook.ipynb` | Jupyter notebook — LiteLLMMiddleware examples for Ollama, vLLM, Gemini, OpenAI, and Claude; includes multi-turn windowed-flush demo and memory verification cells |
| `media_ingest_demo.py` | Media ingestion demo — upload a personal notes document, poll for extraction, review and confirm facts; no Whisper or Vision key needed |
| `.env.example` | Template for local overrides (copy to `.env`) |

## Demo personas

### Alice — ML engineer

```bash
# 1. Seed her memories (run once)
python seed.py

# 2. Set her as the active user in sample/.env
#    SMRITIKOSH_USER=alice
#    SMRITIKOSH_USER_PASS=alicepass

# 3. Chat
python chatbot.py
```

Alice is a machine learning engineer at a Series B startup. She knows Python,
is learning Rust, runs a RAG pipeline, and uses Neovim.

### Priya — lifestyle & travel

```bash
# 1. Seed her memories (run once)
python seed_priya.py

# 2. Switch the active user in sample/.env
#    SMRITIKOSH_USER=priya
#    SMRITIKOSH_USER_PASS=priyapass

# 3. Chat
python chatbot.py
```

Priya is a homemaker who loves fashion (Chanel, Bottega Veneta), reads literary
fiction, travels to exotic destinations (Maldives, Patagonia, Kyoto), has a
wealthy husband Rohan and two kids — Aanya (8) and Kabir (5).

Try asking her:
- *"What should I pack for our Japan trip?"*
- *"Can you recommend a book for Kabir?"*
- *"What's on my travel wishlist?"*

## Switching users

Edit `sample/.env` to change `SMRITIKOSH_USER` / `SMRITIKOSH_USER_PASS`.
The local `sample/.env` overrides the project-root `.env`, so you never
need to touch the server config.

You can also override inline without editing the file:

```bash
SMRITIKOSH_USER=priya SMRITIKOSH_USER_PASS=priyapass python chatbot.py
```

## Authentication

`client.py` supports two auth modes:

**Username/password** (default — exchanges credentials for a JWT on startup):

```bash
python chatbot.py
# reads SMRITIKOSH_USER / SMRITIKOSH_USER_PASS from sample/.env
```

**API key** (recommended for integrations — no login round-trip, never expires):

```bash
# Generate a key: sign in to the dashboard → API Keys → New key
SMRITIKOSH_API_KEY=sk-smriti-your-key-here python chatbot.py
```

Or set `SMRITIKOSH_API_KEY` in `sample/.env` to use it every time.

**Read-only API keys** — for analytics tools that should never write, add `"scopes": ["read"]` when generating a key:

```bash
TOKEN=$(curl -s -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "alicepass"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST http://localhost:8080/keys \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "read-only reporting", "scopes": ["read"]}'
```

A read-only key can call any GET endpoint (search, context, identity) but will receive `403 Forbidden` on POST/DELETE memory endpoints.

## Commands inside the chatbot

| Command | What it does |
|---|---|
| `<any text>` | Chat — memory context is injected automatically |
| `/remember <text>` | Manually store a memory |
| `/search <query>` | Search the current user's memories and show scored results |
| `/quit` | Exit |

## Example session (Alice)

```
============================================================
  Smritikosh demo chatbot  (user: alice)
  LLM: ollama / qwen2.5:14b
  Commands: /remember <text>  /search <query>  /quit
============================================================

You: What do I do for work?
Assistant: You are a machine learning engineer at a Series B startup, focused
on data pipelines and ML systems. Your team is currently migrating the training
infrastructure from PyTorch to JAX.

You: /search editor

  Search results for: 'editor'
  [0.921] · My favourite editor is Neovim with the lazy.nvim plugin manager...
  [0.503] · I use a MacBook Pro M3 Max for local development...

You: /quit
Goodbye!
```

## What happens under the hood

| Step | What Smritikosh did |
|---|---|
| `seed.py` ran | Texts → importance scored → embedded → stored in PostgreSQL → facts extracted → written to Neo4j |
| `chat()` called | `/context` retrieved the most relevant memories + Neo4j profile → injected as system prompt |
| LLM responded | Model answered using the injected context |
| Exchange stored | The full Q&A was stored as a new memory event for future sessions |

## Passive extraction demos

These scripts use `priya` as their subject. Run them after `seed_priya.py`.

### `passive_extraction_demo.py`

No API key required — the demo posts transcripts directly to the server.

```bash
python sample/seed_priya.py           # one-time seed
python sample/passive_extraction_demo.py
```

Demonstrates:
- `POST /ingest/session` — passive extraction from a 7-turn conversation
- Idempotency — re-posting the same `session_id` is a safe no-op
- Streaming windows — three partial `POST /ingest/session` calls with `partial=True`; each window sends only new turns via `last_turn_index` tracking
- **Dry-run mode** — add `"dry_run": true` to any ingest call to see what facts *would* be extracted without writing anything. The response includes `extracted_facts` and `dry_run: true`
- Manual facts — `store_fact()` four times with `source_type="ui_manual"`, confidence 1.0
- Verification — `GET /context` confirms all extracted facts appear in retrieval

### `middleware_demo.py`

Uses a fake OpenAI-style client — no real API key required.

```bash
python sample/middleware_demo.py
```

Demonstrates:
- One-line change: `SmritikoshMiddleware(FakeOpenAI(), ...)` instead of `FakeOpenAI()`
- Turn buffering and windowed partial flush every N turns
- `remember()` tool auto-injected into every LLM call; intercepted transparently
- `auto_inject=True` — memory context fetched and prepended before each call
- Final `close()` flush of remaining turns

Swap `FakeOpenAI()` for `openai.OpenAI()` or `anthropic.Anthropic()` in production.

### `litellm_middleware_notebook.ipynb`

A Jupyter notebook demonstrating `LiteLLMMiddleware` across every supported provider.

```bash
pip install jupyter litellm
jupyter notebook sample/litellm_middleware_notebook.ipynb
```

Demonstrates:
- **Ollama** (local) — `ollama_chat/llama3.2`; no API key, no data leaves your machine
- **vLLM** — self-hosted OpenAI-compatible server; `openai/<model>` + `api_base`
- **Google Gemini** — `gemini/gemini-1.5-flash`; requires `GEMINI_API_KEY`
- **OpenAI via LiteLLM** — swap providers with one model-string change
- **Anthropic Claude via LiteLLM** — same interface, different backend
- **Multi-turn windowed flush** — `extract_every_n_turns=3` firing partial ingests in background threads across a 6-turn conversation
- **Verification cells** — search, context retrieval, and memory listing to confirm what was extracted

Swap `SMRITIKOSH_API_KEY` in the setup cell and run any provider section independently.

### `media_ingest_demo.py`

Upload a personal notes document and watch Smritikosh extract facts from it.
No Whisper or Vision API key required — document extraction runs with your core LLM.

```bash
python sample/media_ingest_demo.py
```

Demonstrates:
- `POST /ingest/media` — upload a `.md` document created in-memory; server returns immediately (async processing)
- `GET /ingest/media/{id}/status` — poll until extraction finishes
- Two-tier routing — facts above 0.75 relevance are saved automatically; facts in the 0.60–0.75 band go to **pending** review
- `POST /ingest/media/{id}/confirm` — user confirms (or dismisses) pending facts
- `POST /context` — extracted facts now appear in memory retrieval

The script's summary also shows the equivalent curl commands for voice note, image (receipt/screenshot/whiteboard), and meeting recording uploads — so you can extend to those once you have `WHISPER_PROVIDER` and `VISION_PROVIDER` configured.

### Recommended run order

```bash
# 1. One-time seed
python sample/seed_priya.py

# 2. Session ingest + manual facts (no extra API keys)
python sample/passive_extraction_demo.py

# 3. Middleware + remember() tool (no OpenAI key needed — uses fake client)
python sample/middleware_demo.py

# 4. LiteLLM middleware — multi-provider (Ollama, vLLM, Gemini, OpenAI, Claude)
pip install jupyter litellm
jupyter notebook sample/litellm_middleware_notebook.ipynb

# 5. Media ingestion from a document (no Whisper/Vision key needed)
python sample/media_ingest_demo.py

# 6. Chat with the enriched memory (requires LLM API key)
export ANTHROPIC_API_KEY=sk-ant-...
python sample/chatbot.py
```

---

## Where to look next

- **Dashboard** (`http://localhost:3000`) — log in as the active user, browse the memory timeline and fact graph
- **Review page** (`/dashboard/review`) — approve or dismiss auto-extracted memories; filter by source type (Distilled, SDK, Tool, etc.)
- **Identity page** — see the Neo4j knowledge graph as a 3D/2D force-directed canvas; click any fact node to see which memories contributed to it
- **Voice enrollment** (`/dashboard/settings/voice-enrollment`) — record a 30-second voice sample to enable speaker diarization for meeting recording uploads
- **Upload media** — the `+` / Upload buttons in the memory timeline let you upload voice notes, documents, images, and meeting recordings directly
- **Admin panel** — log in as `admin` to trigger consolidation, synthesis, or check system health
- **Run consolidation** — compresses memories into summaries and extracts more facts:

```bash
curl -X POST "http://localhost:8080/admin/consolidate?user_id=alice"
curl -X POST "http://localhost:8080/admin/consolidate?user_id=priya"
```

- **Run cross-system synthesis** — infers behavioral patterns from connector signals:

```bash
curl -X POST "http://localhost:8080/admin/synthesize?user_id=priya"
```

- **Check embedding health** — verify all stored vectors match the configured dimension (run this after changing `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS`):

```bash
curl http://localhost:8080/admin/embedding-health \
  -H "Authorization: Bearer $TOKEN"
# {"configured_dim": 768, "total_embedded": 25, "stale_events": 0, "healthy": true}
```

- **Re-embed stale events** — if `stale_events > 0` or `null_embeddings > 0`, re-embed everything in the background:

```bash
curl -X POST http://localhost:8080/admin/re-embed \
  -H "Authorization: Bearer $TOKEN"
```
