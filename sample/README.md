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
| `chatbot.py` | The interactive memory-aware chatbot loop |
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

## Where to look next

- **Dashboard** (`http://localhost:3000`) — log in as the active user, browse the memory timeline and fact graph
- **Identity page** — see the Neo4j knowledge graph as a React Flow canvas
- **Admin panel** — log in as `admin` to trigger consolidation or check system health
- **Run consolidation** — compresses memories into summaries and extracts more facts:

```bash
curl -X POST "http://localhost:8080/admin/consolidate?user_id=alice"
curl -X POST "http://localhost:8080/admin/consolidate?user_id=priya"
```
