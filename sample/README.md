# Smritikosh — Sample Project

A minimal memory-aware chatbot that demonstrates the full Smritikosh flow:
store memories → search → inject context → LLM call.

## Prerequisites

- Smritikosh server running on `http://localhost:8080`
- User `alice` created (see [quickstart.md](../quickstart.md) Step 10)
- Memories seeded: run `python seed.py` once before the chatbot

## Setup

```bash
pip install httpx openai   # openai SDK works with ollama, openai, and gemini
```

## Files

| File | Purpose |
|---|---|
| `client.py` | Thin wrapper around the Smritikosh REST API |
| `seed.py` | Pre-loads 10 memories for `alice` — run once before the chatbot |
| `chatbot.py` | The interactive memory-aware chatbot loop |

## Run

```bash
# 1. Seed alice's memories (run once)
python seed.py

# 2. Start the chatbot (reads LLM config from the project's .env automatically)
python chatbot.py
```

No environment variables need to be exported manually — `chatbot.py` reads
`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, and `LLM_API_KEY` directly from
the project's `.env` file.

## Commands inside the chatbot

| Command | What it does |
|---|---|
| `<any text>` | Chat — memory context is injected automatically |
| `/remember <text>` | Manually store a memory |
| `/search <query>` | Search alice’s memories and show scored results |
| `/quit` | Exit |

## Example session

```
============================================================
  Smritikosh demo chatbot  (user: alice)
  LLM: ollama / qwen2.5:14b
  Commands: /remember <text>  /search <query>  /quit
============================================================

You: What do I do for work?

Assistant: You're a machine learning engineer at a Series B startup. You work
mostly on data pipelines and ML systems, and your team is currently migrating
the training infrastructure from PyTorch to JAX.

You: What languages do I prefer?

Assistant: You prefer Python, especially for data pipelines and ML work.
You're also learning Rust in your spare time — you find the borrow checker
challenging but rewarding.

You: How is my RAG pipeline going?

Assistant: You deployed a RAG pipeline last week using pgvector and LangChain.
Latency was higher than expected, so that's something you're likely still
working through.

You: /search editor

  Search results for: 'editor'
  [0.921] · My favourite editor is Neovim with the lazy.nvim plugin manager...
  [0.503] · I use a MacBook Pro M3 Max for local development...

You: /remember I switched from lazy.nvim to rocks.nvim today

  Stored. importance=0.61  facts_extracted=1

You: What is my current Neovim setup?

Assistant: You just switched from lazy.nvim to rocks.nvim today. Before that
you were using Neovim with lazy.nvim as your plugin manager.

You: /quit
Goodbye!
```

Notice the last answer — the bot recalls the `/remember` from earlier in the
same session. Next time you run `chatbot.py`, Alice's bot will still know about
the rocks.nvim switch because it was persisted to Smritikosh.

## What just happened under the hood

| Step | What Smritikosh did |
|---|---|
| `seed.py` ran | 10 texts → importance scored → embedded → stored in PostgreSQL → facts extracted → written to Neo4j |
| `chat()` called | `/context` retrieved the most relevant memories + Neo4j profile → injected as system prompt |
| LLM responded | Model answered using the injected context |
| Exchange stored | The full Q&A was stored as a new memory event for future sessions |
| `/remember` ran | `POST /memory/event` stored the Neovim switch immediately |

## Where to look next

- **Dashboard** (`http://localhost:3000`) — log in as `alice`, browse the memory timeline and fact graph
- **Identity page** — see the Neo4j knowledge graph as a React Flow canvas
- **Admin panel** — log in as `admin` to trigger consolidation or check system health
- **Run consolidation** — compresses alice’s memories into summaries and extracts more facts:

```bash
curl -X POST "http://localhost:8080/admin/consolidate?user_id=alice"
```
