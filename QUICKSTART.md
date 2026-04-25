# Smritikosh — Quick Start

> From a blank machine to a running server and dashboard in 10 steps.

---

## Step 1 — Install system requirements

### Python 3.11+

```bash
# Check existing version
python3 --version

# macOS
brew install python@3.11

# Ubuntu / Debian
sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip

# Windows — download installer from https://www.python.org/downloads/
# Tick "Add python.exe to PATH" during installation.
```

### Node.js 18+

```bash
# Check existing version
node --version

# macOS
brew install node

# Ubuntu / Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Windows — download LTS installer from https://nodejs.org/
```

### Docker Desktop (optional — needed for Option A in Step 6)

```bash
# Verify after install
docker --version
docker compose version
```

- **macOS / Windows:** install from https://www.docker.com/products/docker-desktop/
- **Ubuntu:** follow https://docs.docker.com/engine/install/ubuntu/ then run:
  ```bash
  sudo usermod -aG docker $USER   # log out/in after this
  ```

---

## Step 2 — Get an LLM API key

You need **at least one** provider key:

| Provider | Sign-up URL | Cheapest model to start |
|---|---|---|
| Anthropic (Claude) | console.anthropic.com | `claude-haiku-4-5-20251001` |
| OpenAI | platform.openai.com/api-keys | `gpt-4o-mini` |
| Gemini | aistudio.google.com/app/apikey | `gemini-1.5-flash` |
| Ollama (free, local) | ollama.com | `llama3.2` |

You also need an **embedding model** key. Recommended: OpenAI `text-embedding-3-small`.

---

## Step 3 — Clone the repository

```bash
git clone https://github.com/your-org/smritikosh.git
cd smritikosh
```

---

## Step 4 — Create a Python virtual environment

```bash
python3 -m venv .venv

# Activate — run this every time you open a new terminal
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install all dependencies
pip install -e ".[dev]"
```

Your prompt will show `(.venv)` when the environment is active.

---

## Step 5 — Configure the backend

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

```dotenv
# LLM
LLM_PROVIDER=claude                      # or: openai / gemini / ollama
LLM_MODEL=claude-haiku-4-5-20251001
LLM_API_KEY=sk-ant-...

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...

# Auth
JWT_SECRET=replace-this-with-something-random-and-long
```

Optional settings (defaults shown):

```dotenv
# Rate limiting (per user identity, per minute)
RATE_LIMIT_ENCODE=60/minute
RATE_LIMIT_CONTEXT=60/minute
RATE_LIMIT_SEARCH=120/minute

# Semantic fact decay (background job, runs weekly)
FACT_DECAY_HALF_LIFE_DAYS=60.0           # halve confidence every 60 days
FACT_DECAY_FLOOR=0.1                     # delete facts below this confidence
```

### Optional: media ingestion providers

These settings are **not required** to run Smritikosh. Document ingestion (`.txt`, `.md`, `.pdf`) works with just your core LLM above — no extra keys needed.

To unlock voice note and image uploads, add the relevant keys:

```dotenv
# Voice note transcription (MP3, WAV, M4A, WebM)
WHISPER_PROVIDER=openai                  # or: local (self-hosted)
WHISPER_API_KEY=sk-...                   # same as your OpenAI key, or a separate one
# WHISPER_PROVIDER=local
# WHISPER_BASE_URL=http://localhost:8000 # URL of your local Whisper server

# Image description (receipt, screenshot, whiteboard)
VISION_PROVIDER=openai                   # or: claude / gemini / ollama
VISION_MODEL=gpt-4o-mini
VISION_API_KEY=sk-...

# Meeting recording diarization (identifies which speaker is the user)
# DIARIZATION_PROVIDER=none              # default — falls back to first-person filter
# DIARIZATION_PROVIDER=pyannote          # full diarization (requires HF_TOKEN + GPU)
# HF_TOKEN=hf_...                        # Hugging Face token for pyannote/speaker-diarization-3.1
```

The `sample/media_ingest_demo.py` script demonstrates document ingestion and works with just the core LLM — try it without configuring the optional keys first.

---

## Step 6 — Start the databases

### Option A — Docker Compose (recommended)

```bash
docker compose up -d
docker compose ps     # all three should show "running (healthy)"
```

### Option B — Podman

```bash
# PostgreSQL
podman run -d \
  --name smritikosh-postgres \
  -e POSTGRES_USER=smritikosh \
  -e POSTGRES_PASSWORD=smritikosh \
  -e POSTGRES_DB=smritikosh \
  -p 5432:5432 \
  -v pgdata:/var/lib/postgresql/data:Z \
  postgres

# Neo4j
mkdir -p $HOME/neo4j/data $HOME/neo4j/logs $HOME/neo4j/conf
podman run -d \
  --name neo4j_container \
  --publish=7474:7474 --publish=7687:7687 \
  --volume=$HOME/neo4j/data:/data:Z \
  --volume=$HOME/neo4j/logs:/logs:Z \
  --volume=$HOME/neo4j/conf:/conf:Z \
  --env=NEO4J_AUTH=neo4j/smritikosh \
  docker.io/neo4j:latest

# MongoDB
podman run -d \
  --name smritikosh-mongo \
  -p 27017:27017 \
  -v mongodata:/data/db:Z \
  docker.io/mongo:7
```

| Service | Port | Notes |
|---|---|---|
| PostgreSQL | 5432 | |
| Neo4j | 7687 | Browser UI at http://localhost:7474 |
| MongoDB | 27017 | |

Wait ~15–20 seconds before the next step.

---

## Step 7 — Create the database tables

```bash
alembic upgrade head
```

You should see output ending with something like:
```
INFO  [alembic.runtime.migration] Running upgrade 0015 -> 0016, add fact_contradictions table
INFO  [alembic.runtime.migration] Running upgrade 0016 -> 0017, add media_ingests table
INFO  [alembic.runtime.migration] Running upgrade 0017 -> 0018, add user_voice_profiles table
```

---

## Step 8 — Create your first admin account

```bash
# Allow unauthenticated first registration
echo "BOOTSTRAP_ADMIN=1" >> .env

# Start the server (keep this terminal open)
uvicorn smritikosh.api.main:app --reload --port 8080
```

In a **new terminal**, register the admin account:

```bash
curl -s -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "changeme123", "role": "admin"}' \
  | python3 -m json.tool
```

Then remove `BOOTSTRAP_ADMIN` and restart the server:

```bash
sed -i '/BOOTSTRAP_ADMIN/d' .env
# Ctrl+C the server, then:
uvicorn smritikosh.api.main:app --reload --port 8080
```

Verify:

```bash
curl http://localhost:8080/health
# {"status": "ok", "postgres": "ok", "neo4j": "ok"}
```

---

## Step 9 — Set up the dashboard UI

Open a **new terminal** (keep the server running).

```bash
cd ui
npm install

cp .env.local.example .env.local
sed -i "s/AUTH_SECRET=change-me-in-production/AUTH_SECRET=$(openssl rand -hex 32)/" .env.local

npm run dev
```

Open **http://localhost:3000** and sign in with `admin` / `changeme123`.

---

## Step 10 — Create a regular user for testing

**Via dashboard:** Admin → Users → New user

**Or via API:**

```bash
# Get an admin token
TOKEN=$(curl -s -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "changeme123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Create a user
curl -s -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"username": "alice", "password": "alicepass", "role": "user", "app_ids": ["default"]}' \
  | python3 -m json.tool
```

You are now fully set up. See the [Sample project](README.md#sample-project) section in the README to test the full memory flow.

**Try the demos in order:**

```bash
# Seed a demo user (one-time)
python sample/seed_priya.py

# Passive extraction from a conversation transcript
python sample/passive_extraction_demo.py

# SDK middleware with transparent turn buffering
python sample/middleware_demo.py

# Media ingestion — upload a personal notes document (no extra keys needed)
python sample/media_ingest_demo.py

# Interactive chat with the enriched memory
export ANTHROPIC_API_KEY=sk-ant-...
python sample/chatbot.py
```

**Verify the LLM adapter resolved correctly** — on startup you should see a log line like:

```
INFO  LLMAdapter initialised  chat_provider=claude  chat_model=claude-haiku-4-5-20251001
                              embed_provider=openai  embed_model=text-embedding-3-small  embed_dimensions=1536
```

If you see the wrong provider or model, check that your `.env` has exactly one `LLM_MODEL` line.

**Export a user's memories** at any time:

```bash
curl -o memories_alice.ndjson \
  "http://localhost:8080/memory/export?user_id=alice&app_ids=default" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Step 11 — Generate an API key (optional, for SDK / integrations)

The dashboard and browser sessions use a short-lived JWT automatically. For programmatic access (SDK, CI, external tools) generate a long-lived API key instead.

**Via the dashboard:** sign in as `alice` → **API Keys** in the left sidebar → **New key** → copy the key (shown once only).

**Via the API:**

```bash
# Get alice's JWT first
TOKEN=$(curl -s -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "alicepass"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Generate a key
curl -s -X POST http://localhost:8080/keys \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "local dev", "app_ids": ["default"]}' \
  | python3 -m json.tool
```

Use the key in the sample chatbot:

```bash
SMRITIKOSH_API_KEY=sk-smriti-... python chatbot.py
```

Or set `SMRITIKOSH_API_KEY` in your `.env` to use it automatically every time.
