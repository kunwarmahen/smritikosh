# Smritikosh — End-to-End Walkthrough

This document walks through a complete, real-world memory lifecycle using two
personas: **Alice**, a machine learning engineer, and **Rohan**, a product
manager. By the end you will understand exactly what happens at every layer —
the chatbot CLI, the REST API, the background pipeline, and the UI — when a
user interacts with an AI assistant backed by Smritikosh.

---

## The scenario

Alice and Rohan both use an internal AI assistant at their startup. The
assistant is backed by Smritikosh so it remembers who they are, what they are
working on, and how they prefer to work — across every session, permanently.

---

## Act 1 — First contact (memory encoding)

Alice opens the chatbot for the first time and types:

```
I'm an ML engineer at a Series B startup. My team is migrating from PyTorch
to JAX. I prefer Python and I'm learning Rust on the side. My editor is
Neovim with lazy.nvim.
```

### What happens inside Smritikosh

The chatbot calls `POST /memory/event` with `user_id=alice` and the raw text.
The **Hippocampus** (the central coordinator) fires a synchronous pipeline:

```
Raw text
   │
   ▼
Amygdala ──────────────────────────────────────────────────────────────────
   Scores importance (0.0 – 1.0) using an LLM.
   "migrating from PyTorch to JAX" → 0.72
   High score because it signals a career-level infrastructure decision.
   │
   ▼
LLM Embedder ──────────────────────────────────────────────────────────────
   Converts the full text to a 768-dimensional vector.
   Captures semantic meaning — "JAX migration" and "framework switch" will
   match this event even without the exact words.
   │
   ▼
PostgreSQL + pgvector ─────────────────────────────────────────────────────
   Stores the event row:
     user_id=alice, app_id=default
     raw_text="I'm an ML engineer..."
     embedding=[0.021, -0.143, 0.087, ...]   (768 floats)
     importance_score=0.72
     consolidated=false
     recall_count=0
   │
   ▼
Fact Extractor ────────────────────────────────────────────────────────────
   LLM identifies structured facts and writes them to Neo4j:
     (alice)-[WORKS_AT]->(Series B startup)
     (alice)-[MIGRATING_TO]->(JAX)         confidence=0.90
     (alice)-[PREVIOUSLY_USED]->(PyTorch)  confidence=0.90
     (alice)-[PREFERS]->(Python)           confidence=0.95
     (alice)-[LEARNING]->(Rust)            confidence=0.85
     (alice)-[USES_EDITOR]->(Neovim)       confidence=0.95
     (alice)-[USES_PLUGIN]->(lazy.nvim)    confidence=0.90
```

The API returns immediately:

```json
{
  "event_id": "a3f1c2d4-...",
  "importance_score": 0.72,
  "facts_extracted": 7
}
```

Alice's message is now a **searchable, queryable memory** — indexed by meaning
(vector similarity) and by structured relationship (graph traversal).

---

## Act 2 — The assistant answers (context retrieval)

The next day Alice asks the assistant:

```
What should I focus on this week for our infra migration?
```

### What happens before the LLM sees the question

The chatbot calls `POST /context` with `user_id=alice` and the query text.
Before running search, Smritikosh classifies the query intent using a two-tier
classifier:

```
Query: "What should I focus on this week for our infra migration?"
   │
   ▼
IntentClassifier
   Step 1 — keyword heuristic (always runs, zero latency):
     "infra", "migration" → PROJECT_PLANNING keywords: confidence=0.67
     Confidence ≥ 0.5 threshold → use keyword result directly.
   Step 2 — LLM fallback (only when keyword confidence < 0.5):
     Prompts a cheap model for {primary_intent, secondary_intents, confidence}.
     Falls back to keyword result on any failure.
   Result: intent=PROJECT_PLANNING, via_llm=False
```

The detected intent adjusts the hybrid search weight distribution. A
`PROJECT_PLANNING` query shifts weight towards recency and contextual match;
a `TECHNICAL` query shifts weight towards embedding similarity. Then
Smritikosh embeds the query and runs a **hybrid search** across all of Alice's
stored memories:

```
Query vector  (768-dim for "infra migration focus this week")
   │
   ▼
Hybrid scoring — every event is scored with intent-adjusted weights:

  hybrid_score =
      w_sim  × cosine_similarity(query_vec, event.embedding)
    + w_rec  × exp(−days_since_event / 30)
    + w_imp  × event.importance_score
    + w_freq × min(recall_count, 50) / 50
    + w_ctx  × contextual_match_score

  Weights sum to 1.0 and vary by intent (e.g. TECHNICAL puts 0.55 on similarity;
  HISTORICAL_RECALL puts 0.30 on frequency).

Top-5 events ranked by hybrid_score:
  [0.91]  "team migrating from PyTorch to JAX"           (high sim + high importance)
  [0.74]  "deployed RAG pipeline, latency still high"    (related infrastructure work)
  [0.61]  "productive hours 9 PM – midnight"             (scheduling context)
  [0.58]  "learning Rust, borrow checker finally clicked"
  [0.51]  "MacBook Pro M3 Max for local dev"
```

After ranking, Smritikosh traverses **narrative chains** for the top-3 results.
If a high-scoring event is causally or temporally linked to other events via
`memory_links`, those chain neighbours are included with a small score boost
(`+0.05`). This surfaces contextually adjacent memories even when they wouldn't
have ranked highly on their own.

Smritikosh assembles these into a context block and returns it:

```
[Memory context — alice]
• You are migrating your team's training infrastructure from PyTorch to JAX.
• You deployed a RAG pipeline last week using pgvector and LangChain.
  Latency was higher than expected — still being worked through.
• Your most productive hours are 9 PM to midnight.
• You are learning Rust in your spare time — the ownership model recently
  clicked for you.
• You work on a MacBook Pro M3 Max.
```

This block is injected into the LLM system prompt. The model answers *as if it
knows Alice personally* — without Alice ever repeating herself.

After the search, Smritikosh increments `recall_count` on each of the 5
surfaced events. Memories recalled frequently will score higher next time —
a reinforcement loop that mirrors how human memory strengthens with use.

---

## Act 3 — A new memory mid-session

Alice types later in the same session:

```
/remember I switched from lazy.nvim to rocks.nvim today
```

This stores a new event. Importance: **0.61** (a concrete, durable tool
decision). The fact extractor updates the graph:

```
Before:  (alice)-[USES_PLUGIN]->(lazy.nvim)
After:   (alice)-[USES_PLUGIN]->(rocks.nvim)
         (alice)-[PREVIOUSLY_USED_PLUGIN]->(lazy.nvim)
```

When Alice immediately asks:

```
What is my current Neovim setup?
```

The hybrid search surfaces both the rocks.nvim event (very recent, high
recency score) and the original lazy.nvim memory (older, lower score). The LLM
sees both and responds:

```
Assistant: You just switched from lazy.nvim to rocks.nvim today.
Before that you were using Neovim with lazy.nvim as your plugin manager.
```

The next time Alice starts a fresh session and asks the same question, the
answer will be the same — Smritikosh persisted the switch permanently.

---

## Act 4 — Background consolidation (overnight processing)

Alice has now stored 20+ raw events. The **MemoryScheduler** runs
consolidation every 30 minutes. Here is what it does to Alice's events:

### Step 1 — Consolidation

```
20 unconsolidated events
   │
   ▼
Consolidator groups related events and summarises them with an LLM:

  Group: ML infrastructure
    "migrating PyTorch → JAX"
    "deployed RAG pipeline, latency high"
    "manager asked to evaluate Smritikosh"
  → Summary: "Alice is leading ML infrastructure modernisation at her startup:
    migrating training from PyTorch to JAX and evaluating memory layers for
    an internal LLM assistant. A RAG pipeline she built has higher-than-
    expected latency."

  Group: Personal development
    "learning Rust, borrow checker confusing but rewarding"
    "finished ownership chapter, it clicked"
    "switched lazy.nvim → rocks.nvim"
  → Summary: "Alice actively develops her skills outside work: learning Rust
    (recently understood ownership) and fine-tuning her Neovim setup
    (switched to rocks.nvim)."
```

Each event is marked `consolidated=true`. The summary is stored back on those
events. Future searches can match against the denser, synthesised summary
rather than many sparse raw texts — fewer tokens, higher precision.

### Step 2 — Clustering (every hour)

The scheduler runs the **MemoryClusterer**, which groups all events by semantic
similarity using a greedy centroid algorithm:

```
Cluster 0 — "Machine learning & infrastructure"
  • team migrating from PyTorch to JAX
  • deployed RAG pipeline (pgvector + LangChain)
  • manager evaluating Smritikosh for LLM assistant

Cluster 1 — "Editor & tooling preferences"
  • Neovim with lazy.nvim
  • switched to rocks.nvim

Cluster 2 — "Programming languages"
  • prefers Python for data pipelines
  • learning Rust, borrow checker

Cluster 3 — "Work style & schedule"
  • productive hours 9 PM – midnight
  • dislikes meetings before 10 AM
```

Each event now has a `cluster_id` and `cluster_label`. The Clusters page in
the UI shows these groupings visually.

### Step 3 — Belief mining (every 2 hours)

The **BeliefMiner** looks across all of Alice's consolidated facts and infers
higher-order beliefs:

```
Facts observed:
  - Prefers Python, learning Rust
  - Uses Neovim, switched plugin managers
  - Works late, dislikes early meetings
  - Reads technical books (Pragmatic Programmer)
  - Migrating infra, building RAG pipelines

Beliefs inferred:
  - Alice values deep technical craftsmanship over convenience
  - Alice is self-directed in skill development
  - Alice works best with uninterrupted late-evening focus blocks
```

These beliefs appear in the Identity page and feed back into future context
retrieval, making answers progressively more personalised over time.

---

## Act 5 — A second user (Rohan, the product manager)

Rohan uses the same assistant but is a completely separate user. He stores:

```
I'm the PM for the data platform. We're launching the v2 API in Q2.
My biggest concern is the migration timeline slipping due to infra delays.
I prefer async communication — I check Slack in batches, not constantly.
I like concise, bulleted updates rather than long paragraphs.
```

His memory graph is entirely separate from Alice's. When Rohan asks:

```
What are my main risks for the Q2 launch?
```

Smritikosh searches **only Rohan's events** and returns context specific to
him. The same Smritikosh instance serves both users — isolated by `user_id` at
every query, storage, and retrieval layer.

This is the **multi-tenant** model: one deployment, many users, zero
cross-contamination.

---

## Act 6 — What the UI shows

After a few sessions and a consolidation run, here is what each UI page
reflects:

### Memories page

A timeline of all events, newest first. Each card shows:
- The raw text
- Importance score (coloured bar — higher = more significant)
- Hybrid search score badge (shown when the page is in search mode)
- A ✓ badge if the event has been consolidated
- Cluster label (e.g. "Machine learning & infrastructure")

A search bar at the top calls `POST /memory/search` and re-renders the list
with results sorted by hybrid score. Searching for "JAX" instantly returns the
most semantically relevant events using the same scoring the API uses — not a
keyword filter. The score badge on each result shows exactly why it ranked where
it did.

An **Export** button calls `GET /memory/export` and downloads all events as
NDJSON — one JSON object per line. Every row includes raw text, summary,
importance score, cluster label, and timestamp. Users can verify what the system
holds about them and import it elsewhere.

### Identity page

A synthesised profile built from Neo4j facts + consolidated summaries:

```
Role:        Machine learning engineer
Employer:    Series B startup
Projects:    PyTorch → JAX migration, RAG pipeline (pgvector + LangChain)
Interests:   Rust, Python, Neovim
Preferences: Late-night work, async communication
```

The fact graph is also rendered as an interactive canvas showing nodes
(alice, Rust, JAX, Neovim...) and labelled edges (LEARNING, MIGRATING_TO,
USES_EDITOR...). You can see exactly what the system knows and why.

### Clusters page

Events grouped by topic. Clicking a cluster expands it to show the individual
memory cards. Useful for spotting what domains dominate a user's memory and
whether clustering has produced meaningful groupings.

### Audit trail page

A chronological log of every pipeline operation that touched Alice's data:

```
memory.encoded          14:04  importance=0.72, embedding=✓, facts=7
memory.facts_extracted  14:04  7 facts written to Neo4j
memory.encoded          14:06  importance=0.61, facts=1
memory.consolidated     14:35  events=10, facts_distilled=12
memory.clustered        15:00  clusters=4, events_clustered=20
belief.mined            16:01  beliefs=3
context.built           09:12  top_k=5, hybrid_search
search.performed        09:18  query="editor", results=3
```

This is the provenance layer. For every memory event you can trace the full
lifecycle: when it was encoded, when it was consolidated, when it was recalled,
and whether any feedback shifted its importance score.

---

## Act 7 — Feedback loop (importance adjustment)

Alice gives a thumbs-down on a retrieved memory that was irrelevant to her
question. The **ReinforcementLoop** reduces its importance score:

```
Before:  importance_score = 0.72
Feedback: negative
After:   importance_score = 0.58   (reduced by reinforcement delta)
```

The event will surface less often in future searches. Conversely, a positive
signal on a memory increases its score. Over many interactions, the retrieval
ranking self-calibrates to what is actually useful to each user.

---

## What makes this different from RAG or a vector database

| Capability | Plain vector DB | Smritikosh |
|---|---|---|
| Semantic search | Yes | Yes |
| Recency-aware ranking | No | Yes — exponential decay |
| Importance weighting | No | Yes — Amygdala scoring |
| Recall reinforcement | No | Yes — recall_count feedback |
| Structured fact graph | No | Yes — Neo4j |
| Identity synthesis | No | Yes — dimensions + summary |
| Memory consolidation | No | Yes — LLM summarisation |
| Belief inference | No | Yes — higher-order patterns |
| Provenance/audit | No | Yes — full lineage per event |
| Multi-tenant isolation | Depends | Yes — user_id at every layer |

Smritikosh is not a vector store with an API wrapper. It is a full memory
system: encoding, retrieval, consolidation, belief formation, and provenance
— the same pipeline the human brain runs, mapped to software.

---

## Integration in three lines

```python
from smritikosh import SmritikoshClient

memory = SmritikoshClient(username="alice", password="...")
memory.remember("alice", "I just switched our infra to Kubernetes")
context = memory.get_context("alice", "what are our current deployment risks?")
# inject context into your LLM system prompt
```

Everything else — embedding, scoring, fact extraction, consolidation,
clustering, belief mining — happens automatically in the background.
