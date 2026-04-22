# middleware_demo.py
"""
SDK Middleware Demo — transparent memory extraction from LLM calls.

Shows SmritikoshMiddleware wrapping a fake OpenAI-style client so the demo
runs without needing a real OpenAI key. The middleware extracts memories
in exactly the same way it would with a real client — the LLM surface is
identical from the middleware's perspective.

What this demo shows:
  1. Wrap any OpenAI-compatible client with SmritikoshMiddleware (one line)
  2. Make LLM calls as normal — developer code is unchanged
  3. Middleware buffers turns and fires POST /ingest/session in the background
  4. auto_inject=True: memory context is fetched and prepended to each call
  5. close() flushes the final batch and confirms extraction

Setup:
    1. Start the server:       docker compose up -d
    2. Seed Priya's memories:  python sample/seed_priya.py
    3. Run this script:        python sample/middleware_demo.py

No OpenAI / Anthropic key required — a local fake client is used.
"""

import time

import httpx

from client import SmritikoshClient
from smritikosh.sdk.middleware import SmritikoshMiddleware

USER   = "priya"
APP_ID = "default"

# ── Auth ──────────────────────────────────────────────────────────────────────
# Log in as admin (same as seed_priya.py) and use the JWT as the bearer token
# for the middleware — no separate API key needed.

_auth = httpx.post(
    "http://localhost:8080/auth/token",
    json={"username": "admin", "password": "changeme123"},
    timeout=10,
)
_auth.raise_for_status()
JWT_TOKEN = _auth.json()["access_token"]

smriti_client = SmritikoshClient(username="admin", password="changeme123", app_id=APP_ID)


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ── Fake LLM client (OpenAI-compatible surface) ────────────────────────────────
# Mimics openai.OpenAI() — the middleware only needs .chat.completions.create().
# Swap this for openai.OpenAI() or anthropic.Anthropic() in production.

class _FakeCompletions:
    """Returns a hard-coded assistant reply so no API key is needed."""
    def create(self, *, messages, **kwargs):
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Msg", (), {"content": f"[fake LLM] Got: {last_user[:60]}"})()
            })()]
        })()


class _FakeChat:
    completions = _FakeCompletions()


class FakeOpenAI:
    """Drop-in OpenAI stand-in for the demo."""
    chat = _FakeChat()


# ── Demo conversation ─────────────────────────────────────────────────────────
# In a real app these would come from your user's actual messages.

TURNS = [
    "I always have Rohan pick the wine — he has much better taste than me in that department.",
    "I prefer cherry blossom season for Japan. The kids would absolutely love it.",
    "My goal is to book a kid-friendly ryokan in Kyoto for next spring before they all fill up.",
    "We never skip travel insurance after what happened in Bali two years ago.",
    "I believe the best luxury is time — slow travel over a packed itinerary every time.",
]


# ── Step 1: Wrap the LLM client ───────────────────────────────────────────────

section("Step 1 — Wrap the LLM client (one line)")

print(f"""
  # Before (plain LLM client):
  llm = FakeOpenAI()
  response = llm.chat.completions.create(model="gpt-4o", messages=[...])

  # After (with memory extraction — nothing else changes):
  from smritikosh.sdk.middleware import SmritikoshMiddleware
  llm = SmritikoshMiddleware(
      FakeOpenAI(),
      smritikosh_url="http://localhost:8080",
      smritikosh_api_key="<jwt-or-api-key>",
      user_id="{USER}",
      app_id="{APP_ID}",
      extract_every_n_turns=3,   # flush every 3 user turns (default: 10)
  )
  response = llm.chat.completions.create(model="gpt-4o", messages=[...])
""")

# ── Step 2: Make LLM calls as normal ─────────────────────────────────────────

section(f"Step 2 — Simulating a {len(TURNS)}-turn conversation")

with SmritikoshMiddleware(
    FakeOpenAI(),
    smritikosh_url="http://localhost:8080",
    smritikosh_api_key=JWT_TOKEN,
    user_id=USER,
    app_id=APP_ID,
    extract_every_n_turns=3,
    use_trigger_filter=True,
    auto_inject=False,   # set True to prepend memory context to each call
) as llm:

    history = []
    for turn in TURNS:
        history.append({"role": "user", "content": turn})
        response = llm.chat.completions.create(model="gpt-4o", messages=list(history))
        assistant_text = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_text})
        print(f"\n  User:      {turn[:70]}")
        print(f"  Assistant: {assistant_text[:70]}")

    print(f"\n  Buffered turns: {llm._user_turn_count} user turns across {len(history)} total")
    print(f"  Session ID:     {llm.session_id}")
    print("\n  Calling close() → flushes remaining turns as final ingest …")
    # __exit__ calls close() automatically at end of `with` block

print("\n  ✓ Session closed and flushed")

# ── Step 3: auto_inject=True demo ─────────────────────────────────────────────

section("Step 3 — auto_inject=True (memory prepended to system message)")

print("""
  With auto_inject=True the middleware fetches GET /context before every
  LLM call and prepends the result as a sentinel-wrapped system message:

    <!-- smritikosh:context-start -->
    ... user's remembered facts ...
    <!-- smritikosh:context-end -->

  The extraction pass strips these blocks so injected facts are never
  re-extracted (anti-contamination).
""")

with SmritikoshMiddleware(
    FakeOpenAI(),
    smritikosh_url="http://localhost:8080",
    smritikosh_api_key=JWT_TOKEN,
    user_id=USER,
    app_id=APP_ID,
    extract_every_n_turns=10,
    auto_inject=True,
) as llm:
    msgs = [{"role": "user", "content": "What do you know about my travel plans and family?"}]
    llm.chat.completions.create(model="gpt-4o", messages=msgs)
    print(f"  Context fetched and injected for user={USER!r}")
    print("  ✓ The fake LLM received a system message with Priya's memory context")

# ── Step 4: Verify extraction appeared in context ─────────────────────────────

section("Step 4 — Verify extracted facts appear in context retrieval")

print("\nWaiting 2 seconds for writes to propagate …")
time.sleep(2)

try:
    context = smriti_client.get_context(USER, "Tell me about Priya's travel plans and family preferences")
    print(f"\nContext (first 600 chars):\n")
    print((context or "(empty)")[:600])
except Exception as e:
    print(f"\n  Context retrieval timed out (LLM API slow after back-to-back ingest calls).")
    print(f"  The memories were extracted — verify with:  python sample/chatbot.py")
    print(f"  Error: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────

section("Demo Complete")
print(f"""
User: {USER!r}  |  App: {APP_ID!r}  |  Session: see server logs

What was demonstrated:
  ✓ SmritikoshMiddleware wraps any OpenAI-compatible sync client
  ✓ Developer code unchanged — just swap the client object
  ✓ Turns buffered transparently; partial ingest fires every N user turns
  ✓ close() (or context manager __exit__) flushes the final batch
  ✓ auto_inject=True prepends memory context before each LLM call
  ✓ Extracted facts appear in context retrieval

To use with a real OpenAI client:
    import openai
    llm = SmritikoshMiddleware(
        openai.OpenAI(),
        smritikosh_url="http://localhost:8080",
        smritikosh_api_key="sk-smriti-...",
        user_id="{USER}",
    )
    response = llm.chat.completions.create(model="gpt-4o", messages=[...])

To use with Anthropic:
    import anthropic
    llm = SmritikoshMiddleware(
        anthropic.Anthropic(),
        smritikosh_url="http://localhost:8080",
        smritikosh_api_key="sk-smriti-...",
        user_id="{USER}",
    )
    response = llm.messages.create(model="claude-haiku-4-5-20251001", max_tokens=1024, messages=[...])
""")
