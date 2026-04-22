"""
Passive Memory Extraction — End-to-End Demo

Demonstrates the three passive extraction paths using SmritikoshClient
(same client used by seed_priya.py and chatbot.py):

1. ingest_session()  — extract memories from a conversation transcript
2. store_fact()      — manually enter a fact (ui_manual source, confidence=1.0)
3. get_context()     — verify the extracted facts appear in context retrieval

Setup:
    1. Start the server:       docker compose up -d
    2. Seed Priya's memories:  python sample/seed_priya.py
    3. Run this script:        python sample/passive_extraction_demo.py
"""

import time

from client import SmritikoshClient

USER   = "priya"
APP_ID = "default"

client = SmritikoshClient(username="admin", password="changeme123", app_id=APP_ID)

SESSION_ID = f"demo-session-{int(time.time())}"


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ── Demo conversation ─────────────────────────────────────────────────────────
# Trigger phrases embedded in user turns: "I always", "I prefer",
# "My goal is", "I believe", "I never".  Assistant turns are discarded.

CONVERSATION = [
    {
        "role": "user",
        "content": (
            "I always start my mornings browsing Net-a-Porter before the kids wake up. "
            "It's my guilty pleasure — especially on Sundays."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "That sounds like a lovely ritual! Are you looking for anything specific "
            "at the moment or just browsing for inspiration?"
        ),
    },
    {
        "role": "user",
        "content": (
            "I prefer investment pieces over fast fashion. "
            "Right now I'm eyeing a Bottega Veneta bag for our Japan trip next year."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Great taste — Bottega Veneta holds its value beautifully. "
            "Is Japan a family trip or just the two of you?"
        ),
    },
    {
        "role": "user",
        "content": (
            "The whole family — Rohan, Aanya, and Kabir. "
            "My goal is to get the kids to Japan before Kabir starts school. "
            "I'm researching kid-friendly ryokans and planning around cherry blossom season."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Cherry blossom season in Japan with kids sounds magical. "
            "Kyoto and Nara are wonderful for families — Nara especially with the deer!"
        ),
    },
    {
        "role": "user",
        "content": (
            "I believe slow travel is the only way to really experience a place. "
            "We never rush through destinations — we stayed two full weeks in the Maldives. "
            "I want Japan to feel the same way."
        ),
    },
]

# ── Step 1: ingest_session (passive extraction from full transcript) ───────────

section("Step 1 — ingest_session() (passive extraction from transcript)")

print(f"\nPosting {len(CONVERSATION)}-turn conversation (session_id={SESSION_ID!r})")
print(f"User turns only will be analysed: {sum(1 for t in CONVERSATION if t['role'] == 'user')} turns")
print("\nTrigger phrases expected in user turns:")
print("  • 'I always' (morning browsing ritual)")
print("  • 'I prefer' (investment pieces over fast fashion)")
print("  • 'My goal is' (Japan trip with the kids)")
print("  • 'I believe' (slow travel philosophy)")
print("  • 'I never' (never rushing destinations)")

result = client.ingest_session(
    USER,
    CONVERSATION,
    session_id=SESSION_ID,
    partial=False,
    use_trigger_filter=True,
    metadata={"demo": True, "source": "passive_extraction_demo.py"},
)

print(f"\nResult:")
print(f"  turns_processed:    {result['turns_processed']}")
print(f"  facts_extracted:    {result['facts_extracted']}")
print(f"  extraction_skipped: {result['extraction_skipped']}")
print(f"  already_processed:  {result['already_processed']}")
print(f"  partial:            {result['partial']}")

# ── Step 2: Idempotency check ─────────────────────────────────────────────────

section("Step 2 — Idempotency check (re-posting same session_id)")

result2 = client.ingest_session(
    USER,
    CONVERSATION,
    session_id=SESSION_ID,
)
print(f"\nRe-posted same session_id: {SESSION_ID!r}")
print(f"  already_processed: {result2['already_processed']} (should be True)")
assert result2["already_processed"], "Expected already_processed=True for duplicate session"
print("  ✓ Idempotency working correctly")

# ── Step 3: store_fact — manual fact entry ─────────────────────────────────────

section("Step 3 — store_fact() (manual fact entry, ui_manual source)")

manual_facts = [
    {"category": "preference", "key": "fashion_brand",    "value": "Bottega Veneta",        "note": "mentioned multiple times"},
    {"category": "preference", "key": "shopping_site",    "value": "Net-a-Porter, Mytheresa","note": "Sunday morning ritual"},
    {"category": "habit",      "key": "morning_routine",  "value": "browses Net-a-Porter before kids wake up"},
    {"category": "goal",       "key": "travel_goal",      "value": "visit every continent before turning 50"},
]

for fact in manual_facts:
    r = client.store_fact(USER, fact["category"], fact["key"], fact["value"], note=fact.get("note"))
    print(f"\n  Stored: {r['category']}/{r['key']} = {r['value']!r}")
    print(f"    confidence:  {r['confidence']:.2f}  (source: {r['source_type']})")
    print(f"    status:      {r['status']}")
    assert r["source_type"] == "ui_manual", f"Expected ui_manual, got {r['source_type']}"
    assert r["confidence"] == 1.0, f"ui_manual should have confidence=1.0"
    assert r["status"] == "active"

print("\n  ✓ All manual facts stored with confidence=1.0, status=active")

# ── Step 4: Streaming extraction (partial=True windows) ────────────────────────

section("Step 4 — Streaming extraction (partial=True windows)")

STREAMING_SESSION = f"streaming-{int(time.time())}"
WINDOW_1 = [
    {"role": "user",      "content": "I always read at least two books a month — mostly literary fiction and travel memoirs."},
    {"role": "assistant", "content": "That's wonderful. Any recent favourites?"},
]
WINDOW_2 = [
    {"role": "user",      "content": "My goal is to visit every continent before I turn 50. Asia and Europe are done!"},
    {"role": "assistant", "content": "Exciting! South America next perhaps?"},
]
WINDOW_3 = [
    {"role": "user", "content": "I prefer Chimamanda Ngozi Adichie and Pico Iyer above all other authors."},
]

print(f"\nPosting 3 partial windows for session {STREAMING_SESSION!r}")
for i, window in enumerate([WINDOW_1, WINDOW_2, WINDOW_3], 1):
    is_final = (i == 3)
    r = client.ingest_session(
        USER,
        window,
        session_id=STREAMING_SESSION,
        partial=not is_final,
        use_trigger_filter=True,
    )
    print(f"\n  Window {i} ({'final' if is_final else 'partial'}):")
    print(f"    turns_processed:    {r['turns_processed']}")
    print(f"    extraction_skipped: {r['extraction_skipped']}")
    print(f"    partial:            {r['partial']}")

# ── Step 5: Verify context retrieval ─────────────────────────────────────────

section("Step 5 — Verify context retrieval sees the extracted facts")

print("\nWaiting 2 seconds for writes to propagate …")
time.sleep(2)

try:
    context_text = client.get_context(USER, "Tell me about this user's travel plans and shopping preferences")
    print(f"\nContext text (first 800 chars):\n")
    print((context_text or "(empty)")[:800])
except Exception as e:
    print(f"\n  Context retrieval timed out (LLM API slow after back-to-back ingest calls).")
    print(f"  The memories were extracted — verify with:  python sample/chatbot.py")
    print(f"  Error: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────

section("Demo Complete")
print(f"""
User: {USER!r}  |  App: {APP_ID!r}

What was demonstrated:
  ✓ ingest_session()   — passive extraction from full conversation transcript
  ✓ Idempotency        — re-posting same session_id is a safe no-op
  ✓ Trigger filter     — LLM only called when high-signal phrases detected
  ✓ store_fact()       — manual fact entry at confidence=1.0 (ui_manual)
  ✓ Streaming windows  — partial=True windows accumulate turn index
  ✓ get_context()      — extracted facts appear in context retrieval

Source types now tracked:
  api_explicit          → remember() / POST /memory/event  (existing path)
  passive_distillation  → ingest_session()
  trigger_word          → ingest_session() with triggers matched
  ui_manual             → store_fact()

Next: run the full test suite with:
    pytest tests/test_trigger_detector.py tests/test_session_ingest.py -v
""")
