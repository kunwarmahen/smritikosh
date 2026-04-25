# media_ingest_demo.py
"""
Media Ingestion Demo — extract memories from a personal document.

This demo uploads a personal notes file (created in-memory — no disk file
needed) and shows how Smritikosh extracts durable facts from it automatically.

What this demo shows:
  1. Upload a personal notes document  → POST /ingest/media
  2. Poll until processing is complete → GET /ingest/media/{id}/status
  3. Review automatically-saved and pending facts
  4. Confirm pending facts             → POST /ingest/media/{id}/confirm
  5. Verify the facts appear in memory → POST /context

Requirements:
  - Smritikosh server running at http://localhost:8080
  - Priya's seed data already loaded: python sample/seed_priya.py
  - Core LLM configured in .env (same key used for the rest of the server)

No Whisper or Vision API key required — document extraction runs entirely
with your existing LLM.  Voice notes and images are shown as optional extras
at the end of the script.

Setup:
    1. docker compose up -d
    2. python sample/seed_priya.py   (one-time)
    3. python sample/media_ingest_demo.py
"""

import time

import httpx

from client import SmritikoshClient

USER = "priya"
APP_ID = "default"
BASE_URL = "http://localhost:8080"

# ── Auth ──────────────────────────────────────────────────────────────────────
# The sample client handles login via username/password.
# We also keep a raw JWT for multipart uploads (which need httpx.files, not JSON).

client = SmritikoshClient(username="admin", password="changeme123", app_id=APP_ID)

_auth = httpx.post(
    f"{BASE_URL}/auth/token",
    json={"username": "admin", "password": "changeme123"},
    timeout=10,
)
_auth.raise_for_status()
JWT_TOKEN = _auth.json()["access_token"]
BEARER = {"Authorization": f"Bearer {JWT_TOKEN}"}


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── The document ──────────────────────────────────────────────────────────────
# This is a plain-text personal notes file.  It is created in memory — you do
# not need any file on disk.  The content is deliberately first-person so the
# extractor can identify facts that are *about the user who wrote it*.

PERSONAL_NOTES = """\
# My Notes — Priya

## Daily Routine
I always start my mornings with an oat milk latte before the kids wake up.
I prefer to exercise three times a week — Pilates and long evening walks.
I never skip my Sunday morning browsing on Net-a-Porter.

## Food & Diet
I follow a mostly plant-based diet and buy organic produce whenever possible.
My favourite grocery store is Whole Foods — I shop there every Saturday morning.
I always order sparkling water at restaurants, never still.
I believe good quality ingredients make all the difference in cooking.

## Travel & Family
My goal is to take the whole family — Rohan, Aanya (8), and Kabir (5) —
to Japan before Kabir starts school.
I prefer slow travel over packed itineraries; we stayed two full weeks in the Maldives.
We never rush through a destination — Rohan and I both agree on this.
My next big trip goal is New Zealand for the South Island scenery.

## Fashion
I prefer investment pieces over fast fashion.
My favourite brands are Bottega Veneta and Chanel.
I shop mostly on Net-a-Porter and Mytheresa.

## Reading
I always read at least two books a month — mostly literary fiction and travel memoirs.
I prefer Chimamanda Ngozi Adichie and Pico Iyer above all other authors.
My goal is to finish Americanah before our Japan trip.
"""

# ── Helper functions ──────────────────────────────────────────────────────────


def upload_document(
    user_id: str,
    content: str,
    filename: str,
    context_note: str = "",
) -> dict:
    """Upload a text/markdown file for background memory extraction."""
    resp = httpx.post(
        f"{BASE_URL}/ingest/media",
        headers=BEARER,
        data={
            "user_id": user_id,
            "app_id": APP_ID,
            "content_type": "document",
            "context_note": context_note,
        },
        files={"file": (filename, content.encode("utf-8"), "text/markdown")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def poll_status(
    media_id: str,
    max_attempts: int = 30,
    interval: float = 2.0,
) -> dict:
    """Poll GET /ingest/media/{id}/status until processing finishes."""
    for attempt in range(1, max_attempts + 1):
        resp = httpx.get(
            f"{BASE_URL}/ingest/media/{media_id}/status",
            headers=BEARER,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status in ("complete", "nothing_found", "error"):
            return data
        print(f"  [{attempt}/{max_attempts}] status={status!r} … waiting {interval}s")
        time.sleep(interval)
    raise TimeoutError(
        f"Processing did not finish after {max_attempts} attempts "
        f"({max_attempts * interval:.0f}s). Check server logs."
    )


def confirm_facts(media_id: str, user_id: str, indices: list[int]) -> dict:
    """Confirm a list of pending facts by their list index."""
    resp = httpx.post(
        f"{BASE_URL}/ingest/media/{media_id}/confirm",
        headers={**BEARER, "Content-Type": "application/json"},
        json={"user_id": user_id, "confirmed_indices": indices},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Upload the document
# ═════════════════════════════════════════════════════════════════════════════

section("Step 1 — Upload a personal notes document")

print(f"\nDocument: priya_notes.md ({len(PERSONAL_NOTES)} bytes)")
print("Preview (first 4 lines):")
for line in PERSONAL_NOTES.strip().split("\n")[:4]:
    if line:
        print(f"  {line}")
print("  …")
print(f'\nContext note: "Extract personal preferences, habits, and goals."')

upload_result = upload_document(
    user_id=USER,
    content=PERSONAL_NOTES,
    filename="priya_notes.md",
    context_note="Extract personal preferences, habits, and goals from these notes.",
)

media_id = upload_result["media_id"]
print(f"\nServer accepted the file immediately:")
print(f"  media_id: {media_id}")
print(f"  status:   {upload_result['status']}  ← processing runs in the background")
print(f"\nSmritikosh will now:")
print(f"  1. Parse the markdown document")
print(f"  2. Apply first-person filter (keeps lines with 'I', 'my', 'we')")
print(f"  3. Run LLM fact extraction against Priya's existing knowledge")
print(f"  4. Route facts: relevance > 0.75 → saved; 0.60–0.75 → pending review")

# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Poll until processing is done
# ═════════════════════════════════════════════════════════════════════════════

section("Step 2 — Wait for extraction to complete")

print("\nPolling GET /ingest/media/{id}/status …")
status = poll_status(media_id)

print(f"\nProcessing complete:")
print(f"  status:               {status['status']}")
print(f"  facts_extracted:      {status['facts_extracted']}  (saved immediately, relevance > 0.75)")
print(f"  facts_pending_review: {status['facts_pending_review']}  (need your confirmation)")

# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Review extracted facts
# ═════════════════════════════════════════════════════════════════════════════

section("Step 3 — Extracted facts")

pending = status.get("pending_facts") or []

if status["status"] == "nothing_found":
    print(
        "\n  Nothing personally memorable found in this document.\n"
        "  Try adding a more specific context_note, or check that the document\n"
        "  contains first-person statements about the user."
    )
elif status["facts_extracted"] == 0 and not pending:
    print("\n  No facts extracted. The document may not contain clear personal statements.")
else:
    if status["facts_extracted"] > 0:
        print(
            f"\n  ✓ {status['facts_extracted']} fact(s) saved automatically — "
            "these are already in Priya's memory graph."
        )
    else:
        print("\n  No facts were saved automatically (relevance threshold not met).")

    if pending:
        print(f"\n  {len(pending)} fact(s) are waiting for your review:")
        print(
            "  (These scored 0.60–0.75 relevance — the system is not fully confident\n"
            "   and wants you to decide before writing them to memory.)\n"
        )
        for i, fact in enumerate(pending):
            content = fact.get("content") or f"{fact.get('key', '?')} = {fact.get('value', '?')}"
            confidence = fact.get("confidence", 0.0)
            category = fact.get("category", "?")
            print(f"  [{i}] {content}")
            print(f"      category={category!r}  confidence={confidence:.2f}")
    else:
        print("\n  No facts pending review.")

# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Confirm pending facts
# ═════════════════════════════════════════════════════════════════════════════

if pending:
    section(f"Step 4 — Confirm all {len(pending)} pending fact(s)")

    print(
        "\n  In the UI, the user sees a review modal and ticks the facts they\n"
        "  want to keep. This demo auto-confirms everything.\n"
    )

    confirm_result = confirm_facts(media_id, USER, list(range(len(pending))))
    print(f"  Server response: {confirm_result.get('message', 'OK')}")
    remaining = confirm_result.get("facts_pending_review", 0)
    print(f"  facts_pending_review remaining: {remaining}")
    print(f"  ✓ All pending facts moved to active status in Priya's memory graph.")
else:
    section("Step 4 — No pending facts to confirm (skipped)")
    print("\n  All extracted facts were saved automatically.")

# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — Verify the facts appear in context retrieval
# ═════════════════════════════════════════════════════════════════════════════

section("Step 5 — Verify facts appear in context retrieval")

print("\nWaiting 2 seconds for writes to propagate …")
time.sleep(2)

query = "Tell me about Priya's food habits, travel plans, and reading preferences"
print(f"Query: {query!r}\n")

try:
    context_text = client.get_context(USER, query)
    print("Context retrieved (first 700 chars):\n")
    print((context_text or "(empty context — LLM may be slow, try chatbot.py)").strip()[:700])
except Exception as e:
    print(f"  Context retrieval timed out or failed: {e}")
    print("  The memories are extracted — verify with: python sample/chatbot.py")

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

section("Demo Complete")

print(f"""
User: {USER!r}  |  App: {APP_ID!r}  |  media_id: {media_id!r}

What was demonstrated:
  ✓ POST /ingest/media         — uploaded a personal notes document; server returned
                                 immediately (status=processing) — no blocking wait
  ✓ GET  /ingest/media/*/status — polled until extraction finished
  ✓ POST /ingest/media/*/confirm — user confirmed borderline (pending) facts
  ✓ POST /context               — extracted facts visible in context retrieval

Source type: media_document  (initial confidence = 0.75)

Two-tier routing used by the extraction pipeline:
  relevance > 0.75  → fact saved immediately as "active" (no user action needed)
  relevance 0.60–0.75 → fact goes to "pending" — user reviews before it enters memory
  relevance < 0.60  → fact discarded (too uncertain or not personal enough)

─────────────────────────────────────────────────────────────
Optional: voice note upload (requires WHISPER_PROVIDER in .env)
─────────────────────────────────────────────────────────────
Record a .wav or .mp3 voice note and upload it:

  import httpx
  with open("note.wav", "rb") as f:
      httpx.post(
          "http://localhost:8080/ingest/media",
          headers={{"Authorization": "Bearer <token>"}},
          data={{"user_id": "priya", "app_id": "default", "content_type": "voice_note",
                "context_note": "My personal reminder"}},
          files={{"file": ("note.wav", f, "audio/wav")}},
      )

Set WHISPER_PROVIDER=openai in .env (and WHISPER_API_KEY) to enable transcription.
For self-hosted Whisper: WHISPER_PROVIDER=local + WHISPER_BASE_URL=http://localhost:8000

─────────────────────────────────────────────────────────────
Optional: image upload (requires VISION_PROVIDER in .env)
─────────────────────────────────────────────────────────────
Upload a receipt, screenshot, or whiteboard photo:

  content_type options: receipt | screenshot | whiteboard

  with open("grocery_receipt.jpg", "rb") as f:
      httpx.post(
          "http://localhost:8080/ingest/media",
          headers={{"Authorization": "Bearer <token>"}},
          data={{"user_id": "priya", "app_id": "default", "content_type": "receipt",
                "context_note": "My weekly grocery shop"}},
          files={{"file": ("receipt.jpg", f, "image/jpeg")}},
      )

Set VISION_PROVIDER=openai in .env to enable the vision model.
For Claude: VISION_PROVIDER=claude + VISION_API_KEY=sk-ant-...

─────────────────────────────────────────────────────────────
Optional: meeting recording (requires DIARIZATION_PROVIDER)
─────────────────────────────────────────────────────────────
Upload a meeting or call recording (up to 500 MB):

  content_type: meeting_recording

  Before uploading a meeting, enroll your voice at:
    http://localhost:3000/dashboard/settings/voice-enrollment
  (30-second recording; used to identify which speaker is you)

  Or set DIARIZATION_PROVIDER=none (default) — the pipeline falls back to
  first-person filter on the full transcript when no voice profile is enrolled.

─────────────────────────────────────────────────────────────
UI upload flow (no code needed)
─────────────────────────────────────────────────────────────
  1. Open http://localhost:3000
  2. Go to Memories → click the Upload button (📎 icon)
  3. Choose a tab: Voice | Document | Image | Meeting
  4. Select your file and add an optional context note
  5. Review the extracted facts in the confirmation modal
  6. Save or dismiss — done.

Next steps:
  python sample/chatbot.py    ← chat with Priya's enriched memory
  pytest tests/test_media_processor.py tests/test_media_ingest.py -v
""")
