# chatbot.py
"""
Memory-aware chatbot using Smritikosh for persistent user memory.

Reads LLM config automatically from the project's .env file.
Supports: ollama, anthropic, openai, gemini (openai-compatible).

Usage:
    python chatbot.py

Commands during the chat:
    /remember <text>    — Manually store something as a memory
    /search <query>     — Search Alice's memories and show scored results
    /quit               — Exit
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load the project .env (one level up from this sample/ directory)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()  # fallback: look in cwd

from client import SmritikoshClient  # noqa: E402 (import after env load)

# ── LLM config from .env ──────────────────────────────────────────────────────

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen2.5:14b")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")

# ── Build the LLM client based on provider ────────────────────────────────────

def _make_llm_client():
    if LLM_PROVIDER == "anthropic":
        try:
            import anthropic
        except ImportError:
            sys.exit("anthropic SDK not installed. Run: pip install anthropic")
        return ("anthropic", anthropic.Anthropic(api_key=LLM_API_KEY))

    # ollama, openai, gemini — all use the OpenAI-compatible API
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai SDK not installed. Run: pip install openai")

    base_urls = {
        "ollama": f"{LLM_BASE_URL.rstrip('/')}/v1",
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    }
    base_url = base_urls.get(LLM_PROVIDER, f"{LLM_BASE_URL.rstrip('/')}/v1")
    api_key  = LLM_API_KEY or ("ollama" if LLM_PROVIDER == "ollama" else "no-key")
    return ("openai", OpenAI(base_url=base_url, api_key=api_key))


_provider_type, _llm = _make_llm_client()

# ── Smritikosh client ─────────────────────────────────────────────────────────

SMRITIKOSH_USER      = "alice"
SMRITIKOSH_USER_PASS = "alicepass"

memory       = SmritikoshClient(username=SMRITIKOSH_USER, password=SMRITIKOSH_USER_PASS)
conversation: list[dict] = []

# ── LLM call (normalised across providers) ────────────────────────────────────

def _llm_call(system: str, messages: list[dict]) -> str:
    if _provider_type == "anthropic":
        response = _llm.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    else:
        full_messages = [{"role": "system", "content": system}] + messages
        response = _llm.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1024,
            messages=full_messages,
        )
        return response.choices[0].message.content

# ── Core chat logic ───────────────────────────────────────────────────────────

def chat(user_message: str) -> str:
    context = memory.get_context(SMRITIKOSH_USER, user_message)

    system_prompt = (
        "You are a helpful personal assistant. "
        "Use the memory context below to give personalised, accurate answers. "
        "If the context contains relevant information, use it naturally — "
        "do not say 'according to your memory'. Just answer as if you know the user well.\n\n"
        + context
    )

    conversation.append({"role": "user", "content": user_message})
    assistant_message = _llm_call(system_prompt, conversation)
    conversation.append({"role": "assistant", "content": assistant_message})

    memory.remember(
        SMRITIKOSH_USER,
        f"User asked: {user_message}\nAssistant replied: {assistant_message}",
    )
    return assistant_message


def show_search(query: str) -> None:
    results = memory.search(SMRITIKOSH_USER, query)
    if not results:
        print("  (no results)")
        return
    for r in results:
        score        = r.get("hybrid_score", 0)
        text         = r.get("raw_text", "")[:90]
        consolidated = "✓" if r.get("consolidated") else "·"
        print(f"  [{score:.3f}] {consolidated} {text}...")

# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(f"  Smritikosh demo chatbot  (user: {SMRITIKOSH_USER})")
    print(f"  LLM: {LLM_PROVIDER} / {LLM_MODEL}")
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

        if user_input.startswith("/quit"):
            print("Goodbye!")
            break

        if user_input.startswith("/remember "):
            text   = user_input[len("/remember "):]
            result = memory.remember(SMRITIKOSH_USER, text)
            print(f"  Stored. importance={result['importance_score']:.2f}  "
                  f"facts_extracted={result['facts_extracted']}")
            continue

        if user_input.startswith("/search "):
            query = user_input[len("/search "):]
            print(f"  Search results for: '{query}'")
            show_search(query)
            continue

        reply = chat(user_input)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
