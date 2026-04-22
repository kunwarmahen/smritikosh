"""
SmritikoshMiddleware — transparent memory extraction wrapper for LLM clients.

Wraps an OpenAI or Anthropic sync client. Every create() call is intercepted:
user turns are buffered, sentinel blocks injected (if auto_inject=True), and
ingestion fired automatically in the background.

Usage (OpenAI):
    from smritikosh.sdk.middleware import SmritikoshMiddleware
    import openai

    client = SmritikoshMiddleware(
        openai.OpenAI(),
        smritikosh_url="http://localhost:8080",
        smritikosh_api_key="sk-smriti-...",
        user_id="alice",
        app_id="my-app",
    )
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
    client.close()  # flushes remaining turns

Usage (Anthropic):
    from smritikosh.sdk.middleware import SmritikoshMiddleware
    import anthropic

    client = SmritikoshMiddleware(
        anthropic.Anthropic(),
        smritikosh_url="http://localhost:8080",
        smritikosh_api_key="sk-smriti-...",
        user_id="alice",
    )
    response = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=1024, messages=[...])

Context manager (auto-closes and flushes):
    with SmritikoshMiddleware(openai.OpenAI(), ...) as client:
        response = client.chat.completions.create(...)
"""
from __future__ import annotations

import threading
import uuid
from typing import Any

import httpx

_DEFAULT_URL = "http://localhost:8080"
_DEFAULT_EXTRACT_EVERY = 10


class SmritikoshMiddleware:
    """
    Transparent memory extraction wrapper for OpenAI / Anthropic sync clients.

    All attribute access is proxied through to the underlying LLM client except
    ``chat`` (OpenAI) and ``messages`` (Anthropic), which are intercepted to
    buffer turns and trigger background ingestion.

    Args:
        llm_client:            ``openai.OpenAI()`` or ``anthropic.Anthropic()`` instance.
        smritikosh_url:        Base URL of the running Smritikosh server.
        smritikosh_api_key:    API key (Bearer token) for authentication.
        user_id:               User whose memories should be extracted.
        app_id:                Application namespace (default: ``"default"``).
        session_id:            Idempotency key for this conversation. Auto-generated UUID
                               if not provided — pass the same value to resume a session.
        extract_every_n_turns: Fire a partial ingest after this many cumulative user turns.
                               Set to 0 to disable mid-session extraction (flush only on close).
        use_trigger_filter:    Skip LLM extraction when no trigger phrases are detected in
                               the window. Reduces cost on low-signal windows.
        auto_inject:           If True, retrieve memory context from Smritikosh before each
                               LLM call and prepend it to the system message, wrapped in
                               sentinel blocks so the extraction pass can strip it later.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        smritikosh_url: str = _DEFAULT_URL,
        smritikosh_api_key: str,
        user_id: str,
        app_id: str = "default",
        session_id: str | None = None,
        extract_every_n_turns: int = _DEFAULT_EXTRACT_EVERY,
        use_trigger_filter: bool = True,
        auto_inject: bool = False,
    ) -> None:
        self._llm = llm_client
        self._url = smritikosh_url.rstrip("/")
        self._api_headers = {
            "Authorization": f"Bearer {smritikosh_api_key}",
            "Content-Type": "application/json",
        }
        self.user_id = user_id
        self.app_id = app_id
        self.session_id = session_id or str(uuid.uuid4())
        self._every_n = extract_every_n_turns
        self._trigger_filter = use_trigger_filter
        self._auto_inject = auto_inject

        self._buffer: list[dict] = []
        self._user_turn_count = 0
        self._last_ingested_at = 0   # _user_turn_count snapshot at last flush
        self._lock = threading.Lock()
        self._closed = False
        self._pending_threads: list[threading.Thread] = []
        self._http = httpx.Client(timeout=60.0)

    # ── Transparent proxy ─────────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        """Proxy all attribute access to the underlying LLM client."""
        return getattr(self._llm, name)

    @property
    def chat(self) -> "_OpenAIChatProxy":
        """Intercepted proxy for OpenAI's ``client.chat`` namespace."""
        return _OpenAIChatProxy(self._llm.chat, self)

    @property
    def messages(self) -> "_AnthropicMessagesProxy":
        """Intercepted proxy for Anthropic's ``client.messages`` namespace."""
        return _AnthropicMessagesProxy(self._llm.messages, self)

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush any remaining buffered turns as a final (non-partial) ingest."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            turns = list(self._buffer)
            pending = list(self._pending_threads)

        # Wait for any in-flight partial flushes to commit their last_turn_index
        # before sending the final ingest — prevents the server re-processing turns
        # that were already covered by a partial flush (race condition).
        for t in pending:
            t.join(timeout=30)

        if turns:
            self._flush(partial=False, turns=turns)
        self._http.close()

    def __enter__(self) -> "SmritikoshMiddleware":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _record_and_maybe_flush(self, messages: list[dict]) -> None:
        """
        Buffer incoming messages and fire a partial ingest in a background
        thread if the per-N-turns threshold has been crossed.
        """
        window_to_flush: list[dict] | None = None

        with self._lock:
            if self._closed:
                return
            # Detect cumulative-history callers (standard OpenAI pattern): each call
            # re-sends the full conversation so far plus one new turn at the end.
            # In that case, skip the prefix we've already buffered to avoid duplicates.
            # For single-turn or fresh calls the prefix check won't match, so all
            # messages are added as normal.
            buf_len = len(self._buffer)
            if (
                buf_len > 0
                and len(messages) > buf_len
                and all(
                    messages[i].get("content", "") == self._buffer[i]["content"]
                    for i in range(buf_len)
                )
            ):
                new_messages = messages[buf_len:]
            else:
                new_messages = messages
            for msg in new_messages:
                self._buffer.append(
                    {"role": msg.get("role", "user"), "content": msg.get("content") or ""}
                )
                if msg.get("role") == "user":
                    self._user_turn_count += 1

            if (
                self._every_n > 0
                and self._user_turn_count - self._last_ingested_at >= self._every_n
            ):
                self._last_ingested_at = self._user_turn_count
                window_to_flush = list(self._buffer)

        if window_to_flush is not None:
            t = threading.Thread(
                target=self._flush,
                kwargs={"partial": True, "turns": window_to_flush},
                daemon=True,
            )
            with self._lock:
                # Prune finished threads before adding the new one
                self._pending_threads = [p for p in self._pending_threads if p.is_alive()]
                self._pending_threads.append(t)
            t.start()

    def _flush(self, *, partial: bool, turns: list[dict]) -> None:
        """POST /ingest/session — best-effort, never raises."""
        try:
            self._http.post(
                f"{self._url}/ingest/session",
                json={
                    "user_id": self.user_id,
                    "app_id": self.app_id,
                    "session_id": self.session_id,
                    "turns": turns,
                    "partial": partial,
                    "use_trigger_filter": self._trigger_filter,
                    "metadata": {"source": "sdk_middleware"},
                },
                headers=self._api_headers,
            )
        except Exception:
            pass  # extraction is best-effort; never interrupt the LLM call

    def _get_context_text(self, query: str) -> str:
        """Return memory context_text for ``query``, or '' on any failure."""
        try:
            r = self._http.post(
                f"{self._url}/context",
                json={"user_id": self.user_id, "app_id": self.app_id, "query": query},
                headers=self._api_headers,
                timeout=10.0,
            )
            return r.json().get("context_text", "")
        except Exception:
            return ""

    def _inject_context(self, messages: list[dict]) -> list[dict]:
        """
        Prepend memory context to the system message, wrapped in sentinel blocks.

        If no user message is present, or context retrieval fails, returns
        the original messages list unchanged.
        """
        query = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if not query:
            return messages
        context_text = self._get_context_text(query)
        if not context_text:
            return messages

        sentinel = (
            f"<!-- smritikosh:context-start -->\n{context_text}\n<!-- smritikosh:context-end -->"
        )
        patched = list(messages)
        if patched and patched[0].get("role") == "system":
            patched[0] = {**patched[0], "content": f"{sentinel}\n\n{patched[0]['content']}"}
        else:
            patched.insert(0, {"role": "system", "content": sentinel})
        return patched


# ── OpenAI proxy chain ────────────────────────────────────────────────────────


class _OpenAIChatProxy:
    """Proxies ``client.chat`` — only ``completions`` is intercepted."""

    def __init__(self, chat: Any, middleware: SmritikoshMiddleware) -> None:
        self._chat = chat
        self._mw = middleware

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)

    @property
    def completions(self) -> "_OpenAICompletionsProxy":
        return _OpenAICompletionsProxy(self._chat.completions, self._mw)


class _OpenAICompletionsProxy:
    """Proxies ``client.chat.completions`` — ``create`` is intercepted."""

    def __init__(self, completions: Any, middleware: SmritikoshMiddleware) -> None:
        self._completions = completions
        self._mw = middleware

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    def create(self, *, messages: list[dict], **kwargs: Any) -> Any:
        """
        Intercept ``chat.completions.create()``.

        1. Optionally inject memory context into the system message.
        2. Forward to the real OpenAI client.
        3. Buffer the user turns and fire partial ingestion if threshold is met.
        """
        outgoing = self._mw._inject_context(messages) if self._mw._auto_inject else messages
        response = self._completions.create(messages=outgoing, **kwargs)
        self._mw._record_and_maybe_flush(messages)  # buffer original (pre-injection) turns
        return response


# ── Anthropic proxy ───────────────────────────────────────────────────────────


class _AnthropicMessagesProxy:
    """Proxies ``client.messages`` — ``create`` is intercepted."""

    def __init__(self, messages_ns: Any, middleware: SmritikoshMiddleware) -> None:
        self._ns = messages_ns
        self._mw = middleware

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ns, name)

    def create(self, *, messages: list[dict], **kwargs: Any) -> Any:
        """
        Intercept ``messages.create()``.

        Anthropic passes ``system`` as a top-level kwarg; messages are user/assistant
        pairs only.  We buffer the messages list for extraction.
        If auto_inject is on, a sentinel system block is prepended or merged.
        """
        if self._mw._auto_inject:
            query = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            context_text = self._mw._get_context_text(query) if query else ""
            if context_text:
                sentinel = (
                    f"<!-- smritikosh:context-start -->\n{context_text}\n"
                    f"<!-- smritikosh:context-end -->"
                )
                existing_system = kwargs.get("system", "")
                kwargs = {
                    **kwargs,
                    "system": f"{sentinel}\n\n{existing_system}".strip(),
                }

        response = self._ns.create(messages=messages, **kwargs)
        self._mw._record_and_maybe_flush(messages)
        return response
