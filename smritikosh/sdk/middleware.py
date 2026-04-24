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

import json as _json
import threading
import uuid
from typing import Any

import httpx

_DEFAULT_URL = "http://localhost:8080"
_DEFAULT_EXTRACT_EVERY = 10

# ── remember() tool definitions ───────────────────────────────────────────────

_REMEMBER_TOOL_OPENAI: dict = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Store something important about the user that should be remembered in "
            "future conversations. Call this when the user reveals a clear preference, "
            "fact, goal, or decision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact to remember, in plain English",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "preference", "goal", "skill", "habit",
                        "role", "project", "belief", "context",
                    ],
                },
                "key": {
                    "type": "string",
                    "description": "Short label for the fact (e.g. 'editor', 'timezone')",
                },
                "value": {
                    "type": "string",
                    "description": "The value (e.g. 'neovim', 'UTC+5:30')",
                },
            },
            "required": ["content", "category"],
        },
    },
}

_REMEMBER_TOOL_ANTHROPIC: dict = {
    "name": "remember",
    "description": (
        "Store something important about the user that should be remembered in "
        "future conversations. Call this when the user reveals a clear preference, "
        "fact, goal, or decision."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to remember, in plain English",
            },
            "category": {
                "type": "string",
                "enum": [
                    "preference", "goal", "skill", "habit",
                    "role", "project", "belief", "context",
                ],
            },
            "key": {
                "type": "string",
                "description": "Short label for the fact (e.g. 'editor', 'timezone')",
            },
            "value": {
                "type": "string",
                "description": "The value (e.g. 'neovim', 'UTC+5:30')",
            },
        },
        "required": ["content", "category"],
    },
}


def _blocks_to_anthropic_content(content_blocks: list) -> list[dict]:
    """Serialize Anthropic response content blocks to message-content format."""
    result = []
    for b in content_blocks:
        btype = getattr(b, "type", "")
        if btype == "text":
            result.append({"type": "text", "text": getattr(b, "text", "")})
        elif btype == "tool_use":
            result.append({
                "type": "tool_use",
                "id": getattr(b, "id", ""),
                "name": getattr(b, "name", ""),
                "input": getattr(b, "input", {}),
            })
    return result


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
        enable_remember_tool: bool = True,
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
        self._enable_remember_tool = enable_remember_tool

        self._buffer: list[dict] = []
        self._user_turn_count = 0
        self._last_ingested_at = 0    # _user_turn_count snapshot at last flush
        self._last_flush_buf_idx = 0  # buffer index of the first unsent turn
        self._lock = threading.Lock()
        self._closed = False
        self._pending_threads: list[threading.Thread] = []
        self._http = httpx.Client(timeout=60.0)
        # Depth counter prevents re-entrant remember() handling in follow-up calls
        self._remember_loop_depth = 0

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
        """Flush any remaining unsent turns as a final (non-partial) ingest."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Capture only the turns not yet sent by a partial flush.
            # _last_flush_buf_idx is already up-to-date: it was advanced inside
            # the lock when each partial flush was scheduled, so no re-read after
            # join() is needed.
            turns = list(self._buffer[self._last_flush_buf_idx:])
            pending = list(self._pending_threads)

        # Wait for any in-flight partial flushes to finish before sending the
        # final ingest — ensures ordering (server processes partials first).
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
                # Send only turns accumulated since the last partial flush (streaming window).
                # The server uses its stored last_turn_index to skip already-processed turns,
                # but sending only the new slice avoids re-transmitting the full history.
                window_to_flush = list(self._buffer[self._last_flush_buf_idx:])
                self._last_flush_buf_idx = len(self._buffer)

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

    # ── remember() tool handling ──────────────────────────────────────────────

    def _store_fact_sync(self, tool_input: dict) -> None:
        """POST /memory/fact for a remember() tool call. Best-effort, never raises."""
        try:
            key = tool_input.get("key") or tool_input.get("content", "unknown")[:50]
            value = tool_input.get("value") or tool_input.get("content", "unknown")
            self._http.post(
                f"{self._url}/memory/fact",
                json={
                    "user_id": self.user_id,
                    "app_id": self.app_id,
                    "category": tool_input.get("category", "preference"),
                    "key": key,
                    "value": value,
                    "note": tool_input.get("content"),
                    "source_type": "tool_use",
                },
                headers=self._api_headers,
            )
        except Exception:
            pass

    def _handle_openai_remember(
        self,
        response: Any,
        messages: list[dict],
        create_fn: Any,
        kwargs: dict,
    ) -> Any:
        """
        Inspect an OpenAI-style response for remember() tool calls.

        Works for any provider whose response follows the OpenAI schema —
        that includes openai, LiteLLM (all its providers), vLLM, llama.cpp, etc.

        ``create_fn`` is a callable that accepts ``messages=`` and ``**kwargs``
        and returns another OpenAI-style response, e.g.:
          - ``self._completions.create``  (OpenAI / OpenAI-compat SDK)
          - ``litellm.completion``        (LiteLLM)

        - If ALL tool calls are remember(): save facts and do a transparent follow-up
          LLM call — the app never sees the remember() call.
        - If mixed with other tools: save facts and return the response as-is.
        """
        choices = getattr(response, "choices", None) or []
        if not choices:
            return response
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        remember_calls = [
            tc for tc in tool_calls
            if getattr(tc.function, "name", "") == "remember"
        ]
        if not remember_calls:
            return response

        # Parse and store each fact synchronously (we block on follow-up anyway)
        for tc in remember_calls:
            try:
                args = _json.loads(tc.function.arguments)
            except Exception:
                args = {"content": str(tc.function.arguments), "category": "preference"}
            self._store_fact_sync(args)

        other_calls = [
            tc for tc in tool_calls
            if getattr(tc.function, "name", "") != "remember"
        ]
        if other_calls:
            # Mixed tool calls — facts saved above; return as-is for app to handle others
            return response

        # All calls are remember() — make a transparent follow-up call
        assistant_msg: dict = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": getattr(tc.function, "name", ""),
                        "arguments": getattr(tc.function, "arguments", ""),
                    },
                }
                for tc in remember_calls
            ],
        }
        assistant_content = getattr(message, "content", None)
        if assistant_content is not None:
            assistant_msg["content"] = assistant_content

        tool_results = [
            {"role": "tool", "tool_call_id": tc.id, "content": "Memory saved."}
            for tc in remember_calls
        ]
        new_messages = messages + [assistant_msg] + tool_results

        self._remember_loop_depth += 1
        try:
            follow_up = create_fn(messages=new_messages, **kwargs)
        finally:
            self._remember_loop_depth -= 1
        return follow_up

    def _handle_anthropic_remember(
        self,
        response: Any,
        messages: list[dict],
        messages_ns: Any,
        kwargs: dict,
    ) -> Any:
        """
        Inspect an Anthropic response for remember() tool_use blocks.

        - If ALL tool_use blocks are remember(): save facts and do a transparent follow-up.
        - If mixed: save facts and return the response as-is.
        """
        content_blocks = getattr(response, "content", []) or []
        remember_blocks = [
            b for b in content_blocks
            if getattr(b, "type", "") == "tool_use" and getattr(b, "name", "") == "remember"
        ]
        if not remember_blocks:
            return response

        for b in remember_blocks:
            self._store_fact_sync(getattr(b, "input", {}) or {})

        other_tool_blocks = [
            b for b in content_blocks
            if getattr(b, "type", "") == "tool_use" and getattr(b, "name", "") != "remember"
        ]
        if other_tool_blocks:
            return response

        # All tool_use blocks are remember() — transparent follow-up
        assistant_content = _blocks_to_anthropic_content(content_blocks)
        tool_results_msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": getattr(b, "id", ""),
                    "content": "Memory saved.",
                }
                for b in remember_blocks
            ],
        }
        new_messages = (
            messages
            + [{"role": "assistant", "content": assistant_content}]
            + [tool_results_msg]
        )

        self._remember_loop_depth += 1
        try:
            follow_up = messages_ns.create(messages=new_messages, **kwargs)
        finally:
            self._remember_loop_depth -= 1
        return follow_up

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

        1. Inject the remember() tool definition into tools (if enabled and not in loop).
        2. Optionally inject memory context into the system message.
        3. Forward to the real OpenAI client.
        4. Handle any remember() tool calls in the response.
        5. Buffer the user turns and fire partial ingestion if threshold is met.
        """
        active = self._mw._enable_remember_tool and self._mw._remember_loop_depth == 0

        if active:
            tools = list(kwargs.get("tools") or [])
            if not any(t.get("function", {}).get("name") == "remember" for t in tools):
                tools.append(_REMEMBER_TOOL_OPENAI)
            kwargs = {**kwargs, "tools": tools}

        outgoing = self._mw._inject_context(messages) if self._mw._auto_inject else messages
        response = self._completions.create(messages=outgoing, **kwargs)

        if active:
            response = self._mw._handle_openai_remember(
                response, outgoing, self._completions.create, kwargs
            )

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
        Injects the remember() tool and handles any remember() calls in the response.
        """
        active = self._mw._enable_remember_tool and self._mw._remember_loop_depth == 0

        if active:
            tools = list(kwargs.get("tools") or [])
            if not any(t.get("name") == "remember" for t in tools):
                tools.append(_REMEMBER_TOOL_ANTHROPIC)
            kwargs = {**kwargs, "tools": tools}

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

        if active:
            response = self._mw._handle_anthropic_remember(
                response, messages, self._ns, kwargs
            )

        self._mw._record_and_maybe_flush(messages)
        return response


# ── LiteLLM middleware ────────────────────────────────────────────────────────


class LiteLLMMiddleware(SmritikoshMiddleware):
    """
    Memory-extracting wrapper for ``litellm.completion()`` / ``litellm.acompletion()``.

    Covers every provider LiteLLM supports through a single interface:
      - Gemini   (``model="gemini/gemini-1.5-pro"``)
      - Ollama   (``model="ollama_chat/llama3"``)
      - vLLM     (``model="openai/<model>"``, ``api_base=...``)
      - llama.cpp(``model="openai/<model>"``, ``api_base=...``)
      - OpenAI   (``model="gpt-4o"``)
      - Claude   (``model="claude-haiku-4-5-20251001"``)

    LiteLLM responses follow the OpenAI schema, so all ``remember()`` tool
    injection and interception logic is reused directly.

    Usage::

        import litellm
        from smritikosh.sdk import LiteLLMMiddleware

        mw = LiteLLMMiddleware(
            litellm,
            smritikosh_api_key="sk-smriti-...",
            user_id="alice",
            app_id="my-app",
        )
        response = mw.completion(model="ollama_chat/llama3", messages=[...])
        mw.close()

    Context manager::

        with LiteLLMMiddleware(litellm, ...) as mw:
            response = mw.completion(model="gemini/gemini-1.5-pro", messages=[...])
    """

    def __init__(self, litellm_module: Any, **kwargs: Any) -> None:
        # The litellm module is stored as self._llm via the parent __init__.
        # The chat/messages proxy properties on SmritikoshMiddleware are never
        # called for LiteLLM — callers use completion() / acompletion() instead.
        super().__init__(litellm_module, **kwargs)
        self._litellm = litellm_module

    def completion(self, *, messages: list[dict], **kwargs: Any) -> Any:
        """
        Intercept ``litellm.completion()``.

        Injects the ``remember()`` tool, forwards to LiteLLM, handles any
        ``remember()`` calls transparently, and buffers turns for ingestion.
        """
        active = self._enable_remember_tool and self._remember_loop_depth == 0

        if active:
            tools = list(kwargs.get("tools") or [])
            if not any(t.get("function", {}).get("name") == "remember" for t in tools):
                tools.append(_REMEMBER_TOOL_OPENAI)
            kwargs = {**kwargs, "tools": tools}

        outgoing = self._inject_context(messages) if self._auto_inject else messages
        response = self._litellm.completion(messages=outgoing, **kwargs)

        if active:
            response = self._handle_openai_remember(
                response, outgoing, self._litellm.completion, kwargs
            )

        self._record_and_maybe_flush(messages)
        return response
