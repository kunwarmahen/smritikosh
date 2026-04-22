"""
Tests for SmritikoshMiddleware (smritikosh/sdk/middleware.py).

All tests are offline — no server, no OpenAI/Anthropic credentials required.
The LLM client and httpx transport are fully mocked.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from smritikosh.sdk.middleware import (
    SmritikoshMiddleware,
    _AnthropicMessagesProxy,
    _OpenAIChatProxy,
    _OpenAICompletionsProxy,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _fake_openai_client() -> MagicMock:
    """Return a minimal mock that looks like openai.OpenAI()."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(choices=[])
    return client


def _fake_anthropic_client() -> MagicMock:
    """Return a minimal mock that looks like anthropic.Anthropic()."""
    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[])
    return client


def _make_middleware(
    llm_client: Any = None,
    *,
    extract_every_n_turns: int = 10,
    auto_inject: bool = False,
    use_trigger_filter: bool = True,
    session_id: str = "fixed-session-id",
) -> SmritikoshMiddleware:
    if llm_client is None:
        llm_client = _fake_openai_client()
    mw = SmritikoshMiddleware(
        llm_client,
        smritikosh_api_key="sk-smriti-test",
        user_id="test-user",
        app_id="test-app",
        session_id=session_id,
        extract_every_n_turns=extract_every_n_turns,
        auto_inject=auto_inject,
        use_trigger_filter=use_trigger_filter,
    )
    # Replace httpx.Client with a mock so no real network calls happen
    mw._http = MagicMock()
    mw._http.post.return_value = MagicMock(json=lambda: {"context_text": ""})
    return mw


MSGS_1_USER = [
    {"role": "user", "content": "Hello, I always use dark mode."},
]

MSGS_2_MIXED = [
    {"role": "user", "content": "I prefer Python."},
    {"role": "assistant", "content": "Great choice!"},
]


# ── Construction ──────────────────────────────────────────────────────────────

def test_session_id_auto_generated():
    mw = SmritikoshMiddleware(
        _fake_openai_client(),
        smritikosh_api_key="sk-smriti-test",
        user_id="u",
    )
    mw._http = MagicMock()
    assert mw.session_id  # non-empty
    try:
        uuid.UUID(mw.session_id)  # valid UUID
    finally:
        mw._http.close = MagicMock()
        mw.close()


def test_session_id_explicit():
    mw = _make_middleware(session_id="my-custom-session")
    assert mw.session_id == "my-custom-session"


def test_user_id_and_app_id_stored():
    mw = _make_middleware()
    assert mw.user_id == "test-user"
    assert mw.app_id == "test-app"


# ── Transparent proxy ─────────────────────────────────────────────────────────

def test_getattr_proxied_to_llm():
    llm = _fake_openai_client()
    llm.some_custom_attr = "hello"
    mw = _make_middleware(llm)
    assert mw.some_custom_attr == "hello"


def test_chat_returns_proxy():
    mw = _make_middleware(_fake_openai_client())
    assert isinstance(mw.chat, _OpenAIChatProxy)


def test_messages_returns_proxy():
    mw = _make_middleware(_fake_anthropic_client())
    assert isinstance(mw.messages, _AnthropicMessagesProxy)


def test_openai_chat_proxy_completions():
    mw = _make_middleware(_fake_openai_client())
    assert isinstance(mw.chat.completions, _OpenAICompletionsProxy)


# ── OpenAI interception ───────────────────────────────────────────────────────

def test_openai_create_forwards_to_real_client():
    llm = _fake_openai_client()
    mw = _make_middleware(llm)
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)
    llm.chat.completions.create.assert_called_once_with(
        messages=MSGS_1_USER, model="gpt-4o"
    )


def test_openai_create_returns_llm_response():
    llm = _fake_openai_client()
    sentinel = MagicMock()
    llm.chat.completions.create.return_value = sentinel
    mw = _make_middleware(llm)
    result = mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)
    assert result is sentinel


def test_openai_turns_buffered_after_create():
    mw = _make_middleware(extract_every_n_turns=100)
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_2_MIXED)
    assert mw._user_turn_count == 1  # only 1 user turn in MSGS_2_MIXED
    assert len(mw._buffer) == 2


# ── Anthropic interception ────────────────────────────────────────────────────

def test_anthropic_create_forwards_to_real_client():
    llm = _fake_anthropic_client()
    mw = _make_middleware(llm)
    mw.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=MSGS_1_USER,
    )
    llm.messages.create.assert_called_once_with(
        messages=MSGS_1_USER,
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
    )


def test_anthropic_turns_buffered():
    llm = _fake_anthropic_client()
    mw = _make_middleware(llm, extract_every_n_turns=100)
    mw.messages.create(model="m", max_tokens=10, messages=MSGS_1_USER)
    assert mw._user_turn_count == 1


# ── Background partial flush ──────────────────────────────────────────────────

def test_partial_flush_fires_after_n_turns():
    mw = _make_middleware(extract_every_n_turns=2)
    # 2 separate calls, each with 1 user turn → should trigger after 2nd
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "turn 1"}])
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "turn 2"}])
    # Allow background thread to complete
    time.sleep(0.1)
    assert mw._http.post.called
    call_kwargs = mw._http.post.call_args
    body = call_kwargs[1]["json"] if "json" in (call_kwargs[1] or {}) else call_kwargs.kwargs.get("json", {})
    assert body["partial"] is True
    assert body["session_id"] == "fixed-session-id"


def test_no_flush_before_threshold():
    mw = _make_middleware(extract_every_n_turns=10)
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "turn 1"}])
    time.sleep(0.05)
    # httpx mock should NOT have been called yet (1 turn < 10)
    mw._http.post.assert_not_called()


def test_flush_disabled_when_extract_n_is_zero():
    mw = _make_middleware(extract_every_n_turns=0)
    for _ in range(20):
        mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    time.sleep(0.05)
    mw._http.post.assert_not_called()


# ── close() — final flush ─────────────────────────────────────────────────────

def test_close_sends_final_non_partial_ingest():
    mw = _make_middleware(extract_every_n_turns=100)
    mw.chat.completions.create(model="m", messages=MSGS_1_USER)
    mw.close()
    mw._http.post.assert_called_once()
    body = mw._http.post.call_args.kwargs.get("json") or mw._http.post.call_args[1]["json"]
    assert body["partial"] is False
    assert body["user_id"] == "test-user"
    assert body["app_id"] == "test-app"


def test_close_is_idempotent():
    mw = _make_middleware()
    mw.chat.completions.create(model="m", messages=MSGS_1_USER)
    mw.close()
    mw.close()  # second close should not POST again
    assert mw._http.post.call_count == 1


def test_close_does_nothing_when_buffer_empty():
    mw = _make_middleware()
    mw.close()
    mw._http.post.assert_not_called()


# ── Context manager ───────────────────────────────────────────────────────────

def test_context_manager_calls_close():
    with _make_middleware(extract_every_n_turns=100) as mw:
        mw.chat.completions.create(model="m", messages=MSGS_1_USER)
    # After __exit__, buffer should have been flushed
    mw._http.post.assert_called_once()
    body = mw._http.post.call_args.kwargs.get("json") or mw._http.post.call_args[1]["json"]
    assert body["partial"] is False


# ── auto_inject ───────────────────────────────────────────────────────────────

def test_auto_inject_prepends_sentinel_to_system():
    llm = _fake_openai_client()
    mw = _make_middleware(llm, auto_inject=True)
    mw._http.post.return_value = MagicMock(
        json=lambda: {"context_text": "User likes Python."}
    )

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What do I like?"},
    ]
    mw.chat.completions.create(model="gpt-4o", messages=msgs)

    called_messages = llm.chat.completions.create.call_args.kwargs["messages"]
    assert called_messages[0]["role"] == "system"
    assert "smritikosh:context-start" in called_messages[0]["content"]
    assert "User likes Python." in called_messages[0]["content"]
    assert "You are a helpful assistant." in called_messages[0]["content"]


def test_auto_inject_creates_system_message_if_absent():
    llm = _fake_openai_client()
    mw = _make_middleware(llm, auto_inject=True)
    mw._http.post.return_value = MagicMock(
        json=lambda: {"context_text": "Priya loves travel."}
    )
    msgs = [{"role": "user", "content": "Where should I go?"}]
    mw.chat.completions.create(model="gpt-4o", messages=msgs)

    called_messages = llm.chat.completions.create.call_args.kwargs["messages"]
    assert called_messages[0]["role"] == "system"
    assert "Priya loves travel." in called_messages[0]["content"]


def test_auto_inject_noop_when_context_empty():
    llm = _fake_openai_client()
    mw = _make_middleware(llm, auto_inject=True)
    mw._http.post.return_value = MagicMock(json=lambda: {"context_text": ""})

    msgs = [{"role": "user", "content": "Hello"}]
    mw.chat.completions.create(model="gpt-4o", messages=msgs)

    called_messages = llm.chat.completions.create.call_args.kwargs["messages"]
    assert called_messages == msgs  # unchanged


def test_auto_inject_anthropic_sets_system_kwarg():
    llm = _fake_anthropic_client()
    mw = _make_middleware(llm, auto_inject=True)
    mw._http.post.return_value = MagicMock(
        json=lambda: {"context_text": "Alice uses Neovim."}
    )
    msgs = [{"role": "user", "content": "What editor do I use?"}]
    mw.messages.create(model="m", max_tokens=100, messages=msgs)

    call_kwargs = llm.messages.create.call_args.kwargs
    assert "system" in call_kwargs
    assert "Alice uses Neovim." in call_kwargs["system"]
    assert "smritikosh:context-start" in call_kwargs["system"]


# ── use_trigger_filter flag ───────────────────────────────────────────────────

def test_trigger_filter_flag_forwarded_in_flush():
    mw = _make_middleware(extract_every_n_turns=1, use_trigger_filter=False)
    mw.chat.completions.create(model="m", messages=MSGS_1_USER)
    time.sleep(0.1)
    body = mw._http.post.call_args.kwargs.get("json") or mw._http.post.call_args[1]["json"]
    assert body["use_trigger_filter"] is False


def test_trigger_filter_true_by_default_in_flush():
    mw = _make_middleware(extract_every_n_turns=1)
    mw.chat.completions.create(model="m", messages=MSGS_1_USER)
    time.sleep(0.1)
    body = mw._http.post.call_args.kwargs.get("json") or mw._http.post.call_args[1]["json"]
    assert body["use_trigger_filter"] is True


# ── Error resilience ──────────────────────────────────────────────────────────

def test_flush_failure_does_not_propagate():
    llm = _fake_openai_client()
    mw = _make_middleware(llm, extract_every_n_turns=1)
    mw._http.post.side_effect = Exception("network error")
    # Must not raise
    mw.chat.completions.create(model="m", messages=MSGS_1_USER)
    time.sleep(0.1)  # background thread runs and silently eats the exception


def test_context_fetch_failure_does_not_propagate():
    llm = _fake_openai_client()
    mw = _make_middleware(llm, auto_inject=True)
    mw._http.post.side_effect = Exception("network error")
    # Should still forward to LLM without crashing
    mw.chat.completions.create(model="m", messages=MSGS_1_USER)
    llm.chat.completions.create.assert_called_once()


# ── Thread safety ─────────────────────────────────────────────────────────────

def test_cumulative_history_no_duplicates():
    """Standard OpenAI pattern: each call re-sends full history — turns must not be duplicated."""
    mw = _make_middleware(extract_every_n_turns=100)
    history = []
    for i in range(4):
        history.append({"role": "user", "content": f"user turn {i}"})
        mw.chat.completions.create(model="m", messages=list(history))
        history.append({"role": "assistant", "content": f"assistant reply {i}"})
    # 4 user turns + 3 assistant turns = 7 total (last assistant reply only enters
    # the buffer when it appears in the next call's history — no 5th call here).
    assert mw._user_turn_count == 4
    assert len(mw._buffer) == 7


def test_concurrent_creates_do_not_corrupt_buffer():
    llm = _fake_openai_client()
    mw = _make_middleware(llm, extract_every_n_turns=1000)

    def worker():
        for _ in range(10):
            mw.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "concurrent"}],
            )

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mw._user_turn_count == 50
    assert len(mw._buffer) == 50
