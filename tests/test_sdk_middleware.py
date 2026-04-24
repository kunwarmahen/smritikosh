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
    LiteLLMMiddleware,
    SmritikoshMiddleware,
    _AnthropicMessagesProxy,
    _OpenAIChatProxy,
    _OpenAICompletionsProxy,
    _REMEMBER_TOOL_OPENAI,
    _REMEMBER_TOOL_ANTHROPIC,
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
    llm.chat.completions.create.assert_called_once()
    call_kwargs = llm.chat.completions.create.call_args[1]
    assert call_kwargs["messages"] == MSGS_1_USER
    assert call_kwargs["model"] == "gpt-4o"


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
    llm.messages.create.assert_called_once()
    call_kwargs = llm.messages.create.call_args[1]
    assert call_kwargs["messages"] == MSGS_1_USER
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert call_kwargs["max_tokens"] == 100


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


# ── Phase 8: streaming windowed flushes ──────────────────────────────────────

def _get_ingest_session_body(call_args) -> dict:
    """Extract the JSON body from a mock _http.post call to /ingest/session."""
    return call_args[1].get("json") or call_args.kwargs.get("json", {})


def test_first_partial_flush_sends_initial_window():
    """First partial flush must send the turns accumulated so far, not an empty list."""
    mw = _make_middleware(extract_every_n_turns=2)
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "A"}])
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "B"}])
    time.sleep(0.1)

    ingest_calls = [c for c in mw._http.post.call_args_list if "/ingest/session" in str(c)]
    assert len(ingest_calls) >= 1
    turns = _get_ingest_session_body(ingest_calls[0])["turns"]
    assert len(turns) == 2
    assert turns[0]["content"] == "A"
    assert turns[1]["content"] == "B"


def test_second_partial_flush_sends_only_new_turns():
    """
    After the first partial flush (turns 1-2), the second flush (turns 3-4) must
    contain ONLY turns 3-4, not the full history.
    """
    mw = _make_middleware(extract_every_n_turns=2)

    # First window
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "turn-1"}])
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "turn-2"}])
    time.sleep(0.1)
    first_call_count = mw._http.post.call_count

    # Second window
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "turn-3"}])
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "turn-4"}])
    time.sleep(0.1)

    ingest_calls = [c for c in mw._http.post.call_args_list if "/ingest/session" in str(c)]
    assert len(ingest_calls) >= 2

    second_body = _get_ingest_session_body(ingest_calls[1])
    turns = second_body["turns"]
    # The second window must only contain turns 3 and 4 — NOT 1-4
    assert len(turns) == 2
    assert turns[0]["content"] == "turn-3"
    assert turns[1]["content"] == "turn-4"


def test_close_after_partial_flush_sends_only_remaining_turns():
    """
    After a partial flush sent turns 1-2, close() must send ONLY turns 3+ (not 1-3).
    """
    mw = _make_middleware(extract_every_n_turns=2)

    # First window — triggers partial flush
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "partial-1"}])
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "partial-2"}])
    time.sleep(0.1)

    # One more turn that hasn't been flushed yet
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "final-3"}])
    mw.close()

    ingest_calls = [c for c in mw._http.post.call_args_list if "/ingest/session" in str(c)]
    # At least: one partial + one final
    assert len(ingest_calls) >= 2

    final_body = _get_ingest_session_body(ingest_calls[-1])
    assert final_body["partial"] is False
    turns = final_body["turns"]
    # Final flush must contain ONLY "final-3", not "partial-1" and "partial-2"
    assert len(turns) == 1
    assert turns[0]["content"] == "final-3"


def test_close_with_no_partial_flushes_sends_full_buffer():
    """
    With no prior partial flushes, close() must still send the complete buffer.
    """
    mw = _make_middleware(extract_every_n_turns=100)  # threshold never crossed
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "only-turn"}])
    mw.close()

    ingest_calls = [c for c in mw._http.post.call_args_list if "/ingest/session" in str(c)]
    assert len(ingest_calls) == 1
    body = _get_ingest_session_body(ingest_calls[0])
    assert body["partial"] is False
    assert len(body["turns"]) == 1
    assert body["turns"][0]["content"] == "only-turn"


def test_last_flush_buf_idx_advances_correctly():
    """_last_flush_buf_idx must advance by the number of buffered turns per window."""
    mw = _make_middleware(extract_every_n_turns=2)

    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "u1"}])
    mw.chat.completions.create(model="m", messages=[{"role": "user", "content": "u2"}])
    time.sleep(0.1)

    # After first flush, index should be at 2 (2 user turns buffered)
    assert mw._last_flush_buf_idx == 2


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


# ── Phase 5: remember() tool injection + intercept ────────────────────────────

def _openai_tool_response(tool_id: str, args_json: str) -> MagicMock:
    """Build a mock OpenAI response that contains a single remember() tool call."""
    tc = MagicMock()
    tc.id = tool_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = "remember"
    tc.function.arguments = args_json

    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg
    resp.choices[0].finish_reason = "tool_calls"
    return resp


def _openai_text_response(text: str = "Got it!") -> MagicMock:
    """Build a mock OpenAI response with a plain text reply and no tool calls."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg
    resp.choices[0].finish_reason = "stop"
    return resp


def _anthropic_tool_response(tool_id: str, tool_input: dict) -> MagicMock:
    """Build a mock Anthropic response that contains a single remember() tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = "remember"
    block.input = tool_input

    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "tool_use"
    return resp


def _anthropic_text_response(text: str = "Got it!") -> MagicMock:
    """Build a mock Anthropic response with a plain text reply."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "end_turn"
    return resp


# ── OpenAI: tool injection ────────────────────────────────────────────────────

def test_remember_tool_injected_openai():
    """remember() tool definition should be added to the outgoing tools list."""
    llm = _fake_openai_client()
    mw = _make_middleware(llm)
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)
    call_kwargs = llm.chat.completions.create.call_args[1]
    tools = call_kwargs.get("tools", [])
    assert any(t["function"]["name"] == "remember" for t in tools)


def test_remember_tool_not_duplicated_openai():
    """If remember() is already in tools, it should not be added again."""
    llm = _fake_openai_client()
    mw = _make_middleware(llm)
    existing_tools = [_REMEMBER_TOOL_OPENAI, {"type": "function", "function": {"name": "search"}}]
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER, tools=existing_tools)
    call_kwargs = llm.chat.completions.create.call_args[1]
    tools = call_kwargs.get("tools", [])
    assert sum(1 for t in tools if t["function"]["name"] == "remember") == 1


def test_remember_tool_not_injected_when_disabled():
    """When enable_remember_tool=False, tools list must not be modified."""
    llm = _fake_openai_client()
    mw = SmritikoshMiddleware(
        llm,
        smritikosh_api_key="sk-smriti-test",
        user_id="u",
        enable_remember_tool=False,
    )
    mw._http = MagicMock()
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)
    call_kwargs = llm.chat.completions.create.call_args[1]
    # No tools kwarg should be injected
    assert "tools" not in call_kwargs or not any(
        t.get("function", {}).get("name") == "remember"
        for t in (call_kwargs.get("tools") or [])
    )


# ── OpenAI: all-remember response (transparent follow-up) ─────────────────────

def test_remember_only_transparent_followup_openai():
    """When LLM returns only a remember() call, middleware must do a transparent follow-up."""
    llm = _fake_openai_client()
    args_json = '{"content":"I prefer neovim","category":"preference","key":"editor","value":"neovim"}'
    llm.chat.completions.create.side_effect = [
        _openai_tool_response("call_abc", args_json),
        _openai_text_response("Got it!"),
    ]
    mw = _make_middleware(llm)

    response = mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)

    # Two LLM calls: original + follow-up
    assert llm.chat.completions.create.call_count == 2
    # The returned response is the follow-up (text, not tool_calls)
    assert response.choices[0].message.content == "Got it!"


def test_remember_fact_stored_via_api_openai():
    """POST /memory/fact must be called with source_type='tool_use'."""
    llm = _fake_openai_client()
    args_json = '{"content":"I use dark mode","category":"preference","key":"theme","value":"dark"}'
    llm.chat.completions.create.side_effect = [
        _openai_tool_response("call_xyz", args_json),
        _openai_text_response(),
    ]
    mw = _make_middleware(llm)
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)

    # Find the /memory/fact POST call
    fact_calls = [
        c for c in mw._http.post.call_args_list
        if "/memory/fact" in str(c)
    ]
    assert len(fact_calls) == 1
    payload = fact_calls[0][1]["json"]
    assert payload["source_type"] == "tool_use"
    assert payload["key"] == "theme"
    assert payload["value"] == "dark"
    assert payload["user_id"] == "test-user"


def test_remember_followup_includes_tool_result_openai():
    """The follow-up LLM call must include the synthetic tool_result message."""
    llm = _fake_openai_client()
    args_json = '{"content":"I prefer Python","category":"preference","key":"language","value":"Python"}'
    llm.chat.completions.create.side_effect = [
        _openai_tool_response("call_t1", args_json),
        _openai_text_response(),
    ]
    mw = _make_middleware(llm)
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)

    second_call_messages = llm.chat.completions.create.call_args_list[1][1]["messages"]
    roles = [m["role"] for m in second_call_messages]
    assert "assistant" in roles
    assert "tool" in roles
    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_t1"
    assert tool_msg["content"] == "Memory saved."


def test_remember_content_fallback_when_key_value_absent():
    """If key/value are absent in the tool call, content should be used as fallback."""
    llm = _fake_openai_client()
    # Only content + category provided
    args_json = '{"content":"User always drinks black coffee","category":"habit"}'
    llm.chat.completions.create.side_effect = [
        _openai_tool_response("call_c1", args_json),
        _openai_text_response(),
    ]
    mw = _make_middleware(llm)
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)

    fact_calls = [c for c in mw._http.post.call_args_list if "/memory/fact" in str(c)]
    payload = fact_calls[0][1]["json"]
    assert payload["key"] == "User always drinks black coffee"[:50]
    assert payload["value"] == "User always drinks black coffee"
    assert payload["note"] == "User always drinks black coffee"


# ── OpenAI: mixed tool calls ───────────────────────────────────────────────────

def test_remember_mixed_calls_saves_fact_returns_original():
    """Mixed tool calls: remember() fact is saved, but response is returned as-is."""
    llm = _fake_openai_client()
    args_json = '{"content":"I prefer tabs","category":"preference","key":"indent","value":"tabs"}'

    # Build a response with both remember() and another tool call
    remember_tc = MagicMock()
    remember_tc.id = "call_r1"
    remember_tc.type = "function"
    remember_tc.function = MagicMock()
    remember_tc.function.name = "remember"
    remember_tc.function.arguments = args_json

    other_tc = MagicMock()
    other_tc.id = "call_o1"
    other_tc.type = "function"
    other_tc.function = MagicMock()
    other_tc.function.name = "search"
    other_tc.function.arguments = '{"query":"neovim docs"}'

    mixed_resp = MagicMock()
    mixed_resp.choices = [MagicMock()]
    mixed_resp.choices[0].message = MagicMock()
    mixed_resp.choices[0].message.tool_calls = [remember_tc, other_tc]
    mixed_resp.choices[0].finish_reason = "tool_calls"

    llm.chat.completions.create.return_value = mixed_resp
    mw = _make_middleware(llm)
    response = mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)

    # Only one LLM call — no follow-up for mixed
    assert llm.chat.completions.create.call_count == 1
    # Response returned as-is
    assert response is mixed_resp
    # But the fact was still saved
    fact_calls = [c for c in mw._http.post.call_args_list if "/memory/fact" in str(c)]
    assert len(fact_calls) == 1
    assert fact_calls[0][1]["json"]["source_type"] == "tool_use"


# ── OpenAI: loop depth guard ──────────────────────────────────────────────────

def test_remember_no_double_inject_in_follow_up():
    """remember() should not be re-injected into the follow-up call's tools list."""
    llm = _fake_openai_client()
    args_json = '{"content":"I prefer spaces","category":"preference","key":"indent","value":"spaces"}'
    llm.chat.completions.create.side_effect = [
        _openai_tool_response("call_d1", args_json),
        _openai_text_response(),
    ]
    mw = _make_middleware(llm)
    mw.chat.completions.create(model="gpt-4o", messages=MSGS_1_USER)

    # Both calls should have been made
    assert llm.chat.completions.create.call_count == 2
    # No infinite loops or extra calls
    assert llm.chat.completions.create.call_count == 2


# ── Anthropic: tool injection ─────────────────────────────────────────────────

def test_remember_tool_injected_anthropic():
    """remember() tool definition should be added to the Anthropic tools list."""
    llm = _fake_anthropic_client()
    mw = _make_middleware(llm)
    mw.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": "Hello"}],
    )
    call_kwargs = llm.messages.create.call_args[1]
    tools = call_kwargs.get("tools", [])
    assert any(t["name"] == "remember" for t in tools)


def test_remember_tool_not_duplicated_anthropic():
    """Anthropic: already-present remember() tool must not be added again."""
    llm = _fake_anthropic_client()
    mw = _make_middleware(llm)
    existing_tools = [_REMEMBER_TOOL_ANTHROPIC, {"name": "search", "input_schema": {}}]
    mw.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": "Hello"}],
        tools=existing_tools,
    )
    call_kwargs = llm.messages.create.call_args[1]
    tools = call_kwargs.get("tools", [])
    assert sum(1 for t in tools if t["name"] == "remember") == 1


# ── Anthropic: all-remember response ─────────────────────────────────────────

def test_remember_only_transparent_followup_anthropic():
    """Anthropic: when LLM returns only a remember() block, follow-up must be transparent."""
    llm = _fake_anthropic_client()
    tool_input = {"content": "I use emacs", "category": "preference", "key": "editor", "value": "emacs"}
    llm.messages.create.side_effect = [
        _anthropic_tool_response("toolu_a1", tool_input),
        _anthropic_text_response("Noted!"),
    ]
    mw = _make_middleware(llm)
    response = mw.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": "I use emacs always"}],
    )

    # Two LLM calls: original + follow-up
    assert llm.messages.create.call_count == 2
    # Returned response is the follow-up text response
    assert response.content[0].text == "Noted!"


def test_remember_fact_stored_via_api_anthropic():
    """Anthropic: POST /memory/fact must be called with correct payload."""
    llm = _fake_anthropic_client()
    tool_input = {"content": "I work in fintech", "category": "role", "key": "industry", "value": "fintech"}
    llm.messages.create.side_effect = [
        _anthropic_tool_response("toolu_b2", tool_input),
        _anthropic_text_response(),
    ]
    mw = _make_middleware(llm)
    mw.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": "I work in fintech"}],
    )

    fact_calls = [c for c in mw._http.post.call_args_list if "/memory/fact" in str(c)]
    assert len(fact_calls) == 1
    payload = fact_calls[0][1]["json"]
    assert payload["source_type"] == "tool_use"
    assert payload["key"] == "industry"
    assert payload["value"] == "fintech"


def test_remember_followup_includes_tool_result_anthropic():
    """Anthropic: follow-up messages must include a tool_result block."""
    llm = _fake_anthropic_client()
    tool_input = {"content": "I prefer dark mode", "category": "preference", "key": "theme", "value": "dark"}
    llm.messages.create.side_effect = [
        _anthropic_tool_response("toolu_c3", tool_input),
        _anthropic_text_response(),
    ]
    mw = _make_middleware(llm)
    mw.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": "I like dark mode"}],
    )

    second_call_messages = llm.messages.create.call_args_list[1][1]["messages"]
    # Last message should be a user message with tool_result content
    last_msg = second_call_messages[-1]
    assert last_msg["role"] == "user"
    assert isinstance(last_msg["content"], list)
    tr = last_msg["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "toolu_c3"
    assert tr["content"] == "Memory saved."


# ── LiteLLMMiddleware ─────────────────────────────────────────────────────────
# LiteLLM uses the OpenAI response schema, so most logic is shared.
# These tests verify the wiring via litellm.completion() specifically.

def _fake_litellm_module() -> MagicMock:
    """Return a mock that looks like the litellm module."""
    mod = MagicMock()
    mod.completion.return_value = _openai_text_response()
    return mod


def _make_litellm_middleware(
    litellm_mod: Any = None,
    *,
    session_id: str = "fixed-session-id",
    enable_remember_tool: bool = True,
) -> LiteLLMMiddleware:
    if litellm_mod is None:
        litellm_mod = _fake_litellm_module()
    mw = LiteLLMMiddleware(
        litellm_mod,
        smritikosh_api_key="sk-smriti-test",
        user_id="test-user",
        app_id="test-app",
        session_id=session_id,
        enable_remember_tool=enable_remember_tool,
    )
    mw._http = MagicMock()
    mw._http.post.return_value = MagicMock(json=lambda: {"context_text": ""})
    return mw


def test_litellm_completion_forwards_to_module():
    """LiteLLMMiddleware.completion() must call litellm.completion with the messages."""
    lit = _fake_litellm_module()
    mw = _make_litellm_middleware(lit)
    mw.completion(model="ollama_chat/llama3", messages=MSGS_1_USER)
    lit.completion.assert_called_once()
    call_kwargs = lit.completion.call_args[1]
    assert call_kwargs["messages"] == MSGS_1_USER
    assert call_kwargs["model"] == "ollama_chat/llama3"


def test_litellm_remember_tool_injected():
    """remember() tool must be injected into every litellm.completion() call."""
    lit = _fake_litellm_module()
    mw = _make_litellm_middleware(lit)
    mw.completion(model="gemini/gemini-1.5-pro", messages=MSGS_1_USER)
    tools = lit.completion.call_args[1].get("tools", [])
    assert any(t["function"]["name"] == "remember" for t in tools)


def test_litellm_remember_tool_disabled():
    """When enable_remember_tool=False, no tool is injected."""
    lit = _fake_litellm_module()
    mw = _make_litellm_middleware(lit, enable_remember_tool=False)
    mw.completion(model="ollama_chat/llama3", messages=MSGS_1_USER)
    kwargs = lit.completion.call_args[1]
    tools = kwargs.get("tools") or []
    assert not any(t.get("function", {}).get("name") == "remember" for t in tools)


def test_litellm_remember_only_transparent_followup():
    """All-remember response: middleware does a transparent follow-up via litellm.completion."""
    lit = _fake_litellm_module()
    args_json = '{"content":"I use vim","category":"preference","key":"editor","value":"vim"}'
    lit.completion.side_effect = [
        _openai_tool_response("call_l1", args_json),
        _openai_text_response("Remembered!"),
    ]
    mw = _make_litellm_middleware(lit)
    response = mw.completion(model="ollama_chat/llama3", messages=MSGS_1_USER)

    assert lit.completion.call_count == 2
    assert response.choices[0].message.content == "Remembered!"


def test_litellm_fact_stored_with_tool_use_source():
    """POST /memory/fact must be called with source_type='tool_use'."""
    lit = _fake_litellm_module()
    args_json = '{"content":"I run on Linux","category":"context","key":"os","value":"Linux"}'
    lit.completion.side_effect = [
        _openai_tool_response("call_l2", args_json),
        _openai_text_response(),
    ]
    mw = _make_litellm_middleware(lit)
    mw.completion(model="gemini/gemini-1.5-pro", messages=MSGS_1_USER)

    fact_calls = [c for c in mw._http.post.call_args_list if "/memory/fact" in str(c)]
    assert len(fact_calls) == 1
    payload = fact_calls[0][1]["json"]
    assert payload["source_type"] == "tool_use"
    assert payload["key"] == "os"
    assert payload["value"] == "Linux"


def test_litellm_turns_buffered_for_ingestion():
    """User turns must be buffered by LiteLLMMiddleware for session ingestion."""
    lit = _fake_litellm_module()
    mw = _make_litellm_middleware(lit)
    mw.completion(model="ollama_chat/llama3", messages=MSGS_1_USER)
    assert mw._user_turn_count == 1


def test_litellm_is_subclass_of_smritikosh_middleware():
    """LiteLLMMiddleware must be a SmritikoshMiddleware subclass."""
    assert issubclass(LiteLLMMiddleware, SmritikoshMiddleware)
