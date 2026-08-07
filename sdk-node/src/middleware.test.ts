/**
 * SmritikoshMiddleware unit tests — mirrors tests/test_sdk_middleware.py.
 *
 * Strategy: stub global `fetch` (so no Smritikosh HTTP happens) and pass a
 * fake OpenAI / Anthropic client whose `create` is a vitest mock. Each test
 * verifies one of:
 *   1. the call is forwarded to the real client unchanged,
 *   2. turns are buffered / flushed with the right windowing,
 *   3. remember() tool injection and transparent follow-up behaviour,
 *   4. auto-inject sentinel wrapping.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { SmritikoshMiddleware, withMemory } from "./middleware.js";

// ── fixtures ──────────────────────────────────────────────────────────────────

const API_KEY = "sk-smriti-test";
const URL_BASE = "http://localhost:8080";

interface FakeOpenAI {
  chat: { completions: { create: ReturnType<typeof vi.fn> } };
  someOtherProp: string;
  unrelatedMethod: ReturnType<typeof vi.fn>;
}

function fakeOpenAI(response: unknown = openaiTextResponse()): FakeOpenAI {
  return {
    chat: { completions: { create: vi.fn().mockResolvedValue(response) } },
    someOtherProp: "passthrough",
    unrelatedMethod: vi.fn().mockReturnValue("called"),
  };
}

function fakeAnthropic(response: unknown = anthropicTextResponse()) {
  return { messages: { create: vi.fn().mockResolvedValue(response) } };
}

function openaiTextResponse(text = "Got it!") {
  return { choices: [{ message: { content: text, tool_calls: null } }] };
}

function openaiToolResponse(toolId: string, argsJson: string, name = "remember") {
  return {
    choices: [
      {
        message: {
          content: null,
          tool_calls: [{ id: toolId, type: "function", function: { name, arguments: argsJson } }],
        },
      },
    ],
  };
}

function anthropicTextResponse(text = "Got it!") {
  return { content: [{ type: "text", text }] };
}

function anthropicToolResponse(
  toolId: string,
  input: Record<string, unknown>,
  name = "remember",
) {
  return { content: [{ type: "tool_use", id: toolId, name, input }] };
}

function makeMiddleware(
  llm: unknown,
  overrides: Record<string, unknown> = {},
): SmritikoshMiddleware {
  return new SmritikoshMiddleware(llm, {
    smritikoshUrl: URL_BASE,
    smritikoshApiKey: API_KEY,
    userId: "alice",
    appId: "testapp",
    extractEveryNTurns: 2,
    enableRememberTool: false, // opt in per-test; keeps other tests focused
    ...overrides,
  } as never);
}

/** Stub fetch; returns the mock so tests can inspect Smritikosh calls. */
function stubFetch(contextText = ""): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue({ context_text: contextText }),
    text: vi.fn().mockResolvedValue("{}"),
  } as unknown as Response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function callsTo(fetchMock: ReturnType<typeof vi.fn>, path: string) {
  return fetchMock.mock.calls.filter((c) => String(c[0]).endsWith(path));
}

function bodyOf(call: unknown[]): Record<string, unknown> {
  const init = call[1] as RequestInit;
  return JSON.parse(init.body as string) as Record<string, unknown>;
}

/** Let un-awaited background flush promises settle. */
async function drain(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

const USER = (text: string) => ({ role: "user", content: text });

beforeEach(() => {
  vi.unstubAllGlobals();
});

// ── construction & proxying ───────────────────────────────────────────────────

describe("construction", () => {
  it("auto-generates a session id", () => {
    stubFetch();
    const mw = makeMiddleware(fakeOpenAI());
    expect(mw.sessionId).toMatch(/^[0-9a-f-]{36}$/i);
  });

  it("uses an explicit session id when given", () => {
    stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { sessionId: "session-123" });
    expect(mw.sessionId).toBe("session-123");
  });

  it("stores userId and appId", () => {
    stubFetch();
    const mw = makeMiddleware(fakeOpenAI());
    expect(mw.userId).toBe("alice");
    expect(mw.appId).toBe("testapp");
  });

  it("proxies unknown properties through to the wrapped client", () => {
    stubFetch();
    const llm = fakeOpenAI();
    const mw = withMemory(llm, {
      smritikoshApiKey: API_KEY,
      userId: "alice",
    });
    expect(mw.someOtherProp).toBe("passthrough");
    expect(mw.unrelatedMethod()).toBe("called");
  });
});

// ── OpenAI path ───────────────────────────────────────────────────────────────

describe("openai create", () => {
  it("forwards the call to the real client", async () => {
    stubFetch();
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm);

    await mw.chat.completions.create({
      model: "gpt-4o",
      messages: [USER("hello")],
    } as never);

    expect(llm.chat.completions.create).toHaveBeenCalledOnce();
    const params = llm.chat.completions.create.mock.calls[0]![0];
    expect(params.model).toBe("gpt-4o");
    expect(params.messages).toEqual([USER("hello")]);
  });

  it("returns the underlying response untouched", async () => {
    stubFetch();
    const response = openaiTextResponse("hi there");
    const mw = makeMiddleware(fakeOpenAI(response));

    const result = await mw.chat.completions.create({
      model: "gpt-4o", messages: [USER("hello")],
    } as never);

    expect(result).toBe(response);
  });

  it("buffers turns without flushing below the threshold", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 5 });

    await mw.chat.completions.create({ model: "m", messages: [USER("one")] } as never);
    await drain();

    expect(callsTo(fetchMock, "/ingest/session")).toHaveLength(0);
  });
});

// ── Anthropic path ────────────────────────────────────────────────────────────

describe("anthropic create", () => {
  it("forwards the call to the real client", async () => {
    stubFetch();
    const llm = fakeAnthropic();
    const mw = makeMiddleware(llm);

    await mw.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 1024,
      messages: [USER("hello")],
    } as never);

    expect(llm.messages.create).toHaveBeenCalledOnce();
    const params = llm.messages.create.mock.calls[0]![0];
    expect(params.max_tokens).toBe(1024);
    expect(params.messages).toEqual([USER("hello")]);
  });

  it("buffers turns", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeAnthropic(), { extractEveryNTurns: 1 });

    await mw.messages.create({ model: "m", messages: [USER("remember this")] } as never);
    await drain();

    const body = bodyOf(callsTo(fetchMock, "/ingest/session")[0]!);
    expect(body.turns).toEqual([{ role: "user", content: "remember this" }]);
  });

  it("flattens block-array content to text when buffering", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeAnthropic(), { extractEveryNTurns: 1 });

    await mw.messages.create({
      model: "m",
      messages: [{ role: "user", content: [{ type: "text", text: "block text" }] }],
    } as never);
    await drain();

    const body = bodyOf(callsTo(fetchMock, "/ingest/session")[0]!);
    expect(body.turns).toEqual([{ role: "user", content: "block text" }]);
  });
});

// ── partial flush windowing ───────────────────────────────────────────────────

describe("partial flush", () => {
  it("fires after N user turns", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 2 });

    await mw.chat.completions.create({ model: "m", messages: [USER("one")] } as never);
    await mw.chat.completions.create({ model: "m", messages: [USER("two")] } as never);
    await drain();

    const ingests = callsTo(fetchMock, "/ingest/session");
    expect(ingests).toHaveLength(1);
    expect(bodyOf(ingests[0]!).partial).toBe(true);
  });

  it("is disabled when extractEveryNTurns is 0", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 0 });

    for (const t of ["a", "b", "c", "d"]) {
      await mw.chat.completions.create({ model: "m", messages: [USER(t)] } as never);
    }
    await drain();

    expect(callsTo(fetchMock, "/ingest/session")).toHaveLength(0);
  });

  it("sends only new turns on the second flush", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 1 });

    await mw.chat.completions.create({ model: "m", messages: [USER("first")] } as never);
    await mw.chat.completions.create({ model: "m", messages: [USER("second")] } as never);
    await drain();

    const ingests = callsTo(fetchMock, "/ingest/session");
    expect(bodyOf(ingests[0]!).turns).toEqual([{ role: "user", content: "first" }]);
    expect(bodyOf(ingests[1]!).turns).toEqual([{ role: "user", content: "second" }]);
  });

  it("does not re-buffer the prefix a cumulative-history caller re-sends", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 0 });

    await mw.chat.completions.create({ model: "m", messages: [USER("one")] } as never);
    // Standard OpenAI pattern: resend everything so far plus the new turn.
    await mw.chat.completions.create({
      model: "m",
      messages: [USER("one"), { role: "assistant", content: "reply" }, USER("two")],
    } as never);
    await mw.close();

    const body = bodyOf(callsTo(fetchMock, "/ingest/session")[0]!);
    expect(body.turns).toEqual([
      { role: "user", content: "one" },
      { role: "assistant", content: "reply" },
      { role: "user", content: "two" },
    ]);
  });
});

// ── close ─────────────────────────────────────────────────────────────────────

describe("close", () => {
  it("sends a final non-partial ingest", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 0 });

    await mw.chat.completions.create({ model: "m", messages: [USER("hello")] } as never);
    await mw.close();

    const ingests = callsTo(fetchMock, "/ingest/session");
    expect(ingests).toHaveLength(1);
    const body = bodyOf(ingests[0]!);
    expect(body.partial).toBe(false);
    expect(body.turns).toEqual([{ role: "user", content: "hello" }]);
  });

  it("is idempotent", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 0 });

    await mw.chat.completions.create({ model: "m", messages: [USER("hello")] } as never);
    await mw.close();
    await mw.close();

    expect(callsTo(fetchMock, "/ingest/session")).toHaveLength(1);
  });

  it("does nothing when the buffer is empty", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI());

    await mw.close();

    expect(callsTo(fetchMock, "/ingest/session")).toHaveLength(0);
  });

  it("sends only the turns a partial flush did not cover", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 1 });

    await mw.chat.completions.create({ model: "m", messages: [USER("flushed")] } as never);
    await drain();
    // Buffer a turn that the threshold has not yet swept up.
    (mw as unknown as { buffer: unknown[] }).buffer.push({
      role: "assistant", content: "trailing",
    });
    await mw.close();

    const ingests = callsTo(fetchMock, "/ingest/session");
    expect(bodyOf(ingests.at(-1)!).turns).toEqual([
      { role: "assistant", content: "trailing" },
    ]);
  });

  it("blocks further buffering once closed", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 1 });

    await mw.close();
    await mw.chat.completions.create({ model: "m", messages: [USER("late")] } as never);
    await drain();

    expect(callsTo(fetchMock, "/ingest/session")).toHaveLength(0);
  });
});

// ── auto-inject ───────────────────────────────────────────────────────────────

describe("autoInject", () => {
  it("prepends a sentinel block to an existing system message", async () => {
    stubFetch("User likes neovim.");
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm, { autoInject: true });

    await mw.chat.completions.create({
      model: "m",
      messages: [{ role: "system", content: "Be brief." }, USER("hi")],
    } as never);

    const sent = llm.chat.completions.create.mock.calls[0]![0].messages;
    expect(sent[0].role).toBe("system");
    expect(sent[0].content).toContain("<!-- smritikosh:context-start -->");
    expect(sent[0].content).toContain("User likes neovim.");
    expect(sent[0].content).toContain("Be brief.");
  });

  it("creates a system message when none exists", async () => {
    stubFetch("User likes neovim.");
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm, { autoInject: true });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);

    const sent = llm.chat.completions.create.mock.calls[0]![0].messages;
    expect(sent).toHaveLength(2);
    expect(sent[0].role).toBe("system");
    expect(sent[1]).toEqual(USER("hi"));
  });

  it("is a no-op when context comes back empty", async () => {
    stubFetch("");
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm, { autoInject: true });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);

    expect(llm.chat.completions.create.mock.calls[0]![0].messages).toEqual([USER("hi")]);
  });

  it("sets the system param for Anthropic instead of a message", async () => {
    stubFetch("User likes neovim.");
    const llm = fakeAnthropic();
    const mw = makeMiddleware(llm, { autoInject: true });

    await mw.messages.create({
      model: "m", max_tokens: 100, system: "Be brief.", messages: [USER("hi")],
    } as never);

    const params = llm.messages.create.mock.calls[0]![0];
    expect(params.system).toContain("<!-- smritikosh:context-start -->");
    expect(params.system).toContain("User likes neovim.");
    expect(params.system).toContain("Be brief.");
    expect(params.messages).toEqual([USER("hi")]);
  });

  it("buffers the pre-injection turns, not the sentinel", async () => {
    const fetchMock = stubFetch("User likes neovim.");
    const mw = makeMiddleware(fakeOpenAI(), { autoInject: true, extractEveryNTurns: 1 });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);
    await drain();

    const body = bodyOf(callsTo(fetchMock, "/ingest/session")[0]!);
    expect(body.turns).toEqual([{ role: "user", content: "hi" }]);
  });
});

// ── ingest payload & failure isolation ────────────────────────────────────────

describe("ingest payload", () => {
  it("forwards the trigger-filter flag", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), {
      extractEveryNTurns: 1, useTriggerFilter: false,
    });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);
    await drain();

    expect(bodyOf(callsTo(fetchMock, "/ingest/session")[0]!).use_trigger_filter).toBe(false);
  });

  it("defaults the trigger filter to true", async () => {
    const fetchMock = stubFetch();
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 1 });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);
    await drain();

    const body = bodyOf(callsTo(fetchMock, "/ingest/session")[0]!);
    expect(body.use_trigger_filter).toBe(true);
    expect(body.session_id).toBe(mw.sessionId);
    expect(body.user_id).toBe("alice");
    expect(body.app_id).toBe("testapp");
  });

  it("does not propagate a flush failure to the caller", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    const mw = makeMiddleware(fakeOpenAI(), { extractEveryNTurns: 1 });

    await expect(
      mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never),
    ).resolves.toBeDefined();
    await expect(mw.close()).resolves.toBeUndefined();
  });

  it("does not propagate a context-fetch failure to the caller", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm, { autoInject: true, extractEveryNTurns: 0 });

    await expect(
      mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never),
    ).resolves.toBeDefined();
    // Context unavailable → messages forwarded unchanged.
    expect(llm.chat.completions.create.mock.calls[0]![0].messages).toEqual([USER("hi")]);
  });
});

// ── remember() tool — OpenAI ──────────────────────────────────────────────────

describe("remember tool (openai)", () => {
  it("injects the tool definition", async () => {
    stubFetch();
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);

    const tools = llm.chat.completions.create.mock.calls[0]![0].tools;
    expect(tools).toHaveLength(1);
    expect(tools[0].function.name).toBe("remember");
  });

  it("does not duplicate an already-present remember tool", async () => {
    stubFetch();
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm, { enableRememberTool: true });
    const existing = { type: "function", function: { name: "remember" } };

    await mw.chat.completions.create({
      model: "m", messages: [USER("hi")], tools: [existing],
    } as never);

    expect(llm.chat.completions.create.mock.calls[0]![0].tools).toHaveLength(1);
  });

  it("preserves the caller's other tools", async () => {
    stubFetch();
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm, { enableRememberTool: true });
    const weather = { type: "function", function: { name: "get_weather" } };

    await mw.chat.completions.create({
      model: "m", messages: [USER("hi")], tools: [weather],
    } as never);

    const tools = llm.chat.completions.create.mock.calls[0]![0].tools;
    expect(tools).toHaveLength(2);
    expect(tools[0]).toBe(weather);
  });

  it("is not injected when disabled", async () => {
    stubFetch();
    const llm = fakeOpenAI();
    const mw = makeMiddleware(llm, { enableRememberTool: false });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);

    expect(llm.chat.completions.create.mock.calls[0]![0].tools).toBeUndefined();
  });

  it("stores the fact via POST /memory/fact", async () => {
    const fetchMock = stubFetch();
    const args = JSON.stringify({
      content: "User prefers neovim", category: "preference",
      key: "editor", value: "neovim",
    });
    const llm = fakeOpenAI();
    llm.chat.completions.create
      .mockResolvedValueOnce(openaiToolResponse("call-1", args))
      .mockResolvedValueOnce(openaiTextResponse("Saved!"));
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.chat.completions.create({ model: "m", messages: [USER("I use neovim")] } as never);

    const body = bodyOf(callsTo(fetchMock, "/memory/fact")[0]!);
    expect(body.category).toBe("preference");
    expect(body.key).toBe("editor");
    expect(body.value).toBe("neovim");
    expect(body.source_type).toBe("tool_use");
    expect(body.user_id).toBe("alice");
  });

  it("falls back to content when key/value are absent", async () => {
    const fetchMock = stubFetch();
    const args = JSON.stringify({ content: "User is vegetarian", category: "preference" });
    const llm = fakeOpenAI();
    llm.chat.completions.create
      .mockResolvedValueOnce(openaiToolResponse("call-1", args))
      .mockResolvedValueOnce(openaiTextResponse());
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);

    const body = bodyOf(callsTo(fetchMock, "/memory/fact")[0]!);
    expect(body.key).toBe("User is vegetarian");
    expect(body.value).toBe("User is vegetarian");
  });

  it("makes a transparent follow-up call and returns its response", async () => {
    stubFetch();
    const args = JSON.stringify({ content: "x", category: "preference" });
    const followUp = openaiTextResponse("All set!");
    const llm = fakeOpenAI();
    llm.chat.completions.create
      .mockResolvedValueOnce(openaiToolResponse("call-1", args))
      .mockResolvedValueOnce(followUp);
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    const result = await mw.chat.completions.create({
      model: "m", messages: [USER("hi")],
    } as never);

    expect(result).toBe(followUp);
    expect(llm.chat.completions.create).toHaveBeenCalledTimes(2);
  });

  it("includes the assistant tool_calls and tool result in the follow-up", async () => {
    stubFetch();
    const args = JSON.stringify({ content: "x", category: "preference" });
    const llm = fakeOpenAI();
    llm.chat.completions.create
      .mockResolvedValueOnce(openaiToolResponse("call-1", args))
      .mockResolvedValueOnce(openaiTextResponse());
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);

    const followUpMessages = llm.chat.completions.create.mock.calls[1]![0].messages;
    expect(followUpMessages[0]).toEqual(USER("hi"));
    expect(followUpMessages[1].role).toBe("assistant");
    expect(followUpMessages[1].tool_calls[0].id).toBe("call-1");
    expect(followUpMessages[2]).toEqual({
      role: "tool", tool_call_id: "call-1", content: "Memory saved.",
    });
  });

  it("returns the original response when tool calls are mixed", async () => {
    const fetchMock = stubFetch();
    const mixed = {
      choices: [
        {
          message: {
            content: null,
            tool_calls: [
              {
                id: "call-1", type: "function",
                function: {
                  name: "remember",
                  arguments: JSON.stringify({ content: "x", category: "preference" }),
                },
              },
              {
                id: "call-2", type: "function",
                function: { name: "get_weather", arguments: "{}" },
              },
            ],
          },
        },
      ],
    };
    const llm = fakeOpenAI(mixed);
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    const result = await mw.chat.completions.create({
      model: "m", messages: [USER("hi")],
    } as never);

    // Fact saved, but no follow-up: the app still needs to handle get_weather.
    expect(callsTo(fetchMock, "/memory/fact")).toHaveLength(1);
    expect(result).toBe(mixed);
    expect(llm.chat.completions.create).toHaveBeenCalledOnce();
  });

  it("does not re-handle remember() in the follow-up call", async () => {
    stubFetch();
    const args = JSON.stringify({ content: "x", category: "preference" });
    const llm = fakeOpenAI();
    // Both responses contain a remember() call; only the first must be handled.
    llm.chat.completions.create.mockResolvedValue(openaiToolResponse("call-1", args));
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);

    expect(llm.chat.completions.create).toHaveBeenCalledTimes(2);
  });

  it("tolerates malformed tool arguments", async () => {
    const fetchMock = stubFetch();
    const llm = fakeOpenAI();
    llm.chat.completions.create
      .mockResolvedValueOnce(openaiToolResponse("call-1", "not-json{{"))
      .mockResolvedValueOnce(openaiTextResponse());
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.chat.completions.create({ model: "m", messages: [USER("hi")] } as never);

    const body = bodyOf(callsTo(fetchMock, "/memory/fact")[0]!);
    expect(body.category).toBe("preference");
    expect(body.value).toBe("not-json{{");
  });
});

// ── remember() tool — Anthropic ───────────────────────────────────────────────

describe("remember tool (anthropic)", () => {
  it("injects the tool definition", async () => {
    stubFetch();
    const llm = fakeAnthropic();
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.messages.create({ model: "m", max_tokens: 10, messages: [USER("hi")] } as never);

    const tools = llm.messages.create.mock.calls[0]![0].tools;
    expect(tools).toHaveLength(1);
    expect(tools[0].name).toBe("remember");
    expect(tools[0].input_schema).toBeDefined();
  });

  it("does not duplicate an already-present remember tool", async () => {
    stubFetch();
    const llm = fakeAnthropic();
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.messages.create({
      model: "m", max_tokens: 10, messages: [USER("hi")],
      tools: [{ name: "remember" }],
    } as never);

    expect(llm.messages.create.mock.calls[0]![0].tools).toHaveLength(1);
  });

  it("stores the fact via POST /memory/fact", async () => {
    const fetchMock = stubFetch();
    const llm = fakeAnthropic();
    llm.messages.create
      .mockResolvedValueOnce(
        anthropicToolResponse("tu-1", {
          content: "User prefers dark mode", category: "preference",
          key: "theme", value: "dark",
        }),
      )
      .mockResolvedValueOnce(anthropicTextResponse("Saved!"));
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    await mw.messages.create({ model: "m", max_tokens: 10, messages: [USER("hi")] } as never);

    const body = bodyOf(callsTo(fetchMock, "/memory/fact")[0]!);
    expect(body.key).toBe("theme");
    expect(body.value).toBe("dark");
    expect(body.source_type).toBe("tool_use");
  });

  it("makes a transparent follow-up with a tool_result block", async () => {
    stubFetch();
    const followUp = anthropicTextResponse("All set!");
    const llm = fakeAnthropic();
    llm.messages.create
      .mockResolvedValueOnce(
        anthropicToolResponse("tu-1", { content: "x", category: "preference" }),
      )
      .mockResolvedValueOnce(followUp);
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    const result = await mw.messages.create({
      model: "m", max_tokens: 10, messages: [USER("hi")],
    } as never);

    expect(result).toBe(followUp);
    const followUpMessages = llm.messages.create.mock.calls[1]![0].messages;
    expect(followUpMessages[1].role).toBe("assistant");
    expect(followUpMessages[1].content[0]).toEqual({
      type: "tool_use", id: "tu-1", name: "remember",
      input: { content: "x", category: "preference" },
    });
    expect(followUpMessages[2]).toEqual({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "tu-1", content: "Memory saved." }],
    });
  });

  it("returns the original response when tool_use blocks are mixed", async () => {
    const fetchMock = stubFetch();
    const mixed = {
      content: [
        { type: "tool_use", id: "tu-1", name: "remember", input: { content: "x", category: "preference" } },
        { type: "tool_use", id: "tu-2", name: "get_weather", input: {} },
      ],
    };
    const llm = fakeAnthropic(mixed);
    const mw = makeMiddleware(llm, { enableRememberTool: true });

    const result = await mw.messages.create({
      model: "m", max_tokens: 10, messages: [USER("hi")],
    } as never);

    expect(callsTo(fetchMock, "/memory/fact")).toHaveLength(1);
    expect(result).toBe(mixed);
    expect(llm.messages.create).toHaveBeenCalledOnce();
  });
});
