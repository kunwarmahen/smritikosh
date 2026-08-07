/**
 * SmritikoshMiddleware — transparent memory extraction wrapper for LLM clients.
 *
 * Wraps an OpenAI or Anthropic Node client. Every `create()` call is
 * intercepted: user turns are buffered, the `remember()` tool is injected,
 * memory context is optionally prepended, and ingestion fires in the
 * background.
 *
 * Usage (OpenAI):
 *     import OpenAI from "openai";
 *     import { withMemory } from "smritikosh";
 *
 *     const client = withMemory(new OpenAI(), {
 *       smritikoshUrl: "http://localhost:8080",
 *       smritikoshApiKey: "sk-smriti-...",
 *       userId: "alice",
 *       appId: "my-app",
 *     });
 *     const res = await client.chat.completions.create({ model: "gpt-4o", messages });
 *     await client.close();   // flushes remaining turns
 *
 * Usage (Anthropic):
 *     import Anthropic from "@anthropic-ai/sdk";
 *
 *     const client = withMemory(new Anthropic(), {
 *       smritikoshApiKey: "sk-smriti-...",
 *       userId: "alice",
 *     });
 *     const res = await client.messages.create({
 *       model: "claude-haiku-4-5-20251001", max_tokens: 1024, messages,
 *     });
 *
 * Every property that is not intercepted proxies straight through to the
 * wrapped client, so the middleware is a drop-in replacement.
 *
 * Concurrency note: unlike the Python SDK (which needs a mutex because
 * background flushes run on threads), the buffer bookkeeping here is a single
 * synchronous block. Node's event loop cannot preempt it, so no lock is
 * needed — flushes are fired as un-awaited promises and tracked for close().
 */

// ── constants ─────────────────────────────────────────────────────────────────

const DEFAULT_URL = "http://localhost:8080";
const DEFAULT_EXTRACT_EVERY = 10;
const DEFAULT_TIMEOUT_MS = 60_000;
const CONTEXT_TIMEOUT_MS = 10_000;

const CONTEXT_START = "<!-- smritikosh:context-start -->";
const CONTEXT_END = "<!-- smritikosh:context-end -->";

/**
 * `Symbol.asyncDispose` only exists on Node >= 18.18. Falling back to a private
 * symbol keeps `await using` working where supported without defining a method
 * keyed on the string "undefined" on older runtimes.
 */
const ASYNC_DISPOSE: symbol =
  (Symbol as { asyncDispose?: symbol }).asyncDispose ?? Symbol("asyncDispose");

const REMEMBER_DESCRIPTION =
  "Store something important about the user that should be remembered in " +
  "future conversations. Call this when the user reveals a clear preference, " +
  "fact, goal, or decision.";

const REMEMBER_CATEGORIES = [
  "preference", "goal", "skill", "habit",
  "role", "project", "belief", "context",
] as const;

const REMEMBER_PROPERTIES = {
  content: {
    type: "string",
    description: "The fact to remember, in plain English",
  },
  category: {
    type: "string",
    enum: REMEMBER_CATEGORIES as unknown as string[],
  },
  key: {
    type: "string",
    description: "Short label for the fact (e.g. 'editor', 'timezone')",
  },
  value: {
    type: "string",
    description: "The value (e.g. 'neovim', 'UTC+5:30')",
  },
};

/** OpenAI-shaped `remember()` tool definition. */
export const REMEMBER_TOOL_OPENAI = {
  type: "function",
  function: {
    name: "remember",
    description: REMEMBER_DESCRIPTION,
    parameters: {
      type: "object",
      properties: REMEMBER_PROPERTIES,
      required: ["content", "category"],
    },
  },
} as const;

/** Anthropic-shaped `remember()` tool definition. */
export const REMEMBER_TOOL_ANTHROPIC = {
  name: "remember",
  description: REMEMBER_DESCRIPTION,
  input_schema: {
    type: "object",
    properties: REMEMBER_PROPERTIES,
    required: ["content", "category"],
  },
} as const;

// ── types ─────────────────────────────────────────────────────────────────────

export interface SmritikoshMiddlewareOptions {
  /** Base URL of the running Smritikosh server. */
  smritikoshUrl?: string;
  /** API key (Bearer token) for authentication. */
  smritikoshApiKey: string;
  /** User whose memories should be extracted. */
  userId: string;
  /** Application namespace. */
  appId?: string;
  /**
   * Idempotency key for this conversation. Auto-generated UUID if omitted —
   * pass the same value to resume a session.
   */
  sessionId?: string;
  /**
   * Fire a partial ingest after this many cumulative user turns.
   * 0 disables mid-session extraction (flush on close only).
   */
  extractEveryNTurns?: number;
  /** Skip LLM extraction when no trigger phrases appear in the window. */
  useTriggerFilter?: boolean;
  /**
   * Retrieve memory context before each LLM call and prepend it to the system
   * message, wrapped in sentinel blocks so extraction can strip it later.
   */
  autoInject?: boolean;
  /** Inject the `remember()` tool and handle its calls transparently. */
  enableRememberTool?: boolean;
  /** Timeout for Smritikosh HTTP calls. */
  timeoutMs?: number;
  /**
   * Timeout for the `autoInject` context fetch, which sits on the critical
   * path of every LLM call. Kept short so a slow memory server degrades to
   * "no context" rather than stalling the call — raise it for local models,
   * whose intent classification can exceed the default.
   */
  contextTimeoutMs?: number;
}

/** A buffered conversation turn, as sent to `POST /ingest/session`. */
export interface BufferedTurn {
  role: string;
  content: string;
}

/** Minimal shape of a chat message the middleware understands. */
interface ChatMessage {
  role?: string;
  content?: unknown;
  [key: string]: unknown;
}

interface CreateParams {
  messages: ChatMessage[];
  tools?: unknown[];
  system?: unknown;
  [key: string]: unknown;
}

type CreateFn = (params: Record<string, unknown>) => Promise<unknown>;

// ── helpers ───────────────────────────────────────────────────────────────────

function randomId(): string {
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c?.randomUUID) return c.randomUUID();
  // Fallback for runtimes without Web Crypto (kept dependency-free).
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (ch) => {
    const r = (Math.random() * 16) | 0;
    const v = ch === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Flatten message content to text.
 *
 * Anthropic (and OpenAI vision) messages carry content as an array of blocks;
 * the ingest API takes plain strings, so text blocks are joined and non-text
 * blocks dropped.
 */
function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((b) => {
        if (typeof b === "string") return b;
        const block = b as { type?: string; text?: string };
        return block?.type === "text" ? (block.text ?? "") : "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

/** Serialize Anthropic response content blocks back to message-content form. */
export function blocksToAnthropicContent(blocks: unknown[]): unknown[] {
  const result: unknown[] = [];
  for (const b of blocks ?? []) {
    const block = b as {
      type?: string; text?: string; id?: string; name?: string; input?: unknown;
    };
    if (block?.type === "text") {
      result.push({ type: "text", text: block.text ?? "" });
    } else if (block?.type === "tool_use") {
      result.push({
        type: "tool_use",
        id: block.id ?? "",
        name: block.name ?? "",
        input: block.input ?? {},
      });
    }
  }
  return result;
}

function wrapSentinel(contextText: string): string {
  return `${CONTEXT_START}\n${contextText}\n${CONTEXT_END}`;
}

/** Text of the most recent user message, or "" when there is none. */
function lastUserQuery(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") return contentToText(messages[i]?.content);
  }
  return "";
}

// ── middleware ────────────────────────────────────────────────────────────────

export class SmritikoshMiddleware {
  readonly userId: string;
  readonly appId: string;
  readonly sessionId: string;

  private readonly llm: unknown;
  private readonly url: string;
  private readonly apiHeaders: Record<string, string>;
  private readonly everyN: number;
  private readonly triggerFilter: boolean;
  private readonly autoInject: boolean;
  private readonly enableRememberTool: boolean;
  private readonly timeoutMs: number;
  private readonly contextTimeoutMs: number;

  private buffer: BufferedTurn[] = [];
  private userTurnCount = 0;
  /** userTurnCount snapshot at the last partial flush. */
  private lastIngestedAt = 0;
  /** Buffer index of the first turn not yet sent. */
  private lastFlushBufIdx = 0;
  private closed = false;
  private pendingFlushes = new Set<Promise<void>>();
  /** Prevents re-entrant remember() handling in transparent follow-up calls. */
  private rememberLoopDepth = 0;

  constructor(llmClient: unknown, options: SmritikoshMiddlewareOptions) {
    this.llm = llmClient;
    this.url = (options.smritikoshUrl ?? DEFAULT_URL).replace(/\/$/, "");
    this.apiHeaders = {
      Authorization: `Bearer ${options.smritikoshApiKey}`,
      "Content-Type": "application/json",
    };
    this.userId = options.userId;
    this.appId = options.appId ?? "default";
    this.sessionId = options.sessionId ?? randomId();
    this.everyN = options.extractEveryNTurns ?? DEFAULT_EXTRACT_EVERY;
    this.triggerFilter = options.useTriggerFilter ?? true;
    this.autoInject = options.autoInject ?? false;
    this.enableRememberTool = options.enableRememberTool ?? true;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.contextTimeoutMs = options.contextTimeoutMs ?? CONTEXT_TIMEOUT_MS;

    // Transparently forward anything we don't intercept to the wrapped client.
    return new Proxy(this, {
      get(target, prop, receiver) {
        if (prop in target) return Reflect.get(target, prop, receiver);
        const value = (target.llm as Record<string | symbol, unknown>)?.[prop];
        return typeof value === "function"
          ? value.bind(target.llm)
          : value;
      },
      has(target, prop) {
        return prop in target || prop in (target.llm as object);
      },
    });
  }

  // ── intercepted namespaces ──────────────────────────────────────────────────

  /** Intercepted proxy for OpenAI's `client.chat` namespace. */
  get chat(): { completions: { create: CreateFn } } {
    const realChat = (this.llm as { chat?: Record<string, unknown> })?.chat ?? {};
    const mw = this;
    return new Proxy(realChat, {
      get(target, prop, receiver) {
        if (prop === "completions") {
          const realCompletions =
            (target as { completions?: Record<string, unknown> })?.completions ?? {};
          return new Proxy(realCompletions, {
            get(cTarget, cProp, cReceiver) {
              if (cProp === "create") {
                return (params: CreateParams) =>
                  mw.openaiCreate(realCompletions as Record<string, unknown>, params);
              }
              const v = Reflect.get(cTarget, cProp, cReceiver);
              return typeof v === "function" ? v.bind(cTarget) : v;
            },
          });
        }
        const v = Reflect.get(target, prop, receiver);
        return typeof v === "function" ? v.bind(target) : v;
      },
    }) as { completions: { create: CreateFn } };
  }

  /** Intercepted proxy for Anthropic's `client.messages` namespace. */
  get messages(): { create: CreateFn } {
    const realMessages =
      (this.llm as { messages?: Record<string, unknown> })?.messages ?? {};
    const mw = this;
    return new Proxy(realMessages, {
      get(target, prop, receiver) {
        if (prop === "create") {
          return (params: CreateParams) =>
            mw.anthropicCreate(realMessages as Record<string, unknown>, params);
        }
        const v = Reflect.get(target, prop, receiver);
        return typeof v === "function" ? v.bind(target) : v;
      },
    }) as { create: CreateFn };
  }

  // ── OpenAI interception ─────────────────────────────────────────────────────

  private async openaiCreate(
    completions: Record<string, unknown>,
    params: CreateParams,
  ): Promise<unknown> {
    const messages = params.messages ?? [];
    const active = this.enableRememberTool && this.rememberLoopDepth === 0;

    let outgoingParams: Record<string, unknown> = { ...params };
    if (active) {
      const tools = [...((params.tools as Record<string, unknown>[]) ?? [])];
      const hasRemember = tools.some(
        (t) => (t?.["function"] as { name?: string })?.name === "remember",
      );
      if (!hasRemember) tools.push(REMEMBER_TOOL_OPENAI as unknown as Record<string, unknown>);
      outgoingParams.tools = tools;
    }

    const outgoing = this.autoInject
      ? await this.injectContext(messages)
      : messages;
    outgoingParams.messages = outgoing;

    const createFn = (completions["create"] as CreateFn).bind(completions);
    let response = await createFn(outgoingParams);

    if (active) {
      response = await this.handleOpenAIRemember(
        response, outgoing, createFn, outgoingParams,
      );
    }

    // Buffer the original (pre-injection) turns.
    this.recordAndMaybeFlush(messages);
    return response;
  }

  private async handleOpenAIRemember(
    response: unknown,
    messages: ChatMessage[],
    createFn: CreateFn,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    const choices = (response as { choices?: unknown[] })?.choices ?? [];
    if (!choices.length) return response;

    const message = (choices[0] as { message?: Record<string, unknown> })?.message;
    const toolCalls = (message?.["tool_calls"] as Record<string, unknown>[]) ?? [];
    const nameOf = (tc: Record<string, unknown>) =>
      (tc?.["function"] as { name?: string })?.name ?? "";

    const rememberCalls = toolCalls.filter((tc) => nameOf(tc) === "remember");
    if (!rememberCalls.length) return response;

    for (const tc of rememberCalls) {
      const fn = tc["function"] as { arguments?: string };
      let args: Record<string, unknown>;
      try {
        args = JSON.parse(fn?.arguments ?? "{}") as Record<string, unknown>;
      } catch {
        args = { content: String(fn?.arguments ?? ""), category: "preference" };
      }
      await this.storeFact(args);
    }

    const otherCalls = toolCalls.filter((tc) => nameOf(tc) !== "remember");
    // Mixed tool calls — facts saved; hand the response back for the app to
    // handle the others.
    if (otherCalls.length) return response;

    const assistantMsg: Record<string, unknown> = {
      role: "assistant",
      tool_calls: rememberCalls.map((tc) => ({
        id: tc["id"],
        type: "function",
        function: {
          name: nameOf(tc),
          arguments: (tc["function"] as { arguments?: string })?.arguments ?? "",
        },
      })),
    };
    const assistantContent = message?.["content"];
    if (assistantContent !== null && assistantContent !== undefined) {
      assistantMsg["content"] = assistantContent;
    }

    const toolResults = rememberCalls.map((tc) => ({
      role: "tool",
      tool_call_id: tc["id"],
      content: "Memory saved.",
    }));

    this.rememberLoopDepth++;
    try {
      return await createFn({
        ...params,
        messages: [...messages, assistantMsg, ...toolResults],
      });
    } finally {
      this.rememberLoopDepth--;
    }
  }

  // ── Anthropic interception ──────────────────────────────────────────────────

  private async anthropicCreate(
    messagesNs: Record<string, unknown>,
    params: CreateParams,
  ): Promise<unknown> {
    const messages = params.messages ?? [];
    const active = this.enableRememberTool && this.rememberLoopDepth === 0;

    const outgoingParams: Record<string, unknown> = { ...params };
    if (active) {
      const tools = [...((params.tools as Record<string, unknown>[]) ?? [])];
      if (!tools.some((t) => t?.["name"] === "remember")) {
        tools.push(REMEMBER_TOOL_ANTHROPIC as unknown as Record<string, unknown>);
      }
      outgoingParams.tools = tools;
    }

    // Anthropic takes `system` as a top-level param, not a message.
    if (this.autoInject) {
      const query = lastUserQuery(messages);
      const contextText = query ? await this.getContextText(query) : "";
      if (contextText) {
        const existingSystem = contentToText(params.system) || "";
        outgoingParams["system"] =
          `${wrapSentinel(contextText)}\n\n${existingSystem}`.trim();
      }
    }

    const createFn = (messagesNs["create"] as CreateFn).bind(messagesNs);
    let response = await createFn(outgoingParams);

    if (active) {
      response = await this.handleAnthropicRemember(
        response, messages, createFn, outgoingParams,
      );
    }

    this.recordAndMaybeFlush(messages);
    return response;
  }

  private async handleAnthropicRemember(
    response: unknown,
    messages: ChatMessage[],
    createFn: CreateFn,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    const blocks = ((response as { content?: unknown[] })?.content ?? []) as Record<
      string, unknown
    >[];
    const isToolUse = (b: Record<string, unknown>) => b?.["type"] === "tool_use";

    const rememberBlocks = blocks.filter(
      (b) => isToolUse(b) && b["name"] === "remember",
    );
    if (!rememberBlocks.length) return response;

    for (const b of rememberBlocks) {
      await this.storeFact((b["input"] as Record<string, unknown>) ?? {});
    }

    const otherToolBlocks = blocks.filter(
      (b) => isToolUse(b) && b["name"] !== "remember",
    );
    if (otherToolBlocks.length) return response;

    const toolResultsMsg = {
      role: "user",
      content: rememberBlocks.map((b) => ({
        type: "tool_result",
        tool_use_id: b["id"] ?? "",
        content: "Memory saved.",
      })),
    };

    this.rememberLoopDepth++;
    try {
      return await createFn({
        ...params,
        messages: [
          ...messages,
          { role: "assistant", content: blocksToAnthropicContent(blocks) },
          toolResultsMsg,
        ],
      });
    } finally {
      this.rememberLoopDepth--;
    }
  }

  // ── buffering ───────────────────────────────────────────────────────────────

  /**
   * Buffer incoming messages and fire a partial ingest once the per-N-turns
   * threshold is crossed. Runs to completion synchronously — the flush itself
   * is deliberately not awaited.
   */
  private recordAndMaybeFlush(messages: ChatMessage[]): void {
    if (this.closed) return;

    // Cumulative-history callers (the standard OpenAI pattern) re-send the whole
    // conversation plus one new turn. When the buffered prefix matches, keep
    // only the tail; otherwise treat every message as new.
    const bufLen = this.buffer.length;
    const isCumulative =
      bufLen > 0 &&
      messages.length > bufLen &&
      messages
        .slice(0, bufLen)
        .every((m, i) => contentToText(m?.content) === this.buffer[i]?.content);
    const newMessages = isCumulative ? messages.slice(bufLen) : messages;

    for (const msg of newMessages) {
      this.buffer.push({
        role: msg?.role ?? "user",
        content: contentToText(msg?.content),
      });
      if (msg?.role === "user") this.userTurnCount++;
    }

    if (this.everyN > 0 && this.userTurnCount - this.lastIngestedAt >= this.everyN) {
      this.lastIngestedAt = this.userTurnCount;
      // Send only the turns accumulated since the last flush. The server skips
      // already-processed turns via its stored last_turn_index, but sending the
      // slice avoids re-transmitting the whole history.
      const window = this.buffer.slice(this.lastFlushBufIdx);
      this.lastFlushBufIdx = this.buffer.length;

      const p = this.flush(true, window).finally(() => {
        this.pendingFlushes.delete(p);
      });
      this.pendingFlushes.add(p);
    }
  }

  // ── session lifecycle ───────────────────────────────────────────────────────

  /** Flush any remaining unsent turns as a final (non-partial) ingest. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;

    const turns = this.buffer.slice(this.lastFlushBufIdx);
    // Let in-flight partial flushes land first so the server sees them in order.
    await Promise.allSettled([...this.pendingFlushes]);
    if (turns.length) await this.flush(false, turns);
  }

  /** `await using` support — closes and flushes on scope exit. */
  async [ASYNC_DISPOSE](): Promise<void> {
    await this.close();
  }

  // ── Smritikosh HTTP ─────────────────────────────────────────────────────────

  private async post(
    path: string,
    body: unknown,
    timeoutMs = this.timeoutMs,
  ): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(`${this.url}${path}`, {
        method: "POST",
        headers: this.apiHeaders,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  /** POST /ingest/session — best-effort, never throws. */
  private async flush(partial: boolean, turns: BufferedTurn[]): Promise<void> {
    try {
      await this.post("/ingest/session", {
        user_id: this.userId,
        app_id: this.appId,
        session_id: this.sessionId,
        turns,
        partial,
        use_trigger_filter: this.triggerFilter,
        metadata: { source: "sdk_middleware" },
      });
    } catch {
      // Extraction is best-effort; never interrupt the LLM call.
    }
  }

  /** POST /memory/fact for a remember() tool call. Best-effort, never throws. */
  private async storeFact(toolInput: Record<string, unknown>): Promise<void> {
    try {
      const content = String(toolInput["content"] ?? "unknown");
      const key = (toolInput["key"] as string) || content.slice(0, 50);
      const value = (toolInput["value"] as string) || content;
      await this.post("/memory/fact", {
        user_id: this.userId,
        app_id: this.appId,
        category: toolInput["category"] ?? "preference",
        key,
        value,
        note: toolInput["content"],
        source_type: "tool_use",
      });
    } catch {
      // Fact storage is best-effort.
    }
  }

  /** Memory context_text for `query`, or "" on any failure. */
  private async getContextText(query: string): Promise<string> {
    try {
      const res = await this.post(
        "/context",
        { user_id: this.userId, app_id: this.appId, query },
        this.contextTimeoutMs,
      );
      const data = (await res.json()) as { context_text?: string };
      return data?.context_text ?? "";
    } catch {
      return "";
    }
  }

  /**
   * Prepend memory context to the system message, wrapped in sentinel blocks.
   * Returns the messages unchanged when there is no user turn or no context.
   */
  private async injectContext(messages: ChatMessage[]): Promise<ChatMessage[]> {
    const query = lastUserQuery(messages);
    if (!query) return messages;
    const contextText = await this.getContextText(query);
    if (!contextText) return messages;

    const sentinel = wrapSentinel(contextText);
    const patched = [...messages];
    if (patched[0]?.role === "system") {
      patched[0] = {
        ...patched[0],
        content: `${sentinel}\n\n${contentToText(patched[0].content)}`,
      };
    } else {
      patched.unshift({ role: "system", content: sentinel });
    }
    return patched;
  }
}

// ── ergonomic wrapper ─────────────────────────────────────────────────────────

/**
 * Wrap an LLM client so its calls are memory-extracting.
 *
 * The returned value is typed as the original client intersected with the
 * middleware, so both `client.chat.completions.create(...)` and
 * `client.close()` type-check, and any other property of the wrapped client
 * still passes straight through.
 */
export function withMemory<T extends object>(
  llmClient: T,
  options: SmritikoshMiddlewareOptions,
): T & SmritikoshMiddleware {
  return new SmritikoshMiddleware(llmClient, options) as T & SmritikoshMiddleware;
}
