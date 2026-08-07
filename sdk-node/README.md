# Smritikosh — Node.js / TypeScript SDK

Client and transparent memory middleware for [Smritikosh](https://github.com/kunwarmahen/smritikosh),
the universal AI memory layer.

Targets Node.js ≥ 18 and uses the built-in `fetch` — **no runtime dependencies**.

```bash
npm install smritikosh
```

---

## Two ways to use it

| | What it does | Use when |
|---|---|---|
| `SmritikoshClient` | Explicit API client — `encode`, `buildContext`, `search`, procedures, ingest, admin | You want to control exactly what is stored and retrieved |
| `withMemory(...)` | Wraps your OpenAI / Anthropic client so memory happens invisibly | You want memory with a one-line change |

---

## Transparent middleware

Wrap the LLM client you already have. Every call is intercepted: user turns are
buffered and ingested in the background, a `remember()` tool is offered to the
model, and (optionally) relevant memory is injected into the prompt.

```typescript
import OpenAI from "openai";
import { withMemory } from "smritikosh";

const client = withMemory(new OpenAI(), {
  smritikoshUrl: "http://localhost:8080",
  smritikoshApiKey: "sk-smriti-...",
  userId: "alice",
  appId: "my-app",
});

// Use it exactly like the OpenAI client — memory is automatic.
const res = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "I always deploy on Kubernetes." }],
});

await client.close();   // flushes any unsent turns
```

Anthropic works the same way:

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { withMemory } from "smritikosh";

const client = withMemory(new Anthropic(), {
  smritikoshApiKey: "sk-smriti-...",
  userId: "alice",
});

const res = await client.messages.create({
  model: "claude-haiku-4-5-20251001",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Remind me what I use for deploys." }],
});
```

Anything the middleware does not intercept passes straight through to the wrapped
client, so it is a drop-in replacement.

### Automatic context injection

With `autoInject: true`, the middleware fetches relevant memory before each call
and prepends it to the system message, wrapped in sentinel comments so the
extraction pass can tell injected context from real conversation:

```typescript
const client = withMemory(new OpenAI(), {
  smritikoshApiKey: "sk-smriti-...",
  userId: "alice",
  autoInject: true,
});
```

```
<!-- smritikosh:context-start -->
## User Memory Context
### Who this user is:
Preference: deploy_target=kubernetes
Tool: editor=neovim
<!-- smritikosh:context-end -->

<your original system prompt>
```

### The `remember()` tool

Enabled by default. The model is offered a `remember()` tool and decides for
itself when something is worth keeping — the highest-precision memory source,
because the judgement is made in context.

When the model calls it, the middleware stores the fact
(`POST /memory/fact`, `source_type="tool_use"`) and transparently continues the
conversation, so **your application never sees the tool call**. If the model mixes
`remember()` with your own tools, the fact is saved and the response is handed
back unchanged for you to handle the rest.

Disable with `enableRememberTool: false`.

### Options

| Option | Default | Description |
|---|---|---|
| `smritikoshApiKey` | — | **Required.** Bearer token for the Smritikosh API |
| `userId` | — | **Required.** User whose memories are extracted |
| `smritikoshUrl` | `http://localhost:8080` | Base URL of the Smritikosh server |
| `appId` | `"default"` | Application namespace |
| `sessionId` | random UUID | Idempotency key — reuse it to resume a session |
| `extractEveryNTurns` | `10` | Partial ingest cadence; `0` = flush on `close()` only |
| `useTriggerFilter` | `true` | Skip LLM extraction on windows with no trigger phrases |
| `autoInject` | `false` | Prepend retrieved memory to the system prompt |
| `enableRememberTool` | `true` | Offer the `remember()` tool and handle its calls |
| `timeoutMs` | `60000` | Timeout for Smritikosh HTTP calls |
| `contextTimeoutMs` | `10000` | Timeout for the `autoInject` fetch — raise it for slow local models |

### Lifecycle

Call `close()` when a conversation ends: it waits for in-flight partial flushes,
then sends the remaining turns as a final ingest. Where `await using` is
supported, the middleware disposes itself:

```typescript
await using client = withMemory(new OpenAI(), { /* ... */ });
```

Memory extraction is **best-effort by design** — if Smritikosh is unreachable,
your LLM calls still succeed and nothing is thrown.

---

## Explicit client

```typescript
import { SmritikoshClient } from "smritikosh";

const client = new SmritikoshClient({
  baseUrl: "http://localhost:8080",
  appId: "myapp",
});

const event = await client.encode({
  userId: "alice",
  content: "I prefer TypeScript over plain JavaScript for large projects.",
});

const ctx = await client.buildContext({
  userId: "alice",
  query: "What language does Alice prefer?",
});
if (!ctx.isEmpty()) {
  // ctx.messages is OpenAI-style — prepend to your messages array
  console.log(ctx.contextText);
}
```

Error handling:

```typescript
import { SmritikoshError } from "smritikosh";

try {
  await client.encode({ userId: "alice", content: "..." });
} catch (err) {
  if (err instanceof SmritikoshError) {
    console.error(`API error ${err.status}: ${err.message}`);
  }
}
```

See the [main README](../README.md#nodejs-sdk) for the complete method reference.

---

## Development

```bash
npm install
npm test           # vitest
npm run lint       # tsc --noEmit
npm run build      # dist/esm + dist/cjs + dist/types
```

## License

MIT
