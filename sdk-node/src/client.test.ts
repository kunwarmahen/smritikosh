/**
 * SmritikoshClient unit tests.
 *
 * Strategy: stub the global `fetch` function so no real HTTP requests are made.
 * Each test verifies:
 *   1. The correct HTTP method + URL is called.
 *   2. The request body / query params are correct.
 *   3. The response is correctly mapped from snake_case API fields to
 *      the camelCase TypeScript interface.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { SmritikoshClient } from "./client.js";

// ── fetch mock helpers ─────────────────────────────────────────────────────────

function mockFetch(body: unknown, status = 200) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
  } as unknown as Response;
  return vi.fn().mockResolvedValue(response);
}

function capturedUrl(fetchMock: ReturnType<typeof vi.fn>): string {
  return fetchMock.mock.calls[0]![0] as string;
}

function capturedBody(fetchMock: ReturnType<typeof vi.fn>): unknown {
  const init = fetchMock.mock.calls[0]![1] as RequestInit;
  return init.body ? JSON.parse(init.body as string) : undefined;
}

function capturedMethod(fetchMock: ReturnType<typeof vi.fn>): string {
  const init = fetchMock.mock.calls[0]![1] as RequestInit;
  return (init.method ?? "GET").toUpperCase();
}

// ── shared client fixture ─────────────────────────────────────────────────────

const BASE = "http://localhost:8080";
const APP_ID = "testapp";

function makeClient() {
  return new SmritikoshClient({ baseUrl: BASE, appId: APP_ID, timeoutMs: 5000 });
}

// ── encode ─────────────────────────────────────────────────────────────────────

describe("encode", () => {
  it("posts to /memory/encode", async () => {
    const fetch = mockFetch({
      event_id: "evt-1", user_id: "u1",
      importance_score: 0.8, facts_extracted: 2, extraction_failed: false,
    });
    vi.stubGlobal("fetch", fetch);

    const client = makeClient();
    await client.encode({ userId: "u1", content: "Hello" });

    expect(capturedMethod(fetch)).toBe("POST");
    expect(capturedUrl(fetch)).toBe(`${BASE}/memory/encode`);
  });

  it("maps snake_case response to camelCase", async () => {
    const fetch = mockFetch({
      event_id: "evt-123", user_id: "u1",
      importance_score: 0.72, facts_extracted: 3, extraction_failed: false,
    });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().encode({ userId: "u1", content: "test" });

    expect(result.eventId).toBe("evt-123");
    expect(result.importanceScore).toBe(0.72);
    expect(result.factsExtracted).toBe(3);
    expect(result.extractionFailed).toBe(false);
  });

  it("sends user_id and content in body", async () => {
    const fetch = mockFetch({
      event_id: "e", user_id: "alice",
      importance_score: 0.5, facts_extracted: 0, extraction_failed: false,
    });
    vi.stubGlobal("fetch", fetch);

    await makeClient().encode({ userId: "alice", content: "My content", appId: "myapp" });

    const body = capturedBody(fetch) as Record<string, unknown>;
    expect(body["user_id"]).toBe("alice");
    expect(body["content"]).toBe("My content");
    expect(body["app_id"]).toBe("myapp");
  });

  it("falls back to client-level appId", async () => {
    const fetch = mockFetch({
      event_id: "e", user_id: "u1",
      importance_score: 0.5, facts_extracted: 0, extraction_failed: false,
    });
    vi.stubGlobal("fetch", fetch);

    await makeClient().encode({ userId: "u1", content: "hi" });

    const body = capturedBody(fetch) as Record<string, unknown>;
    expect(body["app_id"]).toBe(APP_ID);
  });
});

// ── buildContext ───────────────────────────────────────────────────────────────

describe("buildContext", () => {
  const apiResponse = {
    user_id: "u1",
    query: "what do I prefer?",
    context_text: "## User Memory Context\n...",
    messages: [{ role: "system", content: "..." }],
    total_memories: 3,
    embedding_failed: false,
    intent: "preference_query",
    reconsolidation_scheduled: true,
  };

  it("posts to /context/build", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    await makeClient().buildContext({ userId: "u1", query: "test" });

    expect(capturedUrl(fetch)).toBe(`${BASE}/context/build`);
    expect(capturedMethod(fetch)).toBe("POST");
  });

  it("maps all fields correctly", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().buildContext({ userId: "u1", query: "test" });

    expect(result.userId).toBe("u1");
    expect(result.contextText).toBe(apiResponse.context_text);
    expect(result.totalMemories).toBe(3);
    expect(result.embeddingFailed).toBe(false);
    expect(result.intent).toBe("preference_query");
    expect(result.reconsolidationScheduled).toBe(true);
    expect(result.messages).toHaveLength(1);
  });

  it("isEmpty() returns false when totalMemories > 0", async () => {
    const fetch = mockFetch({ ...apiResponse, total_memories: 5 });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().buildContext({ userId: "u1", query: "test" });
    expect(result.isEmpty()).toBe(false);
  });

  it("isEmpty() returns true when totalMemories is 0", async () => {
    const fetch = mockFetch({ ...apiResponse, total_memories: 0 });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().buildContext({ userId: "u1", query: "test" });
    expect(result.isEmpty()).toBe(true);
  });

  it("sends fromDate as ISO string", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    const dt = new Date("2024-01-01T00:00:00Z");
    await makeClient().buildContext({ userId: "u1", query: "test", fromDate: dt });

    const body = capturedBody(fetch) as Record<string, unknown>;
    expect(body["from_date"]).toBe("2024-01-01T00:00:00.000Z");
  });

  it("sends string fromDate unchanged", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    await makeClient().buildContext({
      userId: "u1", query: "test", fromDate: "2024-01-01T00:00:00Z",
    });

    const body = capturedBody(fetch) as Record<string, unknown>;
    expect(body["from_date"]).toBe("2024-01-01T00:00:00Z");
  });
});

// ── getRecent ──────────────────────────────────────────────────────────────────

describe("getRecent", () => {
  const apiResponse = {
    events: [
      {
        event_id: "e1", raw_text: "I prefer dark mode",
        importance_score: 0.7, consolidated: false,
        created_at: "2024-01-01T00:00:00+00:00",
      },
    ],
  };

  it("gets /memory/recent with user_id query param", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    await makeClient().getRecent({ userId: "u1" });

    const url = capturedUrl(fetch);
    expect(url).toContain("/memory/recent");
    expect(url).toContain("user_id=u1");
    expect(capturedMethod(fetch)).toBe("GET");
  });

  it("maps response fields to camelCase", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().getRecent({ userId: "u1" });

    expect(result).toHaveLength(1);
    expect(result[0]!.eventId).toBe("e1");
    expect(result[0]!.rawText).toBe("I prefer dark mode");
    expect(result[0]!.importanceScore).toBe(0.7);
    expect(result[0]!.consolidated).toBe(false);
  });

  it("appends limit param when provided", async () => {
    const fetch = mockFetch({ events: [] });
    vi.stubGlobal("fetch", fetch);

    await makeClient().getRecent({ userId: "u1", limit: 5 });

    expect(capturedUrl(fetch)).toContain("limit=5");
  });
});

// ── submitFeedback ────────────────────────────────────────────────────────────

describe("submitFeedback", () => {
  it("posts to /memory/feedback", async () => {
    const fetch = mockFetch({
      feedback_id: "fb-1", event_id: "evt-1", new_importance_score: 0.9,
    });
    vi.stubGlobal("fetch", fetch);

    await makeClient().submitFeedback({
      eventId: "evt-1", userId: "u1", feedbackType: "positive",
    });

    expect(capturedUrl(fetch)).toBe(`${BASE}/memory/feedback`);
    expect(capturedMethod(fetch)).toBe("POST");
  });

  it("maps response to camelCase", async () => {
    const fetch = mockFetch({
      feedback_id: "fb-99", event_id: "evt-1", new_importance_score: 0.82,
    });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().submitFeedback({
      eventId: "evt-1", userId: "u1", feedbackType: "positive",
    });

    expect(result.feedbackId).toBe("fb-99");
    expect(result.newImportanceScore).toBe(0.82);
  });

  it("sends feedback_type in body", async () => {
    const fetch = mockFetch({
      feedback_id: "fb-1", event_id: "e1", new_importance_score: 0.6,
    });
    vi.stubGlobal("fetch", fetch);

    await makeClient().submitFeedback({
      eventId: "e1", userId: "u1", feedbackType: "negative",
    });

    const body = capturedBody(fetch) as Record<string, unknown>;
    expect(body["feedback_type"]).toBe("negative");
  });
});

// ── deleteEvent ───────────────────────────────────────────────────────────────

describe("deleteEvent", () => {
  it("sends DELETE to /memory/event/{id}", async () => {
    const fetch = mockFetch({ deleted: true, event_id: "evt-1" });
    vi.stubGlobal("fetch", fetch);

    await makeClient().deleteEvent({ eventId: "evt-1" });

    expect(capturedMethod(fetch)).toBe("DELETE");
    expect(capturedUrl(fetch)).toContain("/memory/event/evt-1");
  });

  it("returns deleted flag", async () => {
    const fetch = mockFetch({ deleted: false, event_id: "evt-1" });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().deleteEvent({ eventId: "evt-1" });
    expect(result.deleted).toBe(false);
    expect(result.eventId).toBe("evt-1");
  });
});

// ── deleteUserMemory ──────────────────────────────────────────────────────────

describe("deleteUserMemory", () => {
  it("sends DELETE to /memory/user/{id}", async () => {
    const fetch = mockFetch({ events_deleted: 5, user_id: "u1", app_id: "default" });
    vi.stubGlobal("fetch", fetch);

    await makeClient().deleteUserMemory({ userId: "u1" });

    expect(capturedMethod(fetch)).toBe("DELETE");
    expect(capturedUrl(fetch)).toContain("/memory/user/u1");
  });

  it("maps eventsDeleted", async () => {
    const fetch = mockFetch({ events_deleted: 12, user_id: "u1", app_id: "testapp" });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().deleteUserMemory({ userId: "u1" });
    expect(result.eventsDeleted).toBe(12);
    expect(result.appId).toBe("testapp");
  });
});

// ── storeProcedure ────────────────────────────────────────────────────────────

describe("storeProcedure", () => {
  const apiResponse = {
    procedure_id: "proc-1", user_id: "u1",
    trigger: "LLM deployment", instruction: "mention GPU optimization",
    category: "topic_response", priority: 5, is_active: true,
    hit_count: 0, confidence: 1.0, source: "manual",
    created_at: "2024-01-01T00:00:00+00:00",
  };

  it("posts to /procedures", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    await makeClient().storeProcedure({
      userId: "u1", trigger: "LLM deployment", instruction: "mention GPU optimization",
    });

    expect(capturedUrl(fetch)).toBe(`${BASE}/procedures`);
    expect(capturedMethod(fetch)).toBe("POST");
  });

  it("maps response to camelCase", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().storeProcedure({
      userId: "u1", trigger: "t", instruction: "i",
    });

    expect(result.procedureId).toBe("proc-1");
    expect(result.isActive).toBe(true);
    expect(result.hitCount).toBe(0);
    expect(result.createdAt).toBe("2024-01-01T00:00:00+00:00");
  });

  it("sends trigger and instruction in body", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    await makeClient().storeProcedure({
      userId: "u1", trigger: "startup", instruction: "respond with depth", priority: 8,
    });

    const body = capturedBody(fetch) as Record<string, unknown>;
    expect(body["trigger"]).toBe("startup");
    expect(body["priority"]).toBe(8);
  });
});

// ── listProcedures ────────────────────────────────────────────────────────────

describe("listProcedures", () => {
  it("gets /procedures/{userId}", async () => {
    const fetch = mockFetch({ procedures: [] });
    vi.stubGlobal("fetch", fetch);

    await makeClient().listProcedures({ userId: "u1" });

    expect(capturedUrl(fetch)).toContain("/procedures/u1");
    expect(capturedMethod(fetch)).toBe("GET");
  });

  it("maps procedure items to camelCase", async () => {
    const fetch = mockFetch({
      procedures: [{
        procedure_id: "p1", trigger: "t", instruction: "i",
        category: "communication", priority: 7, is_active: true, hit_count: 3,
      }],
    });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().listProcedures({ userId: "u1" });

    expect(result).toHaveLength(1);
    expect(result[0]!.procedureId).toBe("p1");
    expect(result[0]!.isActive).toBe(true);
    expect(result[0]!.hitCount).toBe(3);
  });
});

// ── deleteProcedure ───────────────────────────────────────────────────────────

describe("deleteProcedure", () => {
  it("sends DELETE to /procedures/{id}", async () => {
    const fetch = mockFetch({ deleted: true, procedure_id: "p-1" });
    vi.stubGlobal("fetch", fetch);

    await makeClient().deleteProcedure({ procedureId: "p-1" });

    expect(capturedMethod(fetch)).toBe("DELETE");
    expect(capturedUrl(fetch)).toContain("/procedures/p-1");
  });
});

// ── deleteUserProcedures ──────────────────────────────────────────────────────

describe("deleteUserProcedures", () => {
  it("sends DELETE to /procedures/user/{userId}", async () => {
    const fetch = mockFetch({ procedures_deleted: 4, user_id: "u1", app_id: "default" });
    vi.stubGlobal("fetch", fetch);

    await makeClient().deleteUserProcedures({ userId: "u1" });

    expect(capturedMethod(fetch)).toBe("DELETE");
    expect(capturedUrl(fetch)).toContain("/procedures/user/u1");
  });

  it("maps proceduresDeleted", async () => {
    const fetch = mockFetch({ procedures_deleted: 4, user_id: "u1", app_id: "testapp" });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().deleteUserProcedures({ userId: "u1" });
    expect(result.proceduresDeleted).toBe(4);
  });
});

// ── reconsolidate ─────────────────────────────────────────────────────────────

describe("reconsolidate", () => {
  const apiResponse = {
    event_id: "evt-1", user_id: "u1",
    updated: true, skipped: false, skip_reason: "",
    old_summary: "original", new_summary: "refined",
  };

  it("posts to /admin/reconsolidate", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    await makeClient().reconsolidate({ eventId: "evt-1", query: "test", userId: "u1" });

    expect(capturedUrl(fetch)).toBe(`${BASE}/admin/reconsolidate`);
  });

  it("maps all fields to camelCase", async () => {
    const fetch = mockFetch(apiResponse);
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().reconsolidate({
      eventId: "evt-1", query: "test", userId: "u1",
    });

    expect(result.eventId).toBe("evt-1");
    expect(result.updated).toBe(true);
    expect(result.skipReason).toBe("");
    expect(result.oldSummary).toBe("original");
    expect(result.newSummary).toBe("refined");
  });
});

// ── admin jobs ────────────────────────────────────────────────────────────────

describe("admin jobs", () => {
  const jobResponse = {
    job: "consolidation", users_processed: 2,
    results: [
      { user_id: "u1", app_id: "default", skipped: false, detail: "consolidated=3" },
    ],
  };

  it("adminConsolidate posts to /admin/consolidate", async () => {
    const fetch = mockFetch(jobResponse);
    vi.stubGlobal("fetch", fetch);

    await makeClient().adminConsolidate({ userId: "u1" });
    expect(capturedUrl(fetch)).toBe(`${BASE}/admin/consolidate`);
  });

  it("adminPrune posts to /admin/prune", async () => {
    const fetch = mockFetch({ ...jobResponse, job: "pruning" });
    vi.stubGlobal("fetch", fetch);

    await makeClient().adminPrune();
    expect(capturedUrl(fetch)).toBe(`${BASE}/admin/prune`);
  });

  it("adminCluster posts to /admin/cluster", async () => {
    const fetch = mockFetch({ ...jobResponse, job: "clustering" });
    vi.stubGlobal("fetch", fetch);

    await makeClient().adminCluster();
    expect(capturedUrl(fetch)).toBe(`${BASE}/admin/cluster`);
  });

  it("adminMineBeliefs posts to /admin/mine-beliefs", async () => {
    const fetch = mockFetch({ ...jobResponse, job: "belief_mining" });
    vi.stubGlobal("fetch", fetch);

    await makeClient().adminMineBeliefs();
    expect(capturedUrl(fetch)).toBe(`${BASE}/admin/mine-beliefs`);
  });

  it("maps result fields to camelCase", async () => {
    const fetch = mockFetch(jobResponse);
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().adminConsolidate();
    expect(result.usersProcessed).toBe(2);
    expect(result.results[0]!.userId).toBe("u1");
    expect(result.results[0]!.appId).toBe("default");
  });
});

// ── health ────────────────────────────────────────────────────────────────────

describe("health", () => {
  it("gets /health", async () => {
    const fetch = mockFetch({ status: "ok", version: "0.1.0" });
    vi.stubGlobal("fetch", fetch);

    await makeClient().health();

    expect(capturedUrl(fetch)).toBe(`${BASE}/health`);
    expect(capturedMethod(fetch)).toBe("GET");
  });

  it("maps status and version", async () => {
    const fetch = mockFetch({ status: "ok", version: "0.2.0" });
    vi.stubGlobal("fetch", fetch);

    const result = await makeClient().health();
    expect(result.status).toBe("ok");
    expect(result.version).toBe("0.2.0");
  });
});

// ── error handling ────────────────────────────────────────────────────────────

describe("error handling", () => {
  it("throws SmritikoshError on non-2xx response", async () => {
    const response = {
      ok: false, status: 422,
      text: vi.fn().mockResolvedValue('{"detail":"Validation error"}'),
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    const client = makeClient();
    await expect(
      client.encode({ userId: "u1", content: "test" })
    ).rejects.toThrow("422");
  });

  it("throws SmritikoshError with status code accessible", async () => {
    const response = {
      ok: false, status: 500,
      text: vi.fn().mockResolvedValue("Internal Server Error"),
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    const client = makeClient();
    try {
      await client.health();
      expect.fail("should have thrown");
    } catch (err: unknown) {
      expect((err as { status: number }).status).toBe(500);
    }
  });

  it("throws on fetch network error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network error")));

    await expect(makeClient().health()).rejects.toThrow("network error");
  });
});

// ── trailing slash normalisation ──────────────────────────────────────────────

describe("baseUrl normalisation", () => {
  it("strips trailing slash from baseUrl", async () => {
    const fetch = mockFetch({ status: "ok", version: "0.1.0" });
    vi.stubGlobal("fetch", fetch);

    const client = new SmritikoshClient({ baseUrl: "http://localhost:8080/" });
    await client.health();

    expect(capturedUrl(fetch)).toBe("http://localhost:8080/health");
  });
});
