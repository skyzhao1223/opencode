import { expect, test } from "bun:test"
import { LLMClient, LLMEvent, LanguageModel, type LLMRequest } from "@opencode-ai/ai"
import { OpenAIChat } from "@opencode-ai/ai/protocols"
import { Database } from "@opencode-ai/core/database/database"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { llmClient } from "@opencode-ai/core/effect/app-node-platform"
import { LayerNode } from "@opencode-ai/util/effect/layer-node"
import { Bus } from "@opencode-ai/core/bus"
import { EventTable } from "@opencode-ai/core/event/sql"
import { SessionCompaction } from "@opencode-ai/core/session/compaction"
import { SessionEvent } from "@opencode-ai/core/session/event"
import { SessionMessage } from "@opencode-ai/core/session/message"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { SessionRunnerModel } from "@opencode-ai/core/session/runner/model"
import { Token } from "@opencode-ai/core/util/token"
import { SessionTable } from "@opencode-ai/core/session/sql"
import { SessionStore } from "@opencode-ai/core/session/store"
import { Session } from "@opencode-ai/core/session"
import { Project } from "@opencode-ai/core/project"
import { ProjectTable } from "@opencode-ai/core/project/sql"
import { App } from "@opencode-ai/core/app"
import { Agent } from "@opencode-ai/core/agent"
import { Location } from "@opencode-ai/core/location"
import { AbsolutePath } from "@opencode-ai/core/schema"
import { Money } from "@opencode-ai/schema/money"
import { DateTime, Effect, Fiber, Layer, Schema, Stream } from "effect"
import { asc, eq } from "drizzle-orm"
import { testEffect } from "./lib/effect"

let requests: LLMRequest[] = []
const model = LanguageModel.make({
  id: "summary-model",
  provider: "test",
  route: OpenAIChat.route,
})
const cost = [
  {
    input: Money.USDPerMillionTokens.make(1),
    output: Money.USDPerMillionTokens.make(2),
    cache: {
      read: Money.USDPerMillionTokens.make(0.1),
      write: Money.USDPerMillionTokens.make(0.5),
    },
  },
]
const client = Layer.mock(LLMClient.Service)({
  stream: (request: LLMRequest) => {
    requests.push(request)
    return Stream.make(
      LLMEvent.textDelta({ id: "summary", text: "manual summary" }),
      LLMEvent.stepFinish({
        index: 0,
        reason: { normalized: "stop" },
        usage: {
          inputTokens: 15,
          outputTokens: 6,
          nonCachedInputTokens: 10,
          cacheReadInputTokens: 3,
          cacheWriteInputTokens: 2,
          reasoningTokens: 2,
        },
      }),
      LLMEvent.finish({
        reason: { normalized: "stop" },
      }),
    )
  },
  generate: () => Effect.die("unused"),
})
const resolved = SessionRunnerModel.resolved(model, {
  capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
  cost,
  limit: { context: 200_000, output: 32_000 },
})
const models = Layer.mock(SessionRunnerModel.Service)({
  resolve: () => Effect.succeed(resolved),
})
const agents = Layer.mock(Agent.Service)({
  get: (id) =>
    Effect.succeed(
      id === Agent.ID.make("compaction")
        ? { ...Agent.Info.default(Agent.ID.make("compaction")), system: "You are a summarization assistant." }
        : undefined,
    ),
  resolve: () => Effect.die("unused"),
  select: () => Effect.die("unused"),
  list: () => Effect.succeed([]),
  transform: () => Effect.die("unused"),
  reload: () => Effect.die("unused"),
})
const it = testEffect(
  AppNodeBuilder.build(
    LayerNode.group([Database.node, Bus.node, SessionProjector.node, SessionStore.node, SessionCompaction.node]),
    [
      [Bus.node, Bus.configured({ persist: true })],
      [llmClient, client],
      [SessionRunnerModel.node, models],
      [Agent.node, agents],
    ],
  ),
)

test("compaction prompt preserves detailed work state and relevant files", () => {
  const prompt = SessionCompaction.buildPrompt({ context: ["conversation history"] })

  expect(prompt).toContain("## Work State\n### Completed")
  expect(prompt).toContain("### Active")
  expect(prompt).toContain("### Blocked")
  expect(prompt).toContain("## Relevant Files")
})

test("compaction describes tool media without embedding base64", () => {
  const base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
  const serialized = SessionCompaction.serializeToolContent([
    { type: "text", text: "Image read successfully" },
    {
      type: "file",
      uri: `data:image/png;base64,${base64}`,
      mime: "image/png",
      name: "pixel.png",
    },
  ])

  expect(serialized).toBe("Image read successfully\n[Attached image/png: pixel.png]")
  expect(serialized).not.toContain(base64)
})

test("compaction prompt requires the checkpoint headings in order", () => {
  const prompt = SessionCompaction.buildPrompt({ context: ["Conversation history"] })
  expect(prompt.match(/^#{2,3} .+$/gm)).toEqual([
    "## Objective",
    "## Important Details",
    "## Work State",
    "### Completed",
    "### Active",
    "### Blocked",
    "## Next Move",
    "## Relevant Files",
  ])
  expect(prompt).toContain("one or two brief sentences")
  expect(prompt).toContain("constraints/preferences, decisions and why")
  expect(prompt).toContain("immediate concrete action")
  expect(prompt).toContain("next action if known")
  expect(prompt).toContain("Keep every section, even when empty.")
})

it.effect("auto compaction reserves a buffer below the prompt ceiling", () =>
  Effect.gen(function* () {
    const compaction = yield* SessionCompaction.Service
    const session = Session.Info.make({
      id: Session.ID.make("ses_input_limit"),
      projectID: Project.ID.global,
      cost: Money.USD.zero,
      tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
      time: { created: DateTime.makeUnsafe(0), updated: DateTime.makeUnsafe(0) },
      location: Location.Ref.make({ directory: AbsolutePath.make("/tmp") }),
    })
    const input = (tokens: number, limit: { context: number; input?: number; output: number }) => ({
      session,
      resolved: SessionRunnerModel.resolved(model, {
        capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
        cost: [],
        limit,
      }),
      messages: [
        Schema.decodeUnknownSync(SessionMessage.Assistant)({
          id: SessionMessage.ID.make("msg_assistant"),
          type: "assistant",
          agent: Agent.defaultID,
          model: { id: "test-model", providerID: "test-provider" },
          content: [],
          tokens: { input: tokens, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
          time: { created: 0, completed: 0 },
        }),
      ],
    })

    const inputLimited = { context: 400_000, input: 272_000, output: 128_000 }
    expect(compaction.required(input(251_999, inputLimited))).toBe(false)
    expect(compaction.required(input(252_000, inputLimited))).toBe(true)

    const contextLimited = { context: 100_000, output: 10_000 }
    expect(compaction.required(input(79_999, contextLimited))).toBe(false)
    expect(compaction.required(input(80_000, contextLimited))).toBe(true)

    const outputLimited = { context: 100_000, output: 30_000 }
    expect(compaction.required(input(69_999, outputLimited))).toBe(false)
    expect(compaction.required(input(70_000, outputLimited))).toBe(true)
  }),
)

it.effect("manual compaction summarizes short context instead of no-op", () =>
  Effect.gen(function* () {
    requests = []
    const db = (yield* Database.Service).db
    const compaction = yield* SessionCompaction.Service
    const bus = yield* Bus.Service
    const store = yield* SessionStore.Service
    const sessionID = Session.ID.make("ses_manual_compaction")
    const parentID = Session.ID.make("ses_manual_compaction_parent")
    const userMessage = {
      id: SessionMessage.ID.create(),
      type: "user" as const,
      text: "Manual compaction should include this short conversation.",
      time: { created: DateTime.makeUnsafe(0) },
    }
    yield* db
      .insert(ProjectTable)
      .values({ id: Project.ID.global, worktree: AbsolutePath.make("/project"), sandboxes: [] })
      .onConflictDoNothing()
      .run()
      .pipe(Effect.orDie)
    yield* db
      .insert(SessionTable)
      .values({
        id: sessionID,
        project_id: Project.ID.global,
        parent_id: parentID,
        slug: "manual-compaction",
        directory: "/project",
        title: "Manual compaction",
        version: "test",
      })
      .run()
      .pipe(Effect.orDie)

    const session = yield* store
      .get(sessionID)
      .pipe(
        Effect.flatMap((session) =>
          session ? Effect.succeed(session) : Effect.die("manual compaction test session missing"),
        ),
      )

    const delta = yield* bus
      .subscribe(SessionEvent.Compaction.Delta)
      .pipe(Stream.take(1), Stream.runCollect, Effect.forkScoped)
    yield* Effect.yieldNow
    expect(
      yield* compaction.compactManual({
        session,
        messages: [userMessage],
        inputID: SessionMessage.ID.make("msg_manual_compaction"),
      }),
    ).toEqual({ status: "completed" })
    expect(Array.from(yield* Fiber.join(delta)).map((event) => event.data.text)).toEqual(["manual summary"])

    expect(requests).toHaveLength(1)
    expect(requests[0]?.promptCacheKey).toBe(sessionID)
    expect(requests[0]?.http?.headers).toEqual({
      "x-session-affinity": sessionID,
      "X-Session-Id": sessionID,
      "x-parent-session-id": parentID,
      "User-Agent": App.useragent(App.make()),
      "x-opencode-project": Project.ID.global,
      "x-opencode-session": sessionID,
      "x-opencode-client": "opencode",
    })
    expect(requests[0]?.generation).toBeUndefined()
    expect(JSON.stringify(requests[0]?.system)).toContain("summarization assistant")
    expect(JSON.stringify(requests[0]?.messages)).toContain("Manual compaction should include this short conversation.")
    expect(yield* store.context(sessionID)).toMatchObject([
      { type: "compaction", reason: "manual", summary: "manual summary", recent: "" },
    ])
    expect(yield* store.get(sessionID)).toMatchObject({
      cost: 0.0000233,
      tokens: { input: 10, output: 4, reasoning: 2, cache: { read: 3, write: 2 } },
    })
    expect(
      yield* db
        .select({ type: EventTable.type })
        .from(EventTable)
        .where(eq(EventTable.aggregate_id, sessionID))
        .orderBy(asc(EventTable.seq))
        .all()
        .pipe(Effect.orDie),
    ).toEqual([
      { type: Bus.versionedType(SessionEvent.Compaction.Started.type, 1) },
      { type: Bus.versionedType(SessionEvent.UsageRecorded.type, 1) },
      { type: Bus.versionedType(SessionEvent.Compaction.Ended.type, 1) },
    ])
  }),
)

test("boundHead truncates oversized head to fit the budget", () => {
  const huge = Array.from({ length: 200 }, (_, i) => `m${i}-` + "x".repeat(1_000))
  const budget = 1_000
  const bounded = SessionCompaction.boundHead(huge, budget)

  expect(SessionCompaction.boundHead(["short"], 100)).toBe("short")
  expect(Token.estimate(bounded)).toBeLessThanOrEqual(budget)
  expect(bounded.length).toBeLessThan(huge.join("\n\n").length)
  // Newest messages survive; the oldest are dropped.
  expect(bounded).toContain("m199-")
  expect(bounded).not.toContain("m0-")
})

test("boundHead keeps whole messages without splitting inside a message", () => {
  const messages = [
    `[Assistant tool call]: read_file(...)\n[Tool result]: ${"y".repeat(2_000)}`,
    `[User]: ${"z".repeat(1_000)}`,
    `[Assistant]: ${"w".repeat(1_000)}`,
  ]
  // m0≈513, m1≈252, m2≈253 tokens. Budget 500 keeps the newest message (m2)
  // whole and drops the rest without ever splitting inside a message.
  const budget = 500
  const bounded = SessionCompaction.boundHead(messages, budget)

  expect(Token.estimate(bounded)).toBeLessThanOrEqual(budget)
  expect(bounded).toContain("[Assistant]")
  expect(bounded).not.toContain("[Tool result]")
  expect(bounded).not.toContain("[User]")
})

test("boundHead trims a single oversized message to the budget, keeping its tail", () => {
  const single = ["history-".repeat(1_000)] // ~9k chars ≈ 2250 tokens
  const budget = 100
  const bounded = SessionCompaction.boundHead(single, budget)

  expect(Token.estimate(bounded)).toBeLessThanOrEqual(budget)
  // Tail (newest content) is kept, the head is dropped.
  expect(bounded.endsWith("history-".repeat(1_000))).toBe(false)
  expect(bounded.length).toBeLessThan("history-".repeat(1_000).length)
})

test("boundHead with a zero budget yields empty rather than the full message", () => {
  const single = ["history-".repeat(1_000)]
  expect(SessionCompaction.boundHead(single, 0)).toBe("")
  expect(SessionCompaction.boundHead(["short"], 0)).toBe("")
  expect(SessionCompaction.boundHead([], 0)).toBe("")
})

it.effect("auto compaction bounds the summary request within the context window", () =>
  Effect.gen(function* () {
    requests = []
    const db = (yield* Database.Service).db
    const compaction = yield* SessionCompaction.Service
    const store = yield* SessionStore.Service
    const sessionID = Session.ID.make("ses_auto_bound")
    const parentID = Session.ID.make("ses_auto_bound_parent")
    yield* db
      .insert(ProjectTable)
      .values({ id: Project.ID.global, worktree: AbsolutePath.make("/project"), sandboxes: [] })
      .onConflictDoNothing()
      .run()
      .pipe(Effect.orDie)
    yield* db
      .insert(SessionTable)
      .values({
        id: sessionID,
        project_id: Project.ID.global,
        parent_id: parentID,
        slug: "auto-bound",
        directory: "/project",
        title: "Auto bound",
        version: "test",
      })
      .run()
      .pipe(Effect.orDie)

    const session = yield* store
      .get(sessionID)
      .pipe(Effect.flatMap((found) => (found ? Effect.succeed(found) : Effect.die("auto bound session missing"))))
    const model = LanguageModel.make({
      id: "bound-model",
      provider: "test-provider",
      route: OpenAIChat.route.with({ limits: { context: 4_000, output: 1_000 } }),
    })
    // ~40 messages × 2.5k chars ≈ 25k tokens of history, far over the 4k context.
    const messages = Array.from({ length: 40 }, (_, i) => ({
      id: SessionMessage.ID.make(`msg_${i}`),
      type: "user" as const,
      text: `History message ${i} with plenty of padding. ` + "z".repeat(2_000),
      time: { created: DateTime.makeUnsafe(i) },
    }))

    const outcome = yield* compaction.compact({
      session,
      messages,
      resolved: SessionRunnerModel.resolved(model, {
        capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
        cost: [],
        limit: { context: 4_000, output: 1_000 },
      }),
    })
    expect(outcome.status).toBe("completed")

    expect(requests).toHaveLength(1)
    const summaryText = JSON.stringify(requests[0]?.messages)
    // The summary request must fit the model's context window.
    expect(Token.estimate(summaryText)).toBeLessThan(4_000)
    // The bounded head keeps the newest *head* history (message 10) and drops the
    // oldest head (message 0); the newest recent (message 39) is retained separately.
    expect(summaryText).toContain("History message 10")
    expect(summaryText).not.toContain("History message 0")
  }),
)

