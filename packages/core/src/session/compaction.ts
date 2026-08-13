export * as SessionCompaction from "./compaction.js"

import { LLM, LLMClient, AIError, LLMEvent, Message, type LLMRequest, type LanguageModel } from "@opencode-ai/ai"
import type { StreamOptions } from "@opencode-ai/ai/route"
import { SessionError } from "@opencode-ai/schema/session-error"
import { Document, type Entry } from "@opencode-ai/schema/config"
import { Context, Effect, Layer, Stream } from "effect"
import { Config } from "../config.js"
import { Bus } from "../bus.js"
import { makeLocationNode } from "@opencode-ai/util/effect/app-node"
import { llmClient } from "../effect/app-node-platform.js"
import { SessionEvent } from "./event.js"
import type { SessionMessage } from "./message.js"
import { SessionModelHeaders } from "./model-headers.js"
import { SessionModelHttp } from "./model-http.js"
import { SessionPromptCacheKey } from "./prompt-cache-key.js"
import { App } from "../app.js"
import { SessionRunnerModel } from "./runner/model.js"
import { SessionSchema } from "./schema.js"
import { toSessionError } from "./to-session-error.js"
import { Token } from "../util/token.js"
import type { Info, Ref } from "../model.js"
import { SessionUsage } from "./usage.js"
import { PluginHooks } from "../plugin/hooks.js"
import { Agent } from "../agent.js"

const DEFAULT_BUFFER = 20_000
const DEFAULT_KEEP_TOKENS = 15_000
const OUTPUT_TOKEN_MAX = 32_000
const TOOL_OUTPUT_MAX_CHARS = 2_000
const SUMMARY_TEMPLATE = `Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Important Details
- [constraints/preferences, decisions and why, important facts/assumptions, exact context needed to continue, or "(none)"]

## Work State
### Completed
- [finished work, verified facts, or changes made; otherwise "(none)"]

### Active
- [current work, partial changes, or investigation state; otherwise "(none)"]

### Blocked
- [blockers, failing commands, or unknowns; otherwise "(none)"]

## Next Move
1. [immediate concrete action, or "(none)"]
2. [next action if known, or "(none)"]

## Relevant Files
- [file or directory path: why it matters, or "(none)"]
</template>

Rules:
- Keep every section, even when empty.
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers when known.
- Do not mention the summary process or that context was compacted.`

type Settings = {
  readonly auto: boolean
  readonly buffer: number
  readonly tokens: number
}

type Dependencies = {
  readonly app: App.Info
  readonly bus: Bus.Interface
  readonly llm: {
    readonly stream: (request: LLMRequest, options?: StreamOptions) => Stream.Stream<LLMEvent, AIError>
  }
  readonly models: SessionRunnerModel.Interface
  readonly config: Settings
  readonly hooks: PluginHooks.Interface
}

export type AutoInput = {
  readonly session: SessionSchema.Info
  readonly messages: readonly SessionMessage.Info[]
  readonly model: LanguageModel
  readonly ref: Ref
  readonly cost: Info["cost"]
}

export type ManualInput = {
  readonly session: SessionSchema.Info
  readonly messages: readonly SessionMessage.Info[]
  readonly inputID: SessionMessage.ID
  readonly started?: boolean
}

type RequiredInput = Omit<AutoInput, "ref">

type Plan = {
  readonly session: SessionSchema.Info
  readonly model: LanguageModel
  readonly ref: Ref
  readonly cost: Info["cost"]
  readonly reason: SessionMessage.Compaction["reason"]
  readonly prompt: string
  readonly recent: string
  readonly inputID?: SessionMessage.ID
  readonly started?: boolean
}

export type Outcome =
  | Pick<SessionMessage.CompactionCompleted, "status">
  | Pick<SessionMessage.CompactionFailed, "status" | "error">

export interface Interface {
  readonly required: (input: RequiredInput) => boolean
  readonly compact: (input: AutoInput) => Effect.Effect<Outcome>
  readonly compactManual: (input: ManualInput) => Effect.Effect<Outcome>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionCompaction") {}

const truncate = (value: string) =>
  value.length <= TOOL_OUTPUT_MAX_CHARS ? value : `${value.slice(0, TOOL_OUTPUT_MAX_CHARS)}\n[truncated]`

export const serializeToolContent = (content: SessionMessage.ToolStateCompleted["content"]) =>
  content
    .map((item) =>
      item.type === "text" ? item.text : `[Attached ${item.mime}${item.name === undefined ? "" : `: ${item.name}`}]`,
    )
    .join("\n")

const serialize = (message: SessionMessage.Info) => {
  if (message.type === "user") {
    const files =
      message.files?.map(
        (file) =>
          `[Attached ${file.mime}: ${file.name ?? (file.source.type === "uri" ? file.source.uri : "inline attachment")}]`,
      ) ?? []
    const skills = message.skills?.map((skill) => `[Attached skill: ${skill.name}]\n${skill.text}`) ?? []
    return [`[User]: ${message.text}`, ...skills, ...files].join("\n")
  }
  if (message.type === "location-switched")
    return `[User]: The working directory has been changed to ${message.location.directory}.`
  if (message.type === "assistant") {
    return message.content
      .flatMap((part) => {
        if (part.type === "text") return [`[Assistant]: ${part.text}`]
        if (part.type === "reasoning") return part.text ? [`[Assistant reasoning]: ${part.text}`] : []
        const input = typeof part.state.input === "string" ? part.state.input : JSON.stringify(part.state.input)
        if (part.state.status === "completed")
          return [
            `[Assistant tool call]: ${part.name}(${input})`,
            `[Tool result]: ${truncate(serializeToolContent(part.state.content))}`,
          ]
        if (part.state.status === "error")
          return [`[Assistant tool call]: ${part.name}(${input})`, `[Tool error]: ${part.state.error.message}`]
        return [`[Assistant tool call]: ${part.name}(${input})`]
      })
      .join("\n")
  }
  if (message.type === "system") return `[System update]: ${message.text}`
  if (message.type === "synthetic") return `[Synthetic context]: ${message.text}`
  if (message.type === "skill") return `[Skill activated: ${message.name}]\n${message.text}`
  if (message.type === "shell") return `[Shell]: ${message.command}\n${truncate(message.output?.output ?? "")}`
  return ""
}

const settings = (documents: readonly Entry[]) => {
  const configured = documents
    .filter((entry): entry is Document => entry.type === "document")
    .flatMap((entry) => (entry.info.compaction ? [entry.info.compaction] : []))
  return {
    auto: configured.findLast((value) => value.auto !== undefined)?.auto ?? true,
    buffer: configured.findLast((value) => value.buffer !== undefined)?.buffer ?? DEFAULT_BUFFER,
    tokens: configured.findLast((value) => value.keep?.tokens !== undefined)?.keep?.tokens ?? DEFAULT_KEEP_TOKENS,
  }
}

const select = (
  messages: readonly SessionMessage.Info[],
  tokens: number,
): { readonly headMessages: readonly string[]; readonly recentMessages: readonly string[] } | undefined => {
  const conversation = messages
    .filter((message) => message.type !== "compaction" && message.type !== "system")
    .flatMap((message) => {
      const text = serialize(message)
      return text ? [{ message, text }] : []
    })
  if (conversation.length === 0) return undefined
  let total = 0
  let split = conversation.length
  for (let index = conversation.length - 1; index >= 0; index--) {
    const next = total + Token.estimate(conversation[index].text)
    if (split < conversation.length && next > tokens) break
    total = next
    split = index
  }
  while (split > 0 && conversation[split].message.type !== "user") split--
  if (split === 0) {
    const latestUser = conversation.findLastIndex((item) => item.message.type === "user")
    if (latestUser > 0) split = latestUser
  }
  return {
    headMessages: conversation.slice(0, split).map((item) => item.text),
    recentMessages: conversation.slice(split).map((item) => item.text),
  }
}

/**
 * Bounds a list of already-serialized conversation messages (ordered oldest to
 * newest) to the given token budget, keeping the newest messages and dropping the
 * oldest. Truncation happens on whole-message boundaries so a tool call is never
 * separated from its result. At least the newest message is always kept so the
 * summary always has some grounding; if that single newest message alone exceeds
 * the budget it is cut down to the budget, keeping its tail (the newest content).
 */
export const boundHead = (messages: readonly string[], tokens: number) => {
  const kept: string[] = []
  let used = 0
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]
    const next = used + Token.estimate(message)
    if (used > 0 && next > tokens) break
    kept.push(message)
    used = next
  }
  // Fallback: a single oversized message is trimmed to the budget (tail kept).
  if (kept.length === 1 && Token.estimate(kept[0]) > tokens) {
    const budgetChars = Math.max(0, Math.floor(tokens * 4))
    kept[0] = kept[0].slice(-budgetChars)
  }
  // Restore chronological order (oldest to newest).
  kept.reverse()
  return kept.join("\n\n")
}

export const buildPrompt = (input: { readonly previousSummary?: string; readonly context: readonly string[] }) =>
  [
    input.previousSummary
      ? `Update the anchored summary below using the conversation history above.\nPreserve still-true details, remove stale details, and merge in the new facts.\n<previous-summary>\n${input.previousSummary}\n</previous-summary>`
      : "Create a new anchored summary from the conversation history.",
    SUMMARY_TEMPLATE,
    "The following is the conversation history:",
    ...input.context,
  ].join("\n\n")

const planContent = (
  messages: readonly SessionMessage.Info[],
  tokens: number,
  model?: Pick<LanguageModel, "route">,
) => {
  const selected = select(messages, tokens)
  if (!selected) return
  const previousSummary = messages.findLast(
    (message) => message.type === "compaction" && message.status === "completed",
  )
  const previousRecent = previousSummary?.type === "compaction" ? previousSummary.recent : ""
  const summarizeRecent = !previousRecent && !selected.headMessages.length
  const previousSummaryText = previousSummary?.type === "compaction" ? previousSummary.summary : undefined
  // Reserve room for the summary output, bounded by the model's output limit.
  const output = model?.route.defaults.limits?.output
  const summaryTokens = Math.max(1, Math.min(output ?? 4_096, 4_096))
  const context = model?.route.defaults.limits?.context ?? Number.POSITIVE_INFINITY
  // Fixed prompt overhead that participates in the same budget as the history:
  // instructions, the summary template, and the previous summary.
  const fixedOverhead =
    Token.estimate(previousSummaryText ?? "") +
    Token.estimate(SUMMARY_TEMPLATE) +
    Token.estimate("The following is the conversation history:") +
    (previousSummaryText
      ? Token.estimate(
          "Update the anchored summary below using the conversation history above.\nPreserve still-true details, remove stale details, and merge in the new facts.\n<previous-summary>\n</previous-summary>",
        )
      : Token.estimate("Create a new anchored summary from the conversation history."))
  const historyBudget = Number.isFinite(context)
    ? Math.max(0, Math.floor(context - summaryTokens - fixedOverhead))
    : Number.POSITIVE_INFINITY
  // previousRecent and head share the history budget. previousRecent is the newest
  // history (kept from the previous compaction), so it is preserved first and head
  // (older) takes whatever budget remains.
  const boundPreviousRecent =
    !previousRecent || Token.estimate(previousRecent) <= historyBudget
      ? previousRecent
      : boundHead(previousRecent.split("\n\n"), historyBudget)
  const recentBudget = Math.max(0, historyBudget - Token.estimate(boundPreviousRecent))
  const head = summarizeRecent || recentBudget <= 0 ? "" : boundHead(selected.headMessages, recentBudget)
  // In the summarizeRecent case the recent tail is the only history: bound it too so
  // the request stays within the window even for small contexts.
  const boundedRecent = summarizeRecent ? boundHead(selected.recentMessages, recentBudget) : ""
  return {
    prompt: buildPrompt({
      previousSummary: previousSummaryText,
      context: summarizeRecent ? [boundedRecent] : [boundPreviousRecent, head].filter(Boolean),
    }),
    recent: summarizeRecent ? "" : selected.recentMessages.join("\n\n"),
  }
}

const make = (dependencies: Dependencies) => {
  const config = dependencies.config
  const failed = Effect.fnUntraced(function* (input: {
    readonly sessionID: SessionSchema.ID
    readonly reason: SessionMessage.Compaction["reason"]
    readonly error: SessionError.Error
    readonly inputID?: SessionMessage.ID
  }) {
    yield* dependencies.bus.publish(SessionEvent.Compaction.Failed, input)
    return { status: "failed" as const, error: input.error }
  })
  const execute = Effect.fn("SessionCompaction.execute")(function* (plan: Plan) {
    if (!plan.started)
      yield* dependencies.bus.publish(SessionEvent.Compaction.Started, {
        sessionID: plan.session.id,
        reason: plan.reason,
        recent: plan.recent,
        inputID: plan.inputID,
      })

    const chunks: string[] = []
    let failure: SessionError.Error | undefined
    let usage: SessionUsage.Recorded | undefined
    const recordUsage = Effect.suspend(() =>
      usage
        ? dependencies.bus.publish(SessionEvent.UsageRecorded, {
            sessionID: plan.session.id,
            source: "compaction",
            ...usage,
          })
        : Effect.void,
    )
    yield* dependencies.llm
      .stream(
        LLM.request({
          model: plan.model,
          promptCacheKey: SessionPromptCacheKey.make(plan.session.id),
          http: { headers: SessionModelHeaders.make(plan.session, dependencies.app) },
          messages: [Message.user(plan.prompt)],
          tools: [],
        }),
        {
          http: SessionModelHttp.middleware(dependencies.hooks, {
            sessionID: plan.session.id,
            agent: Agent.ID.make("compaction"),
            model: plan.ref,
          }),
        },
      )
      .pipe(
        Stream.runForEach((event) => {
          if (LLMEvent.is.providerError(event))
            failure = {
              type: event.classification === "context-overflow" ? "provider.invalid-request" : "provider.error",
              message: event.message,
            }
          if (LLMEvent.is.textDelta(event)) {
            chunks.push(event.text)
            return dependencies.bus.publish(SessionEvent.Compaction.Delta, {
              sessionID: plan.session.id,
              text: event.text,
            })
          }
          if (LLMEvent.is.stepFinish(event)) {
            const step = SessionUsage.record(event.usage, plan.cost)
            usage = usage ? SessionUsage.add(usage, step) : step
          }
          return Effect.void
        }),
        Effect.catchTag("AI.Error", (error) =>
          Effect.sync(() => {
            failure = toSessionError(error)
          }),
        ),
        Effect.onInterrupt(() =>
          recordUsage.pipe(
            Effect.andThen(
              plan.reason === "auto"
                ? failed({
                    sessionID: plan.session.id,
                    reason: plan.reason,
                    error: { type: "compaction.interrupted", message: "Compaction was interrupted" },
                    inputID: plan.inputID,
                  }).pipe(Effect.asVoid)
                : Effect.void,
            ),
          ),
        ),
      )
    yield* recordUsage
    const summary = chunks.join("")
    if (failure || !summary.trim()) {
      const error = failure ?? { type: "compaction.failed" as const, message: "Compaction produced no summary" }
      return yield* failed({
        sessionID: plan.session.id,
        reason: plan.reason,
        error,
        inputID: plan.inputID,
      })
    }
    yield* dependencies.bus.publish(SessionEvent.Compaction.Ended, {
      sessionID: plan.session.id,
      reason: plan.reason,
      text: summary,
      recent: plan.recent,
    })
    return { status: "completed" as const }
  })
  const compact = Effect.fn("SessionCompaction.compact")(function* (input: AutoInput) {
    const content = planContent(input.messages, config.tokens, input.model)
    if (content)
      return yield* execute({
        session: input.session,
        model: input.model,
        ref: input.ref,
        cost: input.cost,
        reason: "auto",
        ...content,
      })
    const error = { type: "compaction.unavailable" as const, message: "Nothing to compact yet" }
    return yield* failed({
      sessionID: input.session.id,
      reason: "auto",
      error,
    })
  })
  const required = (input: RequiredInput) => {
    if (!config.auto) return false
    const context = input.model.route.defaults.limits?.context
    if (context === undefined || context <= 0) return false
    const last = input.messages.findLast(
      (message): message is SessionMessage.Assistant & { tokens: NonNullable<SessionMessage.Assistant["tokens"]> } =>
        message.type === "assistant" && message.tokens !== undefined,
    )
    if (!last) return false
    const limits = input.model.route.defaults.limits
    const output = Math.min(limits?.output ?? 0, OUTPUT_TOKEN_MAX)
    const promptCeiling = Math.min(
      limits?.input === undefined ? Number.POSITIVE_INFINITY : limits.input - config.buffer,
      context - Math.max(output, config.buffer),
    )
    const used =
      last.tokens.input + last.tokens.output + last.tokens.reasoning + last.tokens.cache.read + last.tokens.cache.write
    if (used <= 0) return false
    return used >= promptCeiling
  }
  const compactManual = Effect.fn("SessionCompaction.compactManual")(function* (input: ManualInput) {
    // Check for compactable content first so an empty session fails fast without
    // triggering model resolution. The model is resolved afterwards so the summary
    // prompt can be bounded to its context window.
    if (!planContent(input.messages, config.tokens))
      return yield* failed({
        sessionID: input.session.id,
        reason: "manual",
        error: { type: "compaction.unavailable", message: "Nothing to compact yet" },
        inputID: input.inputID,
      })
    const resolved = yield* dependencies.models.resolve(input.session).pipe(
      Effect.catch((cause) =>
        failed({
          sessionID: input.session.id,
          reason: "manual",
          error: toSessionError(cause),
          inputID: input.inputID,
        }),
      ),
    )
    if ("status" in resolved) return resolved
    const content = planContent(input.messages, config.tokens, resolved.model)
    if (!content)
      return yield* failed({
        sessionID: input.session.id,
        reason: "manual",
        error: { type: "compaction.unavailable", message: "Nothing to compact yet" },
        inputID: input.inputID,
      })
    return yield* execute({
      session: input.session,
      model: resolved.model,
      ref: resolved.ref,
      cost: resolved.cost,
      reason: "manual",
      inputID: input.inputID,
      started: input.started,
      ...content,
    })
  })
  return Service.of({
    required,
    compact,
    compactManual,
  })
}

export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const bus = yield* Bus.Service
    const llm = yield* LLMClient.Service
    const config = yield* Config.Service
    const models = yield* SessionRunnerModel.Service
    const app = yield* App.Metadata
    const hooks = yield* PluginHooks.Service
    return make({ bus, llm, models, config: settings(yield* config.entries()), app, hooks })
  }),
)

export const node = makeLocationNode({
  service: Service,
  layer,
  deps: [Bus.node, llmClient, Config.node, SessionRunnerModel.node, App.node, PluginHooks.node],
})
