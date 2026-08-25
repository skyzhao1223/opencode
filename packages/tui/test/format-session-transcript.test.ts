import { test, expect } from "bun:test"
import { formatSessionTranscript } from "../src/routes/session/index"

test("formatSessionTranscript strips NUL characters", () => {
  const session = { id: "ses_1", title: "Test", time: { created: 0, updated: 0 } }
  const messages = [{ type: "shell", command: "echo hi", output: { output: "hi\u0000world" } }]
  const out = formatSessionTranscript(session as any, messages as any, false)
  expect(out).not.toContain("\u0000")
  expect(out).toContain("hi")
  expect(out).toContain("world")
})

test("formatSessionTranscript keeps normal text intact", () => {
  const session = { id: "ses_2", title: "Test", time: { created: 0, updated: 0 } }
  const messages = [{ type: "user", text: "hello world" }]
  const out = formatSessionTranscript(session as any, messages as any, false)
  expect(out).toContain("hello world")
})
