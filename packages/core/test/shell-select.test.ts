import { expect, test } from "bun:test"
import { ShellSelect } from "@opencode-ai/core/shell/select"

const isWindows = process.platform === "win32"

test("acceptable() rejects terminal-only shells such as fish and nu", () => {
  // fish/nu are marked deny in ShellSelect.META; acceptable() must fall back to a
  // usable shell instead of leaking $SHELL into non-interactive script execution.
  const acceptableFish = ShellSelect.acceptable("/bin/fish")
  expect(ShellSelect.name(acceptableFish)).not.toBe("fish")

  const acceptableNu = ShellSelect.acceptable("nu")
  expect(ShellSelect.name(acceptableNu)).not.toBe("nu")
})

test("acceptable() keeps POSIX shells", () => {
  if (isWindows) return
  expect(ShellSelect.name(ShellSelect.acceptable("/bin/bash"))).toBe("bash")
  expect(ShellSelect.name(ShellSelect.acceptable("/bin/zsh"))).toBe("zsh")
  expect(ShellSelect.name(ShellSelect.acceptable("/bin/sh"))).toBe("sh")
})

test("acceptable() falls back to a usable shell when the configured one is not found", () => {
  const shell = ShellSelect.acceptable("/nonexistent/shell")
  expect(shell).toBeDefined()
  if (!isWindows) {
    expect(shell.startsWith("/")).toBe(true)
  }
})

test("name() and login() classify shells", () => {
  expect(ShellSelect.name("/bin/fish")).toBe("fish")
  expect(ShellSelect.login("/bin/fish")).toBe(true)
  expect(ShellSelect.login("/bin/bash")).toBe(true)
  expect(ShellSelect.name("nu")).toBe("nu")
})
