export * as Token from "./token.js"

const CHARS_PER_TOKEN = 4

/**
 * Rough token estimate: ASCII text is ~4 chars/token; non-ASCII characters (e.g.
 * CJK) are closer to 1 char/token, so they are counted individually to avoid
 * grossly underestimating multilingual history.
 */
export const estimate = (input: string) => {
  let ascii = 0
  let nonAscii = 0
  for (const char of input) {
    if (char.charCodeAt(0) > 127) nonAscii++
    else ascii++
  }
  return Math.max(0, Math.round(ascii / CHARS_PER_TOKEN) + nonAscii)
}
