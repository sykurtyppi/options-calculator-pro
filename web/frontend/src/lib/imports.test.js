// Codebase-shape tests: enforce invariants the bundler doesn't catch.
//
// Vite's bundler treats free variables as runtime concerns — `npm run build`
// succeeds even when a component references a symbol it forgot to import.
// PR #60 round 1 shipped exactly that bug (apiFetch missing in 8 files;
// caught by Codex on review). This file adds a static check that grep-scans
// every source file for known runtime-only references that the build won't
// surface.
//
// Cheap-and-cheerful regex grep. If we ever grow more cross-file invariants,
// the right move is ESLint with `no-undef` + `no-restricted-imports`, but
// adding a single one-file test is the lower-friction option for now.

import test from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))
const SRC_ROOT = join(__dirname, '..')


/**
 * Walk SRC_ROOT recursively, yielding every .jsx / .js file path.
 */
function* allSourceFiles() {
  const stack = [SRC_ROOT]
  while (stack.length) {
    const dir = stack.pop()
    for (const name of readdirSync(dir)) {
      const full = join(dir, name)
      const st = statSync(full)
      if (st.isDirectory()) {
        stack.push(full)
        continue
      }
      if (full.endsWith('.jsx') || full.endsWith('.js')) yield full
    }
  }
}


test('every file referencing apiFetch also imports it', () => {
  const violations = []
  for (const path of allSourceFiles()) {
    // Tests are allowed to reference symbols in regex literals, strings,
    // and comments without importing them — they're testing the
    // invariants, not subject to them. The definition site is also
    // exempt for obvious reasons.
    const basename = path.split('/').pop()
    if (basename === 'api.js') continue
    if (basename.endsWith('.test.js')) continue
    const text = readFileSync(path, 'utf8')
    const references = /\bapiFetch\s*\(/.test(text)
    if (!references) continue
    const imports = /from\s+['"][^'"]*\/lib\/api['"]/.test(text)
    if (!imports) {
      violations.push(relative(SRC_ROOT, path))
    }
  }
  assert.deepEqual(
    violations,
    [],
    `Files referencing apiFetch() without importing from lib/api: ${violations.join(', ')}`,
  )
})
