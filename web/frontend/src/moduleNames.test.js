import test from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync } from 'node:fs'
import { dirname, extname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

// Extensions an extensionless import can resolve to (Vite's default order).
const RESOLVABLE = new Set(['.mjs', '.js', '.mts', '.ts', '.jsx', '.tsx', '.json'])
const SRC = dirname(fileURLToPath(import.meta.url))

/**
 * Names in one directory that a case-insensitive filesystem (macOS by
 * default) cannot tell apart as import targets: `./BuildStamp` resolves
 * `.js` before `.jsx`, so beside `buildStamp.js` it loads the wrong module
 * on a Mac while Linux CI loads the right one.
 */
export function moduleNameCollisions(names) {
  const byKey = new Map()
  for (const name of names) {
    const ext = extname(name)
    const key = (RESOLVABLE.has(ext) ? name.slice(0, -ext.length) : name).toLowerCase()
    byKey.set(key, [...(byKey.get(key) ?? []), name])
  }
  return [...byKey.values()].filter((group) => group.length > 1)
}

function collisionsUnder(dir) {
  const entries = readdirSync(dir, { withFileTypes: true })
  const found = moduleNameCollisions(entries.map((e) => e.name))
    .map((group) => group.map((name) => relative(SRC, join(dir, name))))
  for (const entry of entries) {
    if (entry.isDirectory()) found.push(...collisionsUnder(join(dir, entry.name)))
  }
  return found
}

test('flags names that differ only by case or by a resolvable extension', () => {
  assert.deepEqual(moduleNameCollisions(['BuildStamp.jsx', 'buildStamp.js']), [['BuildStamp.jsx', 'buildStamp.js']])
  assert.deepEqual(moduleNameCollisions(['foo.js', 'foo.jsx']), [['foo.js', 'foo.jsx']])
  assert.deepEqual(moduleNameCollisions(['Styles.css', 'styles.css']), [['Styles.css', 'styles.css']])
  assert.deepEqual(moduleNameCollisions(['Lib', 'lib.js']), [['Lib', 'lib.js']])
})

test('allows names a case-insensitive filesystem still tells apart', () => {
  assert.deepEqual(moduleNameCollisions(['BuildStamp.jsx', 'BuildStamp.dom.test.jsx', 'buildIdentity.js']), [])
  assert.deepEqual(moduleNameCollisions(['styles.css', 'styles.js']), [])
})

test('no two source modules in one directory differ only by case or extension', () => {
  assert.deepEqual(collisionsUnder(SRC), [])
})
