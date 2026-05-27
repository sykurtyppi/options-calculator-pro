// Tests for the API_BASE validator. Run via `npm test` (node --test).
//
// These cover the Frontend-audit P1 contract: VITE_API_BASE must only
// allow values that won't leak the session cookie. Anything else throws
// at module load.

import test from 'node:test'
import assert from 'node:assert/strict'

import { resolveRawApiBase, validateApiBase } from './api.js'


// ── resolveRawApiBase ────────────────────────────────────────────────────
// Codex follow-up P1: production default is same-origin (''), but DEV
// mode must fall back to the documented backend port so `npm run dev`
// works without manual .env.development setup.

test('resolveRawApiBase: explicit VITE_API_BASE wins', () => {
  assert.equal(
    resolveRawApiBase({ VITE_API_BASE: 'https://app.example.com', DEV: true }),
    'https://app.example.com',
  )
  // Even in production mode, explicit env is respected.
  assert.equal(
    resolveRawApiBase({ VITE_API_BASE: 'https://api.example.com', DEV: false }),
    'https://api.example.com',
  )
})

test('resolveRawApiBase: unset env + DEV=true → backend localhost', () => {
  assert.equal(
    resolveRawApiBase({ DEV: true }),
    'http://127.0.0.1:8000',
  )
})

test('resolveRawApiBase: unset env + DEV=false → same-origin', () => {
  assert.equal(resolveRawApiBase({ DEV: false }), '')
  assert.equal(resolveRawApiBase({}), '')
  assert.equal(resolveRawApiBase(undefined), '')
})


// ── validateApiBase ──────────────────────────────────────────────────────


test('validateApiBase: empty string → same-origin', () => {
  assert.equal(validateApiBase(''), '')
  assert.equal(validateApiBase(null), '')
  assert.equal(validateApiBase(undefined), '')
  // Whitespace-only also resolves to same-origin (operator probably
  // forgot to fill the .env value).
  assert.equal(validateApiBase('   '), '')
})


test('validateApiBase: localhost http is allowed (dev cross-origin)', () => {
  assert.equal(
    validateApiBase('http://127.0.0.1:8000'),
    'http://127.0.0.1:8000',
  )
  assert.equal(
    validateApiBase('http://localhost:8000'),
    'http://localhost:8000',
  )
  // Default backend port.
  assert.equal(
    validateApiBase('http://127.0.0.1:8000/'),
    'http://127.0.0.1:8000', // trailing slash stripped
  )
})


test('validateApiBase: https is allowed (any host)', () => {
  assert.equal(
    validateApiBase('https://app.example.com'),
    'https://app.example.com',
  )
  assert.equal(
    validateApiBase('https://api.example.com:8443/'),
    'https://api.example.com:8443', // trailing slash stripped
  )
})


test('validateApiBase: plaintext non-localhost is rejected', () => {
  // The whole point of the validator: refuse to ship cookies over
  // plaintext to a non-loopback host. Without this, an env-injection
  // attack on the build environment would silently exfiltrate the
  // session cookie.
  assert.throws(
    () => validateApiBase('http://attacker.com'),
    /Refusing to use VITE_API_BASE/,
  )
  assert.throws(
    () => validateApiBase('http://example.com:8000'),
    /plaintext/,
  )
  // Even a host that LOOKS like localhost but isn't loopback.
  assert.throws(
    () => validateApiBase('http://127.0.0.1.evil.com'),
    /Refusing to use VITE_API_BASE/,
  )
})


test('validateApiBase: malformed URL throws', () => {
  assert.throws(() => validateApiBase('not a url'), /Invalid VITE_API_BASE/)
  assert.throws(() => validateApiBase('://broken'), /Invalid VITE_API_BASE/)
})


test('validateApiBase: non-http(s) protocols are rejected', () => {
  // file:// and ftp:// shouldn't get a free pass just because they're
  // not "plaintext http to non-localhost".
  assert.throws(
    () => validateApiBase('file:///etc/passwd'),
    /Refusing to use VITE_API_BASE/,
  )
  assert.throws(
    () => validateApiBase('javascript:alert(1)'),
    /Refusing to use VITE_API_BASE/,
  )
})
