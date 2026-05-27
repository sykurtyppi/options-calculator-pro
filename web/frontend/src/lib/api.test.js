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


// ── apiFetch — 401 redirect (PR #69) ────────────────────────────────────
//
// Pre-PR-#69, a 401 surfaced as "HTTP 401" in a red error banner with no
// recovery affordance for the operator. PR #69 wires apiFetch to trigger
// a hard navigation to /login on 401 — the same flow the LogoutButton
// uses. These tests pin:
//
//   1. 401 → redirect fires.
//   2. Non-401 responses → no redirect.
//   3. credentials: 'include' is still injected on every call (the PR
//      #59 contract that this fix must not break).
//   4. If the user is already on /login, the redirect is skipped (loop
//      guard — without it, a 401 from the login page itself would
//      navigate to /login again and produce a brief flash).

import {
  apiFetch,
  _setRedirectToLogin,
  _resetRedirectToLogin,
} from './api.js'


/** Capture-mode fetch + window stubs used by the apiFetch tests below.
 *
 * Returns an object with:
 *   - install({ status, pathname }): swap globals with stubs that
 *     return a Response with the given status from fetch() and
 *     pretend window.location.pathname is `pathname` (default '/').
 *   - restore(): put the original globals back.
 *   - lastFetchCall(): the (input, options) tuple of the most recent
 *     fetch invocation. Used to assert credentials:'include' threading.
 *   - redirectCalls(): count of times the captured redirect fired.
 *
 * Pure helper — kept here so the test bodies stay focused on the
 * assertion rather than the stubbing scaffolding.
 */
function _makeFetchHarness() {
  let originalFetch
  let originalWindow
  let lastFetchInput
  let lastFetchOptions
  let redirectFireCount

  return {
    install({ status = 200, pathname = '/' } = {}) {
      originalFetch = globalThis.fetch
      originalWindow = globalThis.window
      lastFetchInput = undefined
      lastFetchOptions = undefined
      redirectFireCount = 0

      globalThis.fetch = (input, options) => {
        lastFetchInput = input
        lastFetchOptions = options
        // Minimal Response stand-in. The production code only inspects
        // .status; we don't need a real Response.
        return Promise.resolve({ status, ok: status >= 200 && status < 300 })
      }
      // Minimal window with a settable location.pathname. We don't
      // install a real window.location.assign here because the
      // redirect path is overridden via _setRedirectToLogin below —
      // this window stub only exists so `_isAlreadyOnLoginPage()`
      // can read pathname.
      globalThis.window = { location: { pathname } }
      _setRedirectToLogin(() => { redirectFireCount += 1 })
    },
    restore() {
      _resetRedirectToLogin()
      globalThis.fetch = originalFetch
      globalThis.window = originalWindow
    },
    lastFetchCall: () => [lastFetchInput, lastFetchOptions],
    redirectCalls: () => redirectFireCount,
  }
}


test('apiFetch: 401 response triggers /login redirect', async () => {
  const h = _makeFetchHarness()
  h.install({ status: 401, pathname: '/' })
  try {
    const res = await apiFetch('/api/recommendations')
    assert.equal(res.status, 401, 'response still resolves to caller')
    assert.equal(
      h.redirectCalls(), 1,
      '401 must trigger exactly one redirect to /login',
    )
  } finally {
    h.restore()
  }
})

test('apiFetch: non-401 responses do not redirect', async () => {
  // 200 OK
  let h = _makeFetchHarness()
  h.install({ status: 200, pathname: '/' })
  try {
    await apiFetch('/api/recommendations')
    assert.equal(h.redirectCalls(), 0, '200 must NOT trigger a redirect')
  } finally {
    h.restore()
  }

  // 500 ISE — caller should see the error, not be redirected
  h = _makeFetchHarness()
  h.install({ status: 500, pathname: '/' })
  try {
    await apiFetch('/api/recommendations')
    assert.equal(h.redirectCalls(), 0, '500 must NOT trigger a redirect')
  } finally {
    h.restore()
  }

  // 403 — distinct semantics from 401 (authenticated but unauthorized)
  h = _makeFetchHarness()
  h.install({ status: 403, pathname: '/' })
  try {
    await apiFetch('/api/recommendations')
    assert.equal(
      h.redirectCalls(), 0,
      '403 has distinct semantics from 401 — must NOT redirect',
    )
  } finally {
    h.restore()
  }
})

test('apiFetch: 401 while already on /login does not redirect (loop guard)', async () => {
  const h = _makeFetchHarness()
  // Simulate the user being on the login page when the 401 fires
  // (e.g. a session-status probe from the login page itself).
  // Without this guard, the page would re-navigate to /login →
  // ugly flash, no useful behaviour change.
  h.install({ status: 401, pathname: '/login' })
  try {
    await apiFetch('/api/recommendations')
    assert.equal(
      h.redirectCalls(), 0,
      'redirect must be skipped when pathname is already /login',
    )
  } finally {
    h.restore()
  }
})

test('apiFetch: credentials: include still injected on every call', async () => {
  // The PR #59 contract: every apiFetch call sends the session
  // cookie. PR #69 must NOT break this — the .then() handler
  // returning the response should preserve the original opts.
  const h = _makeFetchHarness()
  h.install({ status: 200 })
  try {
    await apiFetch('/api/recommendations')
    const [, opts] = h.lastFetchCall()
    assert.equal(
      opts.credentials, 'include',
      'credentials: include must be threaded through to the underlying fetch',
    )
  } finally {
    h.restore()
  }

  // Caller-supplied options other than credentials are still honoured.
  const h2 = _makeFetchHarness()
  h2.install({ status: 200 })
  try {
    await apiFetch('/api/oos/submit', {
      method: 'POST',
      body: JSON.stringify({ symbol: 'AAPL' }),
    })
    const [, opts] = h2.lastFetchCall()
    assert.equal(opts.credentials, 'include')
    assert.equal(opts.method, 'POST')
    assert.equal(opts.body, '{"symbol":"AAPL"}')
  } finally {
    h2.restore()
  }
})

test('apiFetch: 401 still resolves response to caller (not thrown)', async () => {
  // Codex review point: we add a redirect side-effect on 401, but the
  // promise still resolves with the 401 response. Callers that have
  // their own 401 handling (none today, but possible in the future)
  // still see the response. The redirect is purely additive.
  const h = _makeFetchHarness()
  h.install({ status: 401, pathname: '/' })
  try {
    let thrown = null
    let response = null
    try {
      response = await apiFetch('/api/recommendations')
    } catch (e) {
      thrown = e
    }
    assert.equal(thrown, null, 'apiFetch must NOT reject on 401')
    assert.equal(response.status, 401, 'caller still receives 401 response')
  } finally {
    h.restore()
  }
})
