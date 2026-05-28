// Tests for the AbortController guard pattern used in LedgerDiagnosticsPanel
// and DataQualityDiagnosticsPanel (FE1).
//
// Both panels share the same two guards:
//
//   catch (err) { if (err.name !== 'AbortError') setError(...) }
//   finally     { if (!signal?.aborted)           setLoading(false) }
//
// Pre-fix: AbortErrors surfaced as red error banners and loading was
// cleared even when the effect had already been superseded by a newer
// fetch (stale response race). Post-fix: aborted operations are silent.
//
// These guards are pure expressions — testable without DOM or React.

import test from 'node:test'
import assert from 'node:assert/strict'


// Mirrors the catch guard: only forward non-AbortErrors to state setters.
function applyErrorGuard(err, setError) {
  if (err.name !== 'AbortError') setError(err.message || String(err))
}

// Mirrors the finally guard: only clear loading when the signal wasn't aborted.
function applyLoadingGuard(signal, setLoading) {
  if (!signal?.aborted) setLoading(false)
}


// ── catch guard ──────────────────────────────────────────────────────────

test('AbortError is silently swallowed — setError is not called', () => {
  const errors = []
  const abortErr = new DOMException('The operation was aborted.', 'AbortError')
  applyErrorGuard(abortErr, (msg) => errors.push(msg))
  assert.equal(errors.length, 0, 'AbortError must not reach setError')
})

test('Non-abort errors are forwarded to setError', () => {
  const errors = []
  const networkErr = new Error('Network failed')
  applyErrorGuard(networkErr, (msg) => errors.push(msg))
  assert.equal(errors.length, 1)
  assert.equal(errors[0], 'Network failed')
})

test('HTTP status error (e.g. 500) is forwarded to setError', () => {
  const errors = []
  const statusErr = new Error('Ledger query failed (500)')
  applyErrorGuard(statusErr, (msg) => errors.push(msg))
  assert.equal(errors.length, 1)
  assert.match(errors[0], /500/)
})


// ── finally guard ─────────────────────────────────────────────────────────

test('loading is cleared when signal is not aborted', () => {
  const loadingCalls = []
  const controller = new AbortController()
  // signal.aborted is false — fetch completed normally
  applyLoadingGuard(controller.signal, (v) => loadingCalls.push(v))
  assert.deepEqual(loadingCalls, [false], 'setLoading(false) must fire when not aborted')
})

test('loading is NOT cleared when signal is aborted', () => {
  const loadingCalls = []
  const controller = new AbortController()
  controller.abort()
  // signal.aborted is true — effect was cleaned up; a new fetch is in flight
  applyLoadingGuard(controller.signal, (v) => loadingCalls.push(v))
  assert.equal(loadingCalls.length, 0, 'setLoading must not be called when aborted — a newer effect controls loading state')
})

test('loading guard handles undefined signal (manual Refresh button call)', () => {
  // When the Refresh button calls loadRows() with no signal argument,
  // signal is undefined. `!undefined?.aborted` === true, so loading
  // is cleared normally — the guard is a no-op in this path.
  const loadingCalls = []
  applyLoadingGuard(undefined, (v) => loadingCalls.push(v))
  assert.deepEqual(loadingCalls, [false])
})
