// Pure-function tests for the fetch error humanizer (node --test).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { httpErrorMessage, fetchErrorMessage } from './errors.js'

test('httpErrorMessage prefers a meaningful server detail', () => {
  assert.equal(
    httpErrorMessage(429, 'Screener rate limit exceeded. Try again shortly.'),
    'Screener rate limit exceeded. Try again shortly.',
  )
})

test('httpErrorMessage maps known statuses when detail is absent', () => {
  assert.match(httpErrorMessage(429, ''), /rate-limited/i)
  assert.match(httpErrorMessage(500, null), /transient/i)
  assert.match(httpErrorMessage(503, undefined), /temporarily unavailable/i)
})

test('httpErrorMessage falls back to a clean generic for unknown statuses', () => {
  const m = httpErrorMessage(418, '')
  assert.match(m, /HTTP 418/)
  assert.match(m, /try again/i)
})

test('httpErrorMessage ignores whitespace-only detail', () => {
  assert.match(httpErrorMessage(500, '   '), /transient/i)
})

// FastAPI 422 validation errors return `detail` as an ARRAY of objects (and
// some paths an object). The old `body.detail || ...` would stringify that to
// "[object Object]"; the typeof-string guard must prevent that. Lock it.
test('httpErrorMessage never stringifies a non-string detail (422 array / object)', () => {
  const arrayDetail = [{ loc: ['body', 'symbol'], msg: 'field required' }]
  const m = httpErrorMessage(422, arrayDetail)
  assert.match(m, /HTTP 422/)
  assert.doesNotMatch(m, /object Object/)
  // An object detail on a mapped status falls back to the mapped message.
  assert.match(httpErrorMessage(500, { msg: 'boom' }), /transient/i)
})

test('fetchErrorMessage gives a connectivity hint for network TypeErrors', () => {
  const err = new TypeError('Failed to fetch')
  assert.match(fetchErrorMessage(err), /reach the server/i)
})

test('fetchErrorMessage passes through a message we already built', () => {
  const err = new Error('Screener rate limit exceeded. Try again shortly.')
  assert.equal(fetchErrorMessage(err), 'Screener rate limit exceeded. Try again shortly.')
})

test('fetchErrorMessage never returns an empty string', () => {
  assert.ok(fetchErrorMessage(new Error('')).length > 0)
  assert.ok(fetchErrorMessage(undefined).length > 0)
})
