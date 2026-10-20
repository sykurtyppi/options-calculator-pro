import test from 'node:test'
import assert from 'node:assert/strict'
import { describeBuild } from './buildIdentity.js'

const A = 'aaaaaaa1111111111111111111111111111111111'
const B = 'bbbbbbb2222222222222222222222222222222222'

test('names the build and reports no mismatch when the API runs the same commit', () => {
  assert.deepEqual(describeBuild({ git_sha: A, dirty: false }, { api: { git_sha: A, dirty: false } }),
    { label: 'Build aaaaaaa', mismatch: null })
})

test('the same commit with modified or unknown sources on either side is not called a match', () => {
  for (const [ui, api] of [[true, false], [false, true], [null, false], [false, undefined]]) {
    const out = describeBuild({ git_sha: A, dirty: ui }, { api: { git_sha: A, dirty: api } })
    assert.match(out.mismatch, /cannot confirm they match/, `ui=${ui} api=${api}`)
  }
})

test('flags a known difference between the bundle and the API', () => {
  const out = describeBuild({ git_sha: A, dirty: false }, { api: { git_sha: B } })
  assert.equal(out.label, 'Build aaaaaaa')
  assert.match(out.mismatch, /API runs bbbbbbb/)
})

test('an unknown side is unknown, never a match or a mismatch', () => {
  assert.deepEqual(describeBuild(null, { api: { git_sha: B } }), { label: 'Build unknown', mismatch: null })
  assert.deepEqual(describeBuild({ git_sha: A }, null), { label: 'Build aaaaaaa', mismatch: null })
  assert.deepEqual(describeBuild({ git_sha: A }, { api: { git_sha: null } }), { label: 'Build aaaaaaa', mismatch: null })
  // A known commit difference wins over a modified flag.
  assert.match(describeBuild({ git_sha: A, dirty: true }, { api: { git_sha: B, dirty: true } }).mismatch, /API runs bbbbbbb/)
})

test('marks a bundle built from modified sources', () => {
  assert.equal(describeBuild({ git_sha: A, dirty: true }, null).label, 'Build aaaaaaa (modified)')
})
