// Pure-function tests for the earnings-move-history transform (node --test).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildEarningsMoveHistory, shortDateLabel, MIN_TAKEAWAY_EVENTS } from './earningsMoveHistory.js'

test('shortDateLabel formats ISO dates without TZ drift', () => {
  assert.equal(shortDateLabel('2024-05-02'), "May '24")
  assert.equal(shortDateLabel('2023-11-30'), "Nov '23")
  assert.equal(shortDateLabel('garbage'), 'garbage')
})

test('builds chronological rows and caps to the limit (keeping most recent)', () => {
  const history = Array.from({ length: 15 }, (_, i) => ({
    date: `2020-01-${String((i % 28) + 1).padStart(2, '0')}`,
    move_pct: i,
    release_timing: 'after market close',
  }))
  const { data } = buildEarningsMoveHistory(history, 5, 12)
  assert.equal(data.length, 12)
  // Most-recent kept: first retained move is the 4th element (index 3) of input.
  assert.equal(data[0].move, 3)
  assert.equal(data[data.length - 1].move, 14)
})

test('drops non-finite moves', () => {
  const { data } = buildEarningsMoveHistory(
    [
      { date: '2024-01-01', move_pct: 4 },
      { date: '2024-04-01', move_pct: null },
      { date: '2024-07-01', move_pct: 'x' },
      { date: '2024-10-01', move_pct: 6 },
    ],
    5,
  )
  assert.deepEqual(data.map((d) => d.move), [4, 6])
})

test('exceedRate = fraction of moves >= implied', () => {
  const hist = [2, 4, 6, 8].map((m, i) => ({ date: `2024-0${i + 1}-01`, move_pct: m }))
  const { exceedRate } = buildEarningsMoveHistory(hist, 5) // >=5 → {6,8} => 2/4
  assert.equal(exceedRate, 0.5)
})

test('FE-2: directional takeaway is withheld below the sample floor', () => {
  const mk = (n) => Array.from({ length: n }, (_, i) => ({ date: `2024-0${(i % 9) + 1}-01`, move_pct: 3 }))
  // n=1 exceedRate is a coin flip (0 or 1) — must NOT be called cheap/rich.
  assert.equal(buildEarningsMoveHistory(mk(1), 5).takeawaySufficient, false)
  assert.equal(buildEarningsMoveHistory(mk(MIN_TAKEAWAY_EVENTS - 1), 5).takeawaySufficient, false)
  // At/above the floor the directional call is allowed.
  assert.equal(buildEarningsMoveHistory(mk(MIN_TAKEAWAY_EVENTS), 5).takeawaySufficient, true)
  assert.equal(buildEarningsMoveHistory(mk(MIN_TAKEAWAY_EVENTS + 4), 5).takeawaySufficient, true)
})

test('null implied move yields null reference and null exceedRate', () => {
  const { impliedMove, exceedRate } = buildEarningsMoveHistory(
    [{ date: '2024-01-01', move_pct: 4 }],
    null,
  )
  assert.equal(impliedMove, null)
  assert.equal(exceedRate, null)
})

test('non-array history is safe', () => {
  assert.deepEqual(buildEarningsMoveHistory(undefined, 5).data, [])
})

test('median is computed over the displayed bars (even count averages the middle two)', () => {
  const hist = [4, 11, 7, 9].map((m, i) => ({ date: `2024-0${i + 1}-01`, move_pct: m }))
  // Sorted [4,7,9,11] → median (7+9)/2 = 8.
  assert.equal(buildEarningsMoveHistory(hist, 6).median, 8)
  // Odd count → middle element.
  const odd = [4, 9, 7].map((m, i) => ({ date: `2024-0${i + 1}-01`, move_pct: m }))
  assert.equal(buildEarningsMoveHistory(odd, 6).median, 7)
  assert.equal(buildEarningsMoveHistory([], 6).median, null)
})
