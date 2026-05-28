// Tests for OosSplitChart's malformed-row guard (FE2).
//
// The component may receive split rows where pnl is null (a split with no
// trades), sharpe is non-numeric ("n/a" from some backends), or the row
// object itself is null/undefined. Before the fix, any of these caused a
// TypeError crash at render time. After the fix, bad values are coerced
// to 0 so the chart renders safely.
//
// We can't import JSX directly (no Babel in node:test), so we mirror the
// exact data-mapping expression from OosSplitChart.jsx. If the mapping
// changes in the component, update mapSplits here to match.

import test from 'node:test'
import assert from 'node:assert/strict'


// Mirror of the component's data-mapping loop (post-fix).
function mapSplits(splitsDetail) {
  let cumPnl = 0
  return splitsDetail.map((s, i) => {
    const pnlVal = Number(s?.pnl ?? 0)
    const sharpeVal = Number(s?.sharpe) || 0
    cumPnl += pnlVal
    return {
      split: i + 1,
      pnl: Number(pnlVal.toFixed(2)),
      cumPnl: Number(cumPnl.toFixed(2)),
      label: s?.test_start ? s.test_start.slice(0, 7) : `S${i + 1}`,
      sharpe: Number(sharpeVal.toFixed(2)),
      winRate: s?.win_rate != null ? `${(s.win_rate * 100).toFixed(0)}%` : 'n/a',
      trades: s?.trades,
    }
  })
}


test('OosSplitChart: null pnl row maps to pnl=0, does not throw', () => {
  // Pre-fix: `null.toFixed(2)` → TypeError. Post-fix: 0.
  const splits = [{ pnl: null, sharpe: null, test_start: '2024-01-01' }]
  const data = mapSplits(splits)
  assert.equal(data.length, 1)
  assert.equal(data[0].pnl, 0)
  assert.equal(data[0].cumPnl, 0)
  assert.equal(data[0].sharpe, 0)
})

test('OosSplitChart: null pnl does not propagate NaN to cumPnl', () => {
  const splits = [
    { pnl: 100, sharpe: 1.2, test_start: '2024-01-01' },
    { pnl: null, sharpe: null, test_start: '2024-02-01' },
    { pnl: 50,  sharpe: 0.8, test_start: '2024-03-01' },
  ]
  const data = mapSplits(splits)
  assert.equal(data[1].pnl, 0)
  assert.equal(data[1].cumPnl, 100)   // null contributes 0, not NaN
  assert.equal(data[2].cumPnl, 150)
})

test('OosSplitChart: non-numeric sharpe ("n/a") maps to 0, does not throw', () => {
  // Pre-fix: `"n/a".toFixed(2)` → TypeError (strings have no toFixed).
  // Post-fix: Number("n/a") → NaN → NaN || 0 → 0.
  const splits = [{ pnl: 75, sharpe: 'n/a', test_start: '2024-01-01' }]
  const data = mapSplits(splits)
  assert.equal(data[0].sharpe, 0)
})

test('OosSplitChart: null row object maps to safe zero row, does not throw', () => {
  // A null row uses optional chaining (s?.pnl, s?.sharpe, etc.) so every
  // field resolves to a safe default rather than throwing on property access.
  const splits = [null]
  const data = mapSplits(splits)
  assert.equal(data[0].pnl, 0)
  assert.equal(data[0].sharpe, 0)
  assert.equal(data[0].label, 'S1')
  assert.equal(data[0].winRate, 'n/a')
})

test('OosSplitChart: normal rows unaffected by guard', () => {
  const splits = [
    { pnl: 200.567, sharpe: 1.234, test_start: '2024-01-01' },
    { pnl: -50.123, sharpe: -0.5,  test_start: '2024-02-01' },
  ]
  const data = mapSplits(splits)
  assert.equal(data[0].pnl, 200.57)
  assert.equal(data[0].sharpe, 1.23)
  assert.equal(data[1].pnl, -50.12)
  // cumPnl accumulates raw pnlVal, rounded once at the end.
  // 200.567 + (-50.123) = 150.444 → toFixed(2) → 150.44
  assert.equal(data[1].cumPnl, Number((200.567 - 50.123).toFixed(2)))
})

test('OosSplitChart: split label uses test_start when present', () => {
  const splits = [
    { pnl: 0, sharpe: 0, test_start: '2024-06-15' },
    { pnl: 0, sharpe: 0, test_start: null },
  ]
  const data = mapSplits(splits)
  assert.equal(data[0].label, '2024-06')
  assert.equal(data[1].label, 'S2')
})

test('OosSplitChart: zero sharpe (legitimate) is preserved, not masked by || 0', () => {
  // sharpe=0 is a valid value. Number(0) || 0 = 0 || 0 = 0 — correct.
  const splits = [{ pnl: 10, sharpe: 0, test_start: '2024-01-01' }]
  const data = mapSplits(splits)
  assert.equal(data[0].sharpe, 0)
})
