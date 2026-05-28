// Tests for OosSplitChart's null-pnl guard (FE2).
//
// The component can receive split rows where `pnl` is null (e.g. an OOS
// split that produced no trades). Before the fix, `s.pnl.toFixed(2)` threw
// TypeError at render time and crashed the chart. After the fix, null is
// coerced to 0 via `Number(s.pnl ?? 0)` so the chart renders with a zero
// bar for that split.
//
// We can't import the JSX component directly (no Babel transform in the
// node:test runner), so we mirror the exact data-mapping expression from
// OosSplitChart.jsx and test it in isolation. If the mapping expression
// ever changes in the component, update `mapSplits` here to match.

import test from 'node:test'
import assert from 'node:assert/strict'


// Mirror of the component's data-mapping loop.
function mapSplits(splitsDetail) {
  let cumPnl = 0
  return splitsDetail.map((s, i) => {
    const pnlVal = Number(s.pnl ?? 0)
    cumPnl += pnlVal
    return {
      split: i + 1,
      pnl: Number(pnlVal.toFixed(2)),
      cumPnl: Number(cumPnl.toFixed(2)),
      label: s.test_start ? s.test_start.slice(0, 7) : `S${i + 1}`,
      sharpe: Number((s.sharpe ?? 0).toFixed(2)),
    }
  })
}


test('OosSplitChart: null pnl row maps to pnl=0, does not throw', () => {
  // Pre-fix: `null.toFixed(2)` throws TypeError. Post-fix: 0.
  const splits = [{ pnl: null, sharpe: null, test_start: '2024-01-01' }]
  const data = mapSplits(splits)
  assert.equal(data.length, 1)
  assert.equal(data[0].pnl, 0)
  assert.equal(data[0].cumPnl, 0)
  assert.equal(data[0].sharpe, 0)
})

test('OosSplitChart: null pnl does not propagate NaN to cumPnl', () => {
  // Pre-fix: `cumPnl += null` → cumPnl=0 (JS coercion), but then
  // `null.toFixed(2)` crashes before cumPnl is ever stored. Post-fix:
  // pnlVal=0 so cumPnl stays correct across all subsequent splits.
  const splits = [
    { pnl: 100, sharpe: 1.2, test_start: '2024-01-01' },
    { pnl: null, sharpe: null, test_start: '2024-02-01' },
    { pnl: 50, sharpe: 0.8, test_start: '2024-03-01' },
  ]
  const data = mapSplits(splits)
  assert.equal(data[0].pnl, 100)
  assert.equal(data[0].cumPnl, 100)
  assert.equal(data[1].pnl, 0)
  assert.equal(data[1].cumPnl, 100)   // null row contributes 0
  assert.equal(data[2].pnl, 50)
  assert.equal(data[2].cumPnl, 150)   // not NaN
})

test('OosSplitChart: null sharpe maps to 0, not NaN', () => {
  const splits = [{ pnl: 75, sharpe: null, test_start: '2024-01-01' }]
  const data = mapSplits(splits)
  assert.equal(data[0].sharpe, 0)
})

test('OosSplitChart: normal rows unaffected by guard', () => {
  const splits = [
    { pnl: 200.567, sharpe: 1.234, test_start: '2024-01-01' },
    { pnl: -50.123, sharpe: -0.5, test_start: '2024-02-01' },
  ]
  const data = mapSplits(splits)
  assert.equal(data[0].pnl, 200.57)     // toFixed(2) rounds
  assert.equal(data[0].sharpe, 1.23)
  assert.equal(data[1].pnl, -50.12)
  // cumPnl accumulates raw pnlVal (pre-toFixed), rounded once at the end.
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
