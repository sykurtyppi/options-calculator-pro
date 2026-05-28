// Tests for OosSplitChart's malformed-row guard (FE2).
//
// The component may receive split rows where pnl is null or a non-numeric
// string ("n/a"), sharpe is non-numeric, test_start is a non-string, or the
// row object itself is null. Any of these caused a TypeError or NaN
// propagation before the fix. After the fix, a safeFinite helper coerces bad
// values to 0 and typeof guards protect string operations.
//
// We can't import JSX directly (no Babel in node:test), so we mirror the
// exact data-mapping expression from OosSplitChart.jsx. If the mapping
// changes in the component, update safeFinite and mapSplits here to match.

import test from 'node:test'
import assert from 'node:assert/strict'


// Mirror of the component's safeFinite helper.
function safeFinite(v) {
  const n = Number(v ?? 0)
  return isFinite(n) ? n : 0
}

// Mirror of the component's data-mapping loop (post-fix).
function mapSplits(splitsDetail) {
  let cumPnl = 0
  return splitsDetail.map((s, i) => {
    const pnlVal = safeFinite(s?.pnl)
    const sharpeVal = safeFinite(s?.sharpe)
    cumPnl += pnlVal
    return {
      split: i + 1,
      pnl: Number(pnlVal.toFixed(2)),
      cumPnl: Number(cumPnl.toFixed(2)),
      label: typeof s?.test_start === 'string' ? s.test_start.slice(0, 7) : `S${i + 1}`,
      sharpe: Number(sharpeVal.toFixed(2)),
      winRate: s?.win_rate != null ? `${(s.win_rate * 100).toFixed(0)}%` : 'n/a',
      trades: s?.trades,
    }
  })
}


// ── safeFinite helper ────────────────────────────────────────────────────

test('safeFinite: null → 0', () => assert.equal(safeFinite(null), 0))
test('safeFinite: undefined → 0', () => assert.equal(safeFinite(undefined), 0))
test('safeFinite: "n/a" → 0', () => assert.equal(safeFinite('n/a'), 0))
test('safeFinite: NaN → 0', () => assert.equal(safeFinite(NaN), 0))
test('safeFinite: Infinity → 0', () => assert.equal(safeFinite(Infinity), 0))
test('safeFinite: 0 → 0 (zero preserved)', () => assert.equal(safeFinite(0), 0))
test('safeFinite: 1.5 → 1.5 (normal value preserved)', () => assert.equal(safeFinite(1.5), 1.5))
test('safeFinite: -50.12 → -50.12 (negative preserved)', () => assert.equal(safeFinite(-50.12), -50.12))


// ── pnl guard ────────────────────────────────────────────────────────────

test('OosSplitChart: null pnl maps to 0, does not throw', () => {
  const data = mapSplits([{ pnl: null, sharpe: null, test_start: '2024-01-01' }])
  assert.equal(data[0].pnl, 0)
  assert.equal(data[0].cumPnl, 0)
})

test('OosSplitChart: non-numeric pnl ("n/a") maps to 0, cumPnl stays finite', () => {
  // Pre-fix: Number("n/a" ?? 0) = NaN because ?? only substitutes null/undefined.
  // NaN propagates to cumPnl and all subsequent splits.
  const splits = [
    { pnl: 100,   sharpe: 1.0, test_start: '2024-01-01' },
    { pnl: 'n/a', sharpe: 0.5, test_start: '2024-02-01' },
    { pnl: 50,    sharpe: 0.8, test_start: '2024-03-01' },
  ]
  const data = mapSplits(splits)
  assert.equal(data[1].pnl, 0)
  assert.equal(data[1].cumPnl, 100)   // "n/a" contributes 0, not NaN
  assert.equal(data[2].cumPnl, 150)   // not NaN
  assert.ok(isFinite(data[2].cumPnl))
})

test('OosSplitChart: null pnl does not propagate NaN to cumPnl', () => {
  const splits = [
    { pnl: 100, sharpe: 1.2, test_start: '2024-01-01' },
    { pnl: null, sharpe: null, test_start: '2024-02-01' },
    { pnl: 50,  sharpe: 0.8, test_start: '2024-03-01' },
  ]
  const data = mapSplits(splits)
  assert.equal(data[1].pnl, 0)
  assert.equal(data[1].cumPnl, 100)
  assert.equal(data[2].cumPnl, 150)
})


// ── sharpe guard ─────────────────────────────────────────────────────────

test('OosSplitChart: non-numeric sharpe ("n/a") maps to 0, does not throw', () => {
  // Pre-fix: ("n/a" ?? 0).toFixed → TypeError (strings have no toFixed).
  const data = mapSplits([{ pnl: 75, sharpe: 'n/a', test_start: '2024-01-01' }])
  assert.equal(data[0].sharpe, 0)
})

test('OosSplitChart: null sharpe maps to 0', () => {
  const data = mapSplits([{ pnl: 75, sharpe: null, test_start: '2024-01-01' }])
  assert.equal(data[0].sharpe, 0)
})

test('OosSplitChart: zero sharpe is preserved (not masked as "bad")', () => {
  const data = mapSplits([{ pnl: 10, sharpe: 0, test_start: '2024-01-01' }])
  assert.equal(data[0].sharpe, 0)
})


// ── null row guard ───────────────────────────────────────────────────────

test('OosSplitChart: null row object maps to safe zero row, does not throw', () => {
  const data = mapSplits([null])
  assert.equal(data[0].pnl, 0)
  assert.equal(data[0].sharpe, 0)
  assert.equal(data[0].label, 'S1')
  assert.equal(data[0].winRate, 'n/a')
})


// ── label guard ──────────────────────────────────────────────────────────

test('OosSplitChart: string test_start uses slice', () => {
  const data = mapSplits([{ pnl: 0, sharpe: 0, test_start: '2024-06-15' }])
  assert.equal(data[0].label, '2024-06')
})

test('OosSplitChart: non-string test_start (number) falls back to S-index label', () => {
  // Pre-fix: truthy check passes for 123, then (123).slice(0, 7) → TypeError.
  const data = mapSplits([{ pnl: 0, sharpe: 0, test_start: 123 }])
  assert.equal(data[0].label, 'S1')
})

test('OosSplitChart: null test_start falls back to S-index label', () => {
  const data = mapSplits([{ pnl: 0, sharpe: 0, test_start: null }])
  assert.equal(data[0].label, 'S1')
})


// ── normal rows unaffected ───────────────────────────────────────────────

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
