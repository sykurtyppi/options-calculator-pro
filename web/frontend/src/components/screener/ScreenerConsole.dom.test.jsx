// Rendered-component regression test for ScreenerConsole (Vitest + jsdom).
//
// Locks in the fix where selecting a ranked row is a CHEAP action: clicking a
// row must update the selection (so the detail/calibration panel can render)
// WITHOUT firing the heavy single-ticker /api/edge/analyze call — neither via
// the onAnalyzeSymbol prop nor a direct fetch. Previously every row-click
// auto-ran that ~2s live-data analysis, so merely browsing the ranked list
// silently drained the per-IP analyze rate-limit budget and produced 429s.
// The full analysis stays behind the explicit "Full analysis →" button.
import React from 'react'
import { describe, test, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

// Track every /api/edge/analyze call the component makes directly, so the test
// can assert row-selection triggers none. vi.mock is hoisted; the array is
// declared with `var` so the hoisted factory can close over it without a TDZ.
// eslint-disable-next-line no-var
var analyzeCalls = []

vi.mock('../../lib/api', () => ({
  API_BASE: '',
  apiFetch: vi.fn(async (url) => {
    const u = String(url)
    if (u.includes('/api/edge/analyze')) {
      analyzeCalls.push(u)
      return { ok: true, status: 200, json: async () => ({}) }
    }
    if (u.includes('/api/screener/ranked')) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          rows: [
            {
              rank: 1, symbol: 'PYPL', earnings_date: '2026-07-28', dte: 6,
              release_timing: 'BMO', iv_rv_ratio: 0.80, atm_iv: 0.5, ts_ratio: 0.84,
              median_earnings_move_pct: 8.6, sample_size: 20, ranking_score: 0.85,
              status: 'ranked', error_note: null,
            },
            {
              rank: 2, symbol: 'QCOM', earnings_date: '2026-07-29', dte: 7,
              release_timing: 'AMC', iv_rv_ratio: 0.91, atm_iv: 0.7, ts_ratio: 0.79,
              median_earnings_move_pct: 6.8, sample_size: 20, ranking_score: 0.79,
              status: 'ranked', error_note: null,
            },
          ],
        }),
      }
    }
    // Everything else (calibration curve, etc.) → benign empty body so the
    // mount-time hooks and the detail panel don't throw under jsdom.
    return { ok: true, status: 200, json: async () => ({}) }
  }),
}))

import ScreenerConsole from './ScreenerConsole'

beforeEach(() => {
  analyzeCalls.length = 0
})

describe('ScreenerConsole — ranked-row selection is cheap', () => {
  test('clicking a ranked row selects it without firing a full analyze', async () => {
    const onAnalyzeSymbol = vi.fn()
    render(<ScreenerConsole apiBase="" onAnalyzeSymbol={onAnalyzeSymbol} />)

    // Wait for the ranked rows to render (the /api/screener/ranked fetch
    // resolves on mount for the default 'ranked' tab).
    const qcomCell = await screen.findByText('QCOM')

    // Sanity: mounting + the default first-row selection must not have run a
    // full analysis either.
    expect(onAnalyzeSymbol).not.toHaveBeenCalled()
    expect(analyzeCalls).toHaveLength(0)

    fireEvent.click(qcomCell)

    // The heavy single-ticker analysis must NOT fire on selection — this is
    // the whole point of the fix. Selection only drives the (cheap) detail +
    // calibration panel.
    expect(onAnalyzeSymbol).not.toHaveBeenCalled()
    expect(analyzeCalls).toHaveLength(0)
  })
})
