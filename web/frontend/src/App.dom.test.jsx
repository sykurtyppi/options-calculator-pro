// App-level rendered honesty test (Vitest + jsdom).
//
// The isolated component tests cover the new edge/selector cards, but the
// reworded advisory messages from the frontend honesty fix live INLINE in
// App.jsx's legacy metrics block (inside the lazy LegacyAnalysisPanel Suspense).
// This drives the real App through a mocked analyze call with the
// fallback-move-model flag set, and asserts the rendered DOM shows the honest
// advisory wording and never a false "score capped" claim.
import React from 'react'
import { describe, test, expect, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

// vi.mock is hoisted; keep the payload INSIDE the factory to avoid a TDZ on a
// module-scope const. The analyze endpoint returns a result whose metrics carry
// the reduced-evidence flag; every other endpoint returns a benign empty body
// so the mount-time hooks (watchlist, alert config, warehouse) don't throw.
vi.mock('./lib/api', () => ({
  API_BASE: '',
  apiFetch: vi.fn(async (url) => {
    if (String(url).includes('/api/edge/analyze')) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          recommendation: 'No Trade',
          confidence_pct: 50,
          setup_score: 50,
          generated_at: '2026-06-04T12:00:00Z',
          selector_output: null,
          structure_scorecards: [],
          vol_snapshot: null,
          metrics: {
            // Trips the "Reduced-Evidence Signal" block. The flag is real and
            // must still surface — but as an advisory caveat, NOT a score cap.
            fallback_move_model_flag: true,
            days_to_earnings: 5,
            vol_regime: 'Normal',
            data_source: 'yfinance',
            // Deliberately omit term_structure_days / structure_payoff so the
            // recharts charts don't render under jsdom.
          },
        }),
      }
    }
    return { ok: true, status: 200, json: async () => ({}) }
  }),
}))

import App from './App'

async function _runAnalysis() {
  render(<App />)
  fireEvent.change(screen.getByLabelText('Ticker'), { target: { value: 'AAPL' } })
  fireEvent.click(screen.getByRole('button', { name: /Run Edge Analysis/i }))
  // Wait for the result block (and its tab bar) to render.
  await screen.findByRole('tab', { name: /^Decision$/i }, { timeout: 5000 })
}

describe('App — decision-first result tabs', () => {
  test('lands on the Decision tab; the legacy metrics block is not shown by default', async () => {
    await _runAnalysis()
    // Three result views are offered.
    expect(screen.getByRole('tab', { name: /^Decision$/i })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: /Evidence & regime/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /Full metrics/i })).toBeInTheDocument()
    // The dense legacy block (e.g. "Edge & Expectancy") is NOT in the default view.
    expect(document.body.textContent).not.toMatch(/Edge & Expectancy/i)
    // The live forward-evidence strip is surfaced in the Decision view.
    await waitFor(() => expect(document.body.textContent).toMatch(/Live forward evidence/i))
    // With selector_output null (this mock), the Decision tab is not silently
    // empty — it shows a fallback that points at the other tabs.
    expect(document.body.textContent).toMatch(/No structured decision is available/i)
  })

  test('the Full metrics tab shows its content directly (not behind a second collapse)', async () => {
    await _runAnalysis()
    fireEvent.click(screen.getByRole('tab', { name: /Full metrics/i }))
    // The legacy panel must render OPEN (no nested <details> to click again).
    await waitFor(() => expect(document.body.textContent).toMatch(/Edge & Expectancy/i))
    expect(document.querySelector('.legacy-analysis-panel')).not.toBeNull()
    expect(document.querySelector('.legacy-analysis-panel details')).toBeNull()
  })

  test('the operational tail (warehouse / OOS / diagnostics) is collapsed by default', async () => {
    await _runAnalysis()
    // The disclosure exists but its contents are not mounted yet.
    expect(screen.getByRole('button', { name: /Tools & diagnostics/i })).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(/Walk-Forward OOS Report Card/i)
    // Expanding it reveals the tail.
    fireEvent.click(screen.getByRole('button', { name: /Tools & diagnostics/i }))
    await waitFor(() => expect(document.body.textContent).toMatch(/Walk-Forward OOS Report Card/i))
  })

  test('the Full metrics tab reveals the legacy block, with honest advisory wording and no false cap', async () => {
    await _runAnalysis()
    fireEvent.click(screen.getByRole('tab', { name: /Full metrics/i }))

    // The reduced-evidence advisory block lives in the lazy LegacyAnalysisPanel.
    await waitFor(
      () => expect(document.body.textContent).toMatch(/advisory evidence flag/i),
      { timeout: 5000 },
    )
    const body = document.body.textContent
    expect(body).toMatch(/not reduced/i)          // honest: score not reduced
    expect(body).not.toMatch(/capped at/i)        // never the false claim
    expect(body).not.toMatch(/score capped/i)
  })
})
