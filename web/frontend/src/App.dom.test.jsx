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

describe('App — legacy block honesty (rendered)', () => {
  test('a reduced-evidence flag renders advisory wording, never a false score cap', async () => {
    render(<App />)

    fireEvent.change(screen.getByLabelText('Ticker'), { target: { value: 'AAPL' } })
    fireEvent.click(screen.getByRole('button', { name: /Run Edge Analysis/i }))

    // The advisory block sits inside the lazy LegacyAnalysisPanel Suspense, so
    // wait for the chunk to resolve and the text to appear. Assert on
    // body.textContent because the sentence is split by a <strong> node.
    await waitFor(
      () => expect(document.body.textContent).toMatch(/advisory evidence flag/i),
      { timeout: 5000 },
    )

    const body = document.body.textContent
    // Honest: the flag explicitly states the score is not reduced.
    expect(body).toMatch(/not reduced/i)
    // Never the false claim the honesty fix removed.
    expect(body).not.toMatch(/capped at/i)
    expect(body).not.toMatch(/score capped/i)
  })
})
