import React from 'react'
import { describe, test, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'

const apiFetch = vi.fn()
vi.mock('../../lib/api', () => ({ apiFetch: (...a) => apiFetch(...a) }))

import EvidenceReportPanel from './EvidenceReportPanel'

function _resp(payload) {
  return { ok: true, status: 200, json: async () => payload }
}

describe('EvidenceReportPanel integrity sections', () => {
  beforeEach(() => apiFetch.mockReset())

  test('shows picked vs skipped universe results, exit attrition and excluded evidence', async () => {
    apiFetch.mockResolvedValue(_resp({
      universe_shadow: {
        events_recorded: 12,
        open: 3,
        entry_attrition: { entered_after_retry: 2, not_entered: 1, not_entered_by_reason: { no_option_expiries: 1 } },
        by_baseline: {
          always_otm_strangle: {
            all_events: { n: 9, avg_realized_return_pct: -2 },
            selector_actionable: { n: 3, avg_realized_return_pct: 4 },
            selector_not_actionable: { n: 6, avg_realized_return_pct: -5 },
          },
        },
      },
      exit_attrition: {
        selector: { resolved: 8, exit_missing: 2, attrition_rate: 0.2, by_reason: { booked_contract_unavailable: 2 } },
        baselines: { resolved: 20, exit_missing: 0, attrition_rate: 0, by_reason: {} },
      },
      invalidated_outcomes: { n: 2, by_reason: { 'exit_repriced_unheld_strike: 265 call': 2 } },
      legacy_repriced_baselines: { n: 3, by_exit_repricing: { rediscovered_legacy: 3 } },
    }))
    render(<EvidenceReportPanel apiBase="" />)
    await waitFor(() => expect(screen.getByText(/Did the Selector's Skips Pay/i)).toBeInTheDocument())
    const body = document.body.textContent
    expect(body).toMatch(/Always OTM strangle/)
    expect(body).toMatch(/\+9\.0%/)                       // picked minus skipped
    expect(body).toMatch(/2 missing \/ 8 resolved · 20%/)
    expect(body).toMatch(/booked_contract_unavailable/)
    expect(body).toMatch(/Invalidated selector outcomes/)
    expect(body).toMatch(/exit_repriced_unheld_strike/)
    expect(body).toMatch(/Exit re-discovered strikes \(pre-fix\)/)
    expect(body).toMatch(/Entered after a retry/)
  })

  test('explains an empty universe cohort instead of rendering an empty table', async () => {
    apiFetch.mockResolvedValue(_resp({ universe_shadow: { events_recorded: 4, open: 4, by_baseline: {} } }))
    render(<EvidenceReportPanel apiBase="" />)
    await waitFor(() => expect(screen.getByText(/No resolved universe events yet/i)).toBeInTheDocument())
    expect(document.body.textContent).toMatch(/4 recorded, 4 open/)
  })
})
