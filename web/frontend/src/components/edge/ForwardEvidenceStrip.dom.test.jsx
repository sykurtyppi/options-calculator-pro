import React from 'react'
import { describe, test, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'

const apiFetch = vi.fn()
vi.mock('../../lib/api', () => ({ apiFetch: (...a) => apiFetch(...a) }))

import ForwardEvidenceStrip from './ForwardEvidenceStrip'

function _resp(payload) {
  return { ok: true, status: 200, json: async () => payload }
}

describe('ForwardEvidenceStrip', () => {
  beforeEach(() => apiFetch.mockReset())

  test('shows the accruing state when no paper trades have resolved', async () => {
    apiFetch.mockResolvedValue(_resp({ open_outcome_count: 2, resolved_outcome_count: 0 }))
    render(<ForwardEvidenceStrip apiBase="" />)
    await waitFor(() => expect(screen.getByText(/Live forward evidence/i)).toBeInTheDocument())
    expect(document.body.textContent).toMatch(/nothing has resolved yet/i)
    expect(document.body.textContent).toMatch(/2 open/)
    // Reframed to sell the honesty (the marquee differentiator), not read as "no results".
    expect(document.body.textContent).toMatch(/wins and losses/i)
  })

  test('shows resolved stats once paper outcomes exist', async () => {
    apiFetch.mockResolvedValue(_resp({
      resolved_outcome_count: 12,
      open_outcome_count: 3,
      performance_summary: { n: 12, win_rate: 0.58, avg_realized_return_pct: 4.2 },
    }))
    render(<ForwardEvidenceStrip apiBase="" />)
    await waitFor(() => expect(screen.getByText(/Live forward evidence/i)).toBeInTheDocument())
    const body = document.body.textContent
    expect(body).toMatch(/12.*resolved/)
    expect(body).toMatch(/58%/)              // win rate
    expect(body).toMatch(/paper · not execution-grade/i)  // honest label kept
  })

  test('treats resolved-without-summary-stats as accruing, not n/a-filled stats', async () => {
    // counts and stats come from different backend paths; guard against showing
    // "12 resolved · n/a win · realized n/a".
    apiFetch.mockResolvedValue(_resp({ resolved_outcome_count: 12, open_outcome_count: 0 }))
    render(<ForwardEvidenceStrip apiBase="" />)
    await waitFor(() => expect(screen.getByText(/Live forward evidence/i)).toBeInTheDocument())
    expect(document.body.textContent).toMatch(/nothing has resolved yet/i)
    expect(document.body.textContent).not.toMatch(/n\/a win/i)
  })

  test('fails quiet (renders nothing) when the endpoint errors', async () => {
    apiFetch.mockResolvedValue({ ok: false, status: 500, json: async () => ({}) })
    const { container } = render(<ForwardEvidenceStrip apiBase="" />)
    // Nothing rendered, no throw.
    await waitFor(() => expect(container.querySelector('.forward-evidence-strip')).toBeNull())
  })
})
