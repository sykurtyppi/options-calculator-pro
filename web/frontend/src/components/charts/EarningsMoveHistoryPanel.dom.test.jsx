// Rendered tests for EarningsMoveHistoryPanel (Vitest + jsdom).
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import EarningsMoveHistoryPanel from './EarningsMoveHistoryPanel'

const HISTORY = [
  { date: '2025-02-01', move_pct: 4.0, release_timing: 'after market close' },
  { date: '2025-05-02', move_pct: 9.0, release_timing: 'after market close' },
  { date: '2025-08-01', move_pct: 7.0, release_timing: 'after market close' },
  { date: '2025-11-01', move_pct: 11.0, release_timing: 'after market close' },
]

describe('EarningsMoveHistoryPanel', () => {
  test('renders the chart and a caption with a coherent median / implied / exceed count', () => {
    const { container } = render(<EarningsMoveHistoryPanel history={HISTORY} impliedMove={6} />)
    expect(screen.getByRole('heading', { name: /Historical Earnings Moves/i })).toBeInTheDocument()
    expect(container.querySelector('.vol-chart-wrapper')).not.toBeNull()
    // 3 of 4 moves (9, 7, 11) are >= implied 6.
    expect(screen.getByText(/3\/4/)).toBeInTheDocument()
    // Median is computed over the SAME 4 displayed bars: median([4,7,9,11]) = 8.0.
    expect(screen.getByText(/median/i)).toBeInTheDocument()
    expect(screen.getByText(/8\.0%/)).toBeInTheDocument()
  })

  test('takeaway flips with the exceed rate', () => {
    const { rerender } = render(<EarningsMoveHistoryPanel history={HISTORY} impliedMove={6} />)
    expect(screen.getByText(/priced move is on the low side/i)).toBeInTheDocument()
    // Implied above every historical move → priced high.
    rerender(<EarningsMoveHistoryPanel history={HISTORY} impliedMove={20} />)
    expect(screen.getByText(/priced move is on the high side/i)).toBeInTheDocument()
  })

  test('renders nothing when there is no dated history (fallback path)', () => {
    const { container } = render(<EarningsMoveHistoryPanel history={null} impliedMove={6} />)
    expect(container.firstChild).toBeNull()
  })
})
