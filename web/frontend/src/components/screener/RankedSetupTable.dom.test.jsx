// Rendered-component tests for RankedSetupTable.
// Run under Vitest + jsdom (npm run test:dom). These assert the ACTUAL rendered
// DOM for the three row kinds the screener now emits: scored, upcoming
// (pipeline, no score), and the empty state — the fix that replaced the blank
// "0 setups" with a forward pipeline.
import React from 'react'
import { describe, test, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import RankedSetupTable from './RankedSetupTable'

const scoredRow = {
  rank: 1,
  symbol: 'NVDA',
  earnings_date: '2026-06-20',
  dte: 5,
  release_timing: 'AMC',
  iv_rv_ratio: 0.92,
  atm_iv: 55.2,
  ts_ratio: 0.88,
  median_earnings_move_pct: 9.5,
  sample_size: 10,
  ranking_score: 0.74,
  status: 'ranked',
  error_note: null,
}

const upcomingRow = {
  rank: 2,
  symbol: 'AAPL',
  earnings_date: '2026-07-30',
  dte: 40,
  release_timing: 'AMC',
  iv_rv_ratio: null,
  atm_iv: null,
  ts_ratio: null,
  median_earnings_move_pct: null,
  sample_size: null,
  ranking_score: null,
  status: 'upcoming',
  error_note: null,
}

describe('RankedSetupTable', () => {
  test('renders an informative empty state, not a blank "0 setups"', () => {
    const { container } = render(<RankedSetupTable rows={[]} />)
    expect(container.textContent).toMatch(/No upcoming earnings/i)
    expect(container.textContent).toMatch(/Weeks/)
    expect(container.textContent).toMatch(/DTE/)
  })

  test('renders a scored row with its setup score', () => {
    render(<RankedSetupTable rows={[scoredRow]} selectedSymbol={null} onSelect={() => {}} />)
    expect(screen.getByText('NVDA')).toBeInTheDocument()
    expect(screen.getByText('0.74')).toBeInTheDocument()
  })

  test('renders an upcoming row as pipeline (no score bar, "upcoming" label)', () => {
    render(<RankedSetupTable rows={[upcomingRow]} selectedSymbol={null} onSelect={() => {}} />)
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText(/upcoming/i)).toBeInTheDocument()
    // upcoming rows carry no score — the score number must not render
    expect(screen.queryByText('0.74')).not.toBeInTheDocument()
  })

  test('upcoming rows are still clickable (a real symbol to analyze)', () => {
    const onSelect = vi.fn()
    render(<RankedSetupTable rows={[upcomingRow]} selectedSymbol={null} onSelect={onSelect} />)
    fireEvent.click(screen.getByText('AAPL'))
    expect(onSelect).toHaveBeenCalledWith(upcomingRow)
  })
})
