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
  atm_iv: 0.552,   // decimal, as the API sends it (55.2%) — NOT already-percent
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

  test('column headers carry plain-language tooltips (newcomer comprehension)', () => {
    render(<RankedSetupTable rows={[scoredRow]} selectedSymbol={null} onSelect={() => {}} />)
    // The opaque acronyms must be explained on hover, not left bare.
    expect(screen.getByText('DTE').closest('th')).toHaveAttribute('title', expect.stringMatching(/days to earnings/i))
    expect(screen.getByText('TS').closest('th')).toHaveAttribute('title', expect.stringMatching(/term-structure/i))
    expect(screen.getByText('Setup Score').closest('th')).toHaveAttribute('title', expect.stringMatching(/not a calibrated win rate/i))
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

  test('regime-conditioned row flags its low IV/RV as warn, not green (DD-4)', () => {
    // A low IV/RV normally reads green ("cheap"); when the cheapness rests on
    // an elevated-vol regime the ranking discounts, it must be flagged (warn
    // color + ⚠ marker), not shown as a favorable cheap-vol signal.
    const conditioned = {
      ...scoredRow, symbol: 'PYPL', iv_rv_ratio: 0.82, iv_rv_har: 1.45,
      rv_percentile_rank: 87, iv_regime_conditioned: true,
    }
    const { container } = render(
      <RankedSetupTable rows={[conditioned]} selectedSymbol={null} onSelect={() => {}} />,
    )
    expect(container.textContent).toMatch(/0\.82\s*⚠/)
    const ivrvCell = [...container.querySelectorAll('tbody td')].find((td) => /0\.82/.test(td.textContent))
    expect(ivrvCell.style.color).toBe('var(--warn)')
  })

  test('non-conditioned low IV/RV still reads green (no false alarm)', () => {
    const cheap = { ...scoredRow, iv_rv_ratio: 0.82, iv_regime_conditioned: false }
    const { container } = render(
      <RankedSetupTable rows={[cheap]} selectedSymbol={null} onSelect={() => {}} />,
    )
    expect(container.textContent).not.toMatch(/⚠/)
    const ivrvCell = [...container.querySelectorAll('tbody td')].find((td) => /0\.82/.test(td.textContent))
    expect(ivrvCell.style.color).toBe('var(--pos)')
  })

  test('upcoming row never paints a null metric green (null < 1.0 coercion guard)', () => {
    // Regression: `null < 1.0` is true in JS, which used to color the "—"
    // em-dash cells green (signaling a favorable IV/RV or TS the row doesn't have).
    const { container } = render(
      <RankedSetupTable rows={[upcomingRow]} selectedSymbol={null} onSelect={() => {}} />,
    )
    const cells = [...container.querySelectorAll('tbody td')]
    const green = 'rgb(46, 160, 67)' // #2ea043
    const greenCells = cells.filter((td) => td.style.color === green)
    expect(greenCells).toHaveLength(0)
  })
})

describe('RankedSetupTable · ATM IV units', () => {
  // Regression: atm_iv arrives as a DECIMAL from the API
  // (screener_service.iv_front = snapshot.near_term_atm_iv -> "iv30" -> atm_iv).
  // It was rendered with a bare '%' suffix, so a realistic 34.1% implied vol
  // displayed as "0.3%" — an impossible value, in a financial product.
  // The old fixture hid this by supplying an already-percent 55.2.
  const decimalIvRow = {
    rank: 1, symbol: 'COST', earnings_date: '2026-06-20', dte: 5,
    release_timing: 'AMC', iv_rv_ratio: 0.92,
    atm_iv: 0.34076871872222675,   // the exact value observed from the live API
    ts_ratio: 0.88, median_earnings_move_pct: 9.5, sample_size: 10,
    ranking_score: 0.74, status: 'ranked', error_note: null,
  }

  test('renders a decimal IV as a real percentage, not a fraction of one', () => {
    const { container } = render(
      <RankedSetupTable rows={[decimalIvRow]} selectedSymbol={null} onSelect={() => {}} />
    )
    expect(container.textContent).toMatch(/34\.1%/)
    expect(container.textContent).not.toMatch(/0\.3%/)
  })

  test('already-percent columns are unaffected by the IV fix', () => {
    const { container } = render(
      <RankedSetupTable rows={[decimalIvRow]} selectedSymbol={null} onSelect={() => {}} />
    )
    // median_earnings_move_pct is ALREADY a percent (9.5 means 9.5%) and must
    // not be multiplied — the two units keep separate formatters.
    expect(container.textContent).toMatch(/9\.5%/)
  })

  test('a null IV still renders the placeholder, not "NaN%"', () => {
    const { container } = render(
      <RankedSetupTable rows={[{ ...decimalIvRow, atm_iv: null }]} selectedSymbol={null} onSelect={() => {}} />
    )
    expect(container.textContent).not.toMatch(/NaN/)
  })
})
