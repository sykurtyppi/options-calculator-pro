// Rendered-component tests for SelectorDecisionCard — the primary, honest
// decision surface. Runs under Vitest + jsdom (npm run test:dom).
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import SelectorDecisionCard from './SelectorDecisionCard'

const baseOutput = {
  recommendation: 'Candidate',
  best_structure: 'otm_strangle',
  confidence_pct: 62,
  expected_edge_pct: 2.1,
  expected_return_pct: 4.0,
  earnings_date: '2026-07-31',
  release_timing: 'after market close',
  primary_thesis: 'Pre-earnings IV expansion supported by a steep term structure.',
  primary_risks: ['Muted move lets theta dominate', 'Spread widening on entry'],
}

describe('SelectorDecisionCard', () => {
  test('renders nothing without a selector output', () => {
    const { container } = render(<SelectorDecisionCard selectorOutput={null} />)
    expect(container.firstChild).toBeNull()
  })

  test('shows the recommendation, structure title, and confidence', () => {
    render(<SelectorDecisionCard selectorOutput={baseOutput} />)
    expect(screen.getByText('Candidate')).toBeInTheDocument()
    expect(screen.getByText('Otm Strangle')).toBeInTheDocument()
    expect(screen.getByText('62%')).toBeInTheDocument()
  })

  test('confidence is qualified as evidence-backed, NOT probability', () => {
    render(<SelectorDecisionCard selectorOutput={baseOutput} />)
    expect(screen.getByText(/evidence-backed, not probability/i)).toBeInTheDocument()
  })

  test('signals are labelled modeled / score-derived, not empirical forecasts', () => {
    render(<SelectorDecisionCard selectorOutput={baseOutput} />)
    expect(screen.getByText(/modeled, not empirical/i)).toBeInTheDocument()
    expect(screen.getByText(/score-derived only/i)).toBeInTheDocument()
  })

  test('always carries the paper/research + not-a-forecast trust badges', () => {
    render(<SelectorDecisionCard selectorOutput={baseOutput} />)
    expect(screen.getByText(/Paper \/ Research only/i)).toBeInTheDocument()
    expect(screen.getByText(/Score-derived \(not a forecast\)/i)).toBeInTheDocument()
  })

  test('never renders a false score-cap or applied-multiplier claim', () => {
    const { container } = render(<SelectorDecisionCard selectorOutput={baseOutput} />)
    expect(container.textContent).not.toMatch(/capped/i)
    expect(container.textContent).not.toMatch(/adjusted via/i)
    expect(container.textContent).not.toMatch(/Combined Multiplier/i)
  })

  test('No Trade renders the abstain framing', () => {
    render(<SelectorDecisionCard selectorOutput={{ ...baseOutput, recommendation: 'No Trade', best_structure: null }} />)
    expect(screen.getByText('No Trade')).toBeInTheDocument()
    expect(screen.getByText('Edge not sufficient after costs')).toBeInTheDocument()
  })
})
