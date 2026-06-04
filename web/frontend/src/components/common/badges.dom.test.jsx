// Rendered-component tests for the common badges.
// These run under Vitest + jsdom (npm run test:dom), separate from the
// node:test pure-function suites. They assert the ACTUAL rendered DOM, which is
// strictly stronger than the source-grep honesty guards in src/honesty.test.js.
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TickerTierBadge, VolRegimeBadge } from './badges'

describe('TickerTierBadge', () => {
  test('renders the human tier label', () => {
    render(<TickerTierBadge tier="mega_cap" />)
    expect(screen.getByText('Mega-Cap')).toBeInTheDocument()
  })

  test('does NOT render the (unapplied) tier multiplier', () => {
    // Honesty: the calibration multiplier is computed for audit only and never
    // applied to the score. The badge must never show a "· NN%" multiplier,
    // even if a stray mult prop is passed.
    const { container } = render(<TickerTierBadge tier="mega_cap" mult={0.85} />)
    expect(container.textContent).toBe('Mega-Cap')
    expect(container.textContent).not.toMatch(/%/)
    expect(container.textContent).not.toMatch(/85/)
  })

  test('renders nothing for unknown / missing tier', () => {
    const { container: a } = render(<TickerTierBadge tier="unknown" />)
    expect(a.firstChild).toBeNull()
    const { container: b } = render(<TickerTierBadge tier={null} />)
    expect(b.firstChild).toBeNull()
  })
})

describe('VolRegimeBadge', () => {
  test('renders regime with percentile when provided', () => {
    render(<VolRegimeBadge regime="High" pct={82} />)
    expect(screen.getByText(/Vol High/)).toBeInTheDocument()
    expect(screen.getByText(/82th pct/)).toBeInTheDocument()
  })

  test('renders nothing for unknown regime', () => {
    const { container } = render(<VolRegimeBadge regime="unknown" />)
    expect(container.firstChild).toBeNull()
  })
})
