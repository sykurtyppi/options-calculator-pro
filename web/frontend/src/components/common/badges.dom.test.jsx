// Rendered-component tests for the common badges.
// These run under Vitest + jsdom (npm run test:dom), separate from the
// node:test pure-function suites. They assert the ACTUAL rendered DOM, which is
// strictly stronger than the source-grep honesty guards in src/honesty.test.js.
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TickerTierBadge, VolRegimeBadge, releaseTimeBadge } from './badges'

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

describe('releaseTimeBadge', () => {
  // This badge is the single source of truth for release timing across BOTH
  // paths: the analyze view passes the raw snapshot string, the ranked screener
  // passes the normalized code. Both must map to the same BMO/AMC/Intraday badge.
  test('maps the raw analyze-path strings', () => {
    expect(render(releaseTimeBadge('before market open')).container.textContent).toBe('BMO')
    expect(render(releaseTimeBadge('after market close')).container.textContent).toBe('AMC')
    expect(render(releaseTimeBadge('during market hours')).container.textContent).toBe('Intraday')
  })

  test('maps the normalized screener codes', () => {
    expect(render(releaseTimeBadge('BMO')).container.textContent).toBe('BMO')
    expect(render(releaseTimeBadge('AMC')).container.textContent).toBe('AMC')
  })

  test('renders the raw text for unrecognised timing (e.g. UNKNOWN)', () => {
    expect(render(releaseTimeBadge('UNKNOWN')).container.textContent).toBe('UNKNOWN')
  })

  test('renders nothing when timing is missing', () => {
    expect(render(releaseTimeBadge(null)).container.firstChild).toBeNull()
    expect(render(releaseTimeBadge('')).container.firstChild).toBeNull()
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
