// Rendered tests for VolTermPanel (Vitest + jsdom).
//
// The panel must not silently vanish when the provider returns a thin chain —
// it should render an explicit "unavailable" note instead. It still returns
// nothing when the metric is absent (no analysis yet / older payload).
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import VolTermPanel from './VolTermPanel'

describe('VolTermPanel', () => {
  test('renders the unavailable note when the chain is thin (<2 points)', () => {
    render(<VolTermPanel days={[5]} ivs={[0.42]} earningsDte={3} />)
    expect(screen.getByText(/Term structure unavailable/i)).toBeInTheDocument()
    expect(screen.getByText(/thin option\s+chain/i)).toBeInTheDocument()
    // The chart wrapper must NOT be present in the thin case.
    expect(document.querySelector('.vol-chart-wrapper')).toBeNull()
  })

  test('renders the unavailable note for an empty term structure', () => {
    render(<VolTermPanel days={[]} ivs={[]} earningsDte={3} />)
    expect(screen.getByText(/Term structure unavailable/i)).toBeInTheDocument()
  })

  test('renders the chart (not the note) when there are >=2 points', () => {
    const { container } = render(
      <VolTermPanel days={[3, 9, 17]} ivs={[0.5, 0.42, 0.4]} earningsDte={3} />,
    )
    expect(container.querySelector('.vol-chart-wrapper')).not.toBeNull()
    expect(screen.queryByText(/Term structure unavailable/i)).toBeNull()
  })

  test('renders nothing when the metric is absent (days is not an array)', () => {
    const { container } = render(<VolTermPanel days={undefined} />)
    expect(container.firstChild).toBeNull()
  })
})
