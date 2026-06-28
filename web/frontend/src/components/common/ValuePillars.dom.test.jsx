// Rendered tests for ValuePillars (Vitest + jsdom).
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ValuePillars from './ValuePillars'

describe('ValuePillars', () => {
  test('renders three scannable value pillars', () => {
    const { container } = render(<ValuePillars />)
    expect(container.querySelectorAll('.value-pillar')).toHaveLength(3)
    expect(screen.getByText(/Spot mispriced earnings options/i)).toBeInTheDocument()
    expect(screen.getByText(/A curated universe/i)).toBeInTheDocument()
  })

  test('leads the trust pillar with the honesty wedge (losers included)', () => {
    render(<ValuePillars />)
    expect(screen.getByText(/Proof, not promises/i)).toBeInTheDocument()
    expect(screen.getByText(/losers included/i)).toBeInTheDocument()
  })
})
