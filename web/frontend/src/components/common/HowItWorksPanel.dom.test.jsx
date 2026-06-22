// Rendered-component tests for the new-user explainer (HowItWorksPanel).
// Run under Vitest + jsdom (npm run test:dom).
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import HowItWorksPanel from './HowItWorksPanel'

describe('HowItWorksPanel', () => {
  test('is collapsed by default — explainer content is not in the DOM', () => {
    // Uses explicit open state (not <details>), so collapsed truly means absent.
    render(<HowItWorksPanel apiBase="http://x" />)
    expect(screen.getByRole('button', { name: /learn how this works/i })).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: /how this works/i })).not.toBeInTheDocument()
    expect(screen.queryByText(/not a calibrated win rate/i)).not.toBeInTheDocument()
  })

  test('expands to plain-language explainer + glossary on click', () => {
    render(<HowItWorksPanel apiBase="http://x" />)
    fireEvent.click(screen.getByRole('button', { name: /learn how this works/i }))

    const region = screen.getByRole('region', { name: /how this works/i })
    expect(region).toBeInTheDocument()
    // Plain value prop, no jargon-first
    expect(region.textContent).toMatch(/options look mispriced/i)
    // Glossary defines the acronyms the screener uses
    expect(screen.getByText('DTE')).toBeInTheDocument()
    expect(screen.getByText(/TS \(term structure\)/i)).toBeInTheDocument()
    // Honesty framing is front-and-center
    expect(region.textContent).toMatch(/show forward results even when they’re bad/i)
  })

  test('keeps the engineer-facing architecture link as a secondary affordance', () => {
    render(<HowItWorksPanel apiBase="http://api.test" />)
    fireEvent.click(screen.getByRole('button', { name: /learn how this works/i }))
    const link = screen.getByRole('link', { name: /technical architecture/i })
    expect(link).toHaveAttribute('href', 'http://api.test/product-docs/architecture.md')
  })

  test('toggles closed again', () => {
    render(<HowItWorksPanel apiBase="http://x" />)
    const btn = screen.getByRole('button', { name: /learn how this works/i })
    fireEvent.click(btn)
    expect(screen.getByRole('region', { name: /how this works/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /hide explainer/i }))
    expect(screen.queryByRole('region', { name: /how this works/i })).not.toBeInTheDocument()
  })
})
