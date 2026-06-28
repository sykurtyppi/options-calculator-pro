// Rendered tests for AnalysisSkeleton (Vitest + jsdom).
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import AnalysisSkeleton from './AnalysisSkeleton'

describe('AnalysisSkeleton', () => {
  test('is an accessible busy status with screen-reader text', () => {
    render(<AnalysisSkeleton />)
    const status = screen.getByRole('status')
    expect(status).toHaveAttribute('aria-busy', 'true')
    expect(screen.getByText(/Running analysis/i)).toBeInTheDocument()
  })

  test('renders shimmer placeholders', () => {
    const { container } = render(<AnalysisSkeleton />)
    expect(container.querySelectorAll('.skel').length).toBeGreaterThanOrEqual(4)
    expect(container.querySelector('.skel-card')).not.toBeNull()
  })
})
