// Rendered tests for DisplayAtoms (Vitest + jsdom).
import React from 'react'
import { describe, test, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Metric } from './DisplayAtoms'

describe('Metric provenance token (FE-3)', () => {
  test('modeled metric shows a "modeled" chip and the modeled card class', () => {
    const { container } = render(<Metric label="Expected Net Edge" value="+1.2%" provenance="modeled" />)
    expect(screen.getByText('modeled')).toBeInTheDocument()
    expect(container.querySelector('.metric-card-modeled')).not.toBeNull()
    expect(container.querySelector('.metric-card-measured')).toBeNull()
  })

  test('measured metric shows a "measured" chip and the measured card class', () => {
    const { container } = render(<Metric label="Win Rate" value="58%" provenance="measured" />)
    expect(screen.getByText('measured')).toBeInTheDocument()
    expect(container.querySelector('.metric-card-measured')).not.toBeNull()
  })

  test('untagged metric renders neither chip nor provenance class (backward compatible)', () => {
    const { container } = render(<Metric label="IV30" value="92%" />)
    expect(screen.queryByText(/^(measured|modeled)$/)).toBeNull()
    expect(container.querySelector('.metric-card')).not.toBeNull()
    expect(container.querySelector('.metric-card-modeled')).toBeNull()
    expect(container.querySelector('.metric-card-measured')).toBeNull()
  })

  test('an unknown provenance value is ignored, not rendered', () => {
    const { container } = render(<Metric label="X" value="1" provenance="guessed" />)
    expect(container.querySelector('.metric-provenance')).toBeNull()
    expect(container.querySelector('.metric-card-guessed')).toBeNull()
  })
})
