import React from 'react'

import { expiryModeLabel } from './formatters'

const OPTIONS = [
  { value: 'front_after_earnings', shortLabel: 'Front Weekly' },
  { value: 'next_monthly_opex', shortLabel: 'Next Monthly OPEX' },
]

export default function ExpiryModeToggle({ value, onChange }) {
  return (
    <div className="expiry-mode-toggle" role="tablist" aria-label="Expiry methodology">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`expiry-mode-pill${value === option.value ? ' active' : ''}`}
          onClick={() => onChange(option.value)}
          title={expiryModeLabel(option.value)}
        >
          {option.shortLabel}
        </button>
      ))}
    </div>
  )
}
