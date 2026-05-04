import React from 'react'

import { formatPct } from './formatters'

export default function SpreadChangeIndicator({ state, change }) {
  const normalized = state || 'new'
  if (normalized === 'new' || change == null) {
    return <span className="spread-drift spread-drift-new">new</span>
  }

  const directionLabel = normalized === 'improved'
    ? 'improved'
    : normalized === 'worsened'
      ? 'worsened'
      : 'unchanged'

  return (
    <span className={`spread-drift spread-drift-${normalized}`}>
      {directionLabel}
      {normalized !== 'unchanged' ? ` ${formatPct(Math.abs(change), 2)}` : ''}
    </span>
  )
}
