import React from 'react'

export default function QualificationBadge({ status }) {
  const normalized = String(status || 'EXCLUDED').toUpperCase()
  return <span className={`qualification-badge status-${normalized.toLowerCase()}`}>{normalized}</span>
}
