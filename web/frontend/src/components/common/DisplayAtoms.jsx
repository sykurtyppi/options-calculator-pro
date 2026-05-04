import React from 'react'

export function Metric({ label, value, accent = false, tone = 'default', sub }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${accent ? 'accent' : ''} tone-${tone}`}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  )
}

export function SectionTitle({ children }) {
  return <h3 className="section-title">{children}</h3>
}

export function Badge({ children, variant = 'default' }) {
  return <span className={`badge badge-${variant}`}>{children}</span>
}
