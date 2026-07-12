import React from 'react'

// FE-3: `provenance` distinguishes empirical/measured numbers (market data,
// realized OOS/forward stats) from score-derived/modeled diagnostics, which
// otherwise render in identical chrome. 'measured' | 'modeled' add a labeled
// chip + a left-border accent so a user can tell at a glance which is which.
const PROVENANCE_LABEL = { measured: 'measured', modeled: 'modeled' }

export function Metric({ label, value, accent = false, tone = 'default', sub, provenance }) {
  const prov = PROVENANCE_LABEL[provenance] ? provenance : null
  return (
    <div className={`metric-card${prov ? ` metric-card-${prov}` : ''}`}>
      <div className="metric-label">
        {label}
        {prov && <span className={`metric-provenance metric-provenance-${prov}`}>{PROVENANCE_LABEL[prov]}</span>}
      </div>
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
