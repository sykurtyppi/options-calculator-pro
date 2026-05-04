import React from 'react'
import { buildVolSnapshotSummary } from './selectorViewModel'

export default function VolSnapshotPanel({ volSnapshot }) {
  const rows = buildVolSnapshotSummary(volSnapshot)
  if (!rows.length) return null

  return (
    <section className="selector-panel">
      <div className="selector-panel-header">
        <h3>Vol Snapshot</h3>
        <span>Canonical event-volatility state used by every downstream decision layer.</span>
      </div>

      <div className="snapshot-grid">
        {rows.slice(0, 12).map((row) => (
          <div className="snapshot-item" key={row.label}>
            <span>{row.label}</span>
            <strong>{row.value}</strong>
          </div>
        ))}
      </div>

      <details className="selector-detail-toggle">
        <summary>Show full snapshot detail</summary>
        <div className="selector-code-block">
          <pre>{JSON.stringify(volSnapshot, null, 2)}</pre>
        </div>
      </details>
    </section>
  )
}
