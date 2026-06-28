import React from 'react'
import TermStructureChart from './TermStructureChart'

/**
 * Vol-term-structure panel. Renders the chart when there are at least two
 * (dte, iv) points, otherwise an explicit "unavailable" note so a thin option
 * chain doesn't make the panel silently vanish (which read as a broken page).
 * Returns null only when the metric is absent entirely (no analysis / older
 * payload) — i.e. `days` is not an array.
 */
export default function VolTermPanel({ days, ivs, earningsDte }) {
  if (!Array.isArray(days)) return null

  return (
    <div className="selector-panel selector-panel-vol-term">
      <div className="selector-panel-header">
        <h3>Vol Term Structure</h3>
        <span>Implied vol across expirations — steepness drives calendar carry; event premium shows near-term elevation.</span>
      </div>
      {days.length >= 2 ? (
        <TermStructureChart days={days} ivs={ivs} earningsDte={earningsDte} />
      ) : (
        <div className="empty-state">
          Term structure unavailable — the data provider returned a thin option
          chain (fewer than two expirations with valid ATM IV). This is usually
          transient; try again during market hours.
        </div>
      )}
    </div>
  )
}
