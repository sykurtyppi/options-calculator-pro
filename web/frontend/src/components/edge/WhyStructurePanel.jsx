import React from 'react'
import { buildTrustDetails, formatStructureLabel } from './selectorViewModel'

export default function WhyStructurePanel({ selectorOutput, scorecards = [], volSnapshot = null }) {
  if (!selectorOutput) return null

  const whyNot = Object.entries(selectorOutput.why_not_others || {})
  const trustDetails = buildTrustDetails(selectorOutput, scorecards, volSnapshot)
  const runnerUpLabel = (selectorOutput.runner_up_structures || [])
    .map((value) => formatStructureLabel(value))
    .join(', ')

  return (
    <details className="selector-panel selector-trust-details">
      <summary>
        <span>Why this decision?</span>
        <em>Structure fit, risks, provenance</em>
      </summary>
      <div className="selector-explain-grid selector-explain-grid-compact">
        <div className="selector-panel selector-panel-nested">
          <div className="selector-panel-header">
            <h3>Why This Structure Fits</h3>
            {selectorOutput.best_structure && <span>{formatStructureLabel(selectorOutput.best_structure)}</span>}
          </div>
          <ul className="selector-bullet-list">
            {(selectorOutput.why_this_structure || []).slice(0, 5).map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </div>

        <div className="selector-panel selector-panel-nested">
          <div className="selector-panel-header">
            <h3>Trust Notes</h3>
            <span>Short audit trail</span>
          </div>
          <ul className="selector-bullet-list selector-bullet-list-compact">
            {trustDetails.map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </div>

        <div className="selector-panel selector-panel-nested">
          <div className="selector-panel-header">
            <h3>Why Alternatives Trail</h3>
            <span>{runnerUpLabel || 'All alternatives shown'}</span>
          </div>
          <div className="selector-why-not-list">
            {whyNot.map(([structure, reasons]) => (
              <details className="selector-why-not-item" key={structure}>
                <summary>
                  <span>{formatStructureLabel(structure)}</span>
                  <span>{Array.isArray(reasons) ? reasons.length : 0} notes</span>
                </summary>
                <ul className="selector-bullet-list selector-bullet-list-compact">
                  {(reasons || []).map((reason, index) => <li key={index}>{reason}</li>)}
                </ul>
              </details>
            ))}
          </div>
        </div>
      </div>
    </details>
  )
}
