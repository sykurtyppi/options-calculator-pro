import React from 'react'
import {
  buildDecisionQualityBreakdown,
  buildDecisionStability,
  buildEvidenceSummary,
  buildHistoricalAnalogSummary,
} from './selectorViewModel'

export default function EvidenceQualityPanel({ selectorOutput, scorecards, volSnapshot }) {
  const items = buildEvidenceSummary(selectorOutput, scorecards, volSnapshot)
  const breakdown = buildDecisionQualityBreakdown(selectorOutput, scorecards, volSnapshot)
  const stability = buildDecisionStability(selectorOutput, scorecards)
  const analog = buildHistoricalAnalogSummary(selectorOutput, scorecards)
  if (!items.length) return null

  return (
    <section className="selector-panel">
      <div className="selector-panel-header">
        <h3>Decision Evidence</h3>
        <span>Explains what supports the recommendation and how stable it is.</span>
      </div>
      <div className="decision-evidence-top">
        <div className="decision-quality-card">
          <span>Decision Quality</span>
          <strong>{selectorOutput ? `${Math.round(Number(selectorOutput.confidence_pct || 0))}%` : 'n/a'}</strong>
          <em>Blend of score strength, separation, data quality, and evidence depth.</em>
        </div>
        <div className={`decision-stability-card decision-stability-card-${stability.tone}`}>
          <span>Decision Stability</span>
          <strong>{stability.label}</strong>
          <em>{stability.detail}</em>
          <small>Gap vs runner-up: {stability.gapLabel}</small>
        </div>
        <div className="decision-quality-card">
          <span>{analog.label}</span>
          <strong>{analog.value}</strong>
          <em>{analog.detail}</em>
        </div>
      </div>

      <div className="decision-breakdown-list">
        {breakdown.map((item) => (
          <div className="decision-breakdown-item" key={item.label}>
            <div className="decision-breakdown-head">
              <span>{item.label}</span>
              <strong>{item.valueLabel}</strong>
            </div>
            <div className="decision-breakdown-bar">
              <div className="decision-breakdown-fill" style={{ width: `${item.valuePct}%` }} />
            </div>
            <em>{item.detail}</em>
          </div>
        ))}
      </div>

      <div className="evidence-grid">
        {items.map((item) => (
          <div className="evidence-item" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
            <em>{item.detail}</em>
          </div>
        ))}
      </div>
    </section>
  )
}
