import React from 'react'
import {
  buildTrustBadges,
  formatStructureLabel,
  getEdgeTier,
  getModeledSignalDisclaimer,
  getReturnSignalLabel,
  getSelectorPresentation,
} from './selectorViewModel'

function renderTopRisks(risks = []) {
  return (risks || []).slice(0, 3).map((risk, index) => <li key={index}>{risk}</li>)
}

export default function SelectorDecisionCard({ selectorOutput, scorecards = [], volSnapshot = null }) {
  if (!selectorOutput) return null

  const presentation = getSelectorPresentation(selectorOutput)
  const edgeTier = getEdgeTier(selectorOutput.expected_edge_pct)
  const trustBadges = buildTrustBadges(selectorOutput, scorecards, volSnapshot)
  const hasStructure = Boolean(selectorOutput.best_structure)
  const title = hasStructure
    ? formatStructureLabel(selectorOutput.best_structure)
    : selectorOutput.recommendation === 'No Trade'
      ? 'Edge not sufficient after costs'
      : 'No structure selected'

  return (
    <section className={`selector-card selector-card-${presentation.tone}`}>
      <div className="selector-card-head">
        <div>
          <div className="selector-eyebrow">{presentation.eyebrow}</div>
          <div className="selector-title-row">
            <span className={`selector-recommendation selector-recommendation-${presentation.tone}`}>
              {selectorOutput.recommendation}
            </span>
            <h2>{title}</h2>
          </div>
          <p className="selector-helper">{presentation.helper}</p>
          <div className="trust-badge-row">
            {trustBadges.map((badge) => (
              <span className={`trust-badge trust-badge-${badge.tone}`} title={badge.title} key={badge.label}>
                {badge.label}
              </span>
            ))}
          </div>
        </div>
        <div
          className="selector-score-pill"
          title="How strongly the available evidence backs this call, 0–100. It is NOT the probability the trade wins."
        >
          <span>Decision Quality</span>
          <strong>{Math.round(Number(selectorOutput.confidence_pct || 0))}%</strong>
          <em>evidence-backed, not probability</em>
        </div>
      </div>

      <div className="selector-metrics-grid">
        <div
          className="selector-metric"
          title="The model’s rating of how favorable this setup’s structure is. Derived from the volatility model, not from realized trade outcomes."
        >
          <span>Edge Quality</span>
          <strong>{edgeTier.label}</strong>
          <em>modeled, not empirical</em>
        </div>
        <div
          className="selector-metric"
          title="Direction/size the score implies for this setup. Derived from the setup score only — not a forecast of actual return."
        >
          <span>Return Signal</span>
          <strong>{getReturnSignalLabel(selectorOutput.expected_return_pct)}</strong>
          <em>score-derived only</em>
        </div>
        <div className="selector-metric">
          <span>Earnings Event</span>
          <strong>{selectorOutput.earnings_date || 'n/a'}</strong>
          <em>{selectorOutput.release_timing || 'timing unavailable'}</em>
        </div>
      </div>

      <p className="selector-disclaimer">{getModeledSignalDisclaimer()}</p>

      <div className="selector-thesis-row">
        <div className="selector-thesis-block">
          <div className="selector-subtitle">Primary Thesis</div>
          <p>{selectorOutput.primary_thesis || 'No thesis provided.'}</p>
        </div>
        <div className="selector-risk-block">
          <div className="selector-subtitle">Primary Risks</div>
          <ul>{renderTopRisks(selectorOutput.primary_risks)}</ul>
        </div>
      </div>
    </section>
  )
}
