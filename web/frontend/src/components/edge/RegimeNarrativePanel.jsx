import React from 'react'
import {
  buildEdgeNarrative,
  buildRegimeConfidenceMessage,
  buildVolRegimeBanner,
} from './selectorViewModel'

export default function RegimeNarrativePanel({ selectorOutput, scorecards, volSnapshot }) {
  if (!selectorOutput && !volSnapshot) return null

  const regime = buildVolRegimeBanner(selectorOutput, scorecards, volSnapshot)
  const narrative = buildEdgeNarrative(selectorOutput, scorecards, volSnapshot)
  const confidence = buildRegimeConfidenceMessage(selectorOutput, scorecards, volSnapshot)

  return (
    <section className="selector-panel selector-panel-full regime-panel">
      <div className={`regime-banner regime-banner-${regime.tone}`}>
        <div>
          <div className="regime-banner-label">Volatility Regime</div>
          <strong>{regime.headline}</strong>
          <p>{regime.detail}</p>
        </div>
        <div className={`regime-confidence-chip regime-confidence-chip-${confidence.tone}`}>
          <span>{confidence.label}</span>
          <em>{confidence.detail}</em>
        </div>
      </div>

      <div className="regime-stats-grid">
        <div className="regime-stat">
          <span>Regime Win Rate</span>
          <strong>{regime.stats.winRateLabel}</strong>
        </div>
        <div className="regime-stat">
          <span>Regime Avg Return</span>
          <strong>{regime.stats.avgReturnLabel}</strong>
        </div>
        <div className="regime-stat">
          <span>Regime Sample</span>
          <strong>{regime.stats.sampleLabel}</strong>
        </div>
      </div>
      <p className="regime-stats-note">{regime.stats.note}</p>

      <div className="selector-panel-header">
        <h3>Edge Narrative</h3>
        <span>Explains what the market is pricing, what history says, and where the mismatch sits.</span>
      </div>
      <div className="narrative-grid">
        <div className="narrative-item">
          <span>Market Pricing</span>
          <strong>{narrative.marketLine}</strong>
        </div>
        <div className="narrative-item">
          <span>Historical Context</span>
          <strong>{narrative.historyLine}</strong>
        </div>
        <div className="narrative-item">
          <span>Mismatch</span>
          <strong>{narrative.mismatchLine}</strong>
        </div>
      </div>
      <p className="narrative-note">{narrative.note}</p>
    </section>
  )
}
