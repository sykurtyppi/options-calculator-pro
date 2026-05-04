import React from 'react'
import {
  buildHistoricalEvidenceLine,
  buildNoTradeSummary,
} from './selectorViewModel'

function EvidenceStat({ label, value }) {
  return (
    <div className="selector-inline-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default function DecisionEvidenceStrip({ selectorOutput, scorecards, volSnapshot }) {
  if (!selectorOutput) return null

  const evidence = buildHistoricalEvidenceLine(selectorOutput, scorecards)
  const noTrade = buildNoTradeSummary(selectorOutput, scorecards, volSnapshot)

  return (
    <section className={`selector-inline-evidence selector-inline-evidence-${evidence.tone}`}>
      <div className="selector-inline-evidence-head">
        <div>
          <div className="selector-inline-label">Historical Behavior</div>
          <p>{evidence.headline}</p>
          <span>{evidence.subline}</span>
        </div>
        <div className="selector-inline-stats">
          <EvidenceStat label="Comparables" value={evidence.comparableCount} />
          <EvidenceStat label="Positive Outcomes" value={evidence.positiveRateLabel} />
          <EvidenceStat label="Paper Return" value={evidence.averageReturnLabel} />
        </div>
      </div>

      {noTrade && (
        <div className="selector-no-trade-context">
          <div className="selector-no-trade-headline">{noTrade.headline}</div>
          <ul className="selector-bullet-list selector-bullet-list-compact">
            {noTrade.reasons.map((reason, index) => <li key={index}>{reason}</li>)}
          </ul>
          <p>{noTrade.action}</p>
        </div>
      )}
    </section>
  )
}
