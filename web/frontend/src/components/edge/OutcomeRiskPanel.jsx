import React from 'react'
import {
  buildEdgeDurability,
  buildFailureModes,
  buildOutcomeDecomposition,
  buildOutcomeScenarios,
  buildSensitivityWarnings,
  buildStructureAdvantage,
} from './selectorViewModel'

export default function OutcomeRiskPanel({ selectorOutput, scorecards, volSnapshot }) {
  const decomposition = buildOutcomeDecomposition(selectorOutput, scorecards, volSnapshot)
  const scenarios = buildOutcomeScenarios(selectorOutput, scorecards, volSnapshot)
  const durability = buildEdgeDurability(selectorOutput, scorecards, volSnapshot)
  const advantage = buildStructureAdvantage(selectorOutput, scorecards)
  const failureModes = buildFailureModes(selectorOutput, scorecards, volSnapshot)
  const warnings = buildSensitivityWarnings(selectorOutput, scorecards, volSnapshot)
  if (!decomposition.length && !failureModes.length && !scenarios.length) return null

  return (
    <section className="selector-explain-grid">
      <div className="selector-panel">
        <div className="selector-panel-header">
          <h3>Outcome Decomposition</h3>
          <span>Shows what is driving the modeled edge. Values are score-derived scenario signals, not calibrated profit forecasts.</span>
        </div>
        <div className="outcome-grid">
          {decomposition.map((item) => (
            <div className="outcome-item" key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
              <em>{item.detail}</em>
            </div>
          ))}
        </div>

        <div className="selector-panel-header outcome-subsection-header">
          <h3>Outcome Scenarios</h3>
          <span>Simple score-derived stress cases built from IV variability, historical move dispersion, and execution friction.</span>
        </div>
        <div className="outcome-grid">
          {scenarios.map((item) => (
            <div className="outcome-item" key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
              <em>{item.detail}</em>
            </div>
          ))}
        </div>
      </div>

      <div className="selector-panel">
        <div className="selector-panel-header">
          <h3>Edge Robustness</h3>
          <span>Shows whether the modeled edge survives conservative stress on costs, IV expansion, and realized move assumptions.</span>
        </div>
        <div className={`durability-card durability-card-${durability.tone}`}>
          <span>Edge Durability</span>
          <strong>{durability.label}</strong>
          <em>{durability.detail}</em>
          <small>Stressed score-derived edge after simple haircut: {durability.stressedEdgeLabel}</small>
        </div>

        <div className="selector-panel-header outcome-subsection-header">
          <h3>Structure Advantage</h3>
          <span>Explicit winner-vs-runner-up comparison.</span>
        </div>
        <div className="advantage-card">
          <strong>{advantage.headline}</strong>
          <ul className="selector-bullet-list selector-bullet-list-compact">
            {advantage.bullets.map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </div>

        {warnings.length > 0 && (
          <>
            <div className="selector-panel-header outcome-subsection-header">
              <h3>Sensitivity Warnings</h3>
              <span>Flags when the edge depends heavily on a thin sample, tail events, or unusually tight execution.</span>
            </div>
            <div className="sensitivity-warning-box">
              <ul className="selector-bullet-list selector-bullet-list-compact">
                {warnings.map((item, index) => <li key={index}>{item}</li>)}
              </ul>
            </div>
          </>
        )}

        <div className="selector-panel-header outcome-subsection-header">
          <h3>Failure Modes</h3>
          <span>What would make this setup fail even if the score ranks well today.</span>
        </div>
        <ul className="selector-bullet-list">
          {failureModes.map((item, index) => <li key={index}>{item}</li>)}
        </ul>
      </div>
    </section>
  )
}
