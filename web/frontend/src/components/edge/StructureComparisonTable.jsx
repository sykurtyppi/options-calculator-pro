import React from 'react'
import { buildStructureRows } from './selectorViewModel'

export default function StructureComparisonTable({ scorecards, selectorOutput }) {
  const rows = buildStructureRows(scorecards, selectorOutput)
  if (!rows.length) return null

  return (
    <section className="selector-panel selector-panel-full">
      <div className="selector-panel-header">
        <h3>Structure Comparison</h3>
        <span>All supported structures remain visible, including ineligible ones and their reasons. Edge and return columns are qualitative score-derived tiers.</span>
      </div>

      <div className="structure-table-wrap">
        <table className="structure-table">
          <thead>
            <tr>
              <th>Structure</th>
              <th>Status</th>
              <th>Score</th>
              <th>Edge Tier</th>
              <th>Return Signal</th>
              <th>Execution Pen.</th>
              <th>Comparables</th>
              <th>Positive Rate</th>
              <th>Evidence</th>
              <th>Top Evidence Note</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.key}
                className={[
                  row.isBest ? 'structure-row-best' : '',
                  row.isRunnerUp ? 'structure-row-runner-up' : '',
                  !row.isEligible ? 'structure-row-ineligible' : '',
                ].filter(Boolean).join(' ')}
              >
                <td>
                  <div className="structure-name-cell">
                    <strong>{row.structure}</strong>
                    {row.isBest && <span className="structure-chip structure-chip-best">Selected</span>}
                    {row.isRunnerUp && !row.isBest && <span className="structure-chip">Runner-up</span>}
                  </div>
                </td>
                <td>{row.eligibleLabel}</td>
                <td>{row.compositeScore}</td>
                <td>{row.expectedEdge}</td>
                <td>{row.expectedReturn}</td>
                <td>{row.executionPenalty}</td>
                <td>{row.walkForwardCount}</td>
                <td>{row.walkForwardWinRate}</td>
                <td>{row.sampleConfidence}</td>
                <td className="structure-rationale-cell" title={row.rationaleSummary}>{row.rationaleSummary}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
