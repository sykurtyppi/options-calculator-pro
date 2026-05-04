import React from 'react'

export default function LegacyAnalysisPanel({ children }) {
  return (
    <details className="legacy-analysis-panel">
      <summary>
        <div>
          <strong>Legacy Detailed Analysis</strong>
          <span>Secondary transition view: prior edge-analysis outputs remain available below the selector-first evidence layer. Numeric edge fields below are model-derived diagnostics, not empirical return forecasts, and not calibrated to live retail execution costs.</span>
        </div>
      </summary>
      <div className="legacy-analysis-body">
        {children}
      </div>
    </details>
  )
}
