import React from 'react'

/**
 * Container for the raw-metrics drill-down (the "Full metrics" tab content).
 *
 * Renders open: the tab itself is now the disclosure, so this must NOT wrap its
 * children in a collapsed <details> (that double-collapse meant clicking "Full
 * metrics" still showed only a summary line). The honest disclaimer header is
 * kept as context for the model-derived numbers below.
 */
export default function LegacyAnalysisPanel({ children }) {
  return (
    <section className="legacy-analysis-panel">
      <div className="legacy-analysis-head">
        <strong>Full metrics</strong>
        <span>Model-derived diagnostics — not empirical return forecasts, and not calibrated to live retail execution costs.</span>
      </div>
      <div className="legacy-analysis-body">
        {children}
      </div>
    </section>
  )
}
