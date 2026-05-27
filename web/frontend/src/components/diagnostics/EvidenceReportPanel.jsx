import React, { useEffect, useMemo, useState } from 'react'
import { apiFetch } from '../../lib/api'
import { SectionTitle } from '../common/DisplayAtoms'
import {
  buildBaselineComparisonRows,
  buildEvidenceQualitySummary,
  buildEvidenceReportSummary,
  buildEvidenceWarnings,
  buildExecutionRealismSummary,
  buildQuoteQualityRows,
  buildSimpleIvRvFilter,
  buildSurfaceQualitySummary,
} from './evidenceReportViewModel'

export default function EvidenceReportPanel({ apiBase }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function loadReport() {
    setLoading(true)
    setError('')
    try {
      const response = await apiFetch(`${apiBase}/api/diagnostics/evidence-report`)
      if (!response.ok) throw new Error(`Evidence report failed (${response.status})`)
      setPayload(await response.json())
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadReport()
  }, [apiBase])

  const summary = useMemo(() => buildEvidenceReportSummary(payload || {}), [payload])
  const baselines = useMemo(() => buildBaselineComparisonRows(payload || {}), [payload])
  const warnings = useMemo(() => buildEvidenceWarnings(payload || {}), [payload])
  const quoteRows = useMemo(() => buildQuoteQualityRows(payload || {}), [payload])
  const ivrv = useMemo(() => buildSimpleIvRvFilter(payload || {}), [payload])
  const evidenceQuality = useMemo(() => buildEvidenceQualitySummary(payload || {}), [payload])
  const executionRealism = useMemo(() => buildExecutionRealismSummary(payload || {}), [payload])
  const surfaceQuality = useMemo(() => buildSurfaceQualitySummary(payload || {}), [payload])

  return (
    <section className="oos-block evidence-report-block">
      <div className="oos-head">
        <div>
          <SectionTitle>Evidence Report</SectionTitle>
          <p className="oos-help">Automated paper/research evidence comparing selector outcomes with simple baselines. Not execution-grade performance.</p>
        </div>
        <div className="oos-actions">
          <button type="button" onClick={loadReport} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && !payload ? <div className="oos-message">Loading evidence report…</div> : (
        <>
          <div className={`data-quality-warning-box ${summary.readyForPaidBeta ? 'quality-positive' : ''}`}>
            <strong>{summary.readyForPaidBeta ? 'Evidence gate: beta-ready candidate' : `Evidence maturity: ${summary.maturityLabel}`}</strong>
            <span>{summary.activeDays}/{summary.targetDays} days collected · {summary.resolvedOutcomes}/{summary.minimumResolved} resolved selector outcomes · {summary.evidenceLabel}</span>
          </div>

          <div className="provider-telemetry-summary-grid">
            <div className="data-quality-card"><span>Evidence Days</span><strong>{summary.activeDays}</strong><em>Minimum {summary.minimumDays}</em></div>
            <div className="data-quality-card"><span>Resolved Outcomes</span><strong>{summary.resolvedOutcomes}</strong><em>Paper/research only</em></div>
            <div className="data-quality-card"><span>Selector Win Rate</span><strong>{summary.selectorWinRateLabel}</strong><em>Not probability of profit</em></div>
            <div className="data-quality-card"><span>Selector Avg Return</span><strong>{summary.selectorReturnLabel}</strong><em>Paper outcome quality</em></div>
            <div className="data-quality-card"><span>Claimable Rows</span><strong>{evidenceQuality.claimAllowed}</strong><em>{evidenceQuality.claimBlocked} blocked</em></div>
            <div className="data-quality-card"><span>Avg Spread Cost</span><strong>{executionRealism.avgEntrySpreadLabel}</strong><em>Modeled bid/ask</em></div>
            <div className="data-quality-card"><span>Surface Warnings</span><strong>{surfaceQuality.extremeSpreads + surfaceQuality.sparseAtm + surfaceQuality.ivAnomalies}</strong><em>Quote-chain quality</em></div>
            <div className="data-quality-card"><span>Edge Quality</span><strong>{summary.edgeQualityLabel}</strong><em>No claims before thresholds</em></div>
          </div>

          {warnings.length > 0 && (
            <div className="data-quality-warning-box">
              <strong>Evidence warnings</strong>
              <ul className="selector-bullet-list selector-bullet-list-compact">
                {warnings.map((warning, index) => <li key={index}>{warning}</li>)}
              </ul>
            </div>
          )}

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>Selector vs Baselines</h3>
              <span>Shadow baselines do not update calibration or structure priors.</span>
            </div>
            <div className="structure-table-wrap">
              <table className="structure-table forward-performance-table">
                <thead>
                  <tr><th>Approach</th><th>n</th><th>Win Rate</th><th>Avg Return</th><th>Avg Expansion</th></tr>
                </thead>
                <tbody>
                  {baselines.map((row) => (
                    <tr key={row.name}>
                      <td><strong>{row.label}</strong></td>
                      <td>{row.n}</td>
                      <td>{row.winRateLabel}</td>
                      <td>{row.returnLabel}</td>
                      <td>{row.expansionLabel}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="selector-explain-grid data-quality-grid">
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Simple IV/RV Filter</h3>
                <span>{ivrv.rule}</span>
              </div>
              <p className="oos-help">Kept {ivrv.n} selected outcomes, skipped {ivrv.skippedByFilter}. Avg return {ivrv.returnLabel}, win rate {ivrv.winRateLabel}.</p>
            </div>
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Quote Source Mix</h3>
                <span>Entry quote provenance for shadow baseline evidence.</span>
              </div>
              {quoteRows.length ? quoteRows.map((row) => (
                <div className="quality-source-row" key={row.source}>
                  <span>{row.source}</span><strong>{row.count}</strong>
                </div>
              )) : <div className="oos-message">No baseline quote evidence yet.</div>}
            </div>
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Evidence Quality Gate</h3>
                <span>{evidenceQuality.label}</span>
              </div>
              <div className="quality-source-row"><span>Degraded evidence</span><strong>{evidenceQuality.degraded}</strong></div>
              <div className="quality-source-row"><span>Record-only rows</span><strong>{evidenceQuality.recordOnly}</strong></div>
              <div className="quality-source-row"><span>Execution-grade rows</span><strong>{evidenceQuality.executionGrade}</strong></div>
            </div>
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Execution Scenarios</h3>
                <span>{executionRealism.label}</span>
              </div>
              <div className="quality-source-row"><span>Selector entries</span><strong>{executionRealism.selectorEntryRows}</strong></div>
              <div className="quality-source-row"><span>Baseline entries</span><strong>{executionRealism.baselineEntryRows}</strong></div>
              <div className="quality-source-row"><span>Resolved exits</span><strong>{executionRealism.exitScenarioRows}</strong></div>
            </div>
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Surface Quality</h3>
                <span>{surfaceQuality.label}</span>
              </div>
              <div className="quality-source-row"><span>Extreme spreads</span><strong>{surfaceQuality.extremeSpreads}</strong></div>
              <div className="quality-source-row"><span>Sparse ATM surfaces</span><strong>{surfaceQuality.sparseAtm}</strong></div>
              <div className="quality-source-row"><span>IV anomalies</span><strong>{surfaceQuality.ivAnomalies}</strong></div>
            </div>
          </div>
        </>
      )}
    </section>
  )
}
