import React, { useEffect, useMemo, useState } from 'react'
import { apiFetch } from '../../lib/api'
import { SectionTitle } from '../common/DisplayAtoms'
import {
  buildBaselineComparisonRows,
  buildEvidenceQualitySummary,
  buildEvidenceReportSummary,
  buildEvidenceWarnings,
  buildExcludedEvidenceSummary,
  buildExecutionRealismSummary,
  buildExitAttritionRows,
  buildOutlierDependence,
  buildQuoteQualityRows,
  buildSimpleIvRvFilter,
  buildSurfaceQualitySummary,
  buildUncertaintyRows,
  buildUniverseShadowRows,
  buildUniverseShadowSummary,
} from './evidenceReportViewModel'

export default function EvidenceReportPanel({ apiBase }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function loadReport(signal) {
    setLoading(true)
    setError('')
    try {
      const response = await apiFetch(`${apiBase}/api/diagnostics/evidence-report`, { signal })
      if (!response.ok) throw new Error(`Evidence report failed (${response.status})`)
      setPayload(await response.json())
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message || String(err))
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    loadReport(controller.signal)
    return () => controller.abort()
  }, [apiBase])

  const summary = useMemo(() => buildEvidenceReportSummary(payload || {}), [payload])
  const baselines = useMemo(() => buildBaselineComparisonRows(payload || {}), [payload])
  const warnings = useMemo(() => buildEvidenceWarnings(payload || {}), [payload])
  const quoteRows = useMemo(() => buildQuoteQualityRows(payload || {}), [payload])
  const ivrv = useMemo(() => buildSimpleIvRvFilter(payload || {}), [payload])
  const evidenceQuality = useMemo(() => buildEvidenceQualitySummary(payload || {}), [payload])
  const executionRealism = useMemo(() => buildExecutionRealismSummary(payload || {}), [payload])
  const surfaceQuality = useMemo(() => buildSurfaceQualitySummary(payload || {}), [payload])
  const universeRows = useMemo(() => buildUniverseShadowRows(payload || {}), [payload])
  const universe = useMemo(() => buildUniverseShadowSummary(payload || {}), [payload])
  const attritionRows = useMemo(() => buildExitAttritionRows(payload || {}), [payload])
  const excluded = useMemo(() => buildExcludedEvidenceSummary(payload || {}), [payload])
  const uncertaintyRows = useMemo(() => buildUncertaintyRows(payload || {}), [payload])
  const outliers = useMemo(() => buildOutlierDependence(payload || {}), [payload])
  const uncertaintyMethod = (payload?.uncertainty || {}).method || ''

  return (
    <section className="oos-block evidence-report-block">
      <div className="oos-head">
        <div>
          <SectionTitle>Evidence Report</SectionTitle>
          <p className="oos-help">Automated paper/research evidence comparing selector outcomes with simple baselines. Not execution-grade performance.</p>
        </div>
        <div className="oos-actions">
          <button type="button" onClick={() => loadReport()} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && !payload ? <div className="oos-message">Loading evidence report…</div> : (
        <>
          <div className={`data-quality-warning-box ${summary.readyForPaidBeta ? 'quality-positive' : ''}`}>
            <strong>{summary.readyForPaidBeta ? 'Evidence gate: beta-ready candidate' : `Evidence maturity: ${summary.maturityLabel}`}</strong>
            <span>{summary.activeDays}/{summary.targetDays} days collected · {summary.resolvedOutcomes}/{summary.minimumResolved} resolved selector outcomes · {summary.evidenceLabel}</span>
            {!summary.readyForPaidBeta && summary.gateBlockingReasons.length > 0 && (
              <span>Not beta-ready: {summary.gateBlockingReasons.join('; ')}</span>
            )}
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

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>How Sure Are We?</h3>
              <span>{uncertaintyMethod || 'Bootstrap intervals for each comparison.'}</span>
            </div>
            {uncertaintyRows.length ? (
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr><th>Comparison</th><th>n</th><th>Mean</th><th>95% interval</th><th>Reading</th></tr>
                  </thead>
                  <tbody>
                    {uncertaintyRows.map((row) => (
                      <tr key={row.key}>
                        <td><strong>{row.label}</strong></td>
                        <td>{row.n}</td>
                        <td>{row.meanLabel}</td>
                        <td>{row.intervalLabel}</td>
                        <td>{row.verdictLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="oos-message">No resolved evidence to measure yet.</div>}
            {outliers.available && (
              <p className={`oos-help ${outliers.fragile ? 'error-banner' : ''}`}>
                Without the {outliers.k} best selector results the mean is {outliers.withoutTopLabel}
                {' '}(without the {outliers.k} worst: {outliers.withoutBottomLabel}); the {outliers.k} largest gains
                {' '}are {outliers.topShareLabel} of all gains.
                {outliers.fragile ? ' The positive average depends on a few trades.' : ''}
              </p>
            )}
          </div>

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>Did the Selector's Skips Pay?</h3>
              <span>Every eligible event, shadow-entered whatever the selector said. If skipped events paid as well as picked ones, the selector's filtering is not adding value.</span>
            </div>
            {universeRows.length ? (
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr><th>Baseline</th><th>All events</th><th>Selector picked</th><th>Selector skipped</th><th>Picked − skipped</th></tr>
                  </thead>
                  <tbody>
                    {universeRows.map((row) => (
                      <tr key={row.name}>
                        <td><strong>{row.label}</strong></td>
                        <td>{row.allReturnLabel} <em>(n={row.allN})</em></td>
                        <td>{row.pickedReturnLabel} <em>(n={row.pickedN})</em></td>
                        <td>{row.skippedReturnLabel} <em>(n={row.skippedN})</em></td>
                        <td>{row.selectorEdgeLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="oos-message">No resolved universe events yet ({universe.eventsRecorded} recorded, {universe.open} open).</div>}
          </div>

          <div className="selector-explain-grid data-quality-grid">
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Exit Attrition</h3>
                <span>Entered positions that could not be valued on T-1. High or concentrated attrition biases resolved results.</span>
              </div>
              {attritionRows.map((row) => (
                <div key={row.key}>
                  <div className="quality-source-row">
                    <span>{row.label}</span>
                    <strong>{row.missing} missing / {row.resolved} resolved · {row.rateLabel}</strong>
                  </div>
                  {row.reasons.map((reason) => (
                    <div className="quality-source-row" key={`${row.key}-${reason.label}`}>
                      <span>↳ {reason.label}</span><strong>{reason.count}</strong>
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Universe Coverage</h3>
                <span>Failed entries are retried daily while the event is in the DTE window.</span>
              </div>
              <div className="quality-source-row"><span>Events recorded</span><strong>{universe.eventsRecorded}</strong></div>
              <div className="quality-source-row"><span>Entered after a retry</span><strong>{universe.enteredAfterRetry}</strong></div>
              <div className="quality-source-row"><span>Never entered</span><strong>{universe.notEntered}</strong></div>
              {universe.notEnteredReasons.map((reason) => (
                <div className="quality-source-row" key={`entry-${reason.label}`}>
                  <span>↳ {reason.label}</span><strong>{reason.count}</strong>
                </div>
              ))}
            </div>
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Excluded Evidence</h3>
                <span>Kept for audit, left out of every comparison above.</span>
              </div>
              <div className="quality-source-row"><span>Invalidated selector outcomes</span><strong>{excluded.invalidatedN}</strong></div>
              {excluded.invalidatedReasons.map((reason) => (
                <div className="quality-source-row" key={`inv-${reason.label}`}>
                  <span>↳ {reason.label}</span><strong>{reason.count}</strong>
                </div>
              ))}
              <div className="quality-source-row"><span>Unverified baseline exits</span><strong>{excluded.legacyBaselinesN}</strong></div>
              {excluded.legacyReasons.map((reason) => (
                <div className="quality-source-row" key={`legacy-${reason.label}`}>
                  <span>↳ {reason.label}</span><strong>{reason.count}</strong>
                </div>
              ))}
              <div className="quality-source-row"><span>Non-finite results (selector / baseline)</span><strong>{excluded.nonFiniteSelectorN} / {excluded.nonFiniteBaselineN}</strong></div>
              <div className="quality-source-row"><span>Replay/backtest rows (not forward evidence)</span><strong>{excluded.replayN}</strong></div>
            </div>
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
