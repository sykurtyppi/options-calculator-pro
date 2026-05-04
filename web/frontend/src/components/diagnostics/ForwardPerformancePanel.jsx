import React, { useEffect, useMemo, useState } from 'react'
import { SectionTitle } from '../common/DisplayAtoms'
import {
  buildBenchmarkComparisonRows,
  buildCalibrationBucketRows,
  buildCalibrationReportRows,
  buildClaimablePerformanceRows,
  buildComparisonRows,
  buildForwardPerformanceSummary,
  buildForwardPerformanceWarnings,
  buildRecentResolvedRows,
  buildStructurePerformanceRows,
} from './forwardPerformanceViewModel'

export default function ForwardPerformancePanel({ apiBase }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function loadPerformance() {
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${apiBase}/api/diagnostics/forward-performance`)
      if (!response.ok) throw new Error(`Forward-performance diagnostics failed (${response.status})`)
      setPayload(await response.json())
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPerformance()
  }, [apiBase])

  const summary = useMemo(() => buildForwardPerformanceSummary(payload || {}), [payload])
  const warnings = useMemo(() => buildForwardPerformanceWarnings(payload || {}), [payload])
  const structureRows = useMemo(() => buildStructurePerformanceRows(payload?.by_structure || {}), [payload])
  const bucketRows = useMemo(() => buildCalibrationBucketRows(payload?.calibration_buckets || []), [payload])
  const calibrationReportRows = useMemo(() => buildCalibrationReportRows(payload?.calibration_report || {}), [payload])
  const benchmarkRows = useMemo(() => buildBenchmarkComparisonRows(payload || {}), [payload])
  const claimableRows = useMemo(() => buildClaimablePerformanceRows(payload || {}), [payload])
  const comparisonRows = useMemo(() => buildComparisonRows(payload || {}), [payload])
  const recentRows = useMemo(() => buildRecentResolvedRows(payload?.recent_resolved_outcomes || []), [payload])

  return (
    <section className="oos-block forward-performance-block">
      <div className="oos-head">
        <div>
          <SectionTitle>Forward Performance</SectionTitle>
          <p className="oos-help">Read-only paper/research outcome diagnostics for selector recommendations. These are not execution-grade live performance claims.</p>
        </div>
        <div className="oos-actions">
          <button type="button" onClick={loadPerformance} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && !payload ? <div className="oos-message">Loading forward-performance diagnostics…</div> : (
        <>
          <div className="data-quality-warning-box forward-performance-note">
            <strong>Evidence label: {summary.evidenceLabel}</strong>
            <span>Modeled values are score-derived diagnostics captured at decision time; realized values come from paper/research outcome records.</span>
          </div>

          <div className="provider-telemetry-summary-grid">
            <div className="data-quality-card">
              <span>Resolved Outcomes</span>
              <strong>{summary.resolvedCount}</strong>
              <em>{summary.sampleLabel}</em>
            </div>
            <div className="data-quality-card">
              <span>Paper Win Rate</span>
              <strong>{summary.winRateLabel}</strong>
              <em>Not probability of profit</em>
            </div>
            <div className="data-quality-card">
              <span>Modeled Return</span>
              <strong>{summary.avgModeledLabel}</strong>
              <em>Score-derived at entry</em>
            </div>
            <div className="data-quality-card">
              <span>Realized Return</span>
              <strong>{summary.avgRealizedLabel}</strong>
              <em>Paper/research outcome</em>
            </div>
            <div className="data-quality-card">
              <span>Model Error</span>
              <strong>{summary.avgErrorLabel}</strong>
              <em>Realized minus modeled</em>
            </div>
            <div className="data-quality-card">
              <span>No Trade</span>
              <strong>{summary.noTradeCount}</strong>
              <em>{summary.totalRecommendations} total recommendations</em>
            </div>
          </div>

          {warnings.length > 0 && (
            <div className="data-quality-warning-box">
              <strong>Forward-performance warnings</strong>
              <ul className="selector-bullet-list selector-bullet-list-compact">
                {warnings.map((warning, index) => <li key={index}>{warning}</li>)}
              </ul>
            </div>
          )}

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>Structure Breakdown</h3>
              <span>Win/loss and average modeled-vs-realized paper return by selected structure.</span>
            </div>
            {structureRows.length ? (
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr>
                      <th>Structure</th>
                      <th>n</th>
                      <th>W/L</th>
                      <th>Win Rate</th>
                      <th>Modeled</th>
                      <th>Realized</th>
                      <th>IV Expansion</th>
                    </tr>
                  </thead>
                  <tbody>
                    {structureRows.map((row) => (
                      <tr key={row.structure}>
                        <td><strong>{row.structure}</strong></td>
                        <td>{row.n}</td>
                        <td>{row.wins}/{row.losses}</td>
                        <td>{row.winRateLabel}</td>
                        <td>{row.avgModeledLabel}</td>
                        <td>{row.avgRealizedLabel}</td>
                        <td>{row.avgExpansionLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="oos-message">No resolved paper outcomes yet.</div>}
          </div>

          <div className="selector-explain-grid data-quality-grid">
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Score Buckets</h3>
                <span>Outcome behavior by selector confidence / setup-score bucket.</span>
              </div>
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr>
                      <th>Bucket</th>
                      <th>n</th>
                      <th>Win Rate</th>
                      <th>Modeled</th>
                      <th>Realized</th>
                      <th>Median</th>
                      <th>Error</th>
                      <th>Signal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bucketRows.map((row) => (
                      <tr key={row.bucket}>
                        <td>{row.bucket}</td>
                        <td>{row.n}</td>
                        <td>{row.winRateLabel}</td>
                        <td>{row.avgModeledLabel}</td>
                        <td>{row.avgRealizedLabel}</td>
                        <td>{row.medianRealizedLabel}</td>
                        <td>{row.avgErrorLabel}</td>
                        <td>{row.warningLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Stale / Quality Comparison</h3>
                <span>Paper outcomes grouped by earnings-source freshness and data-quality tier.</span>
              </div>
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr>
                      <th>Group</th>
                      <th>n</th>
                      <th>Win Rate</th>
                      <th>Modeled</th>
                      <th>Realized</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonRows.map((row) => (
                      <tr key={row.group}>
                        <td>{row.group}</td>
                        <td>{row.n}</td>
                        <td>{row.winRateLabel}</td>
                        <td>{row.avgModeledLabel}</td>
                        <td>{row.avgRealizedLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>Empirical Calibration Report</h3>
              <span>Bucketed paper outcomes by score, evidence quality, surface quality, spread cost, and earnings confidence.</span>
            </div>
            {calibrationReportRows.length ? (
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr>
                      <th>Category</th>
                      <th>Bucket</th>
                      <th>n</th>
                      <th>W/L</th>
                      <th>Win Rate</th>
                      <th>Realized</th>
                      <th>Median</th>
                      <th>Claimable</th>
                      <th>Signal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calibrationReportRows.map((row) => (
                      <tr key={`${row.category}-${row.bucket}`}>
                        <td>{row.category}</td>
                        <td><strong>{row.bucket}</strong></td>
                        <td>{row.n}</td>
                        <td>{row.wins}/{row.losses}</td>
                        <td>{row.winRateLabel}</td>
                        <td>{row.avgRealizedLabel}</td>
                        <td>{row.medianRealizedLabel}</td>
                        <td>{row.claimableCount}</td>
                        <td>{row.warningLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="oos-message">No resolved paper outcomes yet.</div>}
          </div>

          <div className="selector-explain-grid data-quality-grid">
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Benchmark Comparison</h3>
                <span>Selector versus transparent baselines. Measured only, not optimized.</span>
              </div>
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr>
                      <th>Benchmark</th>
                      <th>n</th>
                      <th>W/L</th>
                      <th>Win Rate</th>
                      <th>Realized</th>
                      <th>Median</th>
                      <th>Skipped</th>
                      <th>Signal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkRows.map((row) => (
                      <tr key={row.key} title={row.rule}>
                        <td><strong>{row.label}</strong></td>
                        <td>{row.n}</td>
                        <td>{row.wins}/{row.losses}</td>
                        <td>{row.winRateLabel}</td>
                        <td>{row.avgRealizedLabel}</td>
                        <td>{row.medianRealizedLabel}</td>
                        <td>{row.skippedByFilter}</td>
                        <td>{row.warningLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Claimable Evidence</h3>
                <span>Separates evidence usable for claims from record-only observations.</span>
              </div>
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr>
                      <th>Group</th>
                      <th>n</th>
                      <th>Win Rate</th>
                      <th>Realized</th>
                      <th>Median</th>
                      <th>Signal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {claimableRows.map((row) => (
                      <tr key={row.key}>
                        <td><strong>{row.label}</strong></td>
                        <td>{row.n}</td>
                        <td>{row.winRateLabel}</td>
                        <td>{row.avgRealizedLabel}</td>
                        <td>{row.medianRealizedLabel}</td>
                        <td>{row.warningLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>Recent Resolved Outcomes</h3>
              <span>Linked paper/research rows with quote provenance and stale-source status.</span>
            </div>
            {recentRows.length ? (
              <div className="structure-table-wrap">
                <table className="structure-table forward-performance-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Structure</th>
                      <th>Source</th>
                      <th>Modeled</th>
                      <th>Realized</th>
                      <th>Expansion</th>
                      <th>Quality</th>
                      <th>Stale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentRows.map((row, index) => (
                      <tr key={`${row.trade_id || row.recommendation_id}-${index}`}>
                        <td><strong>{row.symbol}</strong></td>
                        <td>{row.structure}</td>
                        <td>{row.sourceLabel}</td>
                        <td>{row.modeledLabel}</td>
                        <td>{row.realizedLabel}</td>
                        <td>{row.expansionLabel}</td>
                        <td>{row.qualityLabel}</td>
                        <td>{row.staleLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="oos-message">No resolved paper outcomes yet.</div>}
          </div>
        </>
      )}
    </section>
  )
}
