import React, { useEffect, useMemo, useState } from 'react'
import { SectionTitle } from '../common/DisplayAtoms'
import {
  buildDataQualityWarnings,
  buildHealthTone,
  buildProviderRows,
  buildQualityBucketRows,
  buildQualitySummary,
  buildWeakRecommendationRows,
} from './dataQualityDiagnosticsViewModel'

export default function DataQualityDiagnosticsPanel({ apiBase }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function loadDiagnostics() {
    setLoading(true)
    setError('')
    try {
      const response = await apiFetch(`${apiBase}/api/diagnostics/data-quality`)
      if (!response.ok) throw new Error(`Data-quality diagnostics failed (${response.status})`)
      setPayload(await response.json())
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDiagnostics()
  }, [apiBase])

  const summary = useMemo(() => buildQualitySummary(payload || {}), [payload])
  const warnings = useMemo(() => buildDataQualityWarnings(payload || {}), [payload])
  const tone = useMemo(() => buildHealthTone(payload || {}), [payload])
  const providerRows = useMemo(() => buildProviderRows(payload?.source_breakdown || {}), [payload])
  const bucketRows = useMemo(() => buildQualityBucketRows(payload?.data_quality_buckets || {}), [payload])
  const weakRows = useMemo(() => buildWeakRecommendationRows(payload?.recent_weak_data_recommendations || []), [payload])

  return (
    <section className={`oos-block data-quality-block data-quality-block-${tone}`}>
      <div className="oos-head">
        <div>
          <SectionTitle>Provider &amp; Data Quality</SectionTitle>
          <p className="oos-help">Read-only observability for stale sources, missing option-chain evidence, provider coverage, and low-quality recommendation records.</p>
        </div>
        <div className="oos-actions">
          <button type="button" onClick={loadDiagnostics} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && !payload ? <div className="oos-message">Loading data-quality diagnostics…</div> : (
        <>
          <div className="data-quality-summary-grid">
            <div className="data-quality-card">
              <span>Total Recommendations</span>
              <strong>{summary.total}</strong>
            </div>
            <div className="data-quality-card">
              <span>Stale Source Rate</span>
              <strong>{summary.staleRateLabel}</strong>
              <em>{summary.staleCount} stale earnings-source records</em>
            </div>
            <div className="data-quality-card">
              <span>Low Quality</span>
              <strong>{summary.lowQualityCount}</strong>
              <em>{summary.lowQualityRateLabel} below threshold</em>
            </div>
            <div className="data-quality-card">
              <span>Missing Chain</span>
              <strong>{summary.missingChainCount}</strong>
              <em>Option-chain source unavailable or unusable</em>
            </div>
          </div>

          {warnings.length > 0 && (
            <div className="data-quality-warning-box">
              <strong>Data-quality warnings</strong>
              <ul className="selector-bullet-list selector-bullet-list-compact">
                {warnings.map((warning, index) => <li key={index}>{warning}</li>)}
              </ul>
            </div>
          )}

          <div className="selector-explain-grid data-quality-grid">
            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Provider / Source Breakdown</h3>
                <span>Grouped from recommendation ledger provenance fields.</span>
              </div>
              <div className="data-quality-provider-list">
                {providerRows.length ? providerRows.slice(0, 12).map((row) => (
                  <div className={row.isUnknown ? 'data-quality-provider-row unknown' : 'data-quality-provider-row'} key={`${row.category}-${row.source}`}>
                    <span>{row.category}</span>
                    <strong>{row.source}</strong>
                    <em>{row.count}</em>
                  </div>
                )) : <div className="oos-message">No provider records yet.</div>}
              </div>
            </div>

            <div className="selector-panel">
              <div className="selector-panel-header">
                <h3>Quality Buckets</h3>
                <span>Recommendation counts by `data_quality_score` band.</span>
              </div>
              <div className="data-quality-bucket-list">
                {bucketRows.map((row) => (
                  <div className="data-quality-bucket-row" key={row.bucket}>
                    <span>{row.bucket}</span>
                    <strong>{row.count}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>Recent Weak-Data Recommendations</h3>
              <span>Rows are shown when stale, low-quality, low-confidence, or missing option-chain evidence is detected.</span>
            </div>
            {weakRows.length ? (
              <div className="structure-table-wrap">
                <table className="structure-table data-quality-weak-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Decision</th>
                      <th>Quality</th>
                      <th>Earnings Source</th>
                      <th>Option Source</th>
                      <th>Reasons</th>
                    </tr>
                  </thead>
                  <tbody>
                    {weakRows.map((row) => (
                      <tr key={row.recommendation_id}>
                        <td><strong>{row.symbol}</strong></td>
                        <td>{row.recommendation || 'n/a'}</td>
                        <td>{row.qualityLabel}</td>
                        <td>{row.earnings_source || 'unknown'} · {row.staleLabel}</td>
                        <td>{row.option_source || 'unknown'}</td>
                        <td>{row.reasonsLabel}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="oos-message">No weak-data recommendation records found.</div>
            )}
          </div>
        </>
      )}
    </section>
  )
}
