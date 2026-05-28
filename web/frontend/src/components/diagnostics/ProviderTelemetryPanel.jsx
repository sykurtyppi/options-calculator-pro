import React, { useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch } from '../../lib/api'
import { SectionTitle } from '../common/DisplayAtoms'
import {
  buildProviderHealthRows,
  buildRecentFailureRows,
  buildTelemetryEndpoint,
  buildTelemetrySummary,
  buildTelemetryWarnings,
} from './providerTelemetryViewModel'

export default function ProviderTelemetryPanel({ apiBase }) {
  const [payload, setPayload] = useState(null)
  const [provider, setProvider] = useState('')
  const [endpointType, setEndpointType] = useState('')
  const [symbol, setSymbol] = useState('')
  const [success, setSuccess] = useState('all')
  const [failuresOnly, setFailuresOnly] = useState(false)
  const [limit, setLimit] = useState(100)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  // Shared across both the filter-change effect and the Refresh button.
  const telemetryFetchRef = useRef(null)

  const endpoint = useMemo(() => buildTelemetryEndpoint(apiBase, {
    provider,
    endpointType,
    symbol,
    success,
    failuresOnly,
    limit,
    offset,
  }), [apiBase, provider, endpointType, symbol, success, failuresOnly, limit, offset])

  async function loadTelemetry(signal) {
    setLoading(true)
    setError('')
    try {
      const response = await apiFetch(endpoint, { signal })
      if (!response.ok) throw new Error(`Provider telemetry failed (${response.status})`)
      setPayload(await response.json())
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message || String(err))
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }

  function startTelemetryFetch() {
    if (telemetryFetchRef.current) telemetryFetchRef.current.abort()
    const controller = new AbortController()
    telemetryFetchRef.current = controller
    loadTelemetry(controller.signal)
  }

  useEffect(() => {
    startTelemetryFetch()
    return () => { if (telemetryFetchRef.current) telemetryFetchRef.current.abort() }
  }, [endpoint])

  const summary = useMemo(() => buildTelemetrySummary(payload || {}), [payload])
  const providerRows = useMemo(() => buildProviderHealthRows(payload?.summary_by_provider || {}), [payload])
  const failureRows = useMemo(() => buildRecentFailureRows(payload?.recent_failures || []), [payload])
  const warnings = useMemo(() => buildTelemetryWarnings(payload || {}), [payload])
  const pageNumber = Math.floor(offset / limit) + 1
  const pageCount = Math.max(1, Math.ceil((payload?.totals?.events || 0) / limit))

  function resetPagedFilter(setter, value) {
    setter(value)
    setOffset(0)
  }

  return (
    <section className="oos-block provider-telemetry-block">
      <div className="oos-head">
        <div>
          <SectionTitle>Provider Telemetry</SectionTitle>
          <p className="oos-help">Direct request-health logs for market data, earnings, quotes, fallback usage, stale cache usage, and latency. Telemetry is best-effort and read-only.</p>
        </div>
        <div className="oos-actions">
          <button type="button" onClick={() => startTelemetryFetch()} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && !payload ? <div className="oos-message">Loading provider telemetry…</div> : (
        <>
          <div className="provider-telemetry-summary-grid">
            <div className="data-quality-card">
              <span>Total Events</span>
              <strong>{summary.events}</strong>
            </div>
            <div className="data-quality-card">
              <span>Failures</span>
              <strong>{summary.failures}</strong>
              <em>{summary.failureRateLabel} failure rate</em>
            </div>
            <div className="data-quality-card">
              <span>Fallbacks</span>
              <strong>{summary.fallbackCount}</strong>
              <em>Provider fallback paths used</em>
            </div>
            <div className="data-quality-card">
              <span>Stale Used</span>
              <strong>{summary.staleCount}</strong>
              <em>Stale cache/source events</em>
            </div>
            <div className="data-quality-card">
              <span>Avg Latency</span>
              <strong>{summary.avgLatencyLabel}</strong>
              <em>Filtered provider events</em>
            </div>
            <div className="data-quality-card">
              <span>Telemetry DB</span>
              <strong>{summary.rowCount}</strong>
              <em>{summary.dbSizeLabel}</em>
            </div>
          </div>

          <div className="warehouse-controls provider-telemetry-controls">
            <label className="oos-field">
              <span>Provider</span>
              <input value={provider} onChange={(event) => resetPagedFilter(setProvider, event.target.value)} placeholder="marketdata_app" />
            </label>
            <label className="oos-field">
              <span>Endpoint</span>
              <input value={endpointType} onChange={(event) => resetPagedFilter(setEndpointType, event.target.value)} placeholder="options_chain" />
            </label>
            <label className="oos-field">
              <span>Symbol</span>
              <input value={symbol} maxLength={10} onChange={(event) => resetPagedFilter(setSymbol, event.target.value.toUpperCase())} placeholder="AAPL" />
            </label>
            <label className="oos-field">
              <span>Status</span>
              <select value={success} onChange={(event) => resetPagedFilter(setSuccess, event.target.value)}>
                <option value="all">All</option>
                <option value="true">Success</option>
                <option value="false">Failure</option>
              </select>
            </label>
            <label className="oos-field">
              <span>Rows</span>
              <select value={limit} onChange={(event) => { setLimit(Number(event.target.value) || 100); setOffset(0) }}>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={250}>250</option>
              </select>
            </label>
            <label className="provider-telemetry-toggle">
              <input
                type="checkbox"
                checked={failuresOnly}
                onChange={(event) => { setFailuresOnly(event.target.checked); setOffset(0) }}
              />
              <span>Recent failures only</span>
            </label>
          </div>

          {warnings.length > 0 && (
            <div className="data-quality-warning-box">
              <strong>Provider health warnings</strong>
              <ul className="selector-bullet-list selector-bullet-list-compact">
                {warnings.map((warning, index) => <li key={index}>{warning}</li>)}
              </ul>
            </div>
          )}

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>Provider Health</h3>
              <span>Failure rate, latency, fallback, and stale counts grouped by provider.</span>
            </div>
            {providerRows.length ? (
              <div className="structure-table-wrap">
                <table className="structure-table provider-telemetry-table">
                  <thead>
                    <tr>
                      <th>Provider</th>
                      <th>Events</th>
                      <th>Failures</th>
                      <th>Failure Rate</th>
                      <th>Avg Latency</th>
                      <th>Fallbacks</th>
                      <th>Stale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {providerRows.map((row) => (
                      <tr key={row.provider} className={`provider-telemetry-row-${row.tone}`}>
                        <td><strong>{row.provider}</strong></td>
                        <td>{row.total}</td>
                        <td>{row.failures}</td>
                        <td>{row.failureRateLabel}</td>
                        <td>{row.avgLatencyLabel}</td>
                        <td>{row.fallbackCount}</td>
                        <td>{row.staleCount}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="oos-message">No provider telemetry recorded yet.</div>}
            <div className="ledger-pagination">
              <button
                type="button"
                className="secondary-button"
                disabled={offset <= 0 || loading}
                onClick={() => setOffset(Math.max(0, offset - limit))}
              >
                Previous
              </button>
              <span>Page {pageNumber} of {pageCount} · {payload?.totals?.events || 0} filtered events</span>
              <button
                type="button"
                className="secondary-button"
                disabled={!payload?.has_more || loading}
                onClick={() => setOffset(offset + limit)}
              >
                Next
              </button>
            </div>
          </div>

          <div className="selector-panel selector-panel-full">
            <div className="selector-panel-header">
              <h3>Recent Failures</h3>
              <span>Sanitized categories only; no URLs, tokens, or provider secrets are stored.</span>
            </div>
            {failureRows.length ? (
              <div className="structure-table-wrap">
                <table className="structure-table provider-telemetry-table">
                  <thead>
                    <tr>
                      <th>Provider</th>
                      <th>Endpoint</th>
                      <th>Symbol</th>
                      <th>Error</th>
                      <th>Latency</th>
                      <th>Fallback</th>
                      <th>Stale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {failureRows.map((row, index) => (
                      <tr key={`${row.provider}-${row.endpoint}-${row.symbol}-${index}`}>
                        <td><strong>{row.provider}</strong></td>
                        <td>{row.endpoint}</td>
                        <td>{row.symbol}</td>
                        <td>{row.error}</td>
                        <td>{row.latencyLabel}</td>
                        <td>{row.fallback_used ? 'yes' : 'no'}</td>
                        <td>{row.stale_used ? 'yes' : 'no'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="oos-message">No provider failures recorded.</div>}
          </div>
        </>
      )}
    </section>
  )
}
