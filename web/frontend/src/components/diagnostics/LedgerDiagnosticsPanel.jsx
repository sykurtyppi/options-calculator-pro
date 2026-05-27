import React, { useEffect, useMemo, useState } from 'react'
import { apiFetch } from '../../lib/api'
import { Badge, SectionTitle } from '../common/DisplayAtoms'
import {
  buildLedgerDetailModel,
  buildLedgerEndpoint,
  buildLedgerExportUrls,
  buildLedgerRows,
  buildLedgerSummaryPills,
  formatLedgerTimestamp,
  getLedgerEmptyMessage,
} from './ledgerDiagnosticsViewModel'

function compactJson(value) {
  return JSON.stringify(value || {}, null, 2)
}

function SummaryPills({ summary }) {
  const pills = buildLedgerSummaryPills(summary || {})

  return (
    <div className="ledger-summary-grid">
      <div className="ledger-summary-pill">
        <span>Total Records</span>
        <strong>{pills.total}</strong>
      </div>
      <div className="ledger-summary-pill">
        <span>Most Selected</span>
        <strong>{pills.topStructure}</strong>
      </div>
      <div className="ledger-summary-pill">
        <span>Fresh / Unknown</span>
        <strong>{pills.freshOrUnknown}</strong>
      </div>
      <div className="ledger-summary-pill ledger-summary-pill-warn">
        <span>Stale Source</span>
        <strong>{pills.stale}</strong>
      </div>
    </div>
  )
}

function RecommendationTable({ rows, selectedId, onSelect, loading }) {
  const displayRows = buildLedgerRows(rows)
  const emptyMessage = getLedgerEmptyMessage(displayRows, loading)
  if (emptyMessage) {
    return <div className="oos-message">{emptyMessage}</div>
  }

  return (
    <div className="structure-table-wrap ledger-table-wrap">
      <table className="structure-table ledger-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Symbol</th>
            <th>Decision</th>
            <th>Structure / Abstain</th>
            <th>Quality</th>
            <th>Earnings Source</th>
            <th>Outcome</th>
            <th>Inspect</th>
          </tr>
        </thead>
        <tbody>
          {displayRows.map((row) => {
            const isSelected = row.recommendation_id === selectedId
            return (
              <tr key={row.recommendation_id} className={isSelected ? 'ledger-row-selected' : ''}>
                <td>{row.displayTimestamp}</td>
                <td><strong>{row.symbol}</strong></td>
                <td>{row.recommendation || 'n/a'}</td>
                <td><span title={row.no_trade_reason || ''}>{row.displayStructureOrReason}</span></td>
                <td>{row.displayQuality}</td>
                <td>
                  <span>{row.displayEarningsSource}</span>
                  {row.staleFlagVisible && <Badge variant="move-risk-elevated">Stale</Badge>}
                </td>
                <td>{row.outcomeLabel}</td>
                <td>
                  <button type="button" className="secondary-button" onClick={() => onSelect(row.recommendation_id)}>
                    Inspect
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function DetailPanel({ detail, linkage, loading }) {
  const scorecards = detail?.structure_scorecards || []
  const quote = detail?.quote_provenance || {}
  const model = buildLedgerDetailModel(detail, linkage)

  if (loading) {
    return <div className="oos-message">Loading ledger record…</div>
  }
  if (!detail) return null

  return (
    <section className="selector-panel selector-panel-full ledger-detail-panel">
      <div className="selector-panel-header">
        <h3>Recommendation Drill-Down</h3>
        <span>Read-only audit record: snapshot, selector reasoning, scorecards, quote provenance, and paper-trade linkage.</span>
      </div>

      <div className="ledger-detail-grid">
        <div className="ledger-detail-card">
          <span>Recommendation ID</span>
          <strong>{model.recommendationId}</strong>
        </div>
        <div className="ledger-detail-card">
          <span>Selected Structure</span>
          <strong>{model.selectedStructure}</strong>
        </div>
        <div className="ledger-detail-card">
          <span>Outcome Link</span>
          <strong>{model.paperTradeLabel}</strong>
        </div>
        <div className="ledger-detail-card">
          <span>Quote Quality</span>
          <strong>{model.quoteQuality}</strong>
        </div>
      </div>

      <div className="ledger-explanation-grid">
        <div>
          <h4>Selector Explanation</h4>
          <p>{model.primaryThesis}</p>
          <ul className="selector-bullet-list selector-bullet-list-compact">
            {model.primaryRisks.map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </div>
        <div>
          <h4>Quote Provenance</h4>
          <p>
            Source: <strong>{model.quoteSource}</strong>
            {model.quoteTimestamp ? ` · ${formatLedgerTimestamp(model.quoteTimestamp)}` : ''}
          </p>
          <p className="ledger-muted">Paper/research quotes are audit evidence only, not execution-grade fills.</p>
        </div>
      </div>

      <details className="selector-detail-toggle" open>
        <summary>Structure scorecards ({scorecards.length})</summary>
        <div className="selector-code-block"><pre>{compactJson(scorecards)}</pre></div>
      </details>
      <details className="selector-detail-toggle">
        <summary>Vol snapshot</summary>
        <div className="selector-code-block"><pre>{compactJson(detail.vol_snapshot)}</pre></div>
      </details>
      <details className="selector-detail-toggle">
        <summary>Quote bid / ask / mid payload</summary>
        <div className="selector-code-block"><pre>{compactJson(quote.bid_ask_mid)}</pre></div>
      </details>
    </section>
  )
}

export default function LedgerDiagnosticsPanel({ apiBase }) {
  const [symbol, setSymbol] = useState('')
  const [limit, setLimit] = useState(25)
  const [offset, setOffset] = useState(0)
  const [rows, setRows] = useState([])
  const [summary, setSummary] = useState(null)
  const [total, setTotal] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [linkage, setLinkage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [detailLoading, setDetailLoading] = useState(false)
  const [error, setError] = useState('')

  const endpoint = useMemo(() => {
    return buildLedgerEndpoint(apiBase, { symbol, limit, offset })
  }, [apiBase, symbol, limit, offset])

  const exportUrls = useMemo(() => {
    return buildLedgerExportUrls(apiBase, { symbol, limit, offset })
  }, [apiBase, symbol, limit, offset])

  async function loadRows() {
    setLoading(true)
    setError('')
    try {
      const [recentResponse, summaryResponse] = await Promise.all([
        apiFetch(endpoint),
        apiFetch(`${apiBase}/api/diagnostics/recommendations/summary`),
      ])
      if (!recentResponse.ok) throw new Error(`Ledger query failed (${recentResponse.status})`)
      if (!summaryResponse.ok) throw new Error(`Ledger summary failed (${summaryResponse.status})`)
      const recentPayload = await recentResponse.json()
      const summaryPayload = await summaryResponse.json()
      setRows(recentPayload.rows || [])
      setSummary(summaryPayload.summary || null)
      setTotal(Number(recentPayload.total || 0))
      setHasMore(Boolean(recentPayload.has_more))
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  async function loadDetail(recommendationId) {
    if (!recommendationId) return
    setSelectedId(recommendationId)
    setDetailLoading(true)
    setError('')
    try {
      const [detailResponse, linkageResponse] = await Promise.all([
        apiFetch(`${apiBase}/api/diagnostics/recommendations/${encodeURIComponent(recommendationId)}`),
        apiFetch(`${apiBase}/api/diagnostics/recommendations/${encodeURIComponent(recommendationId)}/linkage`),
      ])
      if (!detailResponse.ok) throw new Error(`Ledger detail failed (${detailResponse.status})`)
      if (!linkageResponse.ok) throw new Error(`Ledger linkage failed (${linkageResponse.status})`)
      setDetail(await detailResponse.json())
      setLinkage(await linkageResponse.json())
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setDetailLoading(false)
    }
  }

  useEffect(() => {
    loadRows()
  }, [endpoint])

  function updateSymbol(value) {
    setSymbol(value.toUpperCase())
    setOffset(0)
  }

  function updateLimit(value) {
    setLimit(Number(value) || 25)
    setOffset(0)
  }

  const pageNumber = Math.floor(offset / limit) + 1
  const pageCount = Math.max(1, Math.ceil(total / limit))

  return (
    <section className="oos-block ledger-diagnostics-block">
      <div className="oos-head">
        <div>
          <SectionTitle>Recommendation Ledger</SectionTitle>
          <p className="oos-help">Read-only audit trail for selector recommendations, data provenance, quote quality, and paper-trade outcome linkage.</p>
        </div>
        <div className="oos-actions">
          <a className="secondary-button" href={exportUrls.recommendationCsv}>Export CSV</a>
          <a className="secondary-button" href={exportUrls.recommendationJson}>Export JSON</a>
          <a className="secondary-button" href={exportUrls.linkageCsv}>Export Linkage CSV</a>
          <button type="button" onClick={loadRows} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      <SummaryPills summary={summary} />

      <div className="warehouse-controls ledger-controls">
        <label className="oos-field">
          <span>Symbol Filter</span>
          <input
            value={symbol}
            maxLength={10}
            onChange={(event) => updateSymbol(event.target.value)}
            placeholder="AAPL"
          />
        </label>
        <label className="oos-field">
          <span>Rows</span>
          <select value={limit} onChange={(event) => updateLimit(event.target.value)}>
            <option value={10}>10</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </label>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading ? <div className="oos-message">Loading recommendation ledger…</div> : (
        <RecommendationTable rows={rows} selectedId={selectedId} onSelect={loadDetail} loading={loading} />
      )}
      <div className="ledger-pagination">
        <button
          type="button"
          className="secondary-button"
          disabled={offset <= 0 || loading}
          onClick={() => setOffset(Math.max(0, offset - limit))}
        >
          Previous
        </button>
        <span>Page {pageNumber} of {pageCount} · {total} records</span>
        <button
          type="button"
          className="secondary-button"
          disabled={!hasMore || loading}
          onClick={() => setOffset(offset + limit)}
        >
          Next
        </button>
      </div>
      <DetailPanel detail={detail} linkage={linkage} loading={detailLoading} />
    </section>
  )
}
