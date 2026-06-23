import React, { useEffect, useMemo, useRef, useState } from 'react'

import { apiFetch } from '../../lib/api'
import CalibrationInsight from './CalibrationInsight'
import ExpiryModeToggle from './ExpiryModeToggle'
import QualificationBadge from './QualificationBadge'
import RankedSetupTable from './RankedSetupTable'
import ScreenerFilters from './ScreenerFilters'
import ScreenerTable from './ScreenerTable'
import SetupDetailPanel from './SetupDetailPanel'
import { expiryModeLabel, formatTimestamp } from './formatters'

// ── Legacy screener helpers ───────────────────────────────────────────────────

const DEFAULT_FILTERS = {
  query: '',
  releaseTiming: 'AMC',
  status: 'ALL',
  startDate: '',
  endDate: '',
  maxSpread: '',
  minOi: '',
}

const STATUS_ORDER = {
  QUALIFIED: 0,
  MARGINAL: 1,
  EXCLUDED: 2,
}

function compareValues(left, right, sortKey) {
  if (sortKey === 'status') {
    return (STATUS_ORDER[left.status] ?? 9) - (STATUS_ORDER[right.status] ?? 9)
  }
  if (sortKey === 'earnings_date') {
    return String(left.earnings_date).localeCompare(String(right.earnings_date))
  }
  if (sortKey === 'symbol') {
    return String(left.symbol).localeCompare(String(right.symbol))
  }
  const leftValue = Number(left[sortKey] ?? Number.POSITIVE_INFINITY)
  const rightValue = Number(right[sortKey] ?? Number.POSITIVE_INFINITY)
  return leftValue - rightValue
}

// ── Ranked screener helpers ───────────────────────────────────────────────────

const DEFAULT_RANKED_FILTERS = {
  dteMin: 3,
  dteMax: 10,
  minSampleSize: 4,
  releaseFilter: 'all',
  weeks: 4,
  minScore: '',
}

export default function ScreenerConsole({ apiBase, onAnalyzeSymbol }) {
  // Tab: 'ranked' | 'liquidity'
  const [activeTab, setActiveTab] = useState('ranked')

  // ── Liquidity board state (legacy screener) ───────────────────────────────
  const [expiryMode, setExpiryMode] = useState('front_after_earnings')
  const [filters, setFilters] = useState(DEFAULT_FILTERS)
  const [sortKey, setSortKey] = useState('status')
  const [sortDirection, setSortDirection] = useState('asc')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)
  const [selectedKey, setSelectedKey] = useState('')
  const [detailCache, setDetailCache] = useState({})
  // `detailLoadingKey` is the UI-visible loading indicator (state, so the
  // detail panel re-renders when it changes). `detailLoadingKeyRef` mirrors
  // it but stays out of the detail-fetch useEffect's dependency array — the
  // effect MUST NOT re-run when this changes (see the cleanup-aborts-itself
  // bug fixed in PR-U).
  const [detailLoadingKey, setDetailLoadingKey] = useState('')
  const detailLoadingKeyRef = useRef('')

  // ── Ranked setups state ───────────────────────────────────────────────────
  const [rankedFilters, setRankedFilters] = useState(DEFAULT_RANKED_FILTERS)
  const [rankedLoading, setRankedLoading] = useState(false)
  const [rankedError, setRankedError] = useState('')
  const [rankedData, setRankedData] = useState(null)
  const [selectedRankedSymbol, setSelectedRankedSymbol] = useState('')

  // ── Liquidity board functions ─────────────────────────────────────────────

  async function loadScreener(mode = expiryMode) {
    setLoading(true)
    setError('')
    try {
      const response = await apiFetch(`${apiBase}/api/edge/screener?expiry_mode=${mode}`)
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${response.status}`)
      }
      const payload = await response.json()
      setData(payload)
      const firstRow = payload.rows?.[0]
      if (firstRow) {
        setSelectedKey((current) => current || `${firstRow.symbol}-${firstRow.earnings_date}-${firstRow.expiry_mode}`)
      }
    } catch (loadError) {
      setError(String(loadError.message || loadError))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'liquidity') {
      loadScreener(expiryMode)
    }
  }, [expiryMode, activeTab])

  useEffect(() => {
    setSelectedKey('')
  }, [expiryMode])

  const filteredRows = useMemo(() => {
    const rows = data?.rows || []
    const query = filters.query.trim()
    const maxSpread = filters.maxSpread === '' ? null : Number(filters.maxSpread)
    const minOi = filters.minOi === '' ? null : Number(filters.minOi)

    const nextRows = rows.filter((row) => {
      if (query && !row.symbol.includes(query)) return false
      if (filters.releaseTiming !== 'ALL' && row.release_timing !== filters.releaseTiming) return false
      if (filters.status !== 'ALL' && row.status !== filters.status) return false
      if (filters.startDate && String(row.earnings_date) < filters.startDate) return false
      if (filters.endDate && String(row.earnings_date) > filters.endDate) return false
      if (maxSpread != null && row.avg_spread_pct != null && Number(row.avg_spread_pct) > maxSpread) return false
      if (maxSpread != null && row.avg_spread_pct == null) return false
      if (minOi != null && Math.min(row.call_oi ?? 0, row.put_oi ?? 0) < minOi) return false
      return true
    })

    nextRows.sort((left, right) => {
      const base = compareValues(left, right, sortKey)
      if (base !== 0) return sortDirection === 'asc' ? base : base * -1
      return String(left.symbol).localeCompare(String(right.symbol))
    })
    return nextRows
  }, [data, filters, sortDirection, sortKey])

  const selectedRow = useMemo(
    () => filteredRows.find((row) => `${row.symbol}-${row.earnings_date}-${row.expiry_mode}` === selectedKey) || filteredRows[0] || null,
    [filteredRows, selectedKey],
  )

  useEffect(() => {
    if (!filteredRows.length) return
    const exists = filteredRows.some((row) => `${row.symbol}-${row.earnings_date}-${row.expiry_mode}` === selectedKey)
    if (!exists) {
      const firstRow = filteredRows[0]
      setSelectedKey(`${firstRow.symbol}-${firstRow.earnings_date}-${firstRow.expiry_mode}`)
    }
  }, [filteredRows, selectedKey])

  useEffect(() => {
    if (!selectedRow) return
    const cacheKey = `${selectedRow.symbol}-${selectedRow.expiry_mode}`
    // Read the loading key through the ref, NOT through the state — see the
    // PR-U comment block below for why this matters.
    if (detailCache[cacheKey] || detailLoadingKeyRef.current === cacheKey) return

    // PR-R: AbortController cancels the in-flight fetch at the transport
    // layer the moment a newer row supersedes this one — so older requests
    // can't waste bandwidth or land late and overwrite UI state.
    //
    // PR-U: `detailLoadingKey` used to be in this effect's dependency
    // array AND be set inside the effect via setDetailLoadingKey(cacheKey).
    // That combination self-triggered: the state update caused React to
    // run THIS effect's cleanup (abort()) immediately, then re-run the
    // effect, which early-exited via `detailLoadingKey === cacheKey`. Net
    // result: every single click aborted its own fetch, the detail panel
    // showed "loading…" forever, and detailCache was never populated.
    //
    // The fix moves the "are we already fetching this row?" check to a
    // ref (which doesn't trigger re-renders or dep changes) and keeps the
    // state purely for driving the UI loading indicator. The effect's
    // deps array no longer contains anything that the effect itself
    // mutates.
    const controller = new AbortController()
    detailLoadingKeyRef.current = cacheKey
    setDetailLoadingKey(cacheKey)

    apiFetch(`${apiBase}/api/edge/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: selectedRow.symbol }),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => ({}))
          throw new Error(body.detail || `HTTP ${response.status}`)
        }
        return response.json()
      })
      .then((payload) => {
        if (controller.signal.aborted) return
        setDetailCache((current) => ({
          ...current,
          [cacheKey]: { loading: false, error: '', result: payload },
        }))
      })
      .catch((detailError) => {
        // AbortError fires on cleanup when a newer row takes over —
        // expected and ignored.
        if (controller.signal.aborted || (detailError && detailError.name === 'AbortError')) return
        setDetailCache((current) => ({
          ...current,
          [cacheKey]: { loading: false, error: String(detailError.message || detailError), result: null },
        }))
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          detailLoadingKeyRef.current = ''
          setDetailLoadingKey('')
        }
      })

    return () => {
      controller.abort()
    }
  }, [apiBase, detailCache, selectedRow])

  const detailState = selectedRow
    ? detailCache[`${selectedRow.symbol}-${selectedRow.expiry_mode}`] || { loading: detailLoadingKey === `${selectedRow.symbol}-${selectedRow.expiry_mode}` }
    : null

  // ── Ranked setups functions ───────────────────────────────────────────────

  async function loadRanked(overrides = {}) {
    const f = { ...rankedFilters, ...overrides }
    setRankedLoading(true)
    setRankedError('')
    try {
      const params = new URLSearchParams({
        dte_min: f.dteMin,
        dte_max: f.dteMax,
        min_sample_size: f.minSampleSize,
        release_filter: f.releaseFilter,
        weeks: f.weeks,
      })
      const response = await apiFetch(`${apiBase}/api/screener/ranked?${params}`)
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${response.status}`)
      }
      const payload = await response.json()
      setRankedData(payload)
      // Keep the current selection only if it still exists in this result set;
      // otherwise prefer the top scored setup, then the first upcoming row. This
      // prevents the detail panel from showing a stale symbol from a prior run
      // (e.g. after changing filters so the previously-selected row is gone).
      const newRows = payload.rows || []
      setSelectedRankedSymbol((current) => {
        const stillPresent = current && newRows.some(
          (r) => r.symbol === current && (r.status === 'ranked' || r.status === 'upcoming'),
        )
        if (stillPresent) return current
        const firstRanked = newRows.find((r) => r.status === 'ranked')
        if (firstRanked) return firstRanked.symbol
        const firstUpcoming = newRows.find((r) => r.status === 'upcoming')
        return firstUpcoming ? firstUpcoming.symbol : ''
      })
    } catch (err) {
      setRankedError(String(err.message || err))
    } finally {
      setRankedLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'ranked') {
      loadRanked()
    }
  }, [activeTab])

  const visibleRankedRows = useMemo(() => {
    const rows = rankedData?.rows || []
    const minScore = rankedFilters.minScore === '' ? null : Number(rankedFilters.minScore)
    if (minScore == null) return rows
    return rows.filter((r) => r.ranking_score == null || r.ranking_score >= minScore / 100)
  }, [rankedData, rankedFilters.minScore])

  const selectedRankedRow = useMemo(
    () => visibleRankedRows.find((r) => r.symbol === selectedRankedSymbol) || null,
    [visibleRankedRows, selectedRankedSymbol],
  )

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <section className="analysis-block screener-block">
      <div className="screener-header">
        <div>
          <div className="section-title">Earnings Screener</div>
          <p className="screener-subtitle">
            Pre-earnings long-vega setup quality ranking and liquidity board.
          </p>
        </div>
        {/* Tab toggle */}
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <button
            type="button"
            className={activeTab === 'ranked' ? 'primary-btn' : 'secondary-btn'}
            style={{ fontSize: '0.8rem', padding: '5px 12px' }}
            onClick={() => setActiveTab('ranked')}
          >
            Ranked Setups
          </button>
          <button
            type="button"
            className={activeTab === 'liquidity' ? 'primary-btn' : 'secondary-btn'}
            style={{ fontSize: '0.8rem', padding: '5px 12px' }}
            onClick={() => setActiveTab('liquidity')}
          >
            Liquidity Board
          </button>
        </div>
      </div>

      {/* ── RANKED SETUPS TAB ─────────────────────────────────────────── */}
      {activeTab === 'ranked' && (
        <>
          {/* Ranked filter bar */}
          <div className="screener-filters">
            <div className="filter-grid">
              <label>
                <span>DTE min</span>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={rankedFilters.dteMin}
                  onChange={(e) => setRankedFilters((f) => ({ ...f, dteMin: e.target.value }))}
                />
              </label>
              <label>
                <span>DTE max</span>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={rankedFilters.dteMax}
                  onChange={(e) => setRankedFilters((f) => ({ ...f, dteMax: e.target.value }))}
                />
              </label>
              <label>
                <span>Min sample</span>
                <input
                  type="number"
                  min="1"
                  value={rankedFilters.minSampleSize}
                  onChange={(e) => setRankedFilters((f) => ({ ...f, minSampleSize: e.target.value }))}
                />
              </label>
              <label>
                <span>Release</span>
                <select
                  value={rankedFilters.releaseFilter}
                  onChange={(e) => setRankedFilters((f) => ({ ...f, releaseFilter: e.target.value }))}
                >
                  <option value="all">All</option>
                  <option value="bmo">BMO</option>
                  <option value="amc">AMC</option>
                </select>
              </label>
              <label>
                <span>Weeks ahead</span>
                <input
                  type="number"
                  min="1"
                  max="12"
                  value={rankedFilters.weeks}
                  onChange={(e) => setRankedFilters((f) => ({ ...f, weeks: e.target.value }))}
                />
              </label>
              <label>
                <span>Min score (0–100)</span>
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="5"
                  placeholder="—"
                  value={rankedFilters.minScore}
                  onChange={(e) => setRankedFilters((f) => ({ ...f, minScore: e.target.value }))}
                />
              </label>
              <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                <button
                  type="button"
                  className="secondary-btn"
                  style={{ fontSize: '0.8rem', padding: '5px 12px' }}
                  onClick={() => loadRanked()}
                  disabled={rankedLoading}
                >
                  {rankedLoading ? 'Loading…' : 'Run'}
                </button>
              </div>
            </div>
          </div>

          {/* Ranked summary strip */}
          {rankedData && (
            <div className="screener-summary-strip">
              <div className="summary-pill">
                <span>In window</span>
                <strong>{rankedData.in_entry_window}</strong>
              </div>
              <div className="summary-pill" title="Earnings beyond the entry window, in the look-ahead pipeline">
                <span>Upcoming</span>
                <strong>{rankedData.upcoming_count ?? 0}</strong>
              </div>
              <div className="summary-pill">
                <span>Universe</span>
                <strong>{rankedData.universe_size}</strong>
              </div>
              <div className="summary-pill">
                <span>Last refresh</span>
                <strong>{rankedData.generated_at ? new Date(rankedData.generated_at).toLocaleTimeString() : '—'}</strong>
              </div>
              <div className="summary-pill" title={rankedData.strategy_note}>
                <span>Strategy</span>
                <strong style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  Pre-earnings long-vega
                </strong>
              </div>
            </div>
          )}

          {rankedError && <div className="error-banner">{rankedError}</div>}

          <div className="screener-content-grid">
            <div className="screener-board">
              <div className="screener-board-head">
                <div>
                  <h3>Ranked Setups</h3>
                  {(() => {
                    const rankedCount = visibleRankedRows.filter((r) => r.status === 'ranked').length
                    const upcomingCount = visibleRankedRows.filter((r) => r.status === 'upcoming').length
                    if (rankedCount > 0) {
                      return (
                        <p>
                          {rankedCount} setup{rankedCount === 1 ? '' : 's'} in entry window,
                          sorted by setup quality. Click a row to see details.
                        </p>
                      )
                    }
                    if (upcomingCount > 0) {
                      return (
                        <p>
                          No setups in the entry window right now — {upcomingCount} earnings
                          event{upcomingCount === 1 ? '' : 's'} coming up in the look-ahead window (shown
                          below, soonest first). They enter the window as their date approaches.
                        </p>
                      )
                    }
                    return <p>No upcoming earnings in the look-ahead window.</p>
                  })()}
                </div>
              </div>
              <RankedSetupTable
                rows={visibleRankedRows}
                selectedSymbol={selectedRankedSymbol}
                onSelect={(row) => {
                  setSelectedRankedSymbol(row.symbol)
                  if (onAnalyzeSymbol) onAnalyzeSymbol(row.symbol)
                }}
              />
            </div>

            {/* Detail panel */}
            {selectedRankedRow && (
              <div className="detail-panel" style={{ minWidth: 260 }}>
                <div className="detail-panel-header">
                  <span className="detail-symbol">{selectedRankedRow.symbol}</span>
                  {selectedRankedRow.earnings_date && (
                    <span style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>
                      {' · '}Earns {selectedRankedRow.earnings_date}
                      {selectedRankedRow.dte != null ? ` (${selectedRankedRow.dte}d)` : ''}
                    </span>
                  )}
                </div>

                {selectedRankedRow.status === 'upcoming' ? (
                  /* Upcoming rows aren't scored yet — show why, not an empty chart */
                  <div
                    style={{
                      marginTop: 12,
                      padding: '10px 12px',
                      background: 'var(--surface-sunken)',
                      border: '1px solid var(--line-subtle)',
                      borderRadius: 6,
                      fontSize: '0.8rem',
                      color: 'var(--muted)',
                      lineHeight: 1.5,
                    }}
                  >
                    Not yet in the entry window — no setup score until it&apos;s within{' '}
                    {rankedFilters.dteMax} DTE
                    {selectedRankedRow.dte != null ? ` (currently ${selectedRankedRow.dte}d out)` : ''}.
                    Run the analysis below for a full single-ticker view now.
                  </div>
                ) : (
                  <>
                    {/* Score components */}
                    {selectedRankedRow.score_components && Object.keys(selectedRankedRow.score_components).length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
                          Score components
                        </div>
                        {Object.entries(selectedRankedRow.score_components).map(([k, v]) => (
                          <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', padding: '2px 0', borderBottom: '1px solid var(--line-subtle)' }}>
                            <span style={{ color: 'var(--muted)' }}>{k.replace(/_/g, ' ')}</span>
                            <span style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                              {typeof v === 'number' ? v.toFixed(3) : String(v)}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Calibration insight */}
                    <CalibrationInsight
                      apiBase={apiBase}
                      score={selectedRankedRow.ranking_score}
                    />
                  </>
                )}

                {/* Deep-dive link */}
                {onAnalyzeSymbol && (
                  <button
                    type="button"
                    className="secondary-btn"
                    style={{ marginTop: 14, width: '100%', fontSize: '0.8rem' }}
                    onClick={() => onAnalyzeSymbol(selectedRankedRow.symbol)}
                  >
                    Full analysis → {selectedRankedRow.symbol}
                  </button>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {/* ── LIQUIDITY BOARD TAB ───────────────────────────────────────── */}
      {activeTab === 'liquidity' && (
        <>
          <div className="screener-header" style={{ paddingTop: 0, marginTop: 12 }}>
            <div />
            <div className="screener-actions">
              <ExpiryModeToggle value={expiryMode} onChange={setExpiryMode} />
              <button type="button" className="secondary-btn" onClick={() => loadScreener()} disabled={loading}>
                {loading ? 'Refreshing…' : 'Refresh'}
              </button>
            </div>
          </div>

          <div className="screener-summary-strip">
            <div className="summary-pill">
              <span>Methodology</span>
              <strong>{expiryModeLabel(data?.expiry_mode || expiryMode)}</strong>
            </div>
            <div className="summary-pill">
              <span>Qualified</span>
              <strong>{data?.qualified_count ?? 0}</strong>
            </div>
            <div className="summary-pill">
              <span>Marginal</span>
              <strong>{data?.marginal_count ?? 0}</strong>
            </div>
            <div className="summary-pill">
              <span>Excluded</span>
              <strong>{data?.excluded_count ?? 0}</strong>
            </div>
            <div className="summary-pill">
              <span>Universe</span>
              <strong>{data?.universe_size ?? 0}</strong>
            </div>
            <div className="summary-pill">
              <span>Last refresh</span>
              <strong>{formatTimestamp(data?.generated_at)}</strong>
            </div>
          </div>

          <ScreenerFilters
            filters={filters}
            onFilterChange={(key, value) => setFilters((current) => ({ ...current, [key]: value }))}
            sortKey={sortKey}
            sortDirection={sortDirection}
            onSortKeyChange={setSortKey}
            onSortDirectionChange={setSortDirection}
          />

          {error && <div className="error-banner">{error}</div>}

          <div className="screener-content-grid">
            <div className="screener-board">
              <div className="screener-board-head">
                <div>
                  <h3>Action Board</h3>
                  <p>{filteredRows.length} rows after filters. Click a row for the qualification breakdown.</p>
                </div>
                {selectedRow ? <QualificationBadge status={selectedRow.status} /> : null}
              </div>
              <ScreenerTable
                rows={filteredRows}
                selectedKey={selectedKey}
                onSelect={(row) => setSelectedKey(`${row.symbol}-${row.earnings_date}-${row.expiry_mode}`)}
                sortKey={sortKey}
                sortDirection={sortDirection}
                onSortKeyChange={setSortKey}
                onSortDirectionChange={setSortDirection}
              />
            </div>

            <SetupDetailPanel
              row={selectedRow}
              analysisState={detailState}
              onAnalyzeSymbol={onAnalyzeSymbol}
            />
          </div>
        </>
      )}
    </section>
  )
}
