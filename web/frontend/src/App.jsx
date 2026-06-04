import React, { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react'
import ScreenerConsole from './components/screener/ScreenerConsole'
import SelectorDecisionCard from './components/edge/SelectorDecisionCard'
import DecisionEvidenceStrip from './components/edge/DecisionEvidenceStrip'
import CalibrationInsightPanel from './components/edge/CalibrationInsightPanel'
import RegimeNarrativePanel from './components/edge/RegimeNarrativePanel'
import OutcomeRiskPanel from './components/edge/OutcomeRiskPanel'
import WhyStructurePanel from './components/edge/WhyStructurePanel'
import StructureComparisonTable from './components/edge/StructureComparisonTable'
import VolSnapshotPanel from './components/edge/VolSnapshotPanel'
import EvidenceQualityPanel from './components/edge/EvidenceQualityPanel'
import { Badge, Metric, SectionTitle } from './components/common/DisplayAtoms'
import {
  releaseTimeBadge,
  dataSourceBadge,
  TickerTierBadge,
  VolRegimeBadge,
  MoveRiskBadge,
} from './components/common/badges'
import TermStructureChart from './components/charts/TermStructureChart'
import OosSplitChart from './components/charts/OosSplitChart'
import StructurePayoffChart from './components/charts/StructurePayoffChart'
import CalendarSpreadChart from './components/charts/CalendarSpreadChart'
import WatchlistChips from './components/alerts/WatchlistChips'
import AlertBanner from './components/alerts/AlertBanner'
import AlertConfigPanel from './components/alerts/AlertConfigPanel'
import { useWatchlist } from './lib/hooks/useWatchlist'
import { useAlertConfig } from './lib/hooks/useAlertConfig'
import { exportJson } from './lib/exportJson'
import { DEFAULT_OOS_PARAMS, OOS_PROFILE_LABELS, OOS_PROFILE_PRESETS, TODAY_ISO } from './config/oosConfig'
import {
  fmtMoney,
  fmtNum,
  fmtPct,
  fmtPp,
  fmtSn,
  fmtSpp,
  fmtVol,
  formatTimestamp,
  parseFloatOr,
  parseIntOr,
  toneNeg,
  tonePos,
} from './lib/formatters'

import { API_BASE, apiFetch } from './lib/api'
const LegacyAnalysisPanel = lazy(() => import('./components/edge/LegacyAnalysisPanel'))
const HistoricalWarehousePanel = lazy(() => import('./components/historical/HistoricalWarehousePanel'))
const LedgerDiagnosticsPanel = lazy(() => import('./components/diagnostics/LedgerDiagnosticsPanel'))
const DataQualityDiagnosticsPanel = lazy(() => import('./components/diagnostics/DataQualityDiagnosticsPanel'))
const ProviderTelemetryPanel = lazy(() => import('./components/diagnostics/ProviderTelemetryPanel'))
const ForwardPerformancePanel = lazy(() => import('./components/diagnostics/ForwardPerformancePanel'))
const EvidenceReportPanel = lazy(() => import('./components/diagnostics/EvidenceReportPanel'))

// Inline chart / badge / hook / alert definitions were moved to dedicated
// modules in PR-S so the main App component stays readable:
//   - charts/{TermStructureChart, OosSplitChart, StructurePayoffChart,
//             CalendarSpreadChart}
//   - common/badges (releaseTimeBadge, dataSourceBadge, TickerTierBadge,
//                    VolRegimeBadge, MoveRiskBadge)
//   - lib/hooks/{useWatchlist, useAlertConfig}
//   - alerts/{WatchlistChips, AlertBanner, AlertConfigPanel}
//   - lib/exportJson



// ── Logout button ────────────────────────────────────────────────────────────
//
// Frontend-audit P1-2: the backend has POST /logout (HMAC-verified nonce
// revocation, see PR #59) but there was no UI affordance to call it.
// Renders as a small unobtrusive link-style button in the header. On
// click: POST /logout with credentials, then hard-reload to /login so
// the cleared cookie state is reflected immediately.
function LogoutButton() {
  const [busy, setBusy] = useState(false)
  async function handleLogout() {
    if (busy) return
    setBusy(true)
    try {
      // Errors here are non-blocking — even if the network call fails,
      // we still navigate the user to /login. The backend's
      // delete_cookie is the ONLY thing that clears the cookie on the
      // browser; if that fails, the user re-logs in on their next
      // attempt and the old session expires naturally.
      await apiFetch(`${API_BASE}/logout`, { method: 'POST' })
    } catch {
      // Swallowed on purpose — see comment above.
    } finally {
      // Hard navigation rather than React-router push: ensures any
      // cached fetches / module-level state from the authenticated
      // session can't carry over.
      window.location.assign('/login')
    }
  }
  return (
    <button
      type="button"
      className="logout-button"
      onClick={handleLogout}
      disabled={busy}
      aria-label="Log out"
    >
      {busy ? 'Logging out…' : 'Log out'}
    </button>
  )
}


// ── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const { watchlist, addToWatchlist, removeFromWatchlist, isInWatchlist } = useWatchlist()
  const { config: alertConfig, setConfig: setAlertConfig } = useAlertConfig()
  const [alertPanelOpen, setAlertPanelOpen] = useState(false)

  const [symbol, setSymbol] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const [oosLoading, setOosLoading] = useState(false)
  const [oosError, setOosError] = useState('')
  const [oosResult, setOosResult] = useState(null)
  const [oosParams, setOosParams] = useState(DEFAULT_OOS_PARAMS)
  const [oosElapsedSec, setOosElapsedSec] = useState(0)
  // Polling interval ref — cleared when job completes, errors, or user cancels.
  const oosIntervalRef = useRef(null)
  // Set to false on unmount so runOos doesn't create an interval or set state
  // after the component is gone (covers unmount mid-submit and mid-poll).
  const mountedRef = useRef(true)
  // AbortController for the in-flight /api/edge/analyze fetch. A fresh
  // analysis request aborts whatever was previously in-flight so a slow
  // older response can't land last and clobber the newer result.
  const analyzeAbortRef = useRef(null)
  const [warehouse, setWarehouse] = useState({
    available: false,
    symbols: [],
    symbol: 'SPY',
    tradeDate: '2025-01-02',
    minDte: '20',
    maxDte: '90',
    callPut: 'C',
    limit: '25',
    coverage: null,
    rows: [],
    loadingSymbols: false,
    loadingRows: false,
    error: '',
  })

  useEffect(() => {
    if (!oosLoading) { setOosElapsedSec(0); return }
    const t = setInterval(() => setOosElapsedSec(p => p + 1), 1000)
    return () => clearInterval(t)
  }, [oosLoading])

  // On unmount: mark as unmounted and clear any live poll interval.
  // mountedRef guards runOos against creating a new interval or setting state
  // if the component tears down while /api/oos/submit is still in flight.
  //
  // IMPORTANT: set mountedRef.current = true in the effect body, not just via
  // useRef(true). React StrictMode (active in dev via main.jsx) runs
  // setup → cleanup → setup on every mount. Without the reset in the body,
  // the second setup leaves mountedRef false, so runOos() would return early
  // after every submit and leave oosLoading stuck true.
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      if (oosIntervalRef.current) clearInterval(oosIntervalRef.current)
    }
  }, [])

  useEffect(() => {
    loadWarehouseSymbols()
  }, [])

  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol])

  const hardGateReasons = useMemo(() => {
    const r = []
    const eg = result?.metrics?.hard_gate_reasons
    if (Array.isArray(eg)) r.push(...eg.filter(Boolean))
    if (oosResult?.summary?.overall_pass === false)
      r.push(`OOS gate failed (grade=${oosResult?.summary?.grade || 'N/A'}).`)
    return r
  }, [result, oosResult])

  const shortLegAligned = useMemo(() => {
    const eDte = result?.metrics?.days_to_earnings
    const nDte = result?.metrics?.near_term_dte
    if (eDte == null || nDte == null) return null
    return Number(eDte) < Number(nDte)
  }, [result])

  const noTradeBlocked = hardGateReasons.length > 0
  const recommendationValue = result
    ? (noTradeBlocked ? 'No Trade' : result?.recommendation || '--')
    : '--'
  const confidenceValue = result
    ? `${Number(result.confidence_pct).toFixed(1)}%`
    : '--'

  function updateOosParam(key, value) { setOosParams(p => ({ ...p, [key]: value })) }
  function applyOosPreset(name) {
    setOosParams(p => ({ ...p, ...(OOS_PROFILE_PRESETS[name] || {}), oos_stability_profile: name }))
  }

  function buildOosPayload() {
    return {
      oos_stability_profile: oosParams.oos_stability_profile || 'stability_auto',
      lookback_days: parseIntOr(oosParams.lookback_days, 1095),
      max_backtest_symbols: parseIntOr(oosParams.max_backtest_symbols, 50),
      backtest_start_date: oosParams.backtest_start_date || null,
      backtest_end_date: oosParams.backtest_end_date || null,
      min_signal_score: parseFloatOr(oosParams.min_signal_score, 0.5),
      min_crush_confidence: parseFloatOr(oosParams.min_crush_confidence, 0.3),
      min_crush_magnitude: parseFloatOr(oosParams.min_crush_magnitude, 0.06),
      min_crush_edge: parseFloatOr(oosParams.min_crush_edge, 0.02),
      target_entry_dte: parseIntOr(oosParams.target_entry_dte, 6),
      entry_dte_band: parseIntOr(oosParams.entry_dte_band, 6),
      min_daily_share_volume: parseIntOr(oosParams.min_daily_share_volume, 1_000_000),
      max_abs_momentum_5d: parseFloatOr(oosParams.max_abs_momentum_5d, 0.11),
      oos_train_days: parseIntOr(oosParams.oos_train_days, 189),
      oos_test_days: parseIntOr(oosParams.oos_test_days, 42),
      oos_step_days: parseIntOr(oosParams.oos_step_days, 42),
      oos_top_n_train: parseIntOr(oosParams.oos_top_n_train, 1),
      oos_min_splits: parseIntOr(oosParams.oos_min_splits, 8),
      oos_min_total_test_trades: parseIntOr(oosParams.oos_min_total_test_trades, 80),
      oos_min_trades_per_split: parseFloatOr(oosParams.oos_min_trades_per_split, 5.0),
    }
  }

  function cancelOos() {
    if (oosIntervalRef.current) {
      clearInterval(oosIntervalRef.current)
      oosIntervalRef.current = null
    }
    setOosLoading(false)
    setOosError('OOS run cancelled (job continues on server until complete).')
  }

  async function runForSymbol(sym) {
    const s = (sym || normalizedSymbol).trim().toUpperCase()
    if (!s) return
    if (sym) setSymbol(sym)

    // Cancel any in-flight analyze fetch so a slow older response can't
    // land after a newer one and overwrite setResult/setError with stale
    // data. This was a real race when the user clicked tickers rapidly.
    if (analyzeAbortRef.current) {
      analyzeAbortRef.current.abort()
    }
    const controller = new AbortController()
    analyzeAbortRef.current = controller

    setError(''); setLoading(true); setResult(null)
    try {
      const res = await apiFetch(`${API_BASE}/api/edge/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: s }),
        signal: controller.signal,
      })
      if (controller.signal.aborted) return
      if (!res.ok) {
        const b = await res.json().catch(() => ({}))
        throw new Error(b.detail || `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (err) {
      // AbortError is expected when a newer request superseded this one —
      // silently drop it; the newer call owns the UI state now.
      if (err && err.name === 'AbortError') return
      setError(String(err.message || err))
    } finally {
      // Only release loading/ref state if we're still the current
      // in-flight call. If a newer call took over, leave its state alone.
      if (analyzeAbortRef.current === controller) {
        analyzeAbortRef.current = null
        setLoading(false)
      }
    }
  }

  async function runAnalysis(e) {
    e?.preventDefault()
    return runForSymbol(null)
  }

  async function loadWarehouseSymbols() {
    setWarehouse(p => ({ ...p, loadingSymbols: true, error: '' }))
    try {
      const res = await apiFetch(`${API_BASE}/api/historical/options/symbols`)
      if (!res.ok) {
        const b = await res.json().catch(() => ({}))
        throw new Error(b.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      setWarehouse(p => ({
        ...p,
        available: Boolean(data.available),
        symbols: data.symbols || [],
        loadingSymbols: false,
        error: '',
      }))
    } catch (err) {
      setWarehouse(p => ({ ...p, loadingSymbols: false, error: String(err.message || err) }))
    }
  }

  async function runWarehouseQuery() {
    const sym = warehouse.symbol.trim().toUpperCase()
    if (!sym || !warehouse.tradeDate) return
    setWarehouse(p => ({ ...p, loadingRows: true, error: '', rows: [] }))
    try {
      const coverageRes = await apiFetch(`${API_BASE}/api/historical/options/${encodeURIComponent(sym)}/coverage`)
      if (!coverageRes.ok) {
        const b = await coverageRes.json().catch(() => ({}))
        throw new Error(b.detail || `Coverage HTTP ${coverageRes.status}`)
      }
      const coverageData = await coverageRes.json()

      const params = new URLSearchParams()
      params.set('trade_date', warehouse.tradeDate)
      if (warehouse.minDte) params.set('min_dte', warehouse.minDte)
      if (warehouse.maxDte) params.set('max_dte', warehouse.maxDte)
      if (warehouse.callPut) params.set('call_put', warehouse.callPut)
      if (warehouse.limit) params.set('limit', warehouse.limit)

      const chainRes = await apiFetch(`${API_BASE}/api/historical/options/${encodeURIComponent(sym)}/chain?${params.toString()}`)
      if (!chainRes.ok) {
        const b = await chainRes.json().catch(() => ({}))
        throw new Error(b.detail || `Chain HTTP ${chainRes.status}`)
      }
      const chainData = await chainRes.json()
      setWarehouse(p => ({
        ...p,
        symbol: sym,
        loadingRows: false,
        coverage: coverageData.coverage || [],
        rows: chainData.rows || [],
        error: '',
      }))
    } catch (err) {
      setWarehouse(p => ({ ...p, loadingRows: false, error: String(err.message || err) }))
    }
  }

  async function runOos() {
    // Stop any existing poll before starting a new job.
    if (oosIntervalRef.current) {
      clearInterval(oosIntervalRef.current)
      oosIntervalRef.current = null
    }
    setOosLoading(true); setOosError(''); setOosResult(null)

    let jobId = null
    try {
      // Submit — returns immediately with a job_id.
      const submitRes = await apiFetch(`${API_BASE}/api/oos/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildOosPayload()),
      })
      if (!submitRes.ok) {
        const b = await submitRes.json().catch(() => ({}))
        throw new Error(b.detail || `HTTP ${submitRes.status}`)
      }
      const { job_id } = await submitRes.json()
      jobId = job_id

      // Guard: if component unmounted while submit was in flight, don't
      // create an interval or set any state — the interval would fire into
      // a gone component and React would warn about state on unmounted nodes.
      if (!mountedRef.current) return

      // Poll every 2 s until done, error, or user cancels.
      oosIntervalRef.current = setInterval(async () => {
        if (!mountedRef.current) {
          clearInterval(oosIntervalRef.current)
          oosIntervalRef.current = null
          return
        }
        try {
          const statusRes = await apiFetch(`${API_BASE}/api/oos/status/${jobId}`)
          if (!statusRes.ok) return  // transient error — keep polling
          const statusData = await statusRes.json()
          if (!mountedRef.current) return  // unmounted while status fetch was in flight

          if (statusData.status === 'complete') {
            clearInterval(oosIntervalRef.current)
            oosIntervalRef.current = null
            setOosLoading(false)
            setOosResult(statusData.data)
          } else if (statusData.status === 'error') {
            clearInterval(oosIntervalRef.current)
            oosIntervalRef.current = null
            setOosLoading(false)
            setOosError(statusData.error || 'OOS job failed on server.')
          }
          // 'pending' / 'running' → keep polling; elapsed driven by client counter
        } catch (_pollErr) {
          // Network hiccup during poll — stay quiet and retry next tick
        }
      }, 2000)
    } catch (err) {
      if (oosIntervalRef.current) {
        clearInterval(oosIntervalRef.current)
        oosIntervalRef.current = null
      }
      if (mountedRef.current) {
        setOosLoading(false)
        setOosError(String(err.message || err))
      }
    }
  }

  const m = result?.metrics || {}
  const selectorOutput = result?.selector_output || null
  const structureScorecards = result?.structure_scorecards || []
  const volSnapshot = result?.vol_snapshot || null

  return (
    <div className="page-shell">
      <div className="bg-orb bg-orb-a" />
      <div className="bg-orb bg-orb-b" />
      <main className="terminal-panel">

        {/* ── Header ── */}
        <header className="panel-header">
          <div>
            <h1>Earnings Volatility Research</h1>
            <p>Pre-earnings volatility setup quality analysis · Forward screener plus single-ticker drill-down</p>
          </div>
          <div className="header-badges">
            <a className="docs-link" href={`${API_BASE}/product-docs/architecture.md`} target="_blank" rel="noreferrer">
              Learn how this works
            </a>
            <div className="status-chip">Research Tool · Not Financial Advice</div>
            <LogoutButton />
          </div>
        </header>

        <ScreenerConsole apiBase={API_BASE} onAnalyzeSymbol={runForSymbol} />

        {/* ── Edge Analysis ── */}
        <section className="analysis-block">
          <div className="section-header-row">
            <SectionTitle>Edge Analysis</SectionTitle>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {result && (
                <div className="data-source-row">
                  {dataSourceBadge(m.data_source, m.data_sources)}
                </div>
              )}
              <button
                className="export-btn"
                title="Alert configuration"
                onClick={() => setAlertPanelOpen(p => !p)}
              >🔔 Alerts</button>
            </div>
          </div>
          {alertPanelOpen && <AlertConfigPanel config={alertConfig} setConfig={setAlertConfig} />}

          <WatchlistChips
            watchlist={watchlist}
            onSelect={runForSymbol}
            onRemove={removeFromWatchlist}
          />

          <form className="symbol-form" onSubmit={runAnalysis}>
            <label htmlFor="symbol-input">Ticker</label>
            <input
              id="symbol-input"
              maxLength={10}
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
              autoComplete="off"
            />
            <button type="submit" disabled={loading || !normalizedSymbol}>
              {loading ? 'Analyzing…' : 'Run Edge Analysis'}
            </button>
          </form>

          {error && <div className="error-banner">{error}</div>}
          <AlertBanner config={alertConfig} result={result} />

          {result && (
            <>
              <div className="selector-topbar">
                <div className="selector-topbar-left">
                  {m.earnings_release_time && releaseTimeBadge(m.earnings_release_time)}
                  <VolRegimeBadge regime={m.vol_regime} pct={m.rv_percentile_rank} />
                  <TickerTierBadge tier={m.ticker_tier} />
                  <MoveRiskBadge level={m.move_risk_level} ratio={m.move_risk_ratio} sampleSize={m.move_risk_sample_size} />
                </div>
                <div className="selector-topbar-right">
                  <button
                    className="export-btn"
                    title="Export full analysis as JSON"
                    onClick={() => exportJson(normalizedSymbol, result)}
                  >↓ JSON</button>
                  <button
                    className={`star-btn${isInWatchlist(normalizedSymbol) ? ' starred' : ''}`}
                    title={isInWatchlist(normalizedSymbol) ? 'Remove from watchlist' : 'Add to watchlist'}
                    onClick={() => {
                      if (isInWatchlist(normalizedSymbol)) {
                        removeFromWatchlist(normalizedSymbol)
                      } else {
                        addToWatchlist({
                          symbol: normalizedSymbol,
                          recommendation: selectorOutput?.recommendation || recommendationValue,
                          confidence_pct: Number(selectorOutput?.confidence_pct ?? result.confidence_pct),
                          setup_score: Number(result.setup_score),
                          vol_regime: m.vol_regime,
                          timestamp: new Date().toISOString(),
                        })
                      }
                    }}
                  >{isInWatchlist(normalizedSymbol) ? '★' : '☆'}</button>
                </div>
              </div>

              <SelectorDecisionCard
                selectorOutput={selectorOutput}
                scorecards={structureScorecards}
                volSnapshot={volSnapshot}
              />
              <RegimeNarrativePanel
                selectorOutput={selectorOutput}
                scorecards={structureScorecards}
                volSnapshot={volSnapshot}
              />
              <DecisionEvidenceStrip
                selectorOutput={selectorOutput}
                scorecards={structureScorecards}
                volSnapshot={volSnapshot}
              />
              <div className="selector-secondary-grid selector-secondary-grid-trust">
                <CalibrationInsightPanel
                  apiBase={API_BASE}
                  score={result.setup_score}
                />
                <EvidenceQualityPanel
                  selectorOutput={selectorOutput}
                  scorecards={structureScorecards}
                  volSnapshot={volSnapshot}
                />
              </div>
              <WhyStructurePanel
                selectorOutput={selectorOutput}
                scorecards={structureScorecards}
                volSnapshot={volSnapshot}
              />
              <OutcomeRiskPanel
                selectorOutput={selectorOutput}
                scorecards={structureScorecards}
                volSnapshot={volSnapshot}
              />
              <StructureComparisonTable
                scorecards={structureScorecards}
                selectorOutput={selectorOutput}
              />
              <div className="selector-secondary-grid selector-secondary-grid-snapshot">
                <VolSnapshotPanel volSnapshot={volSnapshot} />
              </div>

              {m.term_structure_days?.length >= 2 && (
                <div className="selector-panel selector-panel-vol-term">
                  <div className="selector-panel-header">
                    <h3>Vol Term Structure</h3>
                    <span>Implied vol across expirations — steepness drives calendar carry; event premium shows near-term elevation.</span>
                  </div>
                  <TermStructureChart
                    days={m.term_structure_days}
                    ivs={m.term_structure_ivs}
                    earningsDte={m.days_to_earnings}
                  />
                </div>
              )}

              {m.structure_payoff && (() => {
                const sp = m.structure_payoff
                const isCalendar = sp.structure === 'call_calendar' || sp.structure === 'put_calendar'
                const isStrangle = sp.structure === 'otm_strangle'
                const structureLabels = {
                  atm_straddle:  'Long ATM Straddle',
                  otm_strangle:  'Long OTM Strangle',
                  call_calendar: 'Call Calendar · Short Near / Long Back+28d',
                  put_calendar:  'Put Calendar · Short Near / Long Back+28d',
                }
                const panelTitle = structureLabels[sp.structure] || sp.structure
                return (
                  <div className="selector-panel selector-panel-structure-payoff">
                    <div className="selector-panel-header">
                      <h3>{panelTitle}</h3>
                      <span>
                        {isCalendar && m.calendar_spread_quality && (() => {
                          const qColors = { Strong: '#22c55e', Moderate: '#58a6ff', Weak: '#f0a020', Poor: '#ef4444', unknown: '#8b949e' }
                          return (
                            <span style={{ fontWeight: 600, color: qColors[m.calendar_spread_quality] || '#8b949e', marginRight: 10 }}>
                              {m.calendar_spread_quality} term structure ·&nbsp;
                            </span>
                          )
                        })()}
                        {isStrangle && sp.wing_pct != null && (
                          <span style={{ marginRight: 10 }}>Wings ±{Number(sp.wing_pct).toFixed(1)}% (at implied move) ·&nbsp;</span>
                        )}
                        <span style={{ color: '#6e7681' }}>IV scenarios: symbol-calibrated where available</span>
                      </span>
                    </div>

                    {/* Calendar-specific: breakeven coverage warning */}
                    {isCalendar && m.calendar_be_vs_implied != null && m.calendar_be_vs_implied < 0.70 && (
                      <div style={{
                        background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.35)',
                        borderRadius: 6, padding: '7px 12px', marginBottom: 8, fontSize: 12, color: '#fca5a5',
                      }}>
                        ⚠ Breakevens cover only {Math.round(m.calendar_be_vs_implied * 100)}% of the implied move.
                        The market is pricing a larger move than this structure profits from.
                      </div>
                    )}

                    <StructurePayoffChart payoff={sp} />

                    {/* Metadata footer */}
                    {isCalendar && sp.iv_near != null && (
                      <div style={{ display: 'flex', gap: 18, marginTop: 6, fontSize: 11, color: '#8b949e', flexWrap: 'wrap' }}>
                        <span>Near IV: {(sp.iv_near * 100).toFixed(1)}%</span>
                        <span>Back IV: {(sp.iv_back * 100).toFixed(1)}%</span>
                        <span>Remaining: {sp.t_remaining_days}d</span>
                        {m.calendar_be_vs_implied != null && (
                          <span style={{ color: m.calendar_be_vs_implied < 0.70 ? '#f0a020' : '#8b949e' }}>
                            BE covers {Math.round(m.calendar_be_vs_implied * 100)}% of implied move
                          </span>
                        )}
                      </div>
                    )}
                    {(sp.structure === 'atm_straddle' || isStrangle) && sp.iv_entry != null && (
                      <div style={{ display: 'flex', gap: 18, marginTop: 6, fontSize: 11, color: '#8b949e', flexWrap: 'wrap' }}>
                        <span>Entry IV: {(sp.iv_entry * 100).toFixed(1)}%</span>
                        <span>DTE: {sp.T_near_days}d</span>
                        <span>P&L eval: 1-day post-event</span>
                      </div>
                    )}
                  </div>
                )
              })()}

              <Suspense fallback={<div className="oos-message">Loading legacy analysis…</div>}>
              <LegacyAnalysisPanel>
              {/* Recommendation strip */}
              {noTradeBlocked && (
                <div className="no-trade-banner">
                  <div className="no-trade-title">NO TRADE — Legacy Hard Gate Fail</div>
                  <ul>
                    {hardGateReasons.map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </div>
              )}

              <div className="rec-strip legacy-rec-strip">
                <div className="rec-left">
                  <span className={`rec-badge rec-${recommendationValue.toLowerCase().replace(' ', '-')}`}>
                    {recommendationValue}
                  </span>
                  <span
                    className="rec-confidence"
                    title="Legacy setup score. Not a calibrated win rate."
                  >{confidenceValue} setup score</span>
                  {m.earnings_release_time && releaseTimeBadge(m.earnings_release_time)}
                </div>
                <div className="rec-right">
                  <span className="rec-detail">Setup {Number(result.setup_score).toFixed(3)}</span>
                  <span className="rec-detail">Composite {fmtNum(m.composite_score, 3)}</span>
                  <span className={`rec-detail edge-quality-${(m.edge_quality || '').toLowerCase().replace(/\s+/g, '-')}`}>
                    {m.edge_quality || 'n/a'}
                  </span>
                </div>
              </div>

              {/* Main metrics grid */}
              <div className="metrics-group">
                <div className="metrics-group-label">Edge &amp; Expectancy</div>
                <div className="metrics-grid">
                  <Metric label="Expected Net Edge" value={fmtSpp(m.expected_net_edge_pct)}
                    tone={tonePos(m.expected_net_edge_pct, 0.25, 0)} />
                  <Metric label="Expected Gross Edge" value={fmtSpp(m.expected_gross_edge_pct)}
                    tone={tonePos(m.expected_gross_edge_pct, 0.5, 0)} />
                  <Metric label="Expectancy Ratio" value={fmtSn(m.expectancy_ratio, 2)}
                    tone={tonePos(m.expectancy_ratio, 0.2, 0)} />
                  <Metric label="Implied / Anchor" value={fmtNum(m.implied_vs_anchor_ratio, 2)}
                    tone={tonePos(m.implied_vs_anchor_ratio, 1.05, 1.0)} />
                  <Metric label="Drawdown Risk" value={fmtPp(m.drawdown_risk_pct, 2)}
                    tone={toneNeg(m.drawdown_risk_pct, 1.25, 2.0)} />
                  {/* FIX 7: clarify this is a model-derived friction score, not a broker-quoted cost */}
                  <Metric label="Friction Score" value={fmtPp(m.tx_cost_estimate_pct, 2)}
                    tone={toneNeg(m.tx_cost_estimate_pct, 0.5, 1.0)}
                    sub="heuristic · not broker-quoted" />
                </div>
              </div>

              {(m.expansion_history_status || m.analysis_mode === 'iv_expansion') && (
                <div className="metrics-group">
                  <div className="metrics-group-label">IV Expansion Setup</div>
                  <div className="metrics-grid">
                    <Metric
                      label="Expected IV Change"
                      value={m.expansion_expected_iv_change != null ? fmtPp(m.expansion_expected_iv_change * 100, 2) : '--'}
                      sub="avg front/back IV points"
                      tone={tonePos(m.expansion_expected_iv_change, 0.01, 0)}
                    />
                    <Metric
                      label="Modeled P&L Mid"
                      value={fmtMoney(m.expansion_expected_pnl_mid)}
                      sub="per 1-lot calendar"
                      tone={tonePos(m.expansion_expected_pnl_mid, 0, 0)}
                    />
                    <Metric
                      label="Modeled P&L Adj"
                      value={fmtMoney(m.expansion_expected_pnl_adjusted)}
                      sub="quoted entry/exit"
                      tone={tonePos(m.expansion_expected_pnl_adjusted, 0, 0)}
                    />
                    <Metric
                      label="Ranking Score"
                      value={fmtNum(m.expansion_ranking_score, 3)}
                      tone={tonePos(m.expansion_ranking_score, 0.65, 0.5)}
                    />
                    <Metric
                      label="Entry / Exit"
                      value={
                        m.expansion_selected_entry_offset_days != null
                          ? `T-${m.expansion_selected_entry_offset_days} / T-${m.expansion_exit_offset_days ?? 1}`
                          : '--'
                      }
                      sub={m.expansion_structure || 'same-strike calendar'}
                    />
                    <Metric
                      label="Historical Sample"
                      value={
                        m.expansion_historical_sample_size != null
                          ? `${m.expansion_priceable_trades || 0}/${m.expansion_historical_sample_size}`
                          : '--'
                      }
                      sub={
                        m.expansion_missing_exit_quotes != null
                          ? `${m.expansion_missing_exit_quotes} missing exits`
                          : (m.expansion_history_status || 'no history')
                      }
                    />
                  </div>
                </div>
              )}

              {/*
                The legacy "Score Breakdown" panel was removed in the frontend
                honesty pass. It presented `confidence_pct_raw` → "adjusted via
                tier × kurtosis × inside-IV rate" with a "Combined Multiplier",
                implying the displayed confidence was the raw score times those
                multipliers. The engine never does this: confidence_pct ==
                confidence_pct_raw (both = selector confidence) and the
                calibration multiplier is computed for audit/inspection only,
                never applied (see web/api/edge_engine.py — _calibration_mult is
                exposed in metrics but confidence_pct is set to the selector
                value). The multiplier components remain in the API JSON for
                anyone auditing; they are no longer presented as an applied
                score adjustment.
              */}

              <div className="metrics-group">
                <div className="metrics-group-label">Volatility Surface</div>
                <div className="metrics-grid">
                  {/* FIX 5: show vol metrics as % (annualized), not raw decimals */}
                  <Metric label="IV30" value={fmtVol(m.iv30)}
                    sub="annualized" />
                  <Metric label="RV30 (YZ)" value={fmtVol(m.rv30)}
                    sub={m.rv_estimator === 'yang_zhang' ? 'Yang-Zhang · annualized' : 'close-to-close · annualized'} />
                  <Metric label="IV / RV30" value={fmtNum(m.iv_rv30, 3)}
                    tone={toneNeg(m.iv_rv30, 1.0, 1.25)} />
                  <Metric label="RV Forecast (HAR)" value={fmtVol(m.rv30_har_forecast)}
                    sub={
                      m.rv30_har_forecast == null
                        ? 'n/a (< 30d hist)'
                        : m.rv_har_is_fallback
                          ? 'RS 30d mean · fallback (< 100d hist)'
                          : 'Corsi 2009 · 1d fwd · annualized'
                    } />
                  <Metric label="IV / RV (fwd)" value={fmtNum(m.iv_rv_har, 3)}
                    tone={toneNeg(m.iv_rv_har, 1.0, 1.25)}
                    sub="vs HAR forecast" />
                  {/* FIX 6: relabel Vol Percentile to clarify it uses Rogers-Satchell, not YZ */}
                  <Metric
                    label="Vol Regime Pct (RS-1y)"
                    value={m.rv_percentile_rank != null ? `${Math.round(m.rv_percentile_rank)}th` : 'n/a'}
                    tone={
                      m.rv_percentile_rank == null ? 'default'
                      : m.rv_percentile_rank >= 75 ? 'bad'
                      : m.rv_percentile_rank <= 25 ? 'good'
                      : 'warn'
                    }
                    sub={m.vol_regime && m.vol_regime !== 'unknown' ? `Regime: ${m.vol_regime} · Rogers-Satchell basis` : 'Rogers-Satchell basis'}
                    accent
                  />
                  <Metric label="IV45" value={fmtVol(m.iv45)}
                    sub="annualized" />
                  {/* FIX 1: label now shows actual tenors, not implied "0-45" */}
                  <Metric
                    label={m.ts_slope_near_dte != null && m.ts_slope_far_dte != null
                      ? `TS Slope (${Math.round(m.ts_slope_near_dte)}d→${Math.round(m.ts_slope_far_dte)}d)`
                      : 'TS Slope'}
                    value={fmtNum(m.term_structure_slope ?? m.term_structure_slope_0_45, 5)}
                  />
                  <Metric label="Near / Back IV" value={fmtNum(m.near_back_iv_ratio, 2)}
                    tone={tonePos(m.near_back_iv_ratio, m.min_near_back_iv_ratio_for_event ?? 1.02, 1.0)} />
                  <Metric label="Smile Curvature" value={fmtNum(m.smile_curvature, 3)}
                    sub={m.smile_concave ? 'concave — jump-risk' : m.smile_points > 0 ? 'convex' : null} />
                  <Metric label="Near-Term Spread" value={fmtPp(m.near_term_spread_pct, 2)}
                    tone={toneNeg(m.near_term_spread_pct, 8.0, 18.0)} />
                  <Metric label="Liquidity Proxy" value={m.liquidity_proxy != null ? Number(m.liquidity_proxy).toFixed(0) : 'n/a'}
                    tone={tonePos(m.liquidity_proxy, 2000, 400)} />
                </div>
                <TermStructureChart
                  days={m.term_structure_days}
                  ivs={m.term_structure_ivs}
                  earningsDte={m.days_to_earnings}
                />
              </div>

              <div className="metrics-group">
                <div className="metrics-group-label">Earnings Profile</div>
                {/* Fix 8: decomposition warning in High vol regime */}
                {m.decomp_regime_warning && (
                  <div style={{
                    background: 'rgba(240,160,32,0.09)', border: '1px solid rgba(240,160,32,0.30)',
                    borderRadius: 6, padding: '7px 12px', marginBottom: 8, fontSize: 12, color: '#f0a020',
                  }}>
                    ⚠ <strong>Vol Regime Warning:</strong> High realized-vol environment detected. Event-move and non-event-move
                    decomposition may be unreliable — baseline volatility is elevated enough to corrupt
                    the earnings-specific signal extraction. Treat implied-move anchors with additional caution.
                  </div>
                )}
                {/* Fix 3: disclosure when move model uses fallback or thin sample */}
                {(m.fallback_move_model_flag || m.low_event_count_flag) && (
                  <div style={{
                    background: 'rgba(88,166,255,0.07)', border: '1px solid rgba(88,166,255,0.25)',
                    borderRadius: 6, padding: '7px 12px', marginBottom: 8, fontSize: 12, color: '#79b8ff',
                    display: 'flex', flexDirection: 'column', gap: 3,
                  }}>
                    <strong>Reduced-Evidence Signal</strong>
                    {m.fallback_move_model_flag && (
                      <span>· Move anchor uses a <strong>fallback model</strong> (no earnings history found). Advisory evidence flag — treat the move estimate as lower-confidence; the setup score is not reduced.</span>
                    )}
                    {m.low_event_count_flag && !m.fallback_move_model_flag && (
                      <span>· Fewer than 8 historical earnings events. Move estimates less reliable. Advisory evidence flag — the setup score is not reduced.</span>
                    )}
                  </div>
                )}
                <div className="metrics-grid">
                  <Metric
                    label="Days to Earnings"
                    value={m.days_to_earnings ?? 'n/a'}
                    sub={m.earnings_release_time
                      ? (m.earnings_release_time.includes('before') ? 'Before Market Open'
                        : m.earnings_release_time.includes('after') ? 'After Market Close'
                        : m.earnings_release_time)
                      : null}
                  />
                  <Metric label="Near-Term Opt DTE" value={m.near_term_dte ?? 'n/a'} />
                  <Metric label="Implied Move (Total)" value={fmtPp(m.implied_move_pct, 2)} />
                  {/* FIX 10: surface the event vs non-event move split */}
                  {m.event_implied_move_pct != null && (
                    <Metric label="Event-Implied Move" value={fmtPp(m.event_implied_move_pct, 2)}
                      sub="earnings-specific component" />
                  )}
                  {m.non_event_move_pct != null && m.non_event_move_pct > 0 && (
                    <Metric label="Non-Event Move" value={fmtPp(m.non_event_move_pct, 2)}
                      sub="baseline daily vol component" />
                  )}
                  <Metric label="Anchor Move (Blend)" value={fmtPp(m.earnings_move_anchor_pct, 2)} />
                  <Metric label="Earnings Move (Med)" value={fmtPp(m.earnings_move_median_pct, 2)} />
                  <Metric label="Earnings Move (P90)" value={fmtPp(m.earnings_move_p90_pct, 2)} />
                  {/* Move-risk advisory readout */}
                  {m.move_risk_ratio != null && (
                    <Metric
                      label="Move Risk Ratio"
                      value={`${Number(m.move_risk_ratio).toFixed(2)}×`}
                      sub={`P90 ÷ Event-Implied · ${m.move_risk_level || 'unknown'} · n=${m.move_risk_sample_size ?? 0}`}
                      tone={
                        m.move_risk_level === 'elevated' ? 'bad'
                        : m.move_risk_level === 'moderate' ? 'warn'
                        : 'good'
                      }
                    />
                  )}
                  <Metric label="Uncertainty Penalty" value={fmtPp(m.move_uncertainty_pct, 2)}
                    tone={toneNeg(m.move_uncertainty_pct, 1.0, 2.0)} />
                  <Metric label="Evidence Weight" value={fmtNum(m.sample_confidence, 2)}
                    sub={`${m.earnings_move_sample_size ?? 0} events · ${m.earnings_move_source || 'none'} · sample-size-based`}
                    tone={tonePos(m.sample_confidence, 0.6, 0.3)} />
                </div>
              </div>

              <div className="metrics-group">
                <div className="metrics-group-label">ATM Greeks (BSM · tenor = earnings DTE · σ = IV30)</div>
                <div className="metrics-grid">
                  <Metric label="Delta (Call)" value={fmtNum(m.atm_delta_call, 3)}
                    sub="directional exposure"
                    tone={m.atm_delta_call != null ? (Math.abs(m.atm_delta_call - 0.5) < 0.05 ? 'good' : 'warn') : 'default'} />
                  <Metric label="Delta (Put)" value={fmtNum(m.atm_delta_put, 3)}
                    sub="directional exposure" />
                  <Metric label="Gamma" value={m.atm_gamma != null ? Number(m.atm_gamma).toExponential(3) : 'n/a'}
                    sub="convexity / $1 move" />
                  <Metric label="Vega" value={m.atm_vega != null ? `$${Number(m.atm_vega).toFixed(3)}` : 'n/a'}
                    sub="per 1pp IV move"
                    tone={tonePos(m.atm_vega, 0.01, 0)} />
                  <Metric label="Theta (Call)" value={m.atm_theta_call != null ? `$${Number(m.atm_theta_call).toFixed(3)}` : 'n/a'}
                    sub="daily decay (short leg earns)" />
                  <Metric label="Theta (Put)" value={m.atm_theta_put != null ? `$${Number(m.atm_theta_put).toFixed(3)}` : 'n/a'}
                    sub="daily decay" />
                  <Metric
                    label="Risk-Free Rate"
                    value={fmtVol(m.pricing_risk_free_rate, 2)}
                    sub={m.pricing_risk_free_rate_source ? `pricing input · ${m.pricing_risk_free_rate_source}` : 'pricing input'}
                  />
                </div>
              </div>

              <div className="metrics-group">
                <div className="metrics-group-label">Position Sizing</div>
                <div className="position-sizing-note">
                  {m.position_sizing_note || "Position sizing guidance not available yet — requires calibrated win rate and empirical edge estimate."}
                </div>
              </div>

              {/* Calendar spread diagram (Fix 4: viability gate) */}
              {m.calendar_payoff && (
                <div className="metrics-group">
                  <div className="metrics-group-label">
                    Calendar Spread · ATM · Short Near / Long Back+28d
                    {/* FIX 8: show per-contract value prominently */}
                    {m.calendar_payoff.entry_debit_per_contract != null && (
                      <span style={{ marginLeft: 10, fontWeight: 400, color: '#8b949e', fontSize: 11 }}>
                        Entry debit ${Number(m.calendar_payoff.entry_debit_per_contract).toFixed(2)}/contract
                      </span>
                    )}
                    {m.calendar_spread_quality && (() => {
                      const qColors = { Strong: '#22c55e', Moderate: '#58a6ff', Weak: '#f0a020', Poor: '#ef4444', unknown: '#8b949e' }
                      return (
                        <span style={{ marginLeft: 12, fontWeight: 600, fontSize: 11, color: qColors[m.calendar_spread_quality] || '#8b949e' }}>
                          {m.calendar_spread_quality} term structure
                        </span>
                      )
                    })()}
                  </div>
                  {/* FIX 2: always show theoretical disclaimer — pricing uses interpolated IV, not live chain */}
                  {m.calendar_payoff.calendar_is_theoretical && (
                    <div style={{
                      background: 'rgba(240,160,32,0.08)', border: '1px solid rgba(240,160,32,0.30)',
                      borderRadius: 6, padding: '6px 12px', marginBottom: 8, fontSize: 11,
                      color: '#f0a020',
                    }}>
                      ⚠ Theoretical illustration — priced from interpolated IV30/IV45, back leg = near+28d.
                      Not guaranteed to match a live quoted chain. Verify debit with your broker before trading.
                    </div>
                  )}
                  {/* Fix 4: warn if breakevens are tighter than implied move */}
                  {m.calendar_be_vs_implied != null && m.calendar_be_vs_implied < 0.70 && (
                    <div style={{
                      background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.35)',
                      borderRadius: 6, padding: '7px 12px', marginBottom: 8, fontSize: 12,
                      color: '#fca5a5',
                    }}>
                      ⚠ Breakevens cover only {Math.round(m.calendar_be_vs_implied * 100)}% of the implied move.
                      The market is pricing a larger move than this structure profits from.
                    </div>
                  )}
                  <CalendarSpreadChart calPayoff={m.calendar_payoff} />
                  {m.calendar_payoff.iv_near != null && (
                    <div style={{ display: 'flex', gap: 18, marginTop: 6, fontSize: 11, color: '#8b949e', flexWrap: 'wrap' }}>
                      <span>Near IV: {(m.calendar_payoff.iv_near * 100).toFixed(1)}%</span>
                      <span>Back IV: {(m.calendar_payoff.iv_back * 100).toFixed(1)}%</span>
                      <span>Remaining: {m.calendar_payoff.t_remaining_days}d</span>
                      {m.calendar_be_vs_implied != null && (
                        <span style={{ color: m.calendar_be_vs_implied < 0.70 ? '#f0a020' : '#8b949e' }}>
                          BE covers {Math.round(m.calendar_be_vs_implied * 100)}% of implied move
                        </span>
                      )}
                    </div>
                  )}
                  {/* Scenario source provenance — never pretend a fallback is calibrated history */}
                  {m.calendar_payoff.calendar_scenario_source && (() => {
                    const src = m.calendar_payoff.calendar_scenario_source
                    const srcMeta = {
                      historical_symbol_calibrated: {
                        label: 'Calibrated (symbol history)',
                        color: '#4ade80',
                        bg: 'rgba(34,197,94,0.10)',
                        border: 'rgba(34,197,94,0.28)',
                        tip: 'Scenarios derived from ≥8 earnings events for this ticker — full statistical power.',
                      },
                      small_sample_estimate: {
                        label: 'Small-sample estimate',
                        color: '#fbbf24',
                        bg: 'rgba(240,160,32,0.10)',
                        border: 'rgba(240,160,32,0.28)',
                        tip: 'Fewer than 8 earnings events available. Scenarios are estimates with elevated uncertainty.',
                      },
                      heuristic_fallback: {
                        label: 'Heuristic fallback',
                        color: '#f87171',
                        bg: 'rgba(239,68,68,0.10)',
                        border: 'rgba(239,68,68,0.28)',
                        tip: 'No usable earnings history. Scenarios built from sector/macro heuristics — treat as illustrative only.',
                      },
                    }
                    const m2 = srcMeta[src] || { label: src, color: '#8b949e', bg: 'rgba(139,148,158,0.08)', border: 'rgba(139,148,158,0.22)', tip: '' }
                    return (
                      <div style={{ marginTop: 7, display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 10, color: '#8b949e' }}>Scenario source:</span>
                        <span
                          title={m2.tip}
                          style={{
                            fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
                            background: m2.bg, color: m2.color, border: `1px solid ${m2.border}`,
                            cursor: 'help',
                          }}
                        >
                          {m2.label}
                        </span>
                      </div>
                    )
                  })()}
                </div>
              )}

              {/* Summary strip */}
              <div className="edge-summary-strip">
                <span><strong>Hard Gate:</strong> {m.hard_no_trade ? 'FAIL' : 'PASS'}</span>
                <span><strong>Short-Leg Align:</strong> {shortLegAligned == null ? 'n/a' : shortLegAligned ? 'PASS' : 'FAIL'}</span>
                <span><strong>Liquidity Gate:</strong> {(m.near_term_liquidity_proxy ?? 0) >= (m.min_near_term_liquidity_proxy_for_trade ?? 400) ? 'PASS' : 'FAIL'}</span>
                <span><strong>Concavity Surcharge:</strong> {fmtPp(m.concavity_risk_surcharge_pct)}</span>
                <span><strong>Data:</strong> {
                  m.data_sources
                    ? `Options: ${m.data_sources.options_source === 'marketdata_app' ? 'MDApp' : 'yfinance'} · Prices/RV: ${m.data_sources.price_rv_source === 'marketdata_app' ? 'MDApp' : 'yfinance'}`
                    : m.data_source === 'marketdata_app' ? 'MarketData.app' : 'yfinance'
                }</span>
                <span><strong>Updated:</strong> {formatTimestamp(result.generated_at || m.generated_at)}</span>
                {/* Stale data: IV/RV non-contemporaneous — advisory flag only */}
                {(m.price_data_stale || (m.price_data_age_days != null && m.price_data_age_days > 1)) && (
                  <span style={{ color: '#f0a020', fontWeight: 600 }} title="IV/RV contemporaneity broken — price bars are older than 2 trading days. Treat the snapshot as lower-confidence; the setup score is not reduced.">
                    ⚠ Stale data ({m.price_data_age_days}d old) · advisory flag (score not reduced)
                  </span>
                )}
                {/*
                  Removed: a "Score capped · {reason}" badge gated on
                  m.confidence_capped. That flag is always False — the engine's
                  cap system is informational-only and never reduces the score
                  (confidence_pct = selector confidence). The badge could never
                  truthfully render, so it was dead code asserting a mechanism
                  that does not exist.
                */}
              </div>

              {/* Rationale */}
              <div className="rationale-card">
                <h2>Decision Rationale</h2>
                <ul>
                  {(result.rationale || []).map((line, i) => <li key={i}>{line}</li>)}
                </ul>
              </div>
              </LegacyAnalysisPanel>
              </Suspense>
            </>
          )}
        </section>

        <Suspense fallback={<div className="oos-message">Loading historical warehouse…</div>}>
          <HistoricalWarehousePanel
            warehouse={warehouse}
            setWarehouse={setWarehouse}
            onLoadSymbols={loadWarehouseSymbols}
            onRunQuery={runWarehouseQuery}
          />
        </Suspense>

        <Suspense fallback={<div className="oos-message">Loading recommendation ledger…</div>}>
          <LedgerDiagnosticsPanel apiBase={API_BASE} />
        </Suspense>

        <Suspense fallback={<div className="oos-message">Loading data-quality diagnostics…</div>}>
          <DataQualityDiagnosticsPanel apiBase={API_BASE} />
        </Suspense>

        <Suspense fallback={<div className="oos-message">Loading provider telemetry…</div>}>
          <ProviderTelemetryPanel apiBase={API_BASE} />
        </Suspense>

        <Suspense fallback={<div className="oos-message">Loading forward performance…</div>}>
          <ForwardPerformancePanel apiBase={API_BASE} />
        </Suspense>

        <Suspense fallback={<div className="oos-message">Loading evidence report…</div>}>
          <EvidenceReportPanel apiBase={API_BASE} />
        </Suspense>

        {/* ── OOS Report Card ── */}
        <section className="oos-block">
          <div className="oos-head">
            <SectionTitle>Walk-Forward OOS Report Card</SectionTitle>
            <div className="oos-actions">
              <button onClick={runOos} disabled={oosLoading}>
                {oosLoading ? `Running… ${oosElapsedSec}s` : 'Run OOS Report Card'}
              </button>
              {oosLoading && (
                <button type="button" className="secondary-button" onClick={cancelOos}>Cancel</button>
              )}
            </div>
          </div>
          <p className="oos-help">Strategy-level walk-forward validation across the S&amp;P 500 backfill universe — not ticker-specific. Results reflect whether the pre-earnings volatility setup quality signal is repeatable across many symbols and time periods. Outputs are research metrics, not performance guarantees. Typically 30–120 s.</p>

          <div className="oos-controls">
            <label className="oos-field">
              <span>Stability Profile</span>
              <select value={oosParams.oos_stability_profile} onChange={e => applyOosPreset(e.target.value)}>
                {Object.entries(OOS_PROFILE_LABELS).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
              </select>
            </label>
            <label className="oos-field"><span>Backtest Start</span>
              <input type="date" value={oosParams.backtest_start_date} onChange={e => updateOosParam('backtest_start_date', e.target.value)} />
            </label>
            <label className="oos-field"><span>Backtest End</span>
              <input type="date" value={oosParams.backtest_end_date} onChange={e => updateOosParam('backtest_end_date', e.target.value)} />
            </label>
            <label className="oos-field"><span>Lookback Days</span>
              <input type="number" value={oosParams.lookback_days} onChange={e => updateOosParam('lookback_days', e.target.value)} />
            </label>
            <label className="oos-field"><span>Max Symbols</span>
              <input type="number" value={oosParams.max_backtest_symbols} onChange={e => updateOosParam('max_backtest_symbols', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Signal</span>
              <input type="number" step="0.01" value={oosParams.min_signal_score} onChange={e => updateOosParam('min_signal_score', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Evidence Conf</span>
              <input type="number" step="0.01" value={oosParams.min_crush_confidence} onChange={e => updateOosParam('min_crush_confidence', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Vol Move</span>
              <input type="number" step="0.01" value={oosParams.min_crush_magnitude} onChange={e => updateOosParam('min_crush_magnitude', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Edge Quality</span>
              <input type="number" step="0.01" value={oosParams.min_crush_edge} onChange={e => updateOosParam('min_crush_edge', e.target.value)} />
            </label>
            <label className="oos-field"><span>Target Entry DTE</span>
              <input type="number" value={oosParams.target_entry_dte} onChange={e => updateOosParam('target_entry_dte', e.target.value)} />
            </label>
            <label className="oos-field"><span>Entry DTE Band</span>
              <input type="number" value={oosParams.entry_dte_band} onChange={e => updateOosParam('entry_dte_band', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Share Volume</span>
              <input type="number" value={oosParams.min_daily_share_volume} onChange={e => updateOosParam('min_daily_share_volume', e.target.value)} />
            </label>
            <label className="oos-field"><span>Max |Mom 5d|</span>
              <input type="number" step="0.01" value={oosParams.max_abs_momentum_5d} onChange={e => updateOosParam('max_abs_momentum_5d', e.target.value)} />
            </label>
            <label className="oos-field"><span>Train Days</span>
              <input type="number" value={oosParams.oos_train_days} onChange={e => updateOosParam('oos_train_days', e.target.value)} />
            </label>
            <label className="oos-field"><span>Test Days</span>
              <input type="number" value={oosParams.oos_test_days} onChange={e => updateOosParam('oos_test_days', e.target.value)} />
            </label>
            <label className="oos-field"><span>Step Days</span>
              <input type="number" value={oosParams.oos_step_days} onChange={e => updateOosParam('oos_step_days', e.target.value)} />
            </label>
            <label className="oos-field"><span>Top-N Train</span>
              <input type="number" value={oosParams.oos_top_n_train} onChange={e => updateOosParam('oos_top_n_train', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Splits</span>
              <input type="number" value={oosParams.oos_min_splits} onChange={e => updateOosParam('oos_min_splits', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Total Trades</span>
              <input type="number" value={oosParams.oos_min_total_test_trades} onChange={e => updateOosParam('oos_min_total_test_trades', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Trades/Split</span>
              <input type="number" step="0.1" value={oosParams.oos_min_trades_per_split} onChange={e => updateOosParam('oos_min_trades_per_split', e.target.value)} />
            </label>
          </div>

          {oosLoading && (
            <div className="oos-message">OOS validation running ({oosElapsedSec}s). Results appear when complete.</div>
          )}
          {oosError && <div className="error-banner">{oosError}</div>}

          {oosResult && (() => {
            const s = oosResult.summary || {}
            const metrics = s.metrics || {}
            const sample = s.sample || {}
            return (
              <>
                <div className="oos-message">
                  Profile used: <strong>{s.stability_profile_used || oosParams.oos_stability_profile}</strong>
                  {s.adaptive_retry_used && <span className="oos-tag">adaptive retry</span>}
                </div>
                {s.message && <div className="oos-message">{s.message}</div>}
                {Array.isArray(s.warnings) && s.warnings.map((w, i) => (
                  <div key={i} className="oos-warning">⚠ {w}</div>
                ))}
                {Array.isArray(s.notes) && s.notes.length > 0 && (
                  <div className="oos-message oos-notes">
                    {s.notes.map((n, i) => <div key={i}>{n}</div>)}
                  </div>
                )}
                <div className="oos-grid">
                  <Metric label="Grade" value={s.grade || '--'} accent />
                  <Metric label="Pass" value={s.overall_pass === true ? 'PASS' : s.overall_pass === false ? 'FAIL' : '--'}
                    tone={s.overall_pass === true ? 'good' : s.overall_pass === false ? 'bad' : 'default'} />
                  <Metric label="OOS Splits" value={s.splits ?? '--'} />
                  <Metric label="Total Trades" value={sample.total_test_trades ?? '--'} />
                  <Metric label="Trades/Split" value={fmtNum(sample.avg_trades_per_split, 1)} />
                  <Metric label="Alpha Mean" value={fmtNum(metrics.alpha?.mean, 4)}
                    tone={tonePos(metrics.alpha?.mean, 0, 0)} />
                  <Metric label="Alpha CI Low" value={fmtNum(metrics.alpha?.low, 4)}
                    tone={tonePos(metrics.alpha?.low, 0, 0)} />
                  <Metric label="Per-Trade Sharpe (avg)" value={fmtNum(metrics.sharpe?.mean, 3)}
                    sub="per-trade, not portfolio-level"
                    tone={tonePos(metrics.sharpe?.mean, 0.5, 0)} />
                  <Metric label="Sharpe CI Low" value={fmtNum(metrics.sharpe?.low, 3)}
                    tone={tonePos(metrics.sharpe?.low, 0, 0)} />
                  {s.cross_split_sharpe != null && (
                    <Metric
                      label="Portfolio Sharpe (cross-split)"
                      value={fmtNum(s.cross_split_sharpe, 3)}
                      sub="annualised over test period · realistic"
                      accent
                      tone={tonePos(s.cross_split_sharpe, 1.0, 0.3)}
                    />
                  )}
                  <Metric label="Win Rate" value={fmtPct(metrics.win_rate?.mean)} />
                  <Metric label="Win Rate CI Low" value={fmtPct(metrics.win_rate?.low)}
                    tone={tonePos(metrics.win_rate?.low, 0.5, 0.4)} />
                  <Metric label="PnL Mean" value={fmtMoney(metrics.pnl?.mean)}
                    tone={tonePos(metrics.pnl?.mean, 0, 0)} />
                  <Metric label="PnL CI Low" value={fmtMoney(metrics.pnl?.low)}
                    tone={tonePos(metrics.pnl?.low, 0, 0)} />
                  {metrics.positive_alpha_split_rate != null && (
                    <Metric label="% Splits Positive α"
                      value={fmtPct(metrics.positive_alpha_split_rate)}
                      tone={tonePos(metrics.positive_alpha_split_rate, 0.6, 0.4)} />
                  )}
                  {metrics.positive_pnl_split_rate != null && (
                    <Metric label="% Splits Profitable"
                      value={fmtPct(metrics.positive_pnl_split_rate)}
                      tone={tonePos(metrics.positive_pnl_split_rate, 0.6, 0.4)} />
                  )}
                </div>
                <OosSplitChart splitsDetail={s.splits_detail} />
              </>
            )
          })()}
        </section>

        <footer className="app-footer">
          Created by Tristan Alejandro | Not financial Advice.
        </footer>
      </main>
    </div>
  )
}
