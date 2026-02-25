import React, { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
const OOS_TIMEOUT_MS = 180000

const DEFAULT_OOS_PARAMS = {
  lookback_days: '730',
  max_backtest_symbols: '20',
  backtest_start_date: '2023-01-01',
  backtest_end_date: '',
  min_signal_score: '0.50',
  min_crush_confidence: '0.30',
  min_crush_magnitude: '0.06',
  min_crush_edge: '0.02',
  target_entry_dte: '6',
  entry_dte_band: '6',
  min_daily_share_volume: '1000000',
  max_abs_momentum_5d: '0.11',
  oos_train_days: '252',
  oos_test_days: '63',
  oos_step_days: '63',
  oos_top_n_train: '1',
  oos_min_splits: '8',
  oos_min_total_test_trades: '80',
  oos_min_trades_per_split: '5.0'
}

function Metric({ label, value, accent = false, tone = 'default' }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${accent ? 'accent' : ''} tone-${tone}`}>{value}</div>
    </div>
  )
}

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  return `${(Number(value) * 100).toFixed(2)}%`
}

function fmtNum(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  return Number(value).toFixed(digits)
}

function fmtMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  return `$${Number(value).toFixed(2)}`
}

function fmtPctPoints(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  return `${Number(value).toFixed(digits)}%`
}

function fmtSignedPctPoints(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  const num = Number(value)
  const sign = num > 0 ? '+' : ''
  return `${sign}${num.toFixed(digits)}%`
}

function fmtSignedNum(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  const num = Number(value)
  const sign = num > 0 ? '+' : ''
  return `${sign}${num.toFixed(digits)}`
}

function fmtBool(value) {
  if (value === true) return 'PASS'
  if (value === false) return 'FAIL'
  return 'n/a'
}

function tonePositiveMetric(value, goodThreshold = 0.0, warnThreshold = 0.0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'default'
  const num = Number(value)
  if (num >= goodThreshold) return 'good'
  if (num > warnThreshold) return 'warn'
  return 'bad'
}

function toneNegativeMetric(value, goodMax = 1.25, warnMax = 2.0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'default'
  const num = Number(value)
  if (num <= goodMax) return 'good'
  if (num <= warnMax) return 'warn'
  return 'bad'
}

function parseIntOr(value, fallback) {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

function parseFloatOr(value, fallback) {
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function App() {
  const [symbol, setSymbol] = useState('AAPL')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const [oosLoading, setOosLoading] = useState(false)
  const [oosError, setOosError] = useState('')
  const [oosResult, setOosResult] = useState(null)
  const [oosParams, setOosParams] = useState(DEFAULT_OOS_PARAMS)
  const [oosElapsedSec, setOosElapsedSec] = useState(0)

  const oosAbortRef = useRef(null)

  useEffect(() => {
    if (!oosLoading) {
      setOosElapsedSec(0)
      return undefined
    }
    const timer = setInterval(() => {
      setOosElapsedSec((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [oosLoading])

  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol])
  const hardGateReasons = useMemo(() => {
    const reasons = []
    const edgeReasons = result?.metrics?.hard_gate_reasons
    if (Array.isArray(edgeReasons)) {
      reasons.push(...edgeReasons.filter(Boolean))
    }
    if (oosResult?.summary?.overall_pass === false) {
      reasons.push(`OOS gate failed (grade=${oosResult?.summary?.grade || 'N/A'}).`)
    }
    return reasons
  }, [result, oosResult])

  const shortLegAligned = useMemo(() => {
    const earningsDte = result?.metrics?.days_to_earnings
    const nearTermDte = result?.metrics?.near_term_dte
    if (earningsDte === null || earningsDte === undefined) return null
    if (nearTermDte === null || nearTermDte === undefined) return null
    return Number(earningsDte) < Number(nearTermDte)
  }, [result])

  const noTradeBlocked = hardGateReasons.length > 0
  const recommendationValue = result ? (noTradeBlocked ? 'No Trade' : (result?.recommendation || '--')) : '--'
  const confidenceValue = result
    ? `${Number(result.confidence_pct).toFixed(1)}%${result?.metrics?.confidence_capped ? ' (capped)' : ''}`
    : '--'

  function updateOosParam(key, value) {
    setOosParams((prev) => ({ ...prev, [key]: value }))
  }

  function buildOosPayload() {
    return {
      lookback_days: parseIntOr(oosParams.lookback_days, 730),
      max_backtest_symbols: parseIntOr(oosParams.max_backtest_symbols, 20),
      backtest_start_date: oosParams.backtest_start_date || null,
      backtest_end_date: oosParams.backtest_end_date || null,
      min_signal_score: parseFloatOr(oosParams.min_signal_score, 0.5),
      min_crush_confidence: parseFloatOr(oosParams.min_crush_confidence, 0.3),
      min_crush_magnitude: parseFloatOr(oosParams.min_crush_magnitude, 0.06),
      min_crush_edge: parseFloatOr(oosParams.min_crush_edge, 0.02),
      target_entry_dte: parseIntOr(oosParams.target_entry_dte, 6),
      entry_dte_band: parseIntOr(oosParams.entry_dte_band, 6),
      min_daily_share_volume: parseIntOr(oosParams.min_daily_share_volume, 1000000),
      max_abs_momentum_5d: parseFloatOr(oosParams.max_abs_momentum_5d, 0.11),
      oos_train_days: parseIntOr(oosParams.oos_train_days, 252),
      oos_test_days: parseIntOr(oosParams.oos_test_days, 63),
      oos_step_days: parseIntOr(oosParams.oos_step_days, 63),
      oos_top_n_train: parseIntOr(oosParams.oos_top_n_train, 1),
      oos_min_splits: parseIntOr(oosParams.oos_min_splits, 8),
      oos_min_total_test_trades: parseIntOr(oosParams.oos_min_total_test_trades, 80),
      oos_min_trades_per_split: parseFloatOr(oosParams.oos_min_trades_per_split, 5.0)
    }
  }

  function cancelOosReportCard() {
    if (oosAbortRef.current) {
      oosAbortRef.current.abort('cancelled')
      oosAbortRef.current = null
    }
  }

  async function runAnalysis(e) {
    e?.preventDefault()
    setError('')
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/api/edge/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: normalizedSymbol })
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${res.status}`)
      }
      const body = await res.json()
      setResult(body)
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setLoading(false)
    }
  }

  async function runOosReportCard() {
    if (oosAbortRef.current) {
      oosAbortRef.current.abort('new_request')
      oosAbortRef.current = null
    }

    const controller = new AbortController()
    oosAbortRef.current = controller
    const timeoutId = setTimeout(() => controller.abort('timeout'), OOS_TIMEOUT_MS)

    setOosLoading(true)
    setOosError('')
    setOosResult(null)

    try {
      const payload = buildOosPayload()
      const res = await fetch(`${API_BASE}/api/oos/report-card`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${res.status}`)
      }
      const body = await res.json()
      setOosResult(body)
    } catch (err) {
      if (err?.name === 'AbortError') {
        const reason = controller.signal.reason
        if (reason === 'timeout') {
          setOosError(`OOS run timed out after ${Math.floor(OOS_TIMEOUT_MS / 1000)}s.`)
        } else {
          setOosError('OOS run cancelled.')
        }
      } else {
        setOosError(String(err.message || err))
      }
    } finally {
      clearTimeout(timeoutId)
      if (oosAbortRef.current === controller) {
        oosAbortRef.current = null
      }
      setOosLoading(false)
    }
  }

  return (
    <div className="page-shell">
      <div className="bg-orb bg-orb-a" />
      <div className="bg-orb bg-orb-b" />
      <main className="terminal-panel">
        <header className="panel-header">
          <div>
            <h1>IV Crush Edge Terminal</h1>
            <p>Single-ticker institutional workflow. No scanner, no noise.</p>
          </div>
          <div className="status-chip">Mode: Production Research</div>
        </header>

        <section className="analysis-block">
          <form className="symbol-form" onSubmit={runAnalysis}>
            <label htmlFor="symbol">Ticker</label>
            <input
              id="symbol"
              maxLength={10}
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
            />
            <button type="submit" disabled={loading || !normalizedSymbol}>
              {loading ? 'Running...' : 'Run Edge Analysis'}
            </button>
          </form>

          {error && <div className="error-banner">{error}</div>}
          {result && noTradeBlocked && (
            <div className="no-trade-banner">
              <div className="no-trade-title">NO TRADE</div>
              <ul>
                {hardGateReasons.map((reason, idx) => (
                  <li key={idx}>{reason}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="metrics-grid">
            <Metric label="Recommendation" value={recommendationValue} accent />
            <Metric label="Confidence" value={confidenceValue} />
            <Metric label="Setup Score" value={result ? Number(result.setup_score).toFixed(3) : '--'} />
            <Metric label="Composite Score" value={fmtNum(result?.metrics?.composite_score, 3)} />
            <Metric
              label="Expected Net Edge"
              value={fmtSignedPctPoints(result?.metrics?.expected_net_edge_pct)}
              tone={tonePositiveMetric(result?.metrics?.expected_net_edge_pct, 0.25, 0.0)}
            />
            <Metric
              label="Expectancy Ratio"
              value={fmtSignedNum(result?.metrics?.expectancy_ratio, 2)}
              tone={tonePositiveMetric(result?.metrics?.expectancy_ratio, 0.2, 0.0)}
            />
            <Metric
              label="Implied/Anchor"
              value={fmtNum(result?.metrics?.implied_vs_anchor_ratio, 2)}
              tone={tonePositiveMetric(result?.metrics?.implied_vs_anchor_ratio, 1.05, 1.0)}
            />
            <Metric
              label="Drawdown Risk"
              value={fmtPctPoints(result?.metrics?.drawdown_risk_pct, 2)}
              tone={toneNegativeMetric(result?.metrics?.drawdown_risk_pct, 1.25, 2.0)}
            />
            <Metric label="Implied Move" value={fmtPctPoints(result?.metrics?.implied_move_pct, 2)} />
            <Metric label="Anchor Move (Blend)" value={fmtPctPoints(result?.metrics?.earnings_move_anchor_pct, 2)} />
            <Metric label="Earnings Move (Med)" value={fmtPctPoints(result?.metrics?.earnings_move_median_pct, 2)} />
            <Metric
              label="Near-Term Spread"
              value={fmtPctPoints(result?.metrics?.near_term_spread_pct, 2)}
              tone={toneNegativeMetric(result?.metrics?.near_term_spread_pct, 8.0, 18.0)}
            />
            <Metric
              label="Near/Back IV"
              value={fmtNum(result?.metrics?.near_back_iv_ratio, 2)}
              tone={tonePositiveMetric(
                result?.metrics?.near_back_iv_ratio,
                result?.metrics?.min_near_back_iv_ratio_for_event ?? 1.02,
                1.0
              )}
            />
            <Metric label="Tx Cost Est" value={fmtPctPoints(result?.metrics?.tx_cost_estimate_pct, 2)} />
            <Metric
              label="Move Uncertainty"
              value={fmtPctPoints(result?.metrics?.move_uncertainty_pct, 2)}
              tone={toneNegativeMetric(result?.metrics?.move_uncertainty_pct, 1.0, 2.0)}
            />
            <Metric label="DTE" value={result?.metrics?.days_to_earnings ?? '--'} />
            <Metric label="IV30" value={fmtNum(result?.metrics?.iv30, 4)} />
            <Metric label="RV30" value={fmtNum(result?.metrics?.rv30, 4)} />
            <Metric label="IV/RV30" value={fmtNum(result?.metrics?.iv_rv30, 3)} />
            <Metric label="TS Slope 0-45" value={fmtNum(result?.metrics?.term_structure_slope_0_45, 5)} />
          </div>

          {result && (
            <div className="edge-summary-strip">
              <span><strong>Edge Quality:</strong> {result?.metrics?.edge_quality || 'n/a'}</span>
              <span><strong>Earnings Sample:</strong> {result?.metrics?.earnings_move_sample_size ?? 0} ({result?.metrics?.earnings_move_source || 'none'})</span>
              <span><strong>Hard Gate:</strong> {result?.metrics?.hard_no_trade ? 'FAIL' : 'PASS'}</span>
              <span><strong>Short-Leg Align:</strong> {fmtBool(shortLegAligned)}</span>
              <span><strong>Near-Term DTE:</strong> {result?.metrics?.near_term_dte ?? 'n/a'}</span>
            </div>
          )}

          <div className="rationale-card">
            <h2>Today&apos;s Action Plan</h2>
            {!result && <p>Run symbol analysis to populate rationale and decision context.</p>}
            {result && (
              <ul>
                {(result.rationale || []).map((line, idx) => (
                  <li key={idx}>{line}</li>
                ))}
              </ul>
            )}
          </div>
        </section>

        <section className="oos-block">
          <div className="oos-controls">
            <label className="oos-field">
              <span>Backtest Start</span>
              <input type="date" value={oosParams.backtest_start_date} onChange={(e) => updateOosParam('backtest_start_date', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Backtest End</span>
              <input type="date" value={oosParams.backtest_end_date} onChange={(e) => updateOosParam('backtest_end_date', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Lookback Days</span>
              <input type="number" value={oosParams.lookback_days} onChange={(e) => updateOosParam('lookback_days', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Max Symbols</span>
              <input type="number" value={oosParams.max_backtest_symbols} onChange={(e) => updateOosParam('max_backtest_symbols', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Min Signal</span>
              <input type="number" step="0.01" value={oosParams.min_signal_score} onChange={(e) => updateOosParam('min_signal_score', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Min Crush Conf</span>
              <input type="number" step="0.01" value={oosParams.min_crush_confidence} onChange={(e) => updateOosParam('min_crush_confidence', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Min Crush Magn</span>
              <input type="number" step="0.01" value={oosParams.min_crush_magnitude} onChange={(e) => updateOosParam('min_crush_magnitude', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Min Crush Edge</span>
              <input type="number" step="0.01" value={oosParams.min_crush_edge} onChange={(e) => updateOosParam('min_crush_edge', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Target Entry DTE</span>
              <input type="number" value={oosParams.target_entry_dte} onChange={(e) => updateOosParam('target_entry_dte', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Entry DTE Band</span>
              <input type="number" value={oosParams.entry_dte_band} onChange={(e) => updateOosParam('entry_dte_band', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Min Share Volume</span>
              <input type="number" value={oosParams.min_daily_share_volume} onChange={(e) => updateOosParam('min_daily_share_volume', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Max |Momentum 5d|</span>
              <input type="number" step="0.01" value={oosParams.max_abs_momentum_5d} onChange={(e) => updateOosParam('max_abs_momentum_5d', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Train Days</span>
              <input type="number" value={oosParams.oos_train_days} onChange={(e) => updateOosParam('oos_train_days', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Test Days</span>
              <input type="number" value={oosParams.oos_test_days} onChange={(e) => updateOosParam('oos_test_days', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Step Days</span>
              <input type="number" value={oosParams.oos_step_days} onChange={(e) => updateOosParam('oos_step_days', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Top-N Train</span>
              <input type="number" value={oosParams.oos_top_n_train} onChange={(e) => updateOosParam('oos_top_n_train', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Min Splits</span>
              <input type="number" value={oosParams.oos_min_splits} onChange={(e) => updateOosParam('oos_min_splits', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Min Total Trades</span>
              <input type="number" value={oosParams.oos_min_total_test_trades} onChange={(e) => updateOosParam('oos_min_total_test_trades', e.target.value)} />
            </label>
            <label className="oos-field">
              <span>Min Trades/Split</span>
              <input type="number" step="0.1" value={oosParams.oos_min_trades_per_split} onChange={(e) => updateOosParam('oos_min_trades_per_split', e.target.value)} />
            </label>
          </div>

          <div className="oos-head">
            <h2>Walk-Forward OOS Report Card</h2>
            <div className="oos-actions">
              <button onClick={runOosReportCard} disabled={oosLoading}>
                {oosLoading ? `Building... ${oosElapsedSec}s` : 'Run OOS Report Card'}
              </button>
              {oosLoading && (
                <button type="button" className="secondary-button" onClick={cancelOosReportCard}>
                  Cancel
                </button>
              )}
            </div>
          </div>

          <p className="oos-help">OOS runs can take 30-120s depending on symbols, lookback, and splits.</p>

          {oosLoading && (
            <div className="oos-message">
              OOS validation in progress ({oosElapsedSec}s elapsed). Results appear when complete.
            </div>
          )}

          {oosError && <div className="error-banner">{oosError}</div>}
          {oosResult && (
            <>
              {oosResult.summary?.message && <div className="oos-message">{oosResult.summary.message}</div>}
              <div className="oos-grid">
                <Metric label="Grade" value={oosResult.summary?.grade || '--'} accent />
                <Metric label="Pass" value={fmtBool(oosResult.summary?.overall_pass)} />
                <Metric label="Splits" value={oosResult.summary?.splits ?? '--'} />
                <Metric label="Total Trades" value={oosResult.summary?.sample?.total_test_trades ?? '--'} />
                <Metric label="Avg Trades/Split" value={fmtNum(oosResult.summary?.sample?.avg_trades_per_split, 2)} />
                <Metric label="Alpha Mean" value={fmtNum(oosResult.summary?.metrics?.alpha?.mean, 4)} />
                <Metric label="Alpha CI Low" value={fmtNum(oosResult.summary?.metrics?.alpha?.low, 4)} />
                <Metric label="Sharpe Mean" value={fmtNum(oosResult.summary?.metrics?.sharpe?.mean, 3)} />
                <Metric label="Sharpe CI Low" value={fmtNum(oosResult.summary?.metrics?.sharpe?.low, 3)} />
                <Metric label="Win Rate Mean" value={fmtPct(oosResult.summary?.metrics?.win_rate?.mean)} />
                <Metric label="Win Rate CI Low" value={fmtPct(oosResult.summary?.metrics?.win_rate?.low)} />
                <Metric label="PnL Mean" value={fmtMoney(oosResult.summary?.metrics?.pnl?.mean)} />
                <Metric label="PnL CI Low" value={fmtMoney(oosResult.summary?.metrics?.pnl?.low)} />
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
