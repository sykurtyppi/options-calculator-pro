import React, { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

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

function App() {
  const [symbol, setSymbol] = useState('AAPL')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const [oosLoading, setOosLoading] = useState(false)
  const [oosError, setOosError] = useState('')
  const [oosResult, setOosResult] = useState(null)

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
    setOosLoading(true)
    setOosError('')
    setOosResult(null)
    try {
      const res = await fetch(`${API_BASE}/api/oos/report-card`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${res.status}`)
      }
      const body = await res.json()
      setOosResult(body)
    } catch (err) {
      setOosError(String(err.message || err))
    } finally {
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
            <Metric label="Tx Cost Est" value={fmtPctPoints(result?.metrics?.tx_cost_estimate_pct, 2)} />
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
          <div className="oos-head">
            <h2>Walk-Forward OOS Report Card</h2>
            <button onClick={runOosReportCard} disabled={oosLoading}>
              {oosLoading ? 'Building...' : 'Run OOS Report Card'}
            </button>
          </div>
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
