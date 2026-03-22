import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, ComposedChart, Area,
} from 'recharts'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'
const OOS_TIMEOUT_MS = 180_000

const TODAY_ISO = new Date().toISOString().split('T')[0]

const DEFAULT_OOS_PARAMS = {
  oos_stability_profile: 'stability_auto',
  lookback_days: '1095',
  max_backtest_symbols: '50',
  backtest_start_date: '2023-01-01',
  backtest_end_date: TODAY_ISO,
  min_signal_score: '0.50',
  min_crush_confidence: '0.30',
  min_crush_magnitude: '0.06',
  min_crush_edge: '0.025',
  target_entry_dte: '6',
  entry_dte_band: '5',
  min_daily_share_volume: '1500000',
  max_abs_momentum_5d: '0.11',
  oos_train_days: '189',
  oos_test_days: '42',
  oos_step_days: '42',
  oos_top_n_train: '1',
  oos_min_splits: '8',
  oos_min_total_test_trades: '80',
  oos_min_trades_per_split: '5.0',
}

const OOS_PROFILE_PRESETS = {
  stability_auto: {
    min_signal_score: '0.50', min_crush_confidence: '0.30', min_crush_magnitude: '0.06',
    min_crush_edge: '0.025', min_daily_share_volume: '1500000', max_abs_momentum_5d: '0.11',
    target_entry_dte: '6', entry_dte_band: '5',
  },
  evidence_balanced: {
    min_signal_score: '0.48', min_crush_confidence: '0.28', min_crush_magnitude: '0.06',
    min_crush_edge: '0.025', min_daily_share_volume: '1500000', max_abs_momentum_5d: '0.11',
    target_entry_dte: '6', entry_dte_band: '5',
  },
  sample_expansion: {
    min_signal_score: '0.45', min_crush_confidence: '0.25', min_crush_magnitude: '0.05',
    min_crush_edge: '0.015', min_daily_share_volume: '1000000', max_abs_momentum_5d: '0.11',
    target_entry_dte: '6', entry_dte_band: '6',
  },
  variance_control: {
    min_signal_score: '0.65', min_crush_confidence: '0.50', min_crush_magnitude: '0.09',
    min_crush_edge: '0.025', min_daily_share_volume: '10000000', max_abs_momentum_5d: '0.09',
    target_entry_dte: '6', entry_dte_band: '4',
  },
  alpha_focus: {
    min_signal_score: '0.65', min_crush_confidence: '0.50', min_crush_magnitude: '0.09',
    min_crush_edge: '0.03', min_daily_share_volume: '5000000', max_abs_momentum_5d: '0.08',
    target_entry_dte: '6', entry_dte_band: '3',
  },
}

const OOS_PROFILE_LABELS = {
  stability_auto: 'Auto',
  evidence_balanced: 'Evidence Balanced',
  sample_expansion: 'Sample Expansion',
  variance_control: 'Variance Control',
  alpha_focus: 'Alpha Focus',
}

// ── Utility formatters ────────────────────────────────────────────────────────

function fmtPct(v) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return `${(Number(v) * 100).toFixed(2)}%`
}
function fmtNum(v, d = 3) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return Number(v).toFixed(d)
}
function fmtMoney(v) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return `$${Number(v).toFixed(2)}`
}
function fmtPp(v, d = 2) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return `${Number(v).toFixed(d)}%`
}
function fmtSpp(v, d = 2) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  const n = Number(v)
  return `${n > 0 ? '+' : ''}${n.toFixed(d)}%`
}
function fmtSn(v, d = 2) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  const n = Number(v)
  return `${n > 0 ? '+' : ''}${n.toFixed(d)}`
}

function tonePos(v, good = 0, warn = 0) {
  if (v == null || Number.isNaN(Number(v))) return 'default'
  const n = Number(v)
  if (n >= good) return 'good'
  if (n > warn) return 'warn'
  return 'bad'
}
function toneNeg(v, good = 1.25, warn = 2.0) {
  if (v == null || Number.isNaN(Number(v))) return 'default'
  const n = Number(v)
  if (n <= good) return 'good'
  if (n <= warn) return 'warn'
  return 'bad'
}

function parseIntOr(v, fb) { const p = parseInt(v, 10); return Number.isFinite(p) ? p : fb }
function parseFloatOr(v, fb) { const p = parseFloat(v); return Number.isFinite(p) ? p : fb }

// ── Atoms ─────────────────────────────────────────────────────────────────────

function Metric({ label, value, accent = false, tone = 'default', sub }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${accent ? 'accent' : ''} tone-${tone}`}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  )
}

function SectionTitle({ children }) {
  return <h3 className="section-title">{children}</h3>
}

function Badge({ children, variant = 'default' }) {
  return <span className={`badge badge-${variant}`}>{children}</span>
}

// ── Earnings release time display ─────────────────────────────────────────────
function releaseTimeBadge(rt) {
  if (!rt) return null
  if (rt.includes('before')) return <Badge variant="bmo">BMO</Badge>
  if (rt.includes('after')) return <Badge variant="amc">AMC</Badge>
  if (rt.includes('during')) return <Badge variant="intraday">Intraday</Badge>
  return <Badge>{rt}</Badge>
}

function dataSourceBadge(ds, dataSources) {
  // FIX 4: show granular provenance — options and price/RV may come from different sources
  if (dataSources) {
    const optSrc = dataSources.options_source === 'marketdata_app' ? 'MDApp' : 'yfinance'
    const rvSrc  = dataSources.price_rv_source === 'marketdata_app' ? 'MDApp' : 'yfinance'
    if (optSrc !== rvSrc) {
      return (
        <span style={{ fontSize: 11, color: '#8b949e' }}>
          <Badge variant={dataSources.options_source === 'marketdata_app' ? 'mda' : 'yf'}>{optSrc}</Badge>
          <span style={{ margin: '0 4px', opacity: 0.6 }}>·</span>
          <span style={{ opacity: 0.8 }}>Prices/RV: {rvSrc}</span>
        </span>
      )
    }
  }
  if (ds === 'marketdata_app') return <Badge variant="mda">MDApp</Badge>
  if (ds === 'yfinance_fallback') return <Badge variant="yf">yfinance</Badge>
  return null
}

// ── Vol term structure chart ──────────────────────────────────────────────────
function TermStructureChart({ days, ivs, earningsDte }) {
  if (!Array.isArray(days) || days.length < 2) return null

  const data = days.map((d, i) => ({
    dte: Math.round(Number(d)),
    iv: ivs[i] != null ? Number((ivs[i] * 100).toFixed(2)) : null,
  })).filter(p => p.iv != null).sort((a, b) => a.dte - b.dte)

  if (data.length < 2) return null

  const ivMin = Math.min(...data.map(d => d.iv))
  const ivMax = Math.max(...data.map(d => d.iv))
  const pad   = Math.max((ivMax - ivMin) * 0.25, 0.5)
  const yMin  = Math.max(0, (ivMin - pad).toFixed(1))
  const yMax  = (ivMax + pad).toFixed(1)

  return (
    <div className="vol-chart-wrapper">
      <div className="vol-chart-label">Vol Term Structure (DTE vs IV%)</div>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={data} margin={{ top: 6, right: 18, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="dte"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickCount={6}
            tick={{ fill: '#8b949e', fontSize: 11 }}
            label={{ value: 'DTE', position: 'insideBottomRight', offset: -4, fill: '#8b949e', fontSize: 11 }}
          />
          <YAxis
            domain={[yMin, yMax]}
            tickFormatter={v => `${v}%`}
            tick={{ fill: '#8b949e', fontSize: 11 }}
            width={42}
          />
          <Tooltip
            formatter={(v) => [`${v}%`, 'IV']}
            labelFormatter={(l) => `DTE ${l}`}
            contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 6, fontSize: 12 }}
            itemStyle={{ color: '#e6edf3' }}
            labelStyle={{ color: '#8b949e' }}
          />
          {earningsDte != null && (
            <ReferenceLine
              x={earningsDte}
              stroke="#f0a020"
              strokeDasharray="4 3"
              label={{ value: `Earnings`, position: 'top', fill: '#f0a020', fontSize: 10 }}
            />
          )}
          <ReferenceLine x={30} stroke="rgba(139,148,158,0.35)" strokeDasharray="2 4" />
          <Line
            type="monotone"
            dataKey="iv"
            stroke="#58a6ff"
            strokeWidth={2}
            dot={{ fill: '#58a6ff', r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Ticker tier badge (Fix 1) ─────────────────────────────────────────────────
function TickerTierBadge({ tier, mult }) {
  if (!tier || tier === 'unknown') return null
  const map = {
    mega_cap:   { label: 'Mega-Cap',   variant: 'tier-mega'  },
    large_cap:  { label: 'Large-Cap',  variant: 'tier-large' },
    mid_cap:    { label: 'Mid-Cap',    variant: 'tier-mid'   },
    small_cap:  { label: 'Small-Cap',  variant: 'tier-small' },
    micro_cap:  { label: 'Micro-Cap',  variant: 'tier-micro' },
  }
  const { label, variant } = map[tier] || { label: tier, variant: 'default' }
  const multLabel = mult != null ? ` · ${Math.round(mult * 100)}%` : ''
  return <Badge variant={variant}>{label}{multLabel}</Badge>
}

// ── Vol regime badge ──────────────────────────────────────────────────────────
function VolRegimeBadge({ regime, pct }) {
  if (!regime || regime === 'unknown') return null
  const variantMap = { High: 'regime-high', Elevated: 'regime-elevated', Normal: 'regime-normal', Low: 'regime-low' }
  const variant = variantMap[regime] || 'default'
  const label = pct != null ? `Vol ${regime} · ${Math.round(pct)}th pct` : `Vol ${regime}`
  return <Badge variant={variant}>{label}</Badge>
}

// ── OOS per-split P&L chart ───────────────────────────────────────────────────
function OosSplitChart({ splitsDetail }) {
  if (!Array.isArray(splitsDetail) || splitsDetail.length < 2) return null

  let cumPnl = 0
  const data = splitsDetail.map((s, i) => {
    cumPnl += s.pnl
    return {
      split: i + 1,
      pnl: Number(s.pnl.toFixed(2)),
      cumPnl: Number(cumPnl.toFixed(2)),
      label: s.test_start ? s.test_start.slice(0, 7) : `S${i + 1}`,
      sharpe: Number((s.sharpe || 0).toFixed(2)),
      winRate: s.win_rate != null ? `${(s.win_rate * 100).toFixed(0)}%` : 'n/a',
      trades: s.trades,
    }
  })

  const maxAbs = Math.max(...data.map(d => Math.abs(d.pnl)), 1)
  const padY   = maxAbs * 0.18

  return (
    <div className="vol-chart-wrapper">
      <div className="vol-chart-label">OOS Per-Split P&amp;L + Cumulative ($)</div>
      <ResponsiveContainer width="100%" height={210}>
        <ComposedChart data={data} margin={{ top: 8, right: 18, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="label"
            tick={{ fill: '#8b949e', fontSize: 10 }}
            interval="preserveStartEnd"
          />
          <YAxis
            tickFormatter={v => `$${v}`}
            tick={{ fill: '#8b949e', fontSize: 10 }}
            domain={[-maxAbs - padY, maxAbs + padY]}
            width={52}
          />
          <Tooltip
            formatter={(value, name) => [
              name === 'pnl' ? `$${value}` : `$${value}`,
              name === 'pnl' ? 'Split P&L' : 'Cumulative P&L',
            ]}
            labelFormatter={(l, payload) => {
              const d = payload?.[0]?.payload
              return d ? `Split ${d.split} · ${d.label} · ${d.winRate} WR · ${d.trades} trades · Sharpe ${d.sharpe}` : l
            }}
            contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 6, fontSize: 11 }}
            itemStyle={{ color: '#e6edf3' }}
            labelStyle={{ color: '#8b949e' }}
          />
          <ReferenceLine y={0} stroke="rgba(139,148,158,0.4)" strokeWidth={1} />
          <Bar dataKey="pnl" radius={[3, 3, 0, 0]} maxBarSize={32}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.pnl >= 0 ? 'rgba(34,197,94,0.65)' : 'rgba(239,68,68,0.65)'} />
            ))}
          </Bar>
          <Line
            type="monotone"
            dataKey="cumPnl"
            stroke="#58a6ff"
            strokeWidth={2}
            dot={false}
            strokeDasharray="4 2"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Calendar spread P&L diagram ───────────────────────────────────────────────
function CalendarSpreadChart({ calPayoff }) {
  if (!calPayoff?.payoff_scenarios?.length) return null

  // FIX 8: prefer per-contract scenarios (multiplied by 100) so Y-axis reads in dollars/contract
  const scenarios = calPayoff.payoff_scenarios_per_contract?.length
    ? calPayoff.payoff_scenarios_per_contract
    : calPayoff.payoff_scenarios

  const data = scenarios.map(s => ({
    move: Number(s.move_pct),
    expand: Number((s.iv_expand_20 ?? 0).toFixed(2)),
    flat: Number((s.iv_flat ?? 0).toFixed(2)),
    crush25: Number((s.iv_crush_25 ?? 0).toFixed(2)),
    crush45: Number((s.iv_crush_45 ?? 0).toFixed(2)),
  }))

  const allVals = data.flatMap(d => [d.expand, d.flat, d.crush25, d.crush45]).filter(v => isFinite(v))
  const minV = Math.min(...allVals)
  const maxV = Math.max(...allVals)
  const pad  = Math.max((maxV - minV) * 0.14, 0.01)

  const breakevens = calPayoff.breakeven_moves_pct || []

  return (
    <div className="vol-chart-wrapper">
      <div className="vol-chart-label">
        Calendar Spread P&amp;L at Near-Leg Expiry&nbsp;·&nbsp;$/contract (100 shares)&nbsp;·&nbsp;
        Entry Debit {calPayoff.entry_debit_per_contract != null
          ? `$${Number(calPayoff.entry_debit_per_contract).toFixed(2)}/contract`
          : calPayoff.entry_debit != null ? `$${Number(calPayoff.entry_debit).toFixed(3)}/share` : 'n/a'}
        &nbsp;(near {calPayoff.t_near_days}d / back {calPayoff.t_back_days}d)
      </div>
      {breakevens.length > 0 && (
        <div className="breakeven-note">
          IV-flat breakevens:&nbsp;
          {breakevens.map((b, i) => (
            <span key={i}>{b > 0 ? '+' : ''}{Number(b).toFixed(1)}%{i < breakevens.length - 1 ? ',\u00a0' : ''}</span>
          ))}
        </div>
      )}
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 24, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="move"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={v => `${v > 0 ? '+' : ''}${v}%`}
            tick={{ fill: '#8b949e', fontSize: 10 }}
            label={{ value: 'Underlying Move at Expiry', position: 'insideBottom', offset: -8, fill: '#8b949e', fontSize: 10 }}
          />
          <YAxis
            domain={[minV - pad, maxV + pad]}
            tickFormatter={v => `$${v.toFixed(2)}`}
            tick={{ fill: '#8b949e', fontSize: 10 }}
            width={58}
          />
          <Tooltip
            formatter={(v, name) => {
              const labels = { expand: 'IV +20%', flat: 'IV Flat', crush25: 'IV −25%', crush45: 'IV −45%' }
              return [`$${Number(v).toFixed(3)}`, labels[name] || name]
            }}
            labelFormatter={l => `Move: ${Number(l) > 0 ? '+' : ''}${Number(l)}%`}
            contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 6, fontSize: 11 }}
            itemStyle={{ color: '#e6edf3' }}
            labelStyle={{ color: '#8b949e' }}
          />
          <ReferenceLine y={0} stroke="rgba(139,148,158,0.4)" strokeWidth={1} />
          {breakevens.map((b, i) => (
            <ReferenceLine
              key={i}
              x={Number(b)}
              stroke="rgba(240,160,32,0.55)"
              strokeDasharray="4 3"
              label={{ value: `BE`, position: 'top', fill: '#f0a020', fontSize: 9 }}
            />
          ))}
          <Line type="monotone" dataKey="expand"  stroke="#22c55e" strokeWidth={1.5} dot={false} />
          <Line type="monotone" dataKey="flat"    stroke="#58a6ff" strokeWidth={2}   dot={false} />
          <Line type="monotone" dataKey="crush25" stroke="#f0a020" strokeWidth={1.5} dot={false} />
          <Line type="monotone" dataKey="crush45" stroke="#ef4444" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: 14, marginTop: 4, fontSize: 11, color: '#8b949e', flexWrap: 'wrap' }}>
        <span style={{ color: '#22c55e' }}>━ IV +20%</span>
        <span style={{ color: '#58a6ff' }}>━ IV Flat</span>
        <span style={{ color: '#f0a020' }}>━ IV −25%</span>
        <span style={{ color: '#ef4444' }}>━ IV −45%</span>
        <span style={{ color: 'rgba(240,160,32,0.6)' }}>╌ BE</span>
      </div>
    </div>
  )
}

// ── Export handler ────────────────────────────────────────────────────────────
function exportJson(symbol, result) {
  if (!result) return
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `${symbol}-edge-${TODAY_ISO}.json`
  a.click()
  URL.revokeObjectURL(url)
}

// ── Watchlist hook (localStorage) ─────────────────────────────────────────────
function useWatchlist() {
  const LS_KEY = 'watchlist_v1'
  const [watchlist, setWatchlist] = useState(() => {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || '[]') } catch { return [] }
  })
  function addToWatchlist(entry) {
    setWatchlist(prev => {
      const filtered = prev.filter(e => e.symbol !== entry.symbol)
      const next = [entry, ...filtered].slice(0, 20)
      try { localStorage.setItem(LS_KEY, JSON.stringify(next)) } catch {}
      return next
    })
  }
  function removeFromWatchlist(sym) {
    setWatchlist(prev => {
      const next = prev.filter(e => e.symbol !== sym)
      try { localStorage.setItem(LS_KEY, JSON.stringify(next)) } catch {}
      return next
    })
  }
  function isInWatchlist(sym) {
    return watchlist.some(e => e.symbol === sym)
  }
  return { watchlist, addToWatchlist, removeFromWatchlist, isInWatchlist }
}

// ── Alert config hook (localStorage) ─────────────────────────────────────────
const ALERT_DEFAULTS = { min_confidence: 70, min_edge: 0.5, vol_regime_filter: 'any', enabled: true }
function useAlertConfig() {
  const LS_KEY = 'alert_config_v1'
  const [config, setConfigState] = useState(() => {
    try { return { ...ALERT_DEFAULTS, ...JSON.parse(localStorage.getItem(LS_KEY) || '{}') } }
    catch { return ALERT_DEFAULTS }
  })
  function setConfig(patch) {
    setConfigState(prev => {
      const next = { ...prev, ...patch }
      try { localStorage.setItem(LS_KEY, JSON.stringify(next)) } catch {}
      return next
    })
  }
  return { config, setConfig }
}

// ── WatchlistChips component ──────────────────────────────────────────────────
function WatchlistChips({ watchlist, onSelect, onRemove }) {
  if (!watchlist.length) return null
  return (
    <div className="watchlist-strip">
      <span className="watchlist-label">Watchlist</span>
      {watchlist.map(e => (
        <span
          key={e.symbol}
          className={`watchlist-chip rec-chip-${(e.recommendation || 'default').toLowerCase().replace(/[\s/]+/g, '-')}`}
        >
          <span className="chip-symbol" onClick={() => onSelect(e.symbol)} title={`Re-run ${e.symbol}`}>
            {e.symbol}
          </span>
          {e.confidence_pct != null && (
            <span className="chip-conf">{Math.round(e.confidence_pct)}%</span>
          )}
          <button className="watchlist-chip-remove" onClick={() => onRemove(e.symbol)} title="Remove">×</button>
        </span>
      ))}
    </div>
  )
}

// ── AlertBanner component ─────────────────────────────────────────────────────
function AlertBanner({ config, result }) {
  if (!config?.enabled || !result) return null
  const m   = result.metrics || {}
  const conf = Number(result.confidence_pct || 0)
  const edge = Number(m.expected_net_edge_pct || 0)   // already in % form (e.g. 6.30)
  const regime = m.vol_regime || ''
  const recOk    = !m.hard_no_trade && result.recommendation !== 'No Trade' && result.recommendation !== 'Pass'
  const confOk   = conf >= config.min_confidence
  const edgeOk   = edge >= config.min_edge
  const regimeOk = config.vol_regime_filter === 'any' || regime === config.vol_regime_filter
  if (!(recOk && confOk && edgeOk && regimeOk)) return null
  return (
    <div className="alert-banner">
      <span className="alert-icon">🔔</span>
      <span className="alert-text">
        Alert: {result.symbol || ''} meets thresholds — {result.recommendation} · {conf.toFixed(1)}% conf · Edge {edge.toFixed(2)}%
      </span>
    </div>
  )
}

// ── AlertConfigPanel component ────────────────────────────────────────────────
function AlertConfigPanel({ config, setConfig }) {
  return (
    <div className="alert-config-panel">
      <div className="oos-controls">
        <label className="oos-field">
          <span>Alerts Enabled</span>
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={e => setConfig({ enabled: e.target.checked })}
          />
        </label>
        <label className="oos-field">
          <span>Min Confidence %</span>
          <input
            type="number" min="0" max="100" step="1"
            value={config.min_confidence}
            onChange={e => setConfig({ min_confidence: Number(e.target.value) })}
          />
        </label>
        <label className="oos-field">
          <span>Min Net Edge %</span>
          <input
            type="number" min="0" step="0.1"
            value={config.min_edge}
            onChange={e => setConfig({ min_edge: Number(e.target.value) })}
          />
        </label>
        <label className="oos-field">
          <span>Vol Regime Filter</span>
          <select
            value={config.vol_regime_filter}
            onChange={e => setConfig({ vol_regime_filter: e.target.value })}
          >
            <option value="any">Any</option>
            <option value="Low">Low</option>
            <option value="Normal">Normal</option>
            <option value="Elevated">Elevated</option>
            <option value="High">High</option>
          </select>
        </label>
      </div>
    </div>
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

  useEffect(() => {
    if (!oosLoading) { setOosElapsedSec(0); return }
    const t = setInterval(() => setOosElapsedSec(p => p + 1), 1000)
    return () => clearInterval(t)
  }, [oosLoading])

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
    ? `${Number(result.confidence_pct).toFixed(1)}%${result?.metrics?.confidence_capped ? ' (cap)' : ''}`
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
    setError(''); setLoading(true); setResult(null)
    try {
      const res = await fetch(`${API_BASE}/api/edge/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: s }),
      })
      if (!res.ok) {
        const b = await res.json().catch(() => ({}))
        throw new Error(b.detail || `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setLoading(false)
    }
  }

  async function runAnalysis(e) {
    e?.preventDefault()
    return runForSymbol(null)
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
      const submitRes = await fetch(`${API_BASE}/api/oos/submit`, {
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

      // Poll every 2 s until done, error, or user cancels.
      oosIntervalRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/api/oos/status/${jobId}`)
          if (!statusRes.ok) return  // transient error — keep polling
          const statusData = await statusRes.json()

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
      setOosLoading(false)
      setOosError(String(err.message || err))
    }
  }

  const m = result?.metrics || {}

  return (
    <div className="page-shell">
      <div className="bg-orb bg-orb-a" />
      <div className="bg-orb bg-orb-b" />
      <main className="terminal-panel">

        {/* ── Header ── */}
        <header className="panel-header">
          <div>
            <h1>IV Crush Edge Terminal</h1>
            <p>Institutional IV crush research. Single-ticker. No scanner noise.</p>
          </div>
          <div className="header-badges">
            <div className="status-chip">Production Research</div>
          </div>
        </header>

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

          {result && noTradeBlocked && (
            <div className="no-trade-banner">
              <div className="no-trade-title">NO TRADE — Hard Gate Fail</div>
              <ul>
                {hardGateReasons.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          )}

          {result && (
            <>
              {/* Recommendation strip */}
              <div className="rec-strip">
                <div className="rec-left">
                  <span className={`rec-badge rec-${recommendationValue.toLowerCase().replace(' ', '-')}`}>
                    {recommendationValue}
                  </span>
                  <span className="rec-confidence">{confidenceValue} confidence</span>
                  {m.earnings_release_time && releaseTimeBadge(m.earnings_release_time)}
                  <VolRegimeBadge regime={m.vol_regime} pct={m.rv_percentile_rank} />
                  <TickerTierBadge tier={m.ticker_tier} mult={m.ticker_tier_mult} />
                </div>
                <div className="rec-right">
                  <span className="rec-detail">Setup {Number(result.setup_score).toFixed(3)}</span>
                  <span className="rec-detail">Composite {fmtNum(m.composite_score, 3)}</span>
                  <span className={`rec-detail edge-quality-${(m.edge_quality || '').toLowerCase().replace(/\s+/g, '-')}`}>
                    {m.edge_quality || 'n/a'}
                  </span>
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
                          recommendation: recommendationValue,
                          confidence_pct: Number(result.confidence_pct),
                          setup_score: Number(result.setup_score),
                          vol_regime: m.vol_regime,
                          timestamp: new Date().toISOString(),
                        })
                      }
                    }}
                  >{isInWatchlist(normalizedSymbol) ? '★' : '☆'}</button>
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

              {/* Confidence calibration breakdown */}
              {m.confidence_pct_raw != null && (
                <div className="metrics-group">
                  <div className="metrics-group-label">
                    Confidence Calibration
                    <span style={{ marginLeft: 10, fontWeight: 400, color: '#8b949e', fontSize: 11 }}>
                      raw → adjusted via tier × kurtosis × crush-rate
                    </span>
                  </div>
                  <div className="metrics-grid">
                    <Metric
                      label="Raw Confidence"
                      value={`${Number(m.confidence_pct_raw).toFixed(1)}%`}
                      sub="before calibration"
                    />
                    <Metric
                      label="Ticker Tier"
                      value={m.ticker_tier ? m.ticker_tier.replace('_', '-') : 'n/a'}
                      sub={m.market_cap_usd != null
                        ? `$${(m.market_cap_usd / 1e9).toFixed(1)}B mkt cap`
                        : 'market cap unavailable'}
                      tone={
                        !m.ticker_tier ? 'default'
                        : ['mega_cap','large_cap'].includes(m.ticker_tier) ? 'good'
                        : m.ticker_tier === 'mid_cap' ? 'warn'
                        : 'bad'
                      }
                    />
                    <Metric
                      label="Tier Multiplier"
                      value={m.ticker_tier_mult != null ? `${Math.round(m.ticker_tier_mult * 100)}%` : 'n/a'}
                      tone={tonePos(m.ticker_tier_mult, 0.90, 0.75)}
                    />
                    <Metric
                      label="Move Kurtosis"
                      value={m.move_kurtosis != null ? fmtNum(m.move_kurtosis, 2) : 'n/a'}
                      sub="excess kurtosis (normal=0)"
                      tone={
                        m.move_kurtosis == null ? 'default'
                        : m.move_kurtosis < 1.0 ? 'good'
                        : m.move_kurtosis < 3.0 ? 'warn'
                        : 'bad'
                      }
                    />
                    <Metric
                      label="Kurtosis Haircut"
                      value={m.kurtosis_conf_mult != null ? `${Math.round(m.kurtosis_conf_mult * 100)}%` : 'n/a'}
                      tone={tonePos(m.kurtosis_conf_mult, 0.90, 0.78)}
                    />
                    <Metric
                      label="Hist. Crush Rate"
                      value={m.hist_crush_rate != null ? `${Math.round(m.hist_crush_rate * 100)}%` : 'n/a'}
                      sub="% past moves < implied"
                      tone={
                        m.hist_crush_rate == null ? 'default'
                        : m.hist_crush_rate >= 0.70 ? 'good'
                        : m.hist_crush_rate >= 0.50 ? 'warn'
                        : 'bad'
                      }
                    />
                    <Metric
                      label="Crush Calibration"
                      value={m.crush_calibration_mult != null ? `${Math.round(m.crush_calibration_mult * 100)}%` : 'n/a'}
                      tone={tonePos(m.crush_calibration_mult, 0.95, 0.83)}
                    />
                    <Metric
                      label="Combined Multiplier"
                      value={m.confidence_calibration_mult != null ? `${Math.round(m.confidence_calibration_mult * 100)}%` : 'n/a'}
                      accent
                      tone={tonePos(m.confidence_calibration_mult, 0.90, 0.75)}
                    />
                  </div>
                </div>
              )}

              <div className="metrics-group">
                <div className="metrics-group-label">Volatility Surface</div>
                <div className="metrics-grid">
                  {/* FIX 5: show vol metrics as % (annualized), not raw decimals */}
                  <Metric label="IV30" value={m.iv30 != null ? `${(m.iv30 * 100).toFixed(1)}%` : 'n/a'}
                    sub="annualized" />
                  <Metric label="RV30 (YZ)" value={m.rv30 != null ? `${(m.rv30 * 100).toFixed(1)}%` : 'n/a'}
                    sub={m.rv_estimator === 'yang_zhang' ? 'Yang-Zhang · annualized' : 'close-to-close · annualized'} />
                  <Metric label="IV / RV30" value={fmtNum(m.iv_rv30, 3)}
                    tone={tonePos(m.iv_rv30, 1.05, 1.0)} />
                  <Metric label="RV Forecast (HAR)" value={m.rv30_har_forecast != null ? `${(m.rv30_har_forecast * 100).toFixed(1)}%` : 'n/a'}
                    sub={m.rv30_har_forecast != null ? 'Corsi 2009 · 1d fwd · annualized' : 'n/a (< 35d hist)'} />
                  <Metric label="IV / RV (fwd)" value={fmtNum(m.iv_rv_har, 3)}
                    tone={tonePos(m.iv_rv_har, 1.05, 1.0)}
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
                  <Metric label="IV45" value={m.iv45 != null ? `${(m.iv45 * 100).toFixed(1)}%` : 'n/a'}
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
                  <Metric label="Uncertainty Penalty" value={fmtPp(m.move_uncertainty_pct, 2)}
                    tone={toneNeg(m.move_uncertainty_pct, 1.0, 2.0)} />
                  <Metric label="Sample Confidence" value={fmtNum(m.sample_confidence, 2)}
                    sub={`${m.earnings_move_sample_size ?? 0} events · ${m.earnings_move_source || 'none'}`}
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
                </div>
              </div>

              {(m.kelly_half_pct != null || m.kelly_full_pct != null) && (
                <div className="metrics-group">
                  <div className="metrics-group-label">Position Sizing · Kelly Criterion (heuristic)</div>
                  <div className="metrics-grid">
                    <Metric
                      label="Half-Kelly (Recommended)"
                      value={m.kelly_half_pct != null ? `${m.kelly_half_pct}%` : 'n/a'}
                      accent
                      tone={tonePos(m.kelly_half_pct, 0.5, 0.1)}
                      sub="% of portfolio to risk"
                    />
                    <Metric
                      label="Full Kelly (Aggressive)"
                      value={m.kelly_full_pct != null ? `${m.kelly_full_pct}%` : 'n/a'}
                      tone={tonePos(m.kelly_full_pct, 1.0, 0.2)}
                      sub="% of portfolio — use half-Kelly in practice"
                    />
                    {m.kelly_half_pct != null && (
                      <Metric
                        label="$10k Portfolio →"
                        value={`$${(m.kelly_half_pct * 100).toFixed(0)} risk`}
                        sub="half-Kelly dollar amount"
                      />
                    )}
                    {m.kelly_half_pct != null && (
                      <Metric
                        label="$50k Portfolio →"
                        value={`$${(m.kelly_half_pct * 500).toFixed(0)} risk`}
                        sub="half-Kelly dollar amount"
                      />
                    )}
                  </div>
                  <div className="kelly-note">
                    Kelly is a heuristic based on edge÷risk×confidence. For options, treat as max debit to risk per trade — not total notional.
                  </div>
                </div>
              )}

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
                {/* FIX 3: stale data warning */}
                {(m.price_data_stale || (m.price_data_age_days != null && m.price_data_age_days > 1)) && (
                  <span style={{ color: '#f0a020', fontWeight: 600 }}>
                    ⚠ Stale data ({m.price_data_age_days}d old)
                  </span>
                )}
              </div>

              {/* Rationale */}
              <div className="rationale-card">
                <h2>Decision Rationale</h2>
                <ul>
                  {(result.rationale || []).map((line, i) => <li key={i}>{line}</li>)}
                </ul>
              </div>
            </>
          )}
        </section>

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
          <p className="oos-help">Strategy-level walk-forward validation across the full S&amp;P 500 backfill universe — not ticker-specific. Results are the same regardless of which ticker you searched because the test measures whether the IV crush signal works as a repeatable strategy across many symbols. Typically 30–120 s.</p>

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
            <label className="oos-field"><span>Min Crush Conf</span>
              <input type="number" step="0.01" value={oosParams.min_crush_confidence} onChange={e => updateOosParam('min_crush_confidence', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Crush Magn</span>
              <input type="number" step="0.01" value={oosParams.min_crush_magnitude} onChange={e => updateOosParam('min_crush_magnitude', e.target.value)} />
            </label>
            <label className="oos-field"><span>Min Crush Edge</span>
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
