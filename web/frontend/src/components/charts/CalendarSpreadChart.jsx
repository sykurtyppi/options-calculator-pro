import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

/**
 * Calendar-spread payoff diagram, kept for the legacy analysis panel.
 *
 * The unified StructurePayoffChart supersedes this for the modern flow,
 * but legacy callers (and saved JSON exports without `structure` set)
 * still need the older calendar-specific rendering. Returns null when
 * no payoff_scenarios are available.
 */
export default function CalendarSpreadChart({ calPayoff }) {
  if (!calPayoff?.payoff_scenarios?.length) return null

  // Prefer per-contract scenarios (multiplied by 100) so Y-axis reads
  // in $/contract.
  const scenarios = calPayoff.payoff_scenarios_per_contract?.length
    ? calPayoff.payoff_scenarios_per_contract
    : calPayoff.payoff_scenarios

  const data = scenarios.map((s) => ({
    move: Number(s.move_pct),
    expand: Number((s.iv_expand_20 ?? 0).toFixed(2)),
    flat: Number((s.iv_flat ?? 0).toFixed(2)),
    crush25: Number((s.iv_crush_25 ?? 0).toFixed(2)),
    crush45: Number((s.iv_crush_45 ?? 0).toFixed(2)),
  }))

  const allVals = data.flatMap((d) => [d.expand, d.flat, d.crush25, d.crush45]).filter((v) => isFinite(v))
  const minV = Math.min(...allVals)
  const maxV = Math.max(...allVals)
  const pad = Math.max((maxV - minV) * 0.14, 0.01)

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
            <span key={i}>{b > 0 ? '+' : ''}{Number(b).toFixed(1)}%{i < breakevens.length - 1 ? ', ' : ''}</span>
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
            tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}%`}
            tick={{ fill: '#8b949e', fontSize: 10 }}
            label={{ value: 'Underlying Move at Expiry', position: 'insideBottom', offset: -8, fill: '#8b949e', fontSize: 10 }}
          />
          <YAxis
            domain={[minV - pad, maxV + pad]}
            tickFormatter={(v) => `$${v.toFixed(2)}`}
            tick={{ fill: '#8b949e', fontSize: 10 }}
            width={58}
          />
          <Tooltip
            formatter={(v, name) => {
              const labels = { expand: 'IV +20%', flat: 'IV Flat', crush25: 'IV −25%', crush45: 'IV −45%' }
              return [`$${Number(v).toFixed(3)}`, labels[name] || name]
            }}
            labelFormatter={(l) => `Move: ${Number(l) > 0 ? '+' : ''}${Number(l)}%`}
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
              label={{ value: 'BE', position: 'top', fill: '#f0a020', fontSize: 9 }}
            />
          ))}
          <Line type="monotone" dataKey="expand" stroke="#22c55e" strokeWidth={1.5} dot={false} />
          <Line type="monotone" dataKey="flat" stroke="#58a6ff" strokeWidth={2} dot={false} />
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
