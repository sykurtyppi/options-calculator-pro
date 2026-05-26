import React from 'react'
import {
  ComposedChart,
  Bar,
  Line,
  Cell,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

/**
 * Per-split P&L bars with a cumulative-P&L overlay line.
 * Returns null when fewer than two splits are available.
 */
export default function OosSplitChart({ splitsDetail }) {
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

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.pnl)), 1)
  const padY = maxAbs * 0.18

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
            tickFormatter={(v) => `$${v}`}
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
