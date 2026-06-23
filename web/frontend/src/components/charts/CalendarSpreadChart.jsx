import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
} from 'recharts'
import { CHART, axisTick, tooltipContentStyle } from './chartTheme'
import { usePayoffZoom } from './usePayoffZoom'
import ChartZoomHint from './ChartZoomHint'

const SERIES_KEYS = ['expand', 'flat', 'crush25', 'crush45']

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

  const breakevens = calPayoff.breakeven_moves_pct || []
  const zoom = usePayoffZoom(data, SERIES_KEYS)

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
      <ResponsiveContainer width="100%" height={240}>
        <LineChart
          data={data}
          margin={{ top: 24, right: 24, bottom: 20, left: 8 }}
          style={{ cursor: 'crosshair', userSelect: 'none' }}
          {...zoom.handlers}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="move"
            type="number"
            domain={zoom.xDomain}
            allowDataOverflow
            tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}%`}
            tick={{ ...axisTick, fontSize: 10 }}
            label={{ value: 'Underlying Move at Expiry', position: 'insideBottom', offset: -8, fill: CHART.axis, fontSize: 10 }}
          />
          <YAxis
            domain={zoom.yDomain}
            allowDataOverflow
            tickFormatter={(v) => `$${v.toFixed(2)}`}
            tick={{ ...axisTick, fontSize: 10 }}
            width={58}
          />
          <Tooltip
            formatter={(v, name) => {
              const labels = { expand: 'IV +20%', flat: 'IV Flat', crush25: 'IV −25%', crush45: 'IV −45%' }
              return [`$${Number(v).toFixed(3)}`, labels[name] || name]
            }}
            labelFormatter={(l) => `Move: ${Number(l) > 0 ? '+' : ''}${Number(l)}%`}
            contentStyle={tooltipContentStyle}
            itemStyle={{ color: CHART.text }}
            labelStyle={{ color: CHART.axis }}
          />
          <ReferenceLine y={0} stroke="rgba(139,148,158,0.4)" strokeWidth={1} />
          {breakevens.map((b, i) => (
            <ReferenceLine
              key={i}
              x={Number(b)}
              stroke="rgba(240,160,32,0.55)"
              strokeDasharray="4 3"
              label={{ value: 'BE', position: 'top', fill: CHART.series.warn, fontSize: 9 }}
            />
          ))}
          <Line type="monotone" dataKey="expand" stroke={CHART.series.pos} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="flat" stroke={CHART.series.accent} strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="crush25" stroke={CHART.series.warn} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="crush45" stroke={CHART.series.neg} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          {zoom.sel && zoom.sel.x1 !== zoom.sel.x2 && (
            <ReferenceArea x1={zoom.sel.x1} x2={zoom.sel.x2} strokeOpacity={0.3} fill={CHART.series.accent} fillOpacity={0.12} />
          )}
        </LineChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: 14, marginTop: 4, fontSize: 11, color: CHART.axis, flexWrap: 'wrap' }}>
        <span style={{ color: CHART.series.pos }}>━ IV +20%</span>
        <span style={{ color: CHART.series.accent }}>━ IV Flat</span>
        <span style={{ color: CHART.series.warn }}>━ IV −25%</span>
        <span style={{ color: CHART.series.neg }}>━ IV −45%</span>
        <span style={{ color: 'rgba(240,160,32,0.6)' }}>╌ BE</span>
      </div>
      <ChartZoomHint zoomed={zoom.zoomed} onReset={zoom.reset} />
    </div>
  )
}
