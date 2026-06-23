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
 * Unified payoff diagram for the four supported structures
 * (atm_straddle / otm_strangle / call_calendar / put_calendar).
 * Plots P&L per contract across four IV scenarios: +20%, flat, -25%, -45%.
 * Returns null if no payoff_scenarios are present.
 */
export default function StructurePayoffChart({ payoff }) {
  if (!payoff?.payoff_scenarios?.length) return null

  const structure = payoff.structure || ''
  const isCalendar = structure === 'call_calendar' || structure === 'put_calendar'
  const isStraddle = structure === 'atm_straddle'
  const isStrangle = structure === 'otm_strangle'

  const scenarios = payoff.payoff_scenarios_per_contract?.length
    ? payoff.payoff_scenarios_per_contract
    : payoff.payoff_scenarios

  const data = scenarios.map((s) => ({
    move: Number(s.move_pct),
    expand: Number((s.iv_expand_20 ?? 0).toFixed(2)),
    flat: Number((s.iv_flat ?? 0).toFixed(2)),
    crush25: Number((s.iv_crush_25 ?? 0).toFixed(2)),
    crush45: Number((s.iv_crush_45 ?? 0).toFixed(2)),
  }))

  const breakevens = payoff.breakeven_moves_pct || []
  const zoom = usePayoffZoom(data, SERIES_KEYS)

  const titleMap = {
    atm_straddle: 'Long ATM Straddle · P&L at 1-Day Post-Event',
    otm_strangle: 'Long OTM Strangle · P&L at 1-Day Post-Event',
    call_calendar: 'Call Calendar Spread · P&L at Near-Leg Expiry',
    put_calendar: 'Put Calendar Spread · P&L at Near-Leg Expiry',
  }
  const chartTitle = titleMap[structure] || 'Structure P&L Diagram'

  const xAxisLabel = isCalendar
    ? 'Underlying Move at Near-Leg Expiry'
    : 'Underlying Move 1-Day Post-Earnings'

  const debitStr = payoff.entry_debit_per_contract != null
    ? `$${Number(payoff.entry_debit_per_contract).toFixed(2)}/contract`
    : payoff.entry_debit != null
      ? `$${Number(payoff.entry_debit).toFixed(3)}/share`
      : 'n/a'

  let metaStr = `Entry Debit ${debitStr}`
  if (isStraddle && payoff.strike != null) {
    metaStr += ` · K=${Number(payoff.strike).toFixed(2)} (ATM)`
  } else if (isStrangle && payoff.strike_call != null) {
    metaStr += ` · Call K=${Number(payoff.strike_call).toFixed(2)} / Put K=${Number(payoff.strike_put).toFixed(2)}`
    if (payoff.wing_pct != null) metaStr += ` · ±${Number(payoff.wing_pct).toFixed(1)}% wings`
  } else if (isCalendar) {
    metaStr += ` · (near ${payoff.t_near_days}d / back ${payoff.t_back_days}d)`
  }

  return (
    <div className="vol-chart-wrapper">
      <div className="vol-chart-label">{chartTitle} · $/contract (100 shares) · {metaStr}</div>
      {breakevens.length > 0 && (
        <div className="breakeven-note">
          IV-flat breakevens:&nbsp;
          {breakevens.map((b, i) => (
            <span key={i}>{b > 0 ? '+' : ''}{Number(b).toFixed(1)}%{i < breakevens.length - 1 ? ', ' : ''}</span>
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
            label={{ value: xAxisLabel, position: 'insideBottom', offset: -8, fill: CHART.axis, fontSize: 10 }}
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
              return [`$${Number(v).toFixed(2)}`, labels[name] || name]
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
        {(isStraddle || isStrangle) && (
          <span style={{ color: CHART.axis, marginLeft: 4 }}>· MTM 1-day post-event</span>
        )}
      </div>
      <ChartZoomHint zoomed={zoom.zoomed} onReset={zoom.reset} />
      {(payoff.is_theoretical || payoff.calendar_is_theoretical) && (
        <div style={{ marginTop: 6, fontSize: 10, color: CHART.axisDim }}>
          {isCalendar
            ? '⚠ Theoretical — priced from interpolated IV30/IV45. Verify debit with live chain.'
            : '⚠ Theoretical — BSM-priced at ATM IV. IV scenarios are symbol-calibrated estimates.'}
        </div>
      )}
    </div>
  )
}
