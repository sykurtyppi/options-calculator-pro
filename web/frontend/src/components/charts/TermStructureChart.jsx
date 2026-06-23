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
import { CHART, axisTick, tooltipContentStyle } from './chartTheme'

/**
 * Volatility term-structure chart: DTE on x-axis vs IV % on y-axis,
 * with an optional vertical earnings reference line. Returns null when
 * fewer than two valid (dte, iv) pairs are available.
 */
export default function TermStructureChart({ days, ivs, earningsDte }) {
  if (!Array.isArray(days) || days.length < 2) return null

  const data = days
    .map((d, i) => ({
      dte: Math.round(Number(d)),
      iv: ivs[i] != null ? Number((ivs[i] * 100).toFixed(2)) : null,
    }))
    .filter((p) => p.iv != null)
    .sort((a, b) => a.dte - b.dte)

  if (data.length < 2) return null

  const ivMin = Math.min(...data.map((d) => d.iv))
  const ivMax = Math.max(...data.map((d) => d.iv))
  const pad = Math.max((ivMax - ivMin) * 0.25, 0.5)
  const yMin = Math.max(0, (ivMin - pad).toFixed(1))
  const yMax = (ivMax + pad).toFixed(1)

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
            tick={axisTick}
            label={{ value: 'DTE', position: 'insideBottomRight', offset: -4, fill: CHART.axis, fontSize: 11 }}
          />
          <YAxis
            domain={[yMin, yMax]}
            tickFormatter={(v) => `${v}%`}
            tick={axisTick}
            width={42}
          />
          <Tooltip
            formatter={(v) => [`${v}%`, 'IV']}
            labelFormatter={(l) => `DTE ${l}`}
            contentStyle={tooltipContentStyle}
            itemStyle={{ color: CHART.text }}
            labelStyle={{ color: CHART.axis }}
          />
          {earningsDte != null && (
            <ReferenceLine
              x={earningsDte}
              stroke={CHART.series.warn}
              strokeDasharray="4 3"
              label={{ value: 'Earnings', position: 'top', fill: CHART.series.warn, fontSize: 10 }}
            />
          )}
          <ReferenceLine x={30} stroke="rgba(139,148,158,0.35)" strokeDasharray="2 4" />
          <Line
            type="monotone"
            dataKey="iv"
            stroke={CHART.series.accent}
            strokeWidth={2}
            dot={{ fill: CHART.series.accent, r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
