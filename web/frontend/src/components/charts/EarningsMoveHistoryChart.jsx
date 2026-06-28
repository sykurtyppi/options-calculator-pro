import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
  ResponsiveContainer,
} from 'recharts'
import { CHART, axisTick, tooltipContentStyle } from './chartTheme'

/**
 * Actual close-to-close stock moves on past earnings days, with the current
 * implied move drawn as a reference line. Bars at/above the line are quarters
 * the stock moved at least as much as the market is now pricing — the "are
 * options cheap?" read. Returns null with fewer than one valid bar.
 */
export default function EarningsMoveHistoryChart({ data, impliedMove }) {
  if (!Array.isArray(data) || data.length < 1) return null

  const moves = data.map((d) => d.move)
  const top = Math.max(...moves, impliedMove != null ? impliedMove : 0)
  const yMax = Math.ceil((top + Math.max(top * 0.15, 0.5)))

  return (
    <div className="vol-chart-wrapper">
      <div className="vol-chart-label">Actual move % by earnings date · dashed = implied now</div>
      <ResponsiveContainer width="100%" height={210}>
        <BarChart data={data} margin={{ top: 24, right: 18, bottom: 4, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis dataKey="label" tick={axisTick} interval={0} />
          <YAxis
            domain={[0, yMax]}
            allowDecimals={false}
            tickFormatter={(v) => `${Math.round(v)}%`}
            tick={axisTick}
            width={48}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
            formatter={(v) => [`${v}%`, 'Move']}
            labelFormatter={(l) => `Earnings ${l}`}
            contentStyle={tooltipContentStyle}
            itemStyle={{ color: CHART.text }}
            labelStyle={{ color: CHART.axis }}
          />
          {impliedMove != null && (
            <ReferenceLine
              y={impliedMove}
              stroke={CHART.series.warn}
              strokeDasharray="4 3"
              label={{ value: `Implied ${impliedMove}%`, position: 'top', fill: CHART.series.warn, fontSize: 10 }}
            />
          )}
          <Bar dataKey="move" isAnimationActive={false} radius={[3, 3, 0, 0]}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={impliedMove != null && d.move >= impliedMove ? CHART.series.pos : CHART.series.accent}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
