import React from 'react'

/**
 * Banner shown when the current analysis result meets all configured
 * alert thresholds. Hidden when alerts are disabled, when result is
 * missing, when the recommendation is a hard no-trade, or when any
 * threshold fails. Returns null in all non-alerting cases.
 */
export default function AlertBanner({ config, result }) {
  if (!config?.enabled || !result) return null
  const m = result.metrics || {}
  const conf = Number(result.confidence_pct || 0)
  const edge = Number(m.expected_net_edge_pct || 0) // already in % form (e.g. 6.30)
  const regime = m.vol_regime || ''

  const recOk = !m.hard_no_trade && result.recommendation !== 'No Trade' && result.recommendation !== 'Pass'
  const confOk = conf >= config.min_confidence
  const edgeOk = edge >= config.min_edge
  const regimeOk = config.vol_regime_filter === 'any' || regime === config.vol_regime_filter

  if (!(recOk && confOk && edgeOk && regimeOk)) return null

  return (
    <div className="alert-banner">
      <span className="alert-icon">🔔</span>
      <span className="alert-text">
        Alert: {result.symbol || ''} meets thresholds — {result.recommendation} · score {conf.toFixed(1)} · Edge {edge.toFixed(2)}%
      </span>
    </div>
  )
}
