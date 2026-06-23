import React from 'react'
import { Badge } from './DisplayAtoms'

/**
 * Earnings release-time badge — BMO / AMC / Intraday based on the raw
 * `release_timing` string from the vol snapshot. Returns null when the
 * string is missing or unrecognised (the caller renders a fallback).
 */
export function releaseTimeBadge(rt) {
  if (!rt) return null
  if (rt.includes('before')) return <Badge variant="bmo">BMO</Badge>
  if (rt.includes('after')) return <Badge variant="amc">AMC</Badge>
  if (rt.includes('during')) return <Badge variant="intraday">Intraday</Badge>
  return <Badge>{rt}</Badge>
}

/**
 * Provenance badge for the option chain + price/RV data source.
 *
 * When `dataSources` is provided and the options source differs from the
 * price/RV source (a real case: MarketData.app chain + yfinance price),
 * both are shown so the user can see the split provenance. Otherwise
 * falls back to a single badge from the legacy `ds` string.
 */
export function dataSourceBadge(ds, dataSources) {
  if (dataSources) {
    const optSrc = dataSources.options_source === 'marketdata_app' ? 'MDApp' : 'yfinance'
    const rvSrc = dataSources.price_rv_source === 'marketdata_app' ? 'MDApp' : 'yfinance'
    if (optSrc !== rvSrc) {
      return (
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>
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

/**
 * Market-cap tier badge. Returns null on unknown/missing tier so the caller can
 * omit cleanly. The tier "multiplier" is deliberately NOT rendered (see note
 * below) — it is an audit-only field the engine never applies to the score.
 */
export function TickerTierBadge({ tier }) {
  // NOTE: the tier "multiplier" is intentionally NOT shown. It is an advisory
  // audit component that the engine never applies to the confidence/setup score
  // (confidence_pct = selector confidence; the calibration multiplier is
  // computed for inspection only). Showing it here implied a score adjustment
  // that does not happen — see the frontend honesty fix.
  if (!tier || tier === 'unknown') return null
  const map = {
    mega_cap: { label: 'Mega-Cap', variant: 'tier-mega' },
    large_cap: { label: 'Large-Cap', variant: 'tier-large' },
    mid_cap: { label: 'Mid-Cap', variant: 'tier-mid' },
    small_cap: { label: 'Small-Cap', variant: 'tier-small' },
    micro_cap: { label: 'Micro-Cap', variant: 'tier-micro' },
  }
  const { label, variant } = map[tier] || { label: tier, variant: 'default' }
  return <Badge variant={variant}>{label}</Badge>
}

/**
 * Volatility-regime badge. `pct` is the rank within recent history.
 */
export function VolRegimeBadge({ regime, pct }) {
  if (!regime || regime === 'unknown') return null
  const variantMap = {
    High: 'regime-high',
    Elevated: 'regime-elevated',
    Normal: 'regime-normal',
    Low: 'regime-low',
  }
  const variant = variantMap[regime] || 'default'
  const label = pct != null ? `Vol ${regime} · ${Math.round(pct)}th pct` : `Vol ${regime}`
  return <Badge variant={variant}>{label}</Badge>
}

/**
 * Move-risk advisory badge. Soft signal only — NOT a hard gate.
 * Reflects the p90-historical / event-implied move ratio.
 */
export function MoveRiskBadge({ level, ratio, sampleSize }) {
  if (!level || level === 'unknown') return null
  const map = {
    low: { label: 'Move Risk: Low', variant: 'move-risk-low' },
    moderate: { label: 'Move Risk: Moderate', variant: 'move-risk-moderate' },
    elevated: { label: 'Move Risk: Elevated', variant: 'move-risk-elevated' },
  }
  const { label, variant } = map[level] || { label: `Move Risk: ${level}`, variant: 'default' }
  const ratioLabel = ratio != null ? ` · P90/Impl ${ratio.toFixed(2)}×` : ''
  const sampleLabel = sampleSize != null ? ` · n=${sampleSize}` : ''
  return (
    <Badge
      variant={variant}
      title="Soft advisory: p90 historical move vs. event-implied move. Not a hard gate."
    >
      {label}{ratioLabel}{sampleLabel}
    </Badge>
  )
}
