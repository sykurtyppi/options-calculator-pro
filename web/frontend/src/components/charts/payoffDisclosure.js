// What a theoretical payoff diagram does and does not represent. The curves
// are model scenarios, not executable P&L: every assumption behind them is
// spelled out here so no panel can present them as more than illustrative.

const UNAVAILABLE_REASONS = {
  near_expiry_before_earnings_reaction:
    'Not shown: the nearest expiry lapses before the earnings reaction prints, so a post-event value would be fiction.',
  no_upcoming_earnings_date: 'Not shown: no upcoming earnings date, so there is no post-event session to value at.',
  no_diagram_for_structure: 'No payoff diagram is modelled for this structure.',
  missing_pricing_inputs: 'Not shown: IV, expiry or implied move unavailable for this structure.',
}

export function payoffUnavailableMessage(reason) {
  if (!reason) return null
  return UNAVAILABLE_REASONS[reason] || `Not shown: ${reason}.`
}

// When the straddle/strangle is valued and with how much life left.
export function payoffHorizonLabel(payoff = {}) {
  const valuation = Number(payoff.valuation_days_after_entry)
  const remaining = Number(payoff.T_remain_days)
  if (!Number.isFinite(valuation) || !Number.isFinite(remaining)) return 'Valued post-event'
  const when = payoff.valuation_basis === 'first_post_event_session'
    ? `first post-event session (${Math.round(valuation)}d after entry)`
    : `${Math.round(valuation)}d after entry (no earnings date)`
  return `Valued at ${when}, ${Math.round(remaining)}d left to expiry`
}

const BACK_IV_SOURCE = {
  iv45: 'back-leg IV = IV45',
  'iv30_x_0.88_fallback': 'back-leg IV = IV30 × 0.88 (fixed fallback, IV45 unavailable)',
}

export function payoffAssumptions(payoff = {}) {
  const items = []
  const assumptions = payoff.assumptions || {}
  if (assumptions.synthetic_contracts) {
    items.push('Illustrative synthetic contracts: strike = spot (may not be a listed strike), back expiry = front + 28d (may not be listed).')
    const backIv = BACK_IV_SOURCE[assumptions.back_iv_source]
    if (backIv) items.push(`Entry ${backIv}.`)
    items.push(
      'Valued at front expiry (front leg at intrinsic). Each scenario applies one IV shock to the back leg'
      + (assumptions.back_iv_held_fixed ? ', held constant across every stock move (no skew).' : '.'),
    )
  } else {
    items.push('One IV for every leg (no skew or smile).')
  }
  items.push('Black–Scholes–Merton, European exercise; listed US equity options are American-style.')
  items.push('Gross of bid/ask, fees and slippage: not an executable P&L.')
  return items
}

// Pricing inputs that silently fell back or cannot be trusted.
export function pricingInputWarnings(metrics = {}) {
  const warnings = []
  if (metrics.pricing_risk_free_rate_source === 'fallback_static') {
    warnings.push('Risk-free rate is a fixed fallback (rate feed unavailable); theoretical values are low-confidence.')
  }
  if (metrics.pricing_dividend_yield_source === 'fallback_zero') {
    warnings.push('Dividend yield is 0: no dividend data, or a non-payer. Dividend payers near an ex-date are mispriced.')
  }
  return warnings
}
