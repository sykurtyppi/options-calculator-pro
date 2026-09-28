import assert from 'node:assert/strict'
import test from 'node:test'

import {
  payoffAssumptions,
  payoffHorizonLabel,
  payoffUnavailableMessage,
  pricingInputWarnings,
} from './payoffDisclosure.js'

test('horizon label states the post-event session and the real remaining life', () => {
  assert.equal(
    payoffHorizonLabel({ valuation_days_after_entry: 5, T_remain_days: 9, valuation_basis: 'first_post_event_session' }),
    'Valued at first post-event session (5d after entry), 9d left to expiry',
  )
  assert.equal(
    payoffHorizonLabel({ valuation_days_after_entry: 1, T_remain_days: 13, valuation_basis: 'one_day_after_entry_no_event_date' }),
    'Valued at 1d after entry (no earnings date), 13d left to expiry',
  )
  // Legacy payloads (no horizon fields) never claim "1 day to expiry".
  assert.equal(payoffHorizonLabel({ T_remain_days: 1 }), 'Valued post-event')
})

test('calendar assumptions disclose synthetic contracts and the 0.88 fallback', () => {
  const items = payoffAssumptions({
    assumptions: {
      synthetic_contracts: true,
      back_iv_source: 'iv30_x_0.88_fallback',
      back_iv_held_fixed: true,
    },
  })
  assert.ok(items.some((item) => item.includes('synthetic contracts')))
  assert.ok(items.some((item) => item.includes('IV30 × 0.88')))
  assert.ok(items.some((item) => item.includes('held constant across every stock move')))
  assert.ok(items.some((item) => item.includes('American-style')))
  assert.ok(items.some((item) => item.includes('not an executable P&L')))
})

test('straddle assumptions disclose one IV and gross economics', () => {
  const items = payoffAssumptions({ structure: 'atm_straddle' })
  assert.ok(items.some((item) => item.includes('no skew')))
  assert.ok(!items.some((item) => item.includes('synthetic')))
  assert.ok(items.some((item) => item.includes('Gross of bid/ask')))
})

test('unavailable reasons are explained', () => {
  assert.match(payoffUnavailableMessage('near_expiry_before_earnings_reaction'), /lapses before the earnings reaction/)
  assert.equal(payoffUnavailableMessage(null), null)
  assert.equal(payoffUnavailableMessage('new_reason'), 'Not shown: new_reason.')
})

test('fallback pricing inputs are flagged', () => {
  const warnings = pricingInputWarnings({
    pricing_risk_free_rate_source: 'fallback_static',
    pricing_dividend_yield_source: 'fallback_zero',
  })
  assert.equal(warnings.length, 2)
  assert.deepEqual(pricingInputWarnings({ pricing_risk_free_rate_source: 'yfinance_^IRX', pricing_dividend_yield_source: 'yfinance' }), [])
})
