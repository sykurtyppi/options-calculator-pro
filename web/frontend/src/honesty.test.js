// Honesty regression guards for the result UI.
//
// The engine never reduces the displayed confidence/setup score:
// confidence_pct == confidence_pct_raw (both = selector confidence), and the
// calibration multiplier (ticker_tier_mult × kurtosis × crush × ML) is computed
// for audit/inspection only — it is exposed in the API JSON but NEVER applied
// (see web/api/edge_engine.py). The "confidence cap" system is likewise
// informational-only: m.confidence_capped is always False and no score is
// reduced.
//
// Earlier the frontend's legacy metrics block contradicted that: it rendered a
// "Score Breakdown" panel implying `confidence = raw × multipliers`, and three
// "Score capped at N" messages asserting a cap that never happens. These guards
// assert the source no longer makes those false claims, and that the honest
// advisory wording is present. They are source-content checks (the project has
// no jsdom/render-test infra yet) — the same pattern used by the backend's
// test_threshold_is_minus40_in_source guard.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const APP = readFileSync(join(__dirname, 'App.jsx'), 'utf8')
const BADGES = readFileSync(join(__dirname, 'components/common/badges.jsx'), 'utf8')

// Strip block + line comments so the guards check live JSX, not our explanatory
// notes (which deliberately quote the old phrasing to document the fix).
function stripComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '')
}

const APP_CODE = stripComments(APP)
const BADGES_CODE = stripComments(BADGES)

test('no false "score capped" claim in the result UI', () => {
  assert.ok(
    !/capped at/i.test(APP_CODE),
    'UI must not claim the score is "capped at N" — the engine never reduces the score',
  )
  assert.ok(
    !/score capped/i.test(APP_CODE),
    'UI must not render a "score capped" badge — confidence_capped is always False',
  )
})

test('no "raw → adjusted via multipliers" Score Breakdown framing', () => {
  assert.ok(
    !/adjusted via/i.test(APP_CODE),
    'UI must not imply the score is "adjusted via" calibration multipliers',
  )
  assert.ok(
    !/Combined Multiplier/.test(APP_CODE),
    'UI must not present a "Combined Multiplier" as an applied score adjustment',
  )
})

test('TickerTierBadge does not append the unapplied tier multiplier', () => {
  assert.ok(
    !/mult\s*!=\s*null/.test(BADGES_CODE),
    'TickerTierBadge must not render the tier multiplier — it is never applied to the score',
  )
  assert.ok(
    !/multLabel/.test(BADGES_CODE),
    'TickerTierBadge must not build a multiplier label',
  )
})

test('honest advisory wording is present where evidence flags fire', () => {
  // The fallback-model / low-event / stale-data flags are real and must still
  // surface — but as advisory caveats, not as score reductions.
  assert.ok(
    /advisory/i.test(APP_CODE),
    'reduced-evidence flags must be presented as advisory caveats',
  )
  assert.ok(
    /not reduced/i.test(APP_CODE),
    'the UI must state explicitly that the score is not reduced',
  )
})
