import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildDecisionQualityBreakdown,
  buildEvidenceSummary,
  buildNoTradeSummary,
  buildStructureRows,
  buildVolSnapshotSummary,
  getSelectorPresentation,
} from './selectorViewModel.js'

const strongSelector = {
  recommendation: 'Best Candidate',
  best_structure: 'atm_straddle',
  confidence_pct: 76,
  expected_edge_pct: 4.2,
  expected_return_pct: 7.0,
  data_quality: 'high',
  data_quality_score: 0.92,
}

const noTradeSelector = {
  recommendation: 'No Trade',
  best_structure: null,
  confidence_pct: 28,
  expected_edge_pct: 0,
  expected_return_pct: 0,
  data_quality: 'low',
  data_quality_score: 0.42,
}

const strongScorecards = [
  {
    structure: 'atm_straddle',
    eligible: true,
    eligibility_flags: [],
    expected_edge_pct: 4.2,
    expected_return_pct: 7.0,
    execution_penalty: 0.03,
    sample_confidence: 0.74,
    walk_forward_history_count: 28,
    walk_forward_win_rate: 0.61,
    walk_forward_avg_return_pct: 2.8,
    walk_forward_rank_score: 0.72,
    composite_structure_score: 0.82,
    rationale_bullets: ['Edge quality is positive and return support is supportive; score-derived diagnostics only.'],
  },
  {
    structure: 'otm_strangle',
    eligible: true,
    eligibility_flags: [],
    expected_edge_pct: 1.4,
    expected_return_pct: 2.6,
    execution_penalty: 0.05,
    sample_confidence: 0.58,
    walk_forward_history_count: 14,
    walk_forward_win_rate: 0.52,
    walk_forward_avg_return_pct: 0.9,
    walk_forward_rank_score: 0.52,
    composite_structure_score: 0.59,
    rationale_bullets: ['Runner-up but weaker evidence.'],
  },
]

const staleLowConfidenceSnapshot = {
  earnings_date: '2026-05-01',
  release_timing: 'UNKNOWN',
  days_to_earnings: 8,
  earnings_source_primary: 'alpha_vantage_calendar',
  earnings_source_confidence: 0.54,
  earnings_source_stale: true,
  iv30: 0.24,
  rv30_yang_zhang: 0.23,
  iv_rv_yz: 1.04,
  event_implied_move_pct: 5.2,
  historical_median_move_pct: 4.8,
  historical_vs_implied_move_ratio: 0.92,
  near_back_iv_ratio: 0.94,
  term_structure_slope: 0.001,
  liquidity_tier: 'mid',
  data_quality: 'low',
  data_quality_score: 0.58,
  price_staleness_minutes: 0,
  chain_staleness_minutes: 0,
}

test('golden strong-candidate payload keeps selector hierarchy stable', () => {
  const presentation = getSelectorPresentation(strongSelector)
  const rows = buildStructureRows(strongScorecards, strongSelector)
  const breakdown = buildDecisionQualityBreakdown(strongSelector, strongScorecards, {
    data_quality_score: 0.92,
  })

  assert.equal(presentation.tone, 'best')
  assert.equal(rows[0].isBest, true)
  assert.equal(rows[0].expectedEdge, 'Positive')
  assert.equal(breakdown.map((item) => item.label).join('|'), 'Score Strength|Separation|Data Quality|Evidence Depth')
})

test('golden no-trade payload explains abstention without looking broken', () => {
  const summary = buildNoTradeSummary(
    noTradeSelector,
    [{ ...strongScorecards[0], expected_edge_pct: -0.5, execution_penalty: 0.11 }],
    { data_quality_score: 0.42, near_term_spread_pct: 9.5 },
  )
  const evidence = buildEvidenceSummary(noTradeSelector, [], { data_quality: 'low', data_quality_score: 0.42 })

  assert.equal(getSelectorPresentation(noTradeSelector).tone, 'no-trade')
  assert.match(summary.headline, /Edge not sufficient after costs/)
  assert.match(summary.reasons.join(' '), /Execution friction|Snapshot quality/)
  assert.match(evidence[0].value, /low/)
})

test('golden low-confidence earnings source surfaces downgraded provenance', () => {
  const rows = buildVolSnapshotSummary(staleLowConfidenceSnapshot)
  const evidence = buildEvidenceSummary(strongSelector, strongScorecards, staleLowConfidenceSnapshot)

  assert.equal(rows.find((row) => row.label === 'Earnings Source').value, 'alpha_vantage_calendar · stale fallback')
  assert.equal(rows.find((row) => row.label === 'Source Confidence').value, '54%')
  assert.match(evidence[0].detail, /stale earnings-source fallback/i)
})

test('golden stale-data fallback remains visible while preserving null-safe rendering', () => {
  const rows = buildVolSnapshotSummary({
    ...staleLowConfidenceSnapshot,
    event_implied_move_pct: null,
    historical_median_move_pct: null,
    term_structure_slope: null,
  })

  assert.equal(rows.find((row) => row.label === 'Event Implied Move').value, 'n/a')
  assert.equal(rows.find((row) => row.label === 'Median Hist Move').value, 'n/a')
  assert.equal(rows.find((row) => row.label === 'Term Slope').value, 'n/a')
  assert.equal(rows.find((row) => row.label === 'Earnings Source').value, 'alpha_vantage_calendar · stale fallback')
})
