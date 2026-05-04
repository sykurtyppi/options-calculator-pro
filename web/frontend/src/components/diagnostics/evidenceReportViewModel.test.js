import test from 'node:test'
import assert from 'node:assert/strict'
import {
  buildBaselineComparisonRows,
  buildEvidenceQualitySummary,
  buildEvidenceReportSummary,
  buildEvidenceWarnings,
  buildExecutionRealismSummary,
  buildQuoteQualityRows,
  buildSimpleIvRvFilter,
  buildSurfaceQualitySummary,
} from './evidenceReportViewModel.js'

const payload = {
  evidence_label: 'paper_research_not_execution_grade',
  commercialization_gate: {
    active_evidence_days: 14,
    minimum_days: 60,
    target_days: 90,
    resolved_selector_outcomes: 2,
    minimum_resolved_sample: 30,
    ready_for_paid_beta: false,
  },
  maturity: {
    maturity_label: 'Insufficient evidence',
    edge_quality_label: 'Withheld: insufficient claimable evidence',
    benchmark_comparison_meaningful: false,
    calibration_interpretation_allowed: false,
    bucket_interpretation_allowed: false,
    warning_flags: ['Calibration interpretation is withheld until enough evidence days are collected.'],
  },
  selector_summary: {
    n: 2,
    win_rate: 0.5,
    avg_realized_return_pct: 1.25,
    avg_realized_expansion_pct: 3.5,
  },
  baseline_comparison: {
    always_atm_straddle: { n: 2, win_rate: 0.5, avg_realized_return_pct: -0.5, avg_realized_expansion_pct: 1.0 },
    no_trade: { n: 2, win_rate: null, avg_realized_return_pct: 0, avg_realized_expansion_pct: 0 },
  },
  simple_iv_rv_filter: {
    n: 1,
    skipped_by_filter: 1,
    win_rate: 1,
    avg_realized_return_pct: 2.4,
    rule: 'Keep selected paper outcomes only when iv_rv_har <= 1.05; otherwise no-trade.',
  },
  quote_quality: {
    entry_sources: { marketdata_app: 3, yfinance: 1 },
  },
  evidence_quality: {
    claim_allowed_count: 0,
    claim_blocked_count: 4,
    execution_grade_count: 0,
    record_only_count: 1,
    degraded_count: 3,
    paper_research_label: 'Recorded rows may be useful for audit/research while still blocked from performance claims.',
  },
  execution_realism: {
    selector_entry_scenario_rows: 2,
    baseline_entry_scenario_rows: 4,
    exit_scenario_rows: 1,
    avg_entry_spread_as_pct_of_premium: 16.25,
  },
  surface_quality: {
    crossed_quote_count: 1,
    zero_bid_count: 2,
    extreme_spread_count: 3,
    sparse_atm_count: 4,
    iv_anomaly_count: 5,
    paper_research_label: 'Surface quality is an evidence gate, not a selector score.',
  },
  warning_flags: ['Resolved selector sample is small.'],
}

test('evidence report summary exposes commercialization gate without overclaiming', () => {
  const summary = buildEvidenceReportSummary(payload)
  assert.equal(summary.readyForPaidBeta, false)
  assert.equal(summary.maturityLabel, 'Insufficient evidence')
  assert.equal(summary.edgeQualityLabel, 'Withheld: insufficient claimable evidence')
  assert.equal(summary.benchmarkMeaningful, false)
  assert.equal(summary.calibrationInterpretationAllowed, false)
  assert.equal(summary.activeDays, 14)
  assert.equal(summary.selectorReturnLabel, '+1.3%')
  assert.equal(summary.selectorWinRateLabel, '50%')
})

test('baseline rows include selector and no-trade comparison', () => {
  const rows = buildBaselineComparisonRows(payload)
  assert.equal(rows[0].name, 'selector')
  assert.ok(rows.some((row) => row.name === 'no_trade' && row.returnLabel === '0.0%'))
  assert.ok(rows.some((row) => row.label === 'Always ATM straddle'))
})

test('quote source rows sort by count', () => {
  const rows = buildQuoteQualityRows(payload)
  assert.deepEqual(rows[0], { source: 'marketdata_app', count: 3 })
})

test('simple IV/RV filter remains observational', () => {
  const model = buildSimpleIvRvFilter(payload)
  assert.equal(model.n, 1)
  assert.equal(model.skippedByFilter, 1)
  assert.match(model.rule, /iv_rv_har <= 1\.05/)
})

test('warnings are passed through for display', () => {
  assert.deepEqual(buildEvidenceWarnings(payload), [
    'Resolved selector sample is small.',
    'Calibration interpretation is withheld until enough evidence days are collected.',
  ])
})

test('evidence quality summary separates recorded rows from claimable evidence', () => {
  const quality = buildEvidenceQualitySummary(payload)
  assert.equal(quality.claimAllowed, 0)
  assert.equal(quality.claimBlocked, 4)
  assert.equal(quality.recordOnly, 1)
  assert.equal(quality.degraded, 3)
})

test('execution realism summary formats modeled spread cost', () => {
  const realism = buildExecutionRealismSummary(payload)
  assert.equal(realism.selectorEntryRows, 2)
  assert.equal(realism.baselineEntryRows, 4)
  assert.equal(realism.exitScenarioRows, 1)
  assert.equal(realism.avgEntrySpreadLabel, '16.3%')
})

test('surface quality summary stays diagnostic and non-performance-oriented', () => {
  const surface = buildSurfaceQualitySummary(payload)
  assert.equal(surface.extremeSpreads, 3)
  assert.equal(surface.sparseAtm, 4)
  assert.equal(surface.ivAnomalies, 5)
  assert.match(surface.label, /evidence gate/)
})
