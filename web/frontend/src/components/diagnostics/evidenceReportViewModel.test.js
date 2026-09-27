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
  buildExcludedEvidenceSummary,
  buildExitAttritionRows,
  buildUniverseShadowRows,
  buildUniverseShadowSummary,
  buildOutlierDependence,
  buildUncertaintyRows,
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
    blocking_reasons: ['at least 60 evidence days', 'enough claimable (execution-grade) evidence'],
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
  assert.deepEqual(summary.gateBlockingReasons, ['at least 60 evidence days', 'enough claimable (execution-grade) evidence'])
  assert.deepEqual(buildEvidenceReportSummary({}).gateBlockingReasons, [])
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


const integrityPayload = {
  universe_shadow: {
    events_recorded: 12,
    open: 3,
    resolved: 9,
    rule: 'Every eligible event is shadow-entered once per baseline.',
    entry_attrition: {
      entered: 30,
      entered_after_retry: 2,
      not_entered: 3,
      not_entered_by_reason: { no_option_expiries: 2, missing_back_leg: 1 },
    },
    by_baseline: {
      always_otm_strangle: {
        all_events: { n: 9, avg_realized_return_pct: -2.0 },
        selector_actionable: { n: 3, avg_realized_return_pct: 4.0 },
        selector_not_actionable: { n: 6, avg_realized_return_pct: -5.0 },
      },
      always_atm_straddle: {
        all_events: { n: 9, avg_realized_return_pct: 1.0 },
        selector_actionable: { n: 0, avg_realized_return_pct: null },
        selector_not_actionable: { n: 9, avg_realized_return_pct: 1.0 },
      },
    },
  },
  exit_attrition: {
    selector: { resolved: 8, exit_missing: 2, attrition_rate: 0.2, by_reason: { booked_contract_unavailable: 2 }, by_structure: { otm_strangle: 2 } },
    baselines: { resolved: 20, exit_missing: 0, attrition_rate: 0, by_reason: {}, by_structure: {} },
  },
  invalidated_outcomes: {
    n: 2,
    resolved_n: 2,
    by_reason: { 'exit_repriced_unheld_strike: exit priced the 265 call': 1, 'notes.evidence_invalidated': 1 },
    note: 'Excluded from every performance figure.',
  },
  legacy_repriced_baselines: {
    n: 4,
    by_exit_repricing: { rediscovered_legacy: 3, unverifiable_entry_context: 1 },
    note: 'Excluded from every comparison.',
  },
}

test('universe rows compare picked vs skipped events per baseline', () => {
  const rows = buildUniverseShadowRows(integrityPayload)
  assert.deepEqual(rows.map((row) => row.label), ['Always ATM straddle', 'Always OTM strangle'])
  const strangle = rows.find((row) => row.name === 'always_otm_strangle')
  assert.equal(strangle.pickedN, 3)
  assert.equal(strangle.skippedN, 6)
  assert.equal(strangle.pickedReturnLabel, '+4.0%')
  assert.equal(strangle.skippedReturnLabel, '-5.0%')
  assert.equal(strangle.selectorEdgeLabel, '+9.0%')
  // No picked events yet: the difference is unknown, not zero.
  assert.equal(rows.find((row) => row.name === 'always_atm_straddle').selectorEdgeLabel, 'n/a')
})

test('universe summary reports coverage and entry attrition', () => {
  const summary = buildUniverseShadowSummary(integrityPayload)
  assert.equal(summary.eventsRecorded, 12)
  assert.equal(summary.enteredAfterRetry, 2)
  assert.equal(summary.notEntered, 3)
  assert.deepEqual(summary.notEnteredReasons, [
    { label: 'no_option_expiries', count: 2 },
    { label: 'missing_back_leg', count: 1 },
  ])
})

test('exit attrition rows expose missing exits and reasons for both cohorts', () => {
  const [selector, baselines] = buildExitAttritionRows(integrityPayload)
  assert.equal(selector.label, 'Selector trades')
  assert.equal(selector.missing, 2)
  assert.equal(selector.rateLabel, '20%')
  assert.deepEqual(selector.reasons, [{ label: 'booked_contract_unavailable', count: 2 }])
  assert.equal(baselines.rateLabel, '0%')
  assert.deepEqual(baselines.reasons, [])
})

test('excluded evidence explains what is left out and why', () => {
  const excluded = buildExcludedEvidenceSummary(integrityPayload)
  assert.equal(excluded.invalidatedN, 2)
  assert.equal(excluded.invalidatedReasons.length, 2)
  assert.equal(excluded.legacyBaselinesN, 4)
  assert.deepEqual(excluded.legacyReasons, [
    { label: 'Exit re-discovered strikes (pre-fix)', count: 3 },
    { label: 'Entry recorded too little to verify', count: 1 },
  ])
})

test('new blocks degrade to empty values on an older report payload', () => {
  assert.deepEqual(buildUniverseShadowRows({}), [])
  assert.equal(buildUniverseShadowSummary({}).eventsRecorded, 0)
  assert.equal(buildExitAttritionRows({})[0].rateLabel, 'n/a')
  assert.equal(buildExcludedEvidenceSummary({}).invalidatedN, 0)
})


const uncertaintyPayload = {
  uncertainty: {
    method: '95% percentile bootstrap, 2000 resamples, fixed seed.',
    selector: {
      n: 25, mean: 6.4, ci_low: 1.2, ci_high: 11.9, verdict: 'above_zero',
      outlier_dependence: { k: 5, mean_without_top_k: -0.5, mean_without_bottom_k: 8.1, top_k_share_of_profit: 0.85, fragile: true },
    },
    selector_minus_baseline: {
      always_atm_straddle: { pairs: 12, mean: 3, ci_low: -1.5, ci_high: 7.25, verdict: 'includes_zero' },
    },
    universe_picked_minus_skipped: {
      always_otm_strangle: { n_first: 4, n_second: 20, mean: 2, ci_low: null, ci_high: null, verdict: 'insufficient_sample' },
    },
  },
}

test('uncertainty rows put every comparison next to its interval', () => {
  const rows = buildUncertaintyRows(uncertaintyPayload)
  assert.deepEqual(rows.map((row) => row.label), [
    'Selector vs no trade',
    'Selector − Always ATM straddle (same events)',
    'Picked − skipped (Always OTM strangle)',
  ])
  assert.equal(rows[0].intervalLabel, '+1.2% to +11.9%')
  assert.equal(rows[0].verdictLabel, 'Above zero')
  assert.equal(rows[1].n, '12')
  assert.equal(rows[1].verdictLabel, 'No detectable difference')
  assert.equal(rows[2].n, '4 vs 20')
  assert.equal(rows[2].intervalLabel, '—')
  assert.equal(rows[2].verdictLabel, 'Too few to tell')
})

test('outlier dependence flags a result carried by a few trades', () => {
  const outliers = buildOutlierDependence(uncertaintyPayload)
  assert.equal(outliers.available, true)
  assert.equal(outliers.fragile, true)
  assert.equal(outliers.withoutTopLabel, '-0.5%')
  assert.equal(outliers.topShareLabel, '85%')
  assert.equal(buildOutlierDependence({}).available, false)
  assert.deepEqual(buildUncertaintyRows({}), [])
})
