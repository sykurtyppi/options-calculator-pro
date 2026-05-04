import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildCalibrationModel,
  buildDecisionQualityBreakdown,
  buildDecisionStability,
  buildEdgeDurability,
  buildEdgeNarrative,
  buildEvidenceSummary,
  buildFailureModes,
  buildHistoricalEvidenceLine,
  buildOutcomeScenarios,
  buildOutcomeDecomposition,
  buildNoTradeSummary,
  buildRegimeConfidenceMessage,
  buildSensitivityWarnings,
  buildStructureAdvantage,
  buildStructureRows,
  buildTrustBadges,
  buildTrustDetails,
  buildVolRegimeBanner,
  buildVolSnapshotSummary,
  getModeledSignalDisclaimer,
  getEdgeLayoutSections,
  getSelectorPresentation,
} from './selectorViewModel.js'

const selectorBase = {
  symbol: 'AAPL',
  as_of: '2026-04-20',
  earnings_date: '2026-04-28',
  release_timing: 'after market close',
  confidence_pct: 74,
  expected_edge_pct: 3.5,
  expected_return_pct: 6.1,
  primary_thesis: 'ATM straddle leads.',
  primary_risks: ['Execution risk', 'Overpricing risk'],
  why_this_structure: ['Composite score leads.'],
  why_not_others: { call_calendar: ['Front-end IV too elevated.'] },
  runner_up_structures: ['otm_strangle', 'call_calendar'],
  data_quality: 'high',
  data_quality_score: 0.91,
}

const scorecards = [
  {
    structure: 'atm_straddle',
    eligible: true,
    eligibility_flags: [],
    expected_edge_pct: 3.5,
    expected_return_pct: 6.1,
    execution_penalty: 0.03,
    theta_drag_penalty: 0.02,
    crowding_penalty: 0.03,
    concavity_penalty: 0.02,
    walk_forward_history_count: 24,
    walk_forward_win_rate: 0.58,
    walk_forward_avg_return_pct: 2.3,
    walk_forward_rank_score: 0.68,
    sample_confidence: 0.71,
    composite_structure_score: 0.74,
    rationale_bullets: ['Composite score leads the structure set.'],
  },
  {
    structure: 'otm_strangle',
    eligible: true,
    eligibility_flags: [],
    expected_edge_pct: 2.1,
    expected_return_pct: 4.4,
    execution_penalty: 0.05,
    theta_drag_penalty: 0.03,
    crowding_penalty: 0.02,
    concavity_penalty: 0.03,
    walk_forward_history_count: 18,
    walk_forward_win_rate: 0.54,
    walk_forward_avg_return_pct: 1.4,
    walk_forward_rank_score: 0.59,
    sample_confidence: 0.62,
    composite_structure_score: 0.63,
    rationale_bullets: ['Tail signal is constructive but not dominant.'],
  },
  {
    structure: 'call_calendar',
    eligible: false,
    eligibility_flags: ['missing_option_chain'],
    expected_edge_pct: 0.0,
    expected_return_pct: 0.0,
    execution_penalty: 0.0,
    theta_drag_penalty: 0.0,
    crowding_penalty: 0.0,
    concavity_penalty: 0.0,
    walk_forward_history_count: 12,
    walk_forward_win_rate: 0.52,
    walk_forward_avg_return_pct: 0.6,
    walk_forward_rank_score: 0.42,
    sample_confidence: 0.52,
    composite_structure_score: 0.0,
    rationale_bullets: [],
  },
]

const volSnapshot = {
  symbol: 'AAPL',
  earnings_date: '2026-04-28',
  release_timing: 'after market close',
  days_to_earnings: 8,
  iv30: 0.25,
  rv30_yang_zhang: 0.21,
  iv_rv_yz: 1.19,
  event_implied_move_pct: 5.9,
  historical_median_move_pct: 7.4,
  historical_vs_implied_move_ratio: 1.26,
  near_back_iv_ratio: 0.88,
  term_structure_slope: 0.0021,
  liquidity_tier: 'high',
  data_quality: 'high',
  data_quality_score: 0.9,
  price_staleness_minutes: 0,
  chain_staleness_minutes: 5,
}

test('selector card presentation supports Best Candidate', () => {
  const presentation = getSelectorPresentation({ ...selectorBase, recommendation: 'Best Candidate' })
  assert.equal(presentation.tone, 'best')
  assert.match(presentation.helper, /clear separation/i)
})

test('selector card presentation supports Watch', () => {
  const presentation = getSelectorPresentation({ ...selectorBase, recommendation: 'Watch' })
  assert.equal(presentation.tone, 'watch')
  assert.match(presentation.helper, /mixed/i)
})

test('selector card presentation supports No Trade', () => {
  const presentation = getSelectorPresentation({ ...selectorBase, recommendation: 'No Trade' })
  assert.equal(presentation.tone, 'no-trade')
  assert.match(presentation.helper, /abstaining/i)
})

test('structure comparison rows show all structures and highlight best while keeping ineligible visible', () => {
  const rows = buildStructureRows(scorecards, { ...selectorBase, best_structure: 'atm_straddle' })
  assert.equal(rows.length, 3)
  assert.equal(rows[0].isBest, true)
  assert.equal(rows[2].isEligible, false)
  assert.equal(rows[0].expectedEdge, 'Positive')
  assert.equal(rows[0].expectedReturn, 'Supportive')
  assert.match(rows[2].rationaleSummary, /Ineligible:/)
})

test('legacy analysis remains last in the page hierarchy', () => {
  const sections = getEdgeLayoutSections()
  assert.deepEqual(sections, [
    'selector-decision',
    'regime-banner',
    'edge-narrative',
    'selector-evidence',
    'calibration-insight',
    'outcome-decomposition',
    'why-structure',
    'structure-comparison',
    'vol-snapshot',
    'evidence-quality',
    'legacy-analysis',
  ])
})

test('vol snapshot summary renders top fields and stays null-safe', () => {
  const rows = buildVolSnapshotSummary({
    ...volSnapshot,
    event_implied_move_pct: null,
    term_structure_slope: null,
  })
  const eventRow = rows.find((row) => row.label === 'Event Implied Move')
  const slopeRow = rows.find((row) => row.label === 'Term Slope')
  assert.equal(eventRow.value, 'n/a')
  assert.equal(slopeRow.value, 'n/a')
})

test('vol snapshot summary exposes earnings source confidence and stale fallback state', () => {
  const rows = buildVolSnapshotSummary({
    ...volSnapshot,
    earnings_source_primary: 'alpha_vantage_calendar',
    earnings_source_confidence: 0.54,
    earnings_source_stale: true,
  })
  const sourceRow = rows.find((row) => row.label === 'Earnings Source')
  const confidenceRow = rows.find((row) => row.label === 'Source Confidence')
  assert.equal(sourceRow.value, 'alpha_vantage_calendar · stale fallback')
  assert.equal(confidenceRow.value, '54%')
})

test('evidence summary remains informative in a no-trade style scenario', () => {
  const summary = buildEvidenceSummary(
    { ...selectorBase, recommendation: 'No Trade', best_structure: null, confidence_pct: 31, data_quality_score: 0.55 },
    [],
    { ...volSnapshot, data_quality: 'mixed', data_quality_score: 0.55 },
  )
  assert.equal(summary.length, 4)
  assert.match(summary[0].value, /mixed/i)
  assert.match(summary[3].detail, /not probability of profit/i)
})

test('historical evidence line stays strong when comparable sample is broad', () => {
  const evidence = buildHistoricalEvidenceLine(
    { ...selectorBase, recommendation: 'Best Candidate', best_structure: 'atm_straddle' },
    scorecards,
  )
  assert.equal(evidence.tone, 'strong')
  assert.match(evidence.headline, /24 comparable setups/)
  assert.match(evidence.headline, /58% positive outcomes/)
  assert.match(evidence.headline, /paper outcome quality Mixed/)
  assert.doesNotMatch(evidence.headline, /\+2\.3%/)
})

test('trust badges surface paper score stale low-quality and low-sample conditions', () => {
  const badges = buildTrustBadges(
    { ...selectorBase, data_quality_score: 0.52 },
    [{ ...scorecards[0], walk_forward_history_count: 5 }],
    {
      ...volSnapshot,
      data_quality_score: 0.52,
      earnings_source_stale: true,
      option_source: 'yfinance_fallback',
      chain_staleness_minutes: 120,
    },
  )
  const labels = badges.map((badge) => badge.label)
  assert.ok(labels.includes('Paper / Research only'))
  assert.ok(labels.includes('Score-derived (not a forecast)'))
  assert.ok(labels.includes('Stale data'))
  assert.ok(labels.includes('Low data quality'))
  assert.ok(labels.includes('Low sample size'))
  assert.ok(labels.includes('Provider degraded'))
})

test('trust details keep explanation short and diagnostic rather than predictive', () => {
  const details = buildTrustDetails(
    { ...selectorBase, best_structure: 'atm_straddle' },
    scorecards,
    { ...volSnapshot, earnings_source_primary: 'alpha_vantage', option_source: 'marketdata_app', underlying_source: 'yfinance' },
  )
  assert.ok(details.some((line) => line.includes('Selected Atm Straddle')))
  assert.ok(details.some((line) => line.includes('Sources: earnings alpha_vantage')))
  assert.ok(details.some((line) => line.includes('not financial advice or return forecasts')))
  assert.ok(details.length <= 10)
})

test('historical evidence line degrades when sample is thin', () => {
  const evidence = buildHistoricalEvidenceLine(
    { ...selectorBase, best_structure: 'atm_straddle' },
    [{ ...scorecards[0], walk_forward_history_count: 5, walk_forward_win_rate: 0.6, walk_forward_avg_return_pct: 0.8 }],
  )
  assert.equal(evidence.tone, 'limited')
  assert.match(evidence.subline, /thin/i)
})

test('decision quality breakdown explains the four evidence pillars', () => {
  const breakdown = buildDecisionQualityBreakdown(
    { ...selectorBase, best_structure: 'atm_straddle' },
    scorecards,
    volSnapshot,
  )
  assert.equal(breakdown.length, 4)
  assert.equal(breakdown[0].label, 'Score Strength')
  assert.equal(breakdown[1].label, 'Separation')
  assert.equal(breakdown[2].label, 'Data Quality')
  assert.equal(breakdown[3].label, 'Evidence Depth')
})

test('decision stability marks close alternatives as unstable', () => {
  const stability = buildDecisionStability(
    { ...selectorBase, best_structure: 'atm_straddle', recommendation: 'Watch' },
    [
      { ...scorecards[0], composite_structure_score: 0.68 },
      { ...scorecards[1], composite_structure_score: 0.65 },
    ],
  )
  assert.equal(stability.label, 'Unstable')
  assert.match(stability.detail, /reshuffle/i)
})

test('calibration model highlights the active bucket and preserves research-prior status', () => {
  const calibration = buildCalibrationModel(
    {
      phase: 'bootstrap_prior',
      n_observations: 12,
      min_for_observational: 40,
      min_for_fit: 120,
      min_for_high_fit: 250,
      buckets: [
        { score_lo: 0.0, score_hi: 0.1, expected_expansion_pct: 1.2, std_pct: 0.8, n: 0 },
        { score_lo: 0.1, score_hi: 0.2, expected_expansion_pct: 1.8, std_pct: 1.0, n: 0 },
        { score_lo: 0.2, score_hi: 0.3, expected_expansion_pct: 2.6, std_pct: 1.2, n: 0 },
      ],
    },
    0.18,
  )
  assert.equal(calibration.isPrior, true)
  assert.equal(calibration.phaseLabel, 'Prior / Ordering Only')
  assert.equal(calibration.activeSummary.rangeLabel, '0.1-0.2')
  assert.match(calibration.activeSummary.detail, /ordering-only guidance/i)
})

test('calibration model distinguishes observational phase from fitted phase', () => {
  const calibration = buildCalibrationModel(
    {
      phase: 'observational',
      n_observations: 58,
      min_for_observational: 40,
      min_for_fit: 120,
      min_for_high_fit: 250,
      buckets: [
        { score_lo: 0.4, score_hi: 0.5, expected_expansion_pct: 4.4, std_pct: 1.8, n: 5 },
        { score_lo: 0.5, score_hi: 0.6, expected_expansion_pct: 5.1, std_pct: 2.0, n: 0 },
      ],
    },
    0.44,
  )
  assert.equal(calibration.isObservational, true)
  assert.equal(calibration.isFitted, false)
  assert.match(calibration.phaseSummary, /No fitted curve/i)
  assert.match(calibration.activeSummary.detail, /Observed bucket average/i)
})

test('no-trade explanation identifies execution-driven abstention clearly', () => {
  const summary = buildNoTradeSummary(
    { ...selectorBase, recommendation: 'No Trade', best_structure: null, data_quality_score: 0.8 },
    [{ ...scorecards[0], expected_edge_pct: 1.0, execution_penalty: 0.12 }],
    { ...volSnapshot, near_term_spread_pct: 9.0 },
  )
  assert.match(summary.headline, /Edge not sufficient after costs/i)
  assert.match(summary.reasons.join(' '), /Execution friction/i)
  assert.match(summary.action, /Tighter spreads/i)
})

test('volatility regime banner explains current regime and structure fit', () => {
  const banner = buildVolRegimeBanner(
    { ...selectorBase, recommendation: 'Best Candidate', best_structure: 'atm_straddle' },
    scorecards,
    { ...volSnapshot, vol_regime_label: 'Low', rv_percentile_rank: 24 },
  )
  assert.equal(banner.tone, 'supportive')
  assert.match(banner.headline, /Low realized volatility/i)
  assert.match(banner.detail, /long-vol expansion structures/i)
  assert.equal(banner.stats.winRateLabel, '58%')
  assert.equal(banner.stats.avgReturnLabel, '+2.3%')
  assert.match(banner.stats.sampleLabel, /24 comparable setups/)
})

test('edge narrative explains pricing, history, and mismatch', () => {
  const narrative = buildEdgeNarrative(
    { ...selectorBase, best_structure: 'atm_straddle' },
    scorecards,
    volSnapshot,
  )
  assert.match(narrative.marketLine, /Market is pricing about 5.90%/)
  assert.match(narrative.historyLine, /7.40%/)
  assert.match(narrative.mismatchLine, /underpriced event risk/i)
})

test('outcome decomposition separates IV, move, and execution drivers', () => {
  const decomposition = buildOutcomeDecomposition(
    { ...selectorBase, best_structure: 'atm_straddle' },
    [{ ...scorecards[0], expected_iv_contribution_pct: 2.6 }],
    volSnapshot,
  )
  assert.equal(decomposition.length, 3)
  assert.equal(decomposition[0].label, 'IV Expansion Contribution')
  assert.match(decomposition[0].value, /signal$/)
  assert.equal(decomposition[1].label, 'Realized Move Contribution')
  assert.equal(decomposition[2].label, 'Execution Friction Risk')
  assert.match(decomposition[2].detail, /not a promise of live retail fills/i)
})

test('outcome scenarios show best, base, and worst case spread', () => {
  const scenarios = buildOutcomeScenarios(
    { ...selectorBase, best_structure: 'atm_straddle' },
    [{ ...scorecards[0], expected_iv_contribution_pct: 2.6 }],
    { ...volSnapshot, historical_move_std_pct: 3.2, historical_move_uncertainty_pct: 2.2 },
  )
  assert.equal(scenarios.length, 3)
  assert.equal(scenarios[0].label, 'Best Case')
  assert.equal(scenarios[1].label, 'Base Case')
  assert.equal(scenarios[2].label, 'Worst Case')
  assert.match(scenarios[1].detail, /not a calibrated return forecast/i)
})

test('edge durability labels stressed edge honestly', () => {
  const durability = buildEdgeDurability(
    { ...selectorBase, best_structure: 'atm_straddle' },
    [{ ...scorecards[0], expected_iv_contribution_pct: 2.6, expected_edge_pct: 3.5 }],
    { ...volSnapshot, historical_move_uncertainty_pct: 1.4 },
  )
  assert.equal(durability.label, 'Fragile')
  assert.match(durability.detail, /would likely erase the modeled edge/i)
})

test('structure advantage explains winner versus runner-up', () => {
  const advantage = buildStructureAdvantage(
    { ...selectorBase, best_structure: 'atm_straddle' },
    scorecards,
  )
  assert.match(advantage.headline, /ATM Straddle leads Otm Strangle/i)
  assert.equal(advantage.bullets.length, 3)
  assert.match(advantage.bullets[0], /Score-derived return signal is/)
})

test('failure modes explain what can break the setup', () => {
  const failures = buildFailureModes(
    { ...selectorBase, best_structure: 'atm_straddle' },
    [{ ...scorecards[0], crowding_penalty: 0.06, execution_penalty: 0.05 }],
    { ...volSnapshot, iv_rv_yz: 1.36, near_term_spread_pct: 6.5, historical_vs_implied_move_ratio: 0.96 },
  )
  assert.match(failures.join(' '), /IV is already elevated/i)
  assert.match(failures.join(' '), /Spreads can widen/i)
  assert.match(failures.join(' '), /theta dominate/i)
})

test('regime-based decision quality message softens when evidence is thin', () => {
  const message = buildRegimeConfidenceMessage(
    { ...selectorBase, best_structure: 'atm_straddle' },
    [{ ...scorecards[0], sample_confidence: 0.44 }],
    { ...volSnapshot, data_quality_score: 0.58, vol_regime_label: 'High' },
  )
  assert.equal(message.tone, 'cautious')
  assert.match(message.detail, /limited or noisy/i)
})

test('sensitivity warnings flag thin samples and spread dependence', () => {
  const warnings = buildSensitivityWarnings(
    { ...selectorBase, best_structure: 'atm_straddle' },
    [{ ...scorecards[0], expected_edge_pct: 2.2, execution_penalty: 0.04 }],
    {
      ...volSnapshot,
      historical_event_count: 6,
      historical_move_uncertainty_pct: 2.4,
      tail_vs_implied_move_ratio: 1.6,
    },
  )
  assert.match(warnings.join(' '), /historical sample is still small/i)
  assert.match(warnings.join(' '), /tail events/i)
  assert.match(warnings.join(' '), /tight spreads/i)
})

test('modeled signal disclaimer is explicit about non-empirical edge numbers', () => {
  assert.match(getModeledSignalDisclaimer(), /not an empirical return forecast/i)
  assert.match(getModeledSignalDisclaimer(), /not calibrated to live retail execution/i)
})
