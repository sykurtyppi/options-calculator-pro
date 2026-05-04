import test from 'node:test'
import assert from 'node:assert/strict'
import {
  buildBenchmarkComparisonRows,
  buildCalibrationBucketRows,
  buildCalibrationReportRows,
  buildClaimablePerformanceRows,
  buildComparisonRows,
  buildForwardPerformanceSummary,
  buildForwardPerformanceWarnings,
  buildRecentResolvedRows,
  buildStructurePerformanceRows,
  formatReturnPct,
} from './forwardPerformanceViewModel.js'

const payload = {
  evidence_label: 'paper_research_not_execution_grade',
  total_recommendations: 4,
  no_trade_count: 1,
  resolved_outcome_count: 2,
  open_outcome_count: 1,
  performance_summary: {
    n: 2,
    wins: 1,
    losses: 1,
    win_rate: 0.5,
    avg_modeled_return_pct: 2,
    avg_realized_return_pct: 0.75,
    avg_model_error_pct: -1.25,
  },
  by_structure: {
    atm_straddle: {
      n: 2,
      wins: 1,
      losses: 1,
      win_rate: 0.5,
      avg_modeled_return_pct: 2,
      avg_realized_return_pct: 0.75,
      avg_realized_expansion_pct: 3.5,
    },
  },
  stale_source_comparison: {
    stale_source: { n: 1, win_rate: 0, avg_realized_return_pct: -1.5 },
    fresh_or_unknown_source: { n: 1, win_rate: 1, avg_realized_return_pct: 3.0 },
  },
  data_quality_comparison: {
    high_quality: { n: 1, win_rate: 1, avg_realized_return_pct: 3.0 },
    low_quality: { n: 1, win_rate: 0, avg_realized_return_pct: -1.5 },
  },
  calibration_buckets: [
    {
      bucket: '75-90',
      n: 1,
      win_rate: 1,
      avg_confidence_pct: 82,
      avg_setup_score: 0.82,
      avg_modeled_return_pct: 2,
      avg_realized_return_pct: 4,
      median_realized_return_pct: 4,
      avg_model_error_pct: 2,
      claimable_count: 1,
      small_sample_warning: true,
    },
  ],
  calibration_report: {
    selector_score_bucket: {
      '0.75-0.90': {
        n: 1,
        wins: 1,
        losses: 0,
        win_rate: 1,
        avg_modeled_return_pct: 2,
        avg_realized_return_pct: 4,
        median_realized_return_pct: 4,
        claimable_count: 1,
        small_sample_warning: true,
        paper_research_warning: 'Paper/research only; not execution-grade live performance.',
      },
    },
    evidence_quality_status: {
      degraded_evidence: {
        n: 2,
        wins: 1,
        losses: 1,
        win_rate: 0.5,
        avg_realized_return_pct: 0.75,
        median_realized_return_pct: 0.75,
        claimable_count: 1,
        small_sample_warning: true,
      },
    },
  },
  benchmark_comparison: {
    selector: {
      n: 2,
      wins: 1,
      losses: 1,
      win_rate: 0.5,
      avg_realized_return_pct: 0.75,
      median_realized_return_pct: 0.75,
      small_sample_warning: true,
      rule: 'Actual selector-selected paper outcomes.',
      paper_research_warning: 'Paper/research only; not execution-grade live performance.',
    },
    always_atm_straddle: {
      n: 1,
      wins: 1,
      losses: 0,
      win_rate: 1,
      avg_realized_return_pct: 1.5,
      median_realized_return_pct: 1.5,
      small_sample_warning: true,
      rule: 'Shadow baseline.',
    },
    no_trade: {
      n: 2,
      wins: 0,
      losses: 0,
      avg_realized_return_pct: 0,
      median_realized_return_pct: 0,
      paper_research_label: 'No-trade baseline assumes zero paper return and zero exposure.',
      rule: 'No trade; zero exposure baseline.',
    },
    simple_iv_rv_filter: {
      n: 1,
      skipped_by_filter: 1,
      avg_realized_return_pct: 4,
      median_realized_return_pct: 4,
      small_sample_warning: true,
      rule: 'Keep selected paper outcomes only when iv_rv_har <= 1.05; otherwise no-trade.',
    },
    paper_research_label: 'Benchmarks are observational paper/research comparisons and are not optimized.',
  },
  claimable_performance: {
    claimable: {
      n: 1,
      win_rate: 1,
      avg_realized_return_pct: 4,
      median_realized_return_pct: 4,
      small_sample_warning: true,
    },
    non_claimable: {
      n: 1,
      win_rate: 0,
      avg_realized_return_pct: -1.5,
      median_realized_return_pct: -1.5,
      small_sample_warning: true,
    },
  },
  recent_resolved_outcomes: [
    {
      symbol: 'AAPL',
      structure: 'atm_straddle',
      source_type: 'paper',
      modeled_return_pct: 2,
      realized_return_pct: 4,
      realized_expansion_pct: 8,
      data_quality_score: 0.88,
      earnings_source_stale: true,
      quote_quality: 'paper_research_mid_not_execution_grade',
    },
  ],
  warning_flags: ['Forward-performance results are paper/research diagnostics, not execution-grade live performance.'],
}

test('forward performance summary labels paper outcomes and modeled versus realized returns', () => {
  const summary = buildForwardPerformanceSummary(payload)

  assert.equal(summary.evidenceLabel, 'paper_research_not_execution_grade')
  assert.equal(summary.totalRecommendations, 4)
  assert.equal(summary.noTradeCount, 1)
  assert.equal(summary.winRateLabel, '50%')
  assert.equal(summary.avgModeledLabel, '+2.0%')
  assert.equal(summary.avgRealizedLabel, '+0.8%')
  assert.equal(summary.avgErrorLabel, '-1.3%')
})

test('structure rows show wins losses and expansion without execution-grade claims', () => {
  const rows = buildStructurePerformanceRows(payload.by_structure)

  assert.equal(rows[0].structure, 'atm_straddle')
  assert.equal(rows[0].wins, 1)
  assert.equal(rows[0].losses, 1)
  assert.equal(rows[0].avgExpansionLabel, '+3.5%')
})

test('calibration bucket rows expose sample count and realized error', () => {
  const rows = buildCalibrationBucketRows(payload.calibration_buckets)

  assert.equal(rows[0].bucket, '75-90')
  assert.equal(rows[0].n, 1)
  assert.equal(rows[0].avgConfidenceLabel, '82%')
  assert.equal(rows[0].avgErrorLabel, '+2.0%')
  assert.equal(rows[0].medianRealizedLabel, '+4.0%')
  assert.equal(rows[0].claimableCount, 1)
  assert.equal(rows[0].warningLabel, 'Small sample')
})

test('calibration report rows preserve paper/research and small-sample labels', () => {
  const rows = buildCalibrationReportRows(payload.calibration_report)
  const evidence = rows.find((row) => row.category === 'Evidence quality')

  assert.equal(rows[0].n, 2)
  assert.equal(evidence.bucket, 'degraded_evidence')
  assert.equal(evidence.warningLabel, 'Small sample')
  assert.ok(evidence.paperResearchLabel.includes('Paper/research') || evidence.paperResearchLabel.includes('paper/research'))
})

test('benchmark rows show transparent baselines without execution-grade language', () => {
  const rows = buildBenchmarkComparisonRows(payload)
  const selector = rows.find((row) => row.key === 'selector')
  const noTrade = rows.find((row) => row.key === 'no_trade')
  const ivRv = rows.find((row) => row.key === 'simple_iv_rv_filter')

  assert.equal(selector.label, 'Selector')
  assert.equal(selector.warningLabel, 'Small sample')
  assert.equal(noTrade.avgRealizedLabel, '0.0%')
  assert.equal(ivRv.skippedByFilter, 1)
  assert.ok(!rows.map((row) => row.rule).join(' ').includes('guaranteed'))
})

test('claimable rows separate record-only observations from claimable evidence', () => {
  const rows = buildClaimablePerformanceRows(payload)
  const claimable = rows.find((row) => row.key === 'claimable')
  const nonClaimable = rows.find((row) => row.key === 'non_claimable')

  assert.equal(claimable.n, 1)
  assert.equal(claimable.avgRealizedLabel, '+4.0%')
  assert.equal(nonClaimable.avgRealizedLabel, '-1.5%')
})

test('stale and quality comparisons remain visible with zero-count fallbacks', () => {
  const rows = buildComparisonRows(payload)
  const stale = rows.find((row) => row.group === 'Stale earnings source')
  const middle = rows.find((row) => row.group === 'Middle data quality')

  assert.equal(stale.n, 1)
  assert.equal(stale.avgRealizedLabel, '-1.5%')
  assert.equal(middle.n, 0)
  assert.equal(middle.winRateLabel, 'n/a')
})

test('recent resolved rows show provenance and stale status', () => {
  const rows = buildRecentResolvedRows(payload.recent_resolved_outcomes)

  assert.equal(rows[0].sourceLabel, 'paper')
  assert.equal(rows[0].qualityLabel, '0.88')
  assert.equal(rows[0].staleLabel, 'stale')
  assert.equal(rows[0].quoteQualityLabel, 'paper_research_mid_not_execution_grade')
})

test('warnings degrade honestly for empty and low-sample systems', () => {
  assert.ok(buildForwardPerformanceWarnings(payload).some((warning) => warning.includes('paper/research')))
  assert.ok(buildForwardPerformanceWarnings({ resolved_outcome_count: 0 }).some((warning) => warning.includes('No resolved')))
  assert.equal(formatReturnPct(null), 'n/a')
})
