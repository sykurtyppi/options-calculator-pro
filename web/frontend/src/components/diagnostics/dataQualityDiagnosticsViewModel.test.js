import test from 'node:test'
import assert from 'node:assert/strict'
import {
  buildDataQualityWarnings,
  buildHealthTone,
  buildProviderRows,
  buildQualityBucketRows,
  buildQualitySummary,
  buildWeakRecommendationRows,
} from './dataQualityDiagnosticsViewModel.js'

const payload = {
  total_recommendations: 5,
  stale_earnings_source_count: 2,
  stale_earnings_source_rate: 0.4,
  missing_option_chain_count: 1,
  low_data_quality_count: 2,
  low_data_quality_rate: 0.4,
  data_quality_buckets: {
    '0.00-0.25': 1,
    '0.25-0.50': 1,
    '0.50-0.75': 1,
    '0.75-1.00': 2,
    unknown: 0,
  },
  source_breakdown: {
    option_source: { marketdata_app: 4, unknown: 1 },
    earnings_source: { alpha_vantage: 3, fmp: 2 },
  },
  recent_weak_data_recommendations: [
    {
      recommendation_id: 'rec_weak',
      symbol: 'TSLA',
      recommendation: 'No Trade',
      data_quality_score: 0.24,
      earnings_source: 'alpha_vantage',
      earnings_source_stale: true,
      option_source: null,
      weak_data_reasons: ['low_data_quality_score', 'missing_option_chain'],
    },
  ],
  warning_flags: ['Stale earnings-source rate is elevated.'],
}

test('data-quality summary surfaces stale and low-quality rates', () => {
  const summary = buildQualitySummary(payload)

  assert.equal(summary.total, 5)
  assert.equal(summary.staleCount, 2)
  assert.equal(summary.staleRateLabel, '40%')
  assert.equal(summary.lowQualityCount, 2)
  assert.equal(summary.lowQualityRateLabel, '40%')
  assert.equal(summary.missingChainCount, 1)
})

test('provider rows expose unknown sources for weak provider observability', () => {
  const rows = buildProviderRows(payload.source_breakdown)
  const unknown = rows.find((row) => row.category === 'option_source' && row.source === 'unknown')

  assert.ok(unknown)
  assert.equal(unknown.count, 1)
  assert.equal(unknown.isUnknown, true)
})

test('quality bucket rows preserve all score bands', () => {
  const rows = buildQualityBucketRows(payload.data_quality_buckets)

  assert.equal(rows.length, 5)
  assert.equal(rows.find((row) => row.bucket === '0.00-0.25').count, 1)
  assert.equal(rows.find((row) => row.bucket === '0.75-1.00').count, 2)
})

test('weak recommendation rows show stale and missing-chain reasons', () => {
  const rows = buildWeakRecommendationRows(payload.recent_weak_data_recommendations)

  assert.equal(rows[0].symbol, 'TSLA')
  assert.equal(rows[0].qualityLabel, '0.24')
  assert.equal(rows[0].staleLabel, 'Stale')
  assert.match(rows[0].reasonsLabel, /missing_option_chain/)
})

test('warnings harden stale and low-quality data presentation', () => {
  const warnings = buildDataQualityWarnings(payload)

  assert.ok(warnings.some((warning) => warning.includes('Stale')))
  assert.ok(warnings.some((warning) => warning.includes('below the data-quality threshold')))
  assert.ok(warnings.some((warning) => warning.includes('option-chain')))
  assert.equal(buildHealthTone(payload), 'weak')
})

test('empty data-quality diagnostics show empty tone without false warning inflation', () => {
  assert.equal(buildHealthTone({ total_recommendations: 0, warning_flags: [] }), 'empty')
  assert.deepEqual(buildDataQualityWarnings({ warning_flags: [] }), [])
})
