import test from 'node:test'
import assert from 'node:assert/strict'
import {
  buildProviderHealthRows,
  buildRecentFailureRows,
  buildTelemetryEndpoint,
  buildTelemetrySummary,
  buildTelemetryWarnings,
  formatBytes,
  formatLatency,
} from './providerTelemetryViewModel.js'

const payload = {
  totals: {
    events: 4,
    failures: 1,
    failure_rate: 0.25,
    fallback_count: 2,
    stale_count: 1,
  },
  operational_health: {
    warning_flags: ['Average provider latency is elevated.'],
    avg_latency_ms: 2123,
    db_size_bytes: 4096,
    row_count: 125,
    retention: { max_age_days: 30, max_rows: 50000 },
  },
  summary_by_provider: {
    marketdata_app: {
      total: 3,
      successes: 2,
      failures: 1,
      failure_rate: 0.333333,
      avg_latency_ms: 182.7,
      fallback_count: 1,
      stale_count: 0,
    },
    yfinance: {
      total: 1,
      successes: 1,
      failures: 0,
      failure_rate: 0,
      avg_latency_ms: 52.1,
      fallback_count: 1,
      stale_count: 1,
    },
  },
  recent_failures: [
    {
      provider_name: 'marketdata_app',
      endpoint_type: 'earnings',
      symbol: 'AAPL',
      error_category: 'rate_limited',
      latency_ms: 240,
      fallback_used: true,
      stale_used: false,
    },
  ],
}

test('provider telemetry summary exposes failure, fallback, and stale counts', () => {
  const summary = buildTelemetrySummary(payload)

  assert.equal(summary.events, 4)
  assert.equal(summary.failures, 1)
  assert.equal(summary.failureRateLabel, '25%')
  assert.equal(summary.fallbackCount, 2)
  assert.equal(summary.staleCount, 1)
  assert.equal(summary.rowCount, 125)
  assert.equal(summary.dbSizeLabel, '4.0 KB')
  assert.equal(summary.avgLatencyLabel, '2123 ms')
})

test('provider health rows include latency and risk tone', () => {
  const rows = buildProviderHealthRows(payload.summary_by_provider)

  assert.equal(rows[0].provider, 'marketdata_app')
  assert.equal(rows[0].failureRateLabel, '33%')
  assert.equal(rows[0].avgLatencyLabel, '183 ms')
  assert.equal(rows[0].tone, 'weak')
  assert.equal(rows[1].tone, 'healthy')
})

test('recent failure rows are sanitized display records', () => {
  const rows = buildRecentFailureRows(payload.recent_failures)

  assert.equal(rows[0].provider, 'marketdata_app')
  assert.equal(rows[0].endpoint, 'earnings')
  assert.equal(rows[0].symbol, 'AAPL')
  assert.equal(rows[0].error, 'rate_limited')
  assert.equal(rows[0].latencyLabel, '240 ms')
})

test('telemetry warnings flag failure, fallback, and stale provider paths', () => {
  const warnings = buildTelemetryWarnings(payload)

  assert.ok(warnings.some((warning) => warning.includes('latency')))
  assert.ok(warnings.some((warning) => warning.includes('failure rate')))
  assert.ok(warnings.some((warning) => warning.includes('Fallback')))
  assert.ok(warnings.some((warning) => warning.includes('Stale')))
})

test('provider telemetry endpoint encodes filters and pagination', () => {
  const endpoint = buildTelemetryEndpoint('http://127.0.0.1:8000', {
    provider: 'marketdata_app',
    endpointType: 'options_chain',
    symbol: 'aapl',
    success: 'false',
    failuresOnly: true,
    limit: 25,
    offset: 50,
  })

  assert.equal(
    endpoint,
    'http://127.0.0.1:8000/api/diagnostics/provider-telemetry?limit=25&offset=50&provider=marketdata_app&endpoint_type=options_chain&symbol=AAPL&success=false&failures_only=true',
  )
  assert.equal(formatBytes(0), '0 B')
  assert.equal(formatBytes(4096), '4.0 KB')
})

test('empty provider telemetry is explicit', () => {
  assert.deepEqual(buildTelemetryWarnings({ totals: { events: 0 } }), ['No direct provider telemetry has been recorded yet.'])
  assert.equal(formatLatency(null), 'n/a')
})
