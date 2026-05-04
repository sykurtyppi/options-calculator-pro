export function formatTelemetryPercent(value, digits = 0) {
  if (value === null || value === undefined || value === '') return 'n/a'
  const n = Number(value)
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : 'n/a'
}

export function formatLatency(value) {
  if (value === null || value === undefined || value === '') return 'n/a'
  const n = Number(value)
  return Number.isFinite(n) ? `${Math.round(n)} ms` : 'n/a'
}

export function buildProviderHealthRows(summary = {}) {
  return Object.entries(summary).map(([provider, item]) => ({
    provider,
    total: Number(item.total || 0),
    successes: Number(item.successes || 0),
    failures: Number(item.failures || 0),
    failureRate: Number(item.failure_rate || 0),
    failureRateLabel: formatTelemetryPercent(item.failure_rate || 0, 0),
    avgLatencyLabel: formatLatency(item.avg_latency_ms),
    fallbackCount: Number(item.fallback_count || 0),
    staleCount: Number(item.stale_count || 0),
    tone: (item.failure_rate || 0) >= 0.25 ? 'weak' : (item.failure_rate || 0) > 0 ? 'watch' : 'healthy',
  })).sort((a, b) => b.total - a.total || a.provider.localeCompare(b.provider))
}

export function buildTelemetrySummary(payload = {}) {
  const totals = payload.totals || {}
  const operational = payload.operational_health || {}
  return {
    events: Number(totals.events || 0),
    failures: Number(totals.failures || 0),
    failureRateLabel: formatTelemetryPercent(totals.failure_rate || 0, 0),
    fallbackCount: Number(totals.fallback_count || 0),
    staleCount: Number(totals.stale_count || 0),
    rowCount: Number(operational.row_count || 0),
    dbSizeLabel: formatBytes(operational.db_size_bytes),
    avgLatencyLabel: formatLatency(operational.avg_latency_ms),
  }
}

export function buildRecentFailureRows(rows = []) {
  return rows.map((row) => ({
    ...row,
    provider: row.provider_name || 'unknown',
    endpoint: row.endpoint_type || 'unknown',
    symbol: row.symbol || 'n/a',
    error: row.error_category || 'unknown_error',
    latencyLabel: formatLatency(row.latency_ms),
  }))
}

export function buildTelemetryWarnings(payload = {}) {
  const warnings = [...(payload.operational_health?.warning_flags || [])]
  const totals = payload.totals || {}
  if (!totals.events) {
    if (!warnings.length) warnings.push('No direct provider telemetry has been recorded yet.')
    return warnings
  }
  if ((totals.failure_rate || 0) >= 0.20 && !warnings.some((warning) => warning.includes('failure rate'))) {
    warnings.push('Provider failure rate is elevated.')
  }
  if ((totals.fallback_count || 0) > 0 && !warnings.some((warning) => warning.includes('Fallback'))) {
    warnings.push('Fallback provider paths were used recently.')
  }
  if ((totals.stale_count || 0) > 0 && !warnings.some((warning) => warning.includes('Stale'))) {
    warnings.push('Stale provider/cache data was used recently.')
  }
  return warnings
}

export function buildTelemetryEndpoint(apiBase, filters = {}) {
  const params = new URLSearchParams({
    limit: String(filters.limit || 100),
    offset: String(Math.max(0, Number(filters.offset) || 0)),
  })
  if (filters.provider) params.set('provider', String(filters.provider).trim())
  if (filters.endpointType) params.set('endpoint_type', String(filters.endpointType).trim())
  if (filters.symbol) params.set('symbol', String(filters.symbol).trim().toUpperCase())
  if (filters.success && filters.success !== 'all') params.set('success', filters.success)
  if (filters.failuresOnly) params.set('failures_only', 'true')
  if (filters.since) params.set('since', filters.since)
  if (filters.until) params.set('until', filters.until)
  return `${apiBase}/api/diagnostics/provider-telemetry?${params.toString()}`
}

export function formatBytes(value) {
  const n = Number(value)
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  if (n < 1024) return `${Math.round(n)} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}
