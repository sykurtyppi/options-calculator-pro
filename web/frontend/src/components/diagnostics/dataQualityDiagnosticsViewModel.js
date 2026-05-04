export function formatPercent(value, digits = 0) {
  const n = Number(value)
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : 'n/a'
}

export function formatScore(value) {
  const n = Number(value)
  return Number.isFinite(n) ? n.toFixed(2) : 'n/a'
}

export function buildQualitySummary(payload = {}) {
  const total = Number(payload.total_recommendations || 0)
  const staleCount = Number(payload.stale_earnings_source_count || 0)
  const lowQualityCount = Number(payload.low_data_quality_count || 0)
  const missingChainCount = Number(payload.missing_option_chain_count || 0)
  return {
    total,
    staleCount,
    staleRateLabel: formatPercent(payload.stale_earnings_source_rate || 0, 0),
    lowQualityCount,
    lowQualityRateLabel: formatPercent(payload.low_data_quality_rate || 0, 0),
    missingChainCount,
  }
}

export function buildProviderRows(sourceBreakdown = {}) {
  return Object.entries(sourceBreakdown).flatMap(([category, counts]) => (
    Object.entries(counts || {}).map(([source, count]) => ({
      category,
      source,
      count: Number(count || 0),
      isUnknown: source === 'unknown',
    }))
  )).sort((a, b) => b.count - a.count || a.category.localeCompare(b.category))
}

export function buildQualityBucketRows(buckets = {}) {
  return Object.entries(buckets).map(([bucket, count]) => ({
    bucket,
    count: Number(count || 0),
  }))
}

export function buildWeakRecommendationRows(rows = []) {
  return rows.map((row) => ({
    ...row,
    qualityLabel: formatScore(row.data_quality_score),
    staleLabel: row.earnings_source_stale ? 'Stale' : 'Fresh/unknown',
    reasonsLabel: (row.weak_data_reasons || []).join(', ') || 'weak data condition',
  }))
}

export function buildDataQualityWarnings(payload = {}) {
  const warnings = [...(payload.warning_flags || [])]
  if ((payload.low_data_quality_count || 0) > 0 && !warnings.some((item) => item.includes('Low data-quality'))) {
    warnings.push('Some recommendations are below the data-quality threshold.')
  }
  if ((payload.stale_earnings_source_count || 0) > 0 && !warnings.some((item) => item.includes('Stale'))) {
    warnings.push('Some recommendations used stale earnings-source evidence.')
  }
  if ((payload.missing_option_chain_count || 0) > 0 && !warnings.some((item) => item.includes('option-chain'))) {
    warnings.push('Some recommendations were missing usable option-chain evidence.')
  }
  return warnings
}

export function buildHealthTone(payload = {}) {
  const warnings = buildDataQualityWarnings(payload)
  if (!payload.total_recommendations) return 'empty'
  if (warnings.length >= 2 || (payload.low_data_quality_rate || 0) > 0.2) return 'weak'
  if (warnings.length === 1) return 'watch'
  return 'healthy'
}
