export function formatRate(value, digits = 0) {
  if (value === null || value === undefined || value === '') return 'n/a'
  const n = Number(value)
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : 'n/a'
}

export function formatReturnPct(value, digits = 1) {
  if (value === null || value === undefined || value === '') return 'n/a'
  const n = Number(value)
  if (!Number.isFinite(n)) return 'n/a'
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(digits)}%`
}

export function formatScore(value) {
  const n = Number(value)
  return Number.isFinite(n) ? n.toFixed(2) : 'n/a'
}

export function buildForwardPerformanceSummary(payload = {}) {
  const summary = payload.performance_summary || {}
  return {
    evidenceLabel: payload.evidence_label || 'paper_research_not_execution_grade',
    totalRecommendations: Number(payload.total_recommendations || 0),
    noTradeCount: Number(payload.no_trade_count || 0),
    resolvedCount: Number(payload.resolved_outcome_count || 0),
    openCount: Number(payload.open_outcome_count || 0),
    winRateLabel: formatRate(summary.win_rate),
    avgModeledLabel: formatReturnPct(summary.avg_modeled_return_pct),
    avgRealizedLabel: formatReturnPct(summary.avg_realized_return_pct),
    avgErrorLabel: formatReturnPct(summary.avg_model_error_pct),
    sampleLabel: `${Number(summary.n || 0)} resolved paper outcomes`,
  }
}

export function buildStructurePerformanceRows(byStructure = {}) {
  return Object.entries(byStructure).map(([structure, item]) => ({
    structure,
    n: Number(item.n || 0),
    wins: Number(item.wins || 0),
    losses: Number(item.losses || 0),
    winRateLabel: formatRate(item.win_rate),
    avgModeledLabel: formatReturnPct(item.avg_modeled_return_pct),
    avgRealizedLabel: formatReturnPct(item.avg_realized_return_pct),
    avgExpansionLabel: formatReturnPct(item.avg_realized_expansion_pct),
  })).sort((a, b) => b.n - a.n || a.structure.localeCompare(b.structure))
}

export function buildCalibrationBucketRows(buckets = []) {
  return buckets.map((bucket) => ({
    bucket: bucket.bucket,
    n: Number(bucket.n || 0),
    winRateLabel: formatRate(bucket.win_rate),
    avgConfidenceLabel: bucket.avg_confidence_pct == null ? 'n/a' : `${Number(bucket.avg_confidence_pct).toFixed(0)}%`,
    avgSetupScoreLabel: formatScore(bucket.avg_setup_score),
    avgModeledLabel: formatReturnPct(bucket.avg_modeled_return_pct),
    avgRealizedLabel: formatReturnPct(bucket.avg_realized_return_pct),
    medianRealizedLabel: formatReturnPct(bucket.median_realized_return_pct),
    avgErrorLabel: formatReturnPct(bucket.avg_model_error_pct),
    claimableCount: Number(bucket.claimable_count || 0),
    warningLabel: bucket.small_sample_warning ? 'Small sample' : 'Observed',
  }))
}

export function buildCalibrationReportRows(report = {}, maxRows = 12) {
  const categoryLabels = {
    selector_score_bucket: 'Selector score',
    selected_structure: 'Structure',
    evidence_quality_status: 'Evidence quality',
    surface_quality_status: 'Surface quality',
    data_quality_score_bucket: 'Data quality',
    spread_cost_bucket: 'Spread cost',
    execution_scenario: 'Execution scenario',
    earnings_source_confidence_bucket: 'Earnings confidence',
  }
  const rows = []
  for (const [key, label] of Object.entries(categoryLabels)) {
    const groups = report[key] || {}
    for (const [bucket, item] of Object.entries(groups)) {
      rows.push({
        category: label,
        bucket,
        n: Number(item.n || 0),
        wins: Number(item.wins || 0),
        losses: Number(item.losses || 0),
        winRateLabel: formatRate(item.win_rate),
        avgModeledLabel: formatReturnPct(item.avg_modeled_return_pct),
        avgRealizedLabel: formatReturnPct(item.avg_realized_return_pct),
        medianRealizedLabel: formatReturnPct(item.median_realized_return_pct),
        claimableCount: Number(item.claimable_count || 0),
        warningLabel: item.small_sample_warning ? 'Small sample' : 'Observed',
        paperResearchLabel: item.paper_research_warning || item.paper_research_label || 'Paper/research only',
      })
    }
  }
  return rows.sort((a, b) => b.n - a.n || a.category.localeCompare(b.category) || a.bucket.localeCompare(b.bucket)).slice(0, maxRows)
}

export function buildComparisonRows(payload = {}) {
  const stale = payload.stale_source_comparison || {}
  const quality = payload.data_quality_comparison || {}
  return [
    { group: 'Fresh/unknown earnings source', ...(stale.fresh_or_unknown_source || {}) },
    { group: 'Stale earnings source', ...(stale.stale_source || {}) },
    { group: 'High data quality', ...(quality.high_quality || {}) },
    { group: 'Middle data quality', ...(quality.middle_quality || {}) },
    { group: 'Low data quality', ...(quality.low_quality || {}) },
  ].map((row) => ({
    group: row.group,
    n: Number(row.n || 0),
    winRateLabel: formatRate(row.win_rate),
    avgModeledLabel: formatReturnPct(row.avg_modeled_return_pct),
    avgRealizedLabel: formatReturnPct(row.avg_realized_return_pct),
  }))
}

export function buildBenchmarkComparisonRows(payload = {}) {
  const benchmarks = payload.benchmark_comparison || {}
  const labels = {
    selector: 'Selector',
    always_atm_straddle: 'Always ATM straddle',
    always_otm_strangle: 'Always OTM strangle',
    no_trade: 'No trade',
    simple_iv_rv_filter: 'Simple IV/RV filter',
    liquidity_only_filter: 'Liquidity-only filter',
    clean_surface_only_filter: 'Clean-surface filter',
  }
  return Object.entries(labels).map(([key, label]) => {
    const item = benchmarks[key] || {}
    return {
      key,
      label,
      n: Number(item.n || 0),
      wins: Number(item.wins || 0),
      losses: Number(item.losses || 0),
      winRateLabel: formatRate(item.win_rate),
      avgRealizedLabel: formatReturnPct(item.avg_realized_return_pct),
      medianRealizedLabel: formatReturnPct(item.median_realized_return_pct),
      skippedByFilter: Number(item.skipped_by_filter || 0),
      warningLabel: item.small_sample_warning ? 'Small sample' : 'Observed',
      rule: item.rule || 'Measured only; not optimized.',
      paperResearchLabel: item.paper_research_warning || item.paper_research_label || benchmarks.paper_research_label || 'Paper/research only',
    }
  })
}

export function buildClaimablePerformanceRows(payload = {}) {
  const groups = payload.claimable_performance || {}
  const labels = {
    claimable: 'Claimable evidence',
    non_claimable: 'Non-claimable evidence',
    legacy_unknown_claimability: 'Legacy unknown',
  }
  return Object.entries(labels).map(([key, label]) => {
    const item = groups[key] || {}
    return {
      key,
      label,
      n: Number(item.n || 0),
      winRateLabel: formatRate(item.win_rate),
      avgRealizedLabel: formatReturnPct(item.avg_realized_return_pct),
      medianRealizedLabel: formatReturnPct(item.median_realized_return_pct),
      warningLabel: item.small_sample_warning ? 'Small sample' : 'Observed',
      paperResearchLabel: item.paper_research_warning || item.paper_research_label || 'Paper/research only',
    }
  })
}

export function buildRecentResolvedRows(rows = []) {
  return rows.map((row) => ({
    ...row,
    sourceLabel: row.source_type || 'paper',
    qualityLabel: formatScore(row.data_quality_score),
    modeledLabel: formatReturnPct(row.modeled_return_pct),
    realizedLabel: formatReturnPct(row.realized_return_pct),
    expansionLabel: formatReturnPct(row.realized_expansion_pct),
    staleLabel: row.earnings_source_stale ? 'stale' : 'fresh/unknown',
    quoteQualityLabel: row.quote_quality || 'paper/research',
  }))
}

export function buildForwardPerformanceWarnings(payload = {}) {
  const warnings = [...(payload.warning_flags || [])]
  if (!payload.resolved_outcome_count && !warnings.some((item) => item.includes('No resolved'))) {
    warnings.push('No resolved paper outcomes are available yet.')
  }
  if ((payload.resolved_outcome_count || 0) > 0 && (payload.resolved_outcome_count || 0) < 30 && !warnings.some((item) => item.includes('sample'))) {
    warnings.push('Resolved sample is small; treat performance reads as directional only.')
  }
  return warnings
}
