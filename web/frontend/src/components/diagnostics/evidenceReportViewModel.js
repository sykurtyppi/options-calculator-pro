import { formatRate, formatReturnPct } from './forwardPerformanceViewModel.js'

export function buildEvidenceReportSummary(payload = {}) {
  const gate = payload.commercialization_gate || {}
  const selector = payload.selector_summary || {}
  return {
    evidenceLabel: payload.evidence_label || 'paper_research_not_execution_grade',
    maturityLabel: (payload.maturity || {}).maturity_label || 'Insufficient evidence',
    edgeQualityLabel: (payload.maturity || {}).edge_quality_label || 'Withheld: insufficient claimable evidence',
    benchmarkMeaningful: Boolean((payload.maturity || {}).benchmark_comparison_meaningful),
    calibrationInterpretationAllowed: Boolean((payload.maturity || {}).calibration_interpretation_allowed),
    bucketInterpretationAllowed: Boolean((payload.maturity || {}).bucket_interpretation_allowed),
    activeDays: Number(gate.active_evidence_days || 0),
    minimumDays: Number(gate.minimum_days || 60),
    targetDays: Number(gate.target_days || 90),
    resolvedOutcomes: Number(gate.resolved_selector_outcomes || selector.n || 0),
    minimumResolved: Number(gate.minimum_resolved_sample || 30),
    readyForPaidBeta: Boolean(gate.ready_for_paid_beta),
    selectorReturnLabel: formatReturnPct(selector.avg_realized_return_pct),
    selectorWinRateLabel: formatRate(selector.win_rate),
  }
}

export function buildBaselineComparisonRows(payload = {}) {
  const selector = payload.selector_summary || {}
  const baselines = payload.baseline_comparison || {}
  const rows = [
    {
      name: 'selector',
      label: 'Selector',
      n: selector.n,
      win_rate: selector.win_rate,
      avg_realized_return_pct: selector.avg_realized_return_pct,
      avg_realized_expansion_pct: selector.avg_realized_expansion_pct,
    },
    ...Object.entries(baselines).map(([name, row]) => ({
      name,
      label: labelBaseline(name),
      ...row,
    })),
  ]
  return rows.map((row) => ({
    name: row.name,
    label: row.label,
    n: Number(row.n || 0),
    winRateLabel: formatRate(row.win_rate),
    returnLabel: formatReturnPct(row.avg_realized_return_pct),
    expansionLabel: formatReturnPct(row.avg_realized_expansion_pct),
  }))
}

export function buildEvidenceWarnings(payload = {}) {
  const maturityWarnings = (payload.maturity || {}).warning_flags || []
  return [...new Set([...(payload.warning_flags || []), ...maturityWarnings])]
}

export function buildQuoteQualityRows(payload = {}) {
  const quote = payload.quote_quality || {}
  return Object.entries(quote.entry_sources || {}).map(([source, count]) => ({
    source,
    count: Number(count || 0),
  })).sort((a, b) => b.count - a.count || a.source.localeCompare(b.source))
}

export function buildEvidenceQualitySummary(payload = {}) {
  const quality = payload.evidence_quality || {}
  return {
    claimAllowed: Number(quality.claim_allowed_count || 0),
    claimBlocked: Number(quality.claim_blocked_count || 0),
    executionGrade: Number(quality.execution_grade_count || 0),
    recordOnly: Number(quality.record_only_count || 0),
    degraded: Number(quality.degraded_count || 0),
    selectorStatusCounts: quality.selector_status_counts || {},
    baselineStatusCounts: quality.baseline_status_counts || {},
    label: quality.paper_research_label || 'Recorded rows may still be blocked from claims.',
  }
}

export function buildExecutionRealismSummary(payload = {}) {
  const realism = payload.execution_realism || {}
  const avgSpread = Number(realism.avg_entry_spread_as_pct_of_premium)
  return {
    selectorEntryRows: Number(realism.selector_entry_scenario_rows || 0),
    baselineEntryRows: Number(realism.baseline_entry_scenario_rows || 0),
    exitScenarioRows: Number(realism.exit_scenario_rows || 0),
    avgEntrySpreadLabel: Number.isFinite(avgSpread) ? `${avgSpread.toFixed(1)}%` : 'N/A',
    label: realism.paper_research_label || 'Execution scenarios are modeled, not broker fills.',
  }
}

export function buildSurfaceQualitySummary(payload = {}) {
  const surface = payload.surface_quality || {}
  return {
    selectorStatusCounts: surface.selector_status_counts || {},
    baselineStatusCounts: surface.baseline_status_counts || {},
    crossedQuotes: Number(surface.crossed_quote_count || 0),
    zeroBids: Number(surface.zero_bid_count || 0),
    extremeSpreads: Number(surface.extreme_spread_count || 0),
    sparseAtm: Number(surface.sparse_atm_count || 0),
    ivAnomalies: Number(surface.iv_anomaly_count || 0),
    label: surface.paper_research_label || 'Surface quality is diagnostic, not a selector score.',
  }
}

export function buildSimpleIvRvFilter(payload = {}) {
  const item = payload.simple_iv_rv_filter || {}
  return {
    n: Number(item.n || 0),
    skippedByFilter: Number(item.skipped_by_filter || 0),
    winRateLabel: formatRate(item.win_rate),
    returnLabel: formatReturnPct(item.avg_realized_return_pct),
    rule: item.rule || 'IV/RV baseline unavailable.',
  }
}

function labelBaseline(name) {
  if (name === 'always_atm_straddle') return 'Always ATM straddle'
  if (name === 'always_otm_strangle') return 'Always OTM strangle'
  if (name === 'no_trade') return 'No trade'
  return String(name || 'unknown').replace(/_/g, ' ')
}
