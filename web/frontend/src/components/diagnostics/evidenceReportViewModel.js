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
    gateBlockingReasons: Array.isArray(gate.blocking_reasons) ? gate.blocking_reasons : [],
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

// Universe shadow cohort: every eligible event, whatever the selector said.
// The key question is whether events the selector SKIPPED paid worse than the
// ones it picked; if not, its skips are not adding value.
export function buildUniverseShadowRows(payload = {}) {
  const universe = payload.universe_shadow || {}
  return Object.entries(universe.by_baseline || {}).map(([name, groups]) => {
    const all = groups.all_events || {}
    const picked = groups.selector_actionable || {}
    const skipped = groups.selector_not_actionable || {}
    const pickedReturn = finiteOrNull(picked.avg_realized_return_pct)
    const skippedReturn = finiteOrNull(skipped.avg_realized_return_pct)
    const edge = pickedReturn !== null && skippedReturn !== null ? pickedReturn - skippedReturn : null
    return {
      name,
      label: labelBaseline(name),
      allN: Number(all.n || 0),
      allReturnLabel: formatReturnPct(all.avg_realized_return_pct),
      pickedN: Number(picked.n || 0),
      pickedReturnLabel: formatReturnPct(picked.avg_realized_return_pct),
      skippedN: Number(skipped.n || 0),
      skippedReturnLabel: formatReturnPct(skipped.avg_realized_return_pct),
      // Picked minus skipped: positive means the selector's picks paid more.
      selectorEdgeLabel: formatReturnPct(edge),
    }
  }).sort((a, b) => a.label.localeCompare(b.label))
}

export function buildUniverseShadowSummary(payload = {}) {
  const universe = payload.universe_shadow || {}
  const attrition = universe.entry_attrition || {}
  return {
    eventsRecorded: Number(universe.events_recorded || 0),
    open: Number(universe.open || 0),
    resolved: Number(universe.resolved || 0),
    entered: Number(attrition.entered || 0),
    enteredAfterRetry: Number(attrition.entered_after_retry || 0),
    notEntered: Number(attrition.not_entered || 0),
    notEnteredReasons: countRows(attrition.not_entered_by_reason),
    rule: universe.rule || 'Universe shadow cohort not recorded yet.',
  }
}

// Positions entered but never valued at T-1. High or concentrated attrition
// means resolved results are selected by quote availability.
export function buildExitAttritionRows(payload = {}) {
  const attrition = payload.exit_attrition || {}
  return [
    ['selector', 'Selector trades'],
    ['baselines', 'Shadow baselines'],
  ].map(([key, label]) => {
    const block = attrition[key] || {}
    return {
      key,
      label,
      resolved: Number(block.resolved || 0),
      missing: Number(block.exit_missing || 0),
      rateLabel: formatRate(block.attrition_rate),
      reasons: countRows(block.by_reason),
      structures: countRows(block.by_structure),
    }
  })
}

// What is being kept OUT of every comparison, and why.
export function buildExcludedEvidenceSummary(payload = {}) {
  const invalidated = payload.invalidated_outcomes || {}
  const legacy = payload.legacy_repriced_baselines || {}
  const nonFinite = payload.non_finite_outcomes || {}
  const replay = payload.excluded_replay_outcomes || {}
  return {
    invalidatedN: Number(invalidated.n || 0),
    invalidatedResolvedN: Number(invalidated.resolved_n || 0),
    invalidatedReasons: countRows(invalidated.by_reason),
    invalidatedNote: invalidated.note || '',
    legacyBaselinesN: Number(legacy.n || 0),
    legacyReasons: countRows(legacy.by_exit_repricing).map((row) => ({ ...row, label: labelExitRepricing(row.label) })),
    legacyNote: legacy.note || '',
    nonFiniteSelectorN: Number(nonFinite.selector_n || 0),
    nonFiniteBaselineN: Number(nonFinite.baseline_n || 0),
    replayN: Number(replay.n || 0),
    replayResolvedN: Number(replay.resolved_n || 0),
  }
}

// Every headline comparison with its interval. "Too few to tell" is shown
// instead of an interval below the backend's minimum sample.
export function buildUncertaintyRows(payload = {}) {
  const block = payload.uncertainty || {}
  const rows = []
  if (block.selector) {
    rows.push(uncertaintyRow('selector', 'Selector vs no trade', block.selector, block.selector.n))
  }
  for (const [name, item] of Object.entries(block.selector_minus_baseline || {})) {
    rows.push(uncertaintyRow(`paired-${name}`, `Selector − ${labelBaseline(name)} (same events)`, item, item.pairs))
  }
  for (const [name, item] of Object.entries(block.universe_picked_minus_skipped || {})) {
    const n = `${Number(item.n_first || 0)} vs ${Number(item.n_second || 0)}`
    rows.push(uncertaintyRow(`universe-${name}`, `Picked − skipped (${labelBaseline(name)})`, item, n))
  }
  return rows
}

export function buildOutlierDependence(payload = {}) {
  const item = ((payload.uncertainty || {}).selector || {}).outlier_dependence || {}
  return {
    available: item.mean_without_top_k !== null && item.mean_without_top_k !== undefined,
    k: Number(item.k || 5),
    withoutTopLabel: formatReturnPct(item.mean_without_top_k),
    withoutBottomLabel: formatReturnPct(item.mean_without_bottom_k),
    topShareLabel: formatRate(item.top_k_share_of_profit),
    fragile: Boolean(item.fragile),
  }
}

function uncertaintyRow(key, label, item, n) {
  const hasInterval = finiteOrNull(item.ci_low) !== null && finiteOrNull(item.ci_high) !== null
  return {
    key,
    label,
    n: n === undefined || n === null ? '0' : String(n),
    meanLabel: formatReturnPct(item.mean),
    intervalLabel: hasInterval ? `${formatReturnPct(item.ci_low)} to ${formatReturnPct(item.ci_high)}` : '—',
    verdictLabel: labelVerdict(item.verdict),
    verdict: item.verdict || 'insufficient_sample',
  }
}

function labelVerdict(value) {
  if (value === 'above_zero') return 'Above zero'
  if (value === 'below_zero') return 'Below zero'
  if (value === 'includes_zero') return 'No detectable difference'
  return 'Too few to tell'
}

function countRows(counts = {}) {
  return Object.entries(counts || {})
    .map(([label, count]) => ({ label, count: Number(count || 0) }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
}

function finiteOrNull(value) {
  if (value === null || value === undefined || value === '') return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function labelExitRepricing(value) {
  if (value === 'rediscovered_legacy') return 'Exit re-discovered strikes (pre-fix)'
  if (value === 'booked_strikes') return 'Labelled booked without leg check (#143)'
  if (value === 'unverifiable_entry_context') return 'Entry recorded too little to verify'
  return String(value || 'unknown').replace(/_/g, ' ')
}

function labelBaseline(name) {
  if (name === 'always_atm_straddle') return 'Always ATM straddle'
  if (name === 'always_otm_strangle') return 'Always OTM strangle'
  if (name === 'always_iron_condor') return 'Always iron condor'
  if (name === 'no_trade') return 'No trade'
  return String(name || 'unknown').replace(/_/g, ' ')
}
