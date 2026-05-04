export function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  return Number(value).toFixed(digits)
}

function clamp01(value) {
  if (value == null || Number.isNaN(Number(value))) return 0
  return Math.max(0, Math.min(1, Number(value)))
}

export function formatPercent(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  return `${Number(value).toFixed(digits)}%`
}

export function formatSignedPercent(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  const n = Number(value)
  return `${n > 0 ? '+' : ''}${n.toFixed(digits)}%`
}

export function getEdgeTier(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return { label: 'Unclear', tone: 'unclear' }
  }
  const n = Number(value)
  if (n <= 0) return { label: 'Negative / Unclear', tone: 'negative' }
  if (n < 1.5) return { label: 'Marginal', tone: 'marginal' }
  return { label: 'Positive', tone: 'positive' }
}

export function getReturnSignalLabel(value) {
  if (value == null || Number.isNaN(Number(value))) return 'Unclear'
  const n = Number(value)
  if (n <= 0) return 'Weak'
  if (n < 3) return 'Mixed'
  return 'Supportive'
}

export function getModeledSignalDisclaimer() {
  return 'Score-derived signal quality only. Not an empirical return forecast and not calibrated to live retail execution.'
}

export function buildTrustBadges(selectorOutput = null, scorecards = [], snapshot = null) {
  if (!selectorOutput && !snapshot) return []
  const best = getBestScorecard(scorecards, selectorOutput)
  const badges = [
    {
      label: 'Paper / Research only',
      tone: 'neutral',
      title: 'Decision support only; not live execution performance.',
    },
    {
      label: 'Score-derived (not a forecast)',
      tone: 'neutral',
      title: 'Selector metrics rank setup quality; they are not calibrated return forecasts.',
    },
  ]

  if (snapshot?.earnings_source_stale || Number(snapshot?.price_staleness_minutes || 0) > 60 || Number(snapshot?.chain_staleness_minutes || 0) > 60) {
    badges.push({
      label: 'Stale data',
      tone: 'warning',
      title: 'One or more source fields are stale or fallback-based.',
    })
  }
  const dataQuality = Number(snapshot?.data_quality_score ?? selectorOutput?.data_quality_score)
  if (Number.isFinite(dataQuality) && dataQuality < 0.60) {
    badges.push({
      label: 'Low data quality',
      tone: 'warning',
      title: 'Data quality is below the research threshold for high-trust decisions.',
    })
  }
  if (best && Number(best.walk_forward_history_count || 0) < 8) {
    badges.push({
      label: 'Low sample size',
      tone: 'warning',
      title: 'Comparable forward evidence is thin.',
    })
  }
  const optionSource = String(snapshot?.option_source || '').toLowerCase()
  const underlyingSource = String(snapshot?.underlying_source || '').toLowerCase()
  if (optionSource.includes('fallback') || underlyingSource.includes('fallback') || optionSource === 'unknown') {
    badges.push({
      label: 'Provider degraded',
      tone: 'warning',
      title: 'One or more market-data providers used fallback or degraded coverage.',
    })
  }
  return badges
}

export function buildTrustDetails(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const notes = []
  if (selectorOutput?.best_structure) {
    notes.push(`Selected ${formatStructureLabel(selectorOutput.best_structure)} because it ranked best on the shared scorecard.`)
  } else {
    notes.push('No structure selected; the engine is abstaining.')
  }
  ;(selectorOutput?.why_this_structure || []).slice(0, 3).forEach((item) => notes.push(item))
  ;(selectorOutput?.primary_risks || []).slice(0, 3).forEach((item) => notes.push(`Risk: ${item}`))
  if (best) {
    notes.push(`${best.walk_forward_history_count ?? 0} comparable paper/research observations inform this structure.`)
  }
  if (snapshot?.earnings_source_primary || snapshot?.option_source || snapshot?.underlying_source) {
    notes.push(`Sources: earnings ${snapshot?.earnings_source_primary || 'unknown'}, options ${snapshot?.option_source || 'unknown'}, underlying ${snapshot?.underlying_source || 'unknown'}.`)
  }
  if (snapshot?.earnings_source_stale) {
    notes.push('Earnings source used stale/fallback evidence.')
  }
  const quality = snapshot?.data_quality_score ?? selectorOutput?.data_quality_score
  if (quality != null) {
    notes.push(`Data quality score: ${formatPercent(Number(quality) * 100, 0)}.`)
  }
  notes.push('Outputs are diagnostic rankings, not financial advice or return forecasts.')
  return notes.filter(Boolean).slice(0, 10)
}

export function formatDateLabel(value) {
  if (!value) return 'n/a'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return String(value)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

export function formatStructureLabel(value) {
  if (!value) return 'No structure selected'
  return String(value)
    .split('_')
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(' ')
}

export function getSelectorPresentation(selectorOutput) {
  const recommendation = selectorOutput?.recommendation || 'No Trade'
  const map = {
    'Best Candidate': {
      tone: 'best',
      helper: 'Strong historical support, clear separation from alternatives, and execution that still leaves room for edge.',
      eyebrow: 'Best structure',
    },
    Candidate: {
      tone: 'candidate',
      helper: 'Positive edge with usable evidence, though the separation versus alternatives is not fully dominant.',
      eyebrow: 'Recommended structure',
    },
    Watch: {
      tone: 'watch',
      helper: 'Evidence is mixed or still developing. Worth monitoring, but not a clean enough setup yet.',
      eyebrow: 'Conditional view',
    },
    'No Trade': {
      tone: 'no-trade',
      helper: 'The engine is abstaining because the edge does not survive costs or the evidence quality is not strong enough.',
      eyebrow: 'Abstain',
    },
  }
  return map[recommendation] || map['No Trade']
}

export function getBestScorecard(scorecards = [], selectorOutput = null) {
  const requested = selectorOutput?.best_structure
  if (requested) {
    const direct = (scorecards || []).find((card) => card.structure === requested)
    if (direct) return direct
  }
  const eligible = (scorecards || []).filter((card) => card.eligible)
  if (eligible.length) {
    return [...eligible].sort((a, b) => {
      const compositeDiff = Number(b.composite_structure_score || 0) - Number(a.composite_structure_score || 0)
      if (compositeDiff !== 0) return compositeDiff
      return Number(b.expected_edge_pct || 0) - Number(a.expected_edge_pct || 0)
    })[0]
  }
  return scorecards?.[0] || null
}

export function getRunnerUpScorecard(scorecards = [], selectorOutput = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const eligible = (scorecards || []).filter((card) => card.eligible && card.structure !== best?.structure)
  if (!eligible.length) return null
  return [...eligible].sort((a, b) => {
    const compositeDiff = Number(b.composite_structure_score || 0) - Number(a.composite_structure_score || 0)
    if (compositeDiff !== 0) return compositeDiff
    return Number(b.expected_edge_pct || 0) - Number(a.expected_edge_pct || 0)
  })[0]
}

export function getScoreGap(scorecards = [], selectorOutput = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const runnerUp = getRunnerUpScorecard(scorecards, selectorOutput)
  if (!best) return 0
  if (!runnerUp) return clamp01(best.composite_structure_score)
  return Math.max(Number(best.composite_structure_score || 0) - Number(runnerUp.composite_structure_score || 0), 0)
}

export function buildStructureRows(scorecards = [], selectorOutput = null) {
  const bestStructure = selectorOutput?.best_structure || null
  const runnerUps = new Set(selectorOutput?.runner_up_structures || [])
  return (scorecards || []).map((card) => ({
    key: card.structure,
    structure: formatStructureLabel(card.structure),
    eligibleLabel: card.eligible ? 'Eligible' : 'Ineligible',
    isEligible: Boolean(card.eligible),
    isBest: card.structure === bestStructure,
    isRunnerUp: runnerUps.has(card.structure),
    compositeScore: formatNumber(card.composite_structure_score, 2),
    expectedEdge: getEdgeTier(card.expected_edge_pct).label,
    expectedReturn: getReturnSignalLabel(card.expected_return_pct),
    executionPenalty: formatNumber(card.execution_penalty, 2),
    walkForwardCount: card.walk_forward_history_count ?? 0,
    walkForwardWinRate: formatPercent((card.walk_forward_win_rate ?? 0) * 100, 0),
    sampleConfidence: formatPercent((card.sample_confidence ?? 0) * 100, 0),
    rationaleSummary: card.eligible
      ? (card.rationale_bullets?.[0] || 'No rationale provided.')
      : `Ineligible: ${(card.eligibility_flags || []).join(', ') || 'No eligibility reason provided.'}`,
    eligibilityFlags: card.eligibility_flags || [],
  }))
}

export function buildVolSnapshotSummary(snapshot = null) {
  if (!snapshot) return []
  return [
    { label: 'Earnings Date', value: formatDateLabel(snapshot.earnings_date) },
    { label: 'Release Timing', value: snapshot.release_timing || 'n/a' },
    {
      label: 'Earnings Source',
      value: [
        snapshot.earnings_source_primary || 'n/a',
        snapshot.earnings_source_stale ? 'stale fallback' : null,
      ].filter(Boolean).join(' · '),
    },
    {
      label: 'Source Confidence',
      value: snapshot.earnings_source_confidence != null
        ? formatPercent(Number(snapshot.earnings_source_confidence) * 100, 0)
        : 'n/a',
    },
    { label: 'Days To Earnings', value: snapshot.days_to_earnings ?? 'n/a' },
    { label: 'IV30', value: formatPercent((snapshot.iv30 ?? null) != null ? Number(snapshot.iv30) * 100 : null, 1) },
    { label: 'RV30 (YZ)', value: formatPercent((snapshot.rv30_yang_zhang ?? null) != null ? Number(snapshot.rv30_yang_zhang) * 100 : null, 1) },
    { label: 'IV / RV', value: formatNumber(snapshot.iv_rv_yz, 2) },
    { label: 'Event Implied Move', value: formatPercent(snapshot.event_implied_move_pct, 2) },
    { label: 'Median Hist Move', value: formatPercent(snapshot.historical_median_move_pct, 2) },
    { label: 'Hist / Implied', value: formatNumber(snapshot.historical_vs_implied_move_ratio, 2) },
    { label: 'Near / Back IV', value: formatNumber(snapshot.near_back_iv_ratio, 2) },
    { label: 'Term Slope', value: formatNumber(snapshot.term_structure_slope, 4) },
    { label: 'Liquidity Tier', value: snapshot.liquidity_tier || 'n/a' },
    { label: 'Data Quality', value: `${snapshot.data_quality || 'n/a'} · ${formatPercent((snapshot.data_quality_score ?? 0) * 100, 0)}` },
    {
      label: 'Staleness',
      value: [
        snapshot.price_staleness_minutes != null ? `Price ${snapshot.price_staleness_minutes}m` : null,
        snapshot.chain_staleness_minutes != null ? `Chain ${snapshot.chain_staleness_minutes}m` : null,
      ].filter(Boolean).join(' · ') || 'n/a',
    },
  ]
}

export function buildEvidenceSummary(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  return [
    {
      label: 'Data Quality',
      value: `${snapshot?.data_quality || selectorOutput?.data_quality || 'n/a'} · ${formatPercent(((snapshot?.data_quality_score ?? selectorOutput?.data_quality_score) ?? 0) * 100, 0)}`,
      detail: snapshot?.earnings_source_stale
        ? 'Snapshot reliability includes a stale earnings-source fallback; treat the event date as lower-confidence.'
        : 'Snapshot reliability and staleness context.',
    },
    {
      label: 'Comparable Setups',
      value: best ? `${best.walk_forward_history_count ?? 0} events` : 'n/a',
      detail: best ? `Positive outcome rate ${formatPercent((best.walk_forward_win_rate ?? 0) * 100, 0)} · rank ${formatNumber(best.walk_forward_rank_score, 2)}` : 'No structure evidence available.',
    },
    {
      label: 'Evidence Depth',
      value: best ? formatPercent((best.sample_confidence ?? 0) * 100, 0) : 'n/a',
      detail: 'Measures how much historical support sits behind the current structure score.',
    },
    {
      label: 'Decision Quality',
      value: selectorOutput ? formatPercent(selectorOutput.confidence_pct, 0) : 'n/a',
      detail: 'Blend of score strength, separation, evidence depth, and data quality; not probability of profit.',
    },
  ]
}

export function buildHistoricalEvidenceLine(selectorOutput = null, scorecards = []) {
  const best = getBestScorecard(scorecards, selectorOutput)
  if (!best) {
    return {
      tone: 'limited',
      headline: 'Historical structure evidence is unavailable for this event.',
      subline: 'The engine can still summarize the snapshot, but there is no comparable walk-forward evidence to anchor the recommendation.',
      comparableCount: 0,
      positiveRateLabel: 'n/a',
      averageReturnLabel: 'n/a',
    }
  }

  const count = Number(best.walk_forward_history_count || 0)
  const positiveRateLabel = formatPercent((best.walk_forward_win_rate ?? 0) * 100, 0)
  const averageReturnLabel = getReturnSignalLabel(best.walk_forward_avg_return_pct)
  let tone = 'strong'
  let subline = 'Comparable outcomes are broad enough to interpret as historical support, not a one-off observation.'
  if (count < 8) {
    tone = 'limited'
    subline = 'Comparable history is thin, so treat the evidence as directional rather than deeply established.'
  } else if (count < 20) {
    tone = 'moderate'
    subline = 'Comparable history exists, but the sample is still moderate rather than deep.'
  }
  return {
    tone,
    headline: `Historically: ${count} comparable setups · ${positiveRateLabel} positive outcomes · paper outcome quality ${averageReturnLabel}`,
    subline,
    comparableCount: count,
    positiveRateLabel,
    averageReturnLabel,
  }
}

export function buildDecisionQualityBreakdown(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const gap = getScoreGap(scorecards, selectorOutput)
  const dataQualityScore = clamp01(snapshot?.data_quality_score ?? selectorOutput?.data_quality_score)
  const breakdown = [
    {
      label: 'Score Strength',
      value: clamp01(best?.composite_structure_score),
      detail: 'How strong the leading structure scores on the selector framework.',
    },
    {
      label: 'Separation',
      value: clamp01(gap / 0.20),
      detail: 'Distance between the top structure and the next-best eligible alternative.',
    },
    {
      label: 'Data Quality',
      value: dataQualityScore,
      detail: 'How reliable the current price, chain, and staleness inputs are.',
    },
    {
      label: 'Evidence Depth',
      value: clamp01(best?.sample_confidence),
      detail: best
        ? `${best.walk_forward_history_count ?? 0} comparable setups inform this structure.`
        : 'Comparable setup history is unavailable.',
    },
  ]

  return breakdown.map((item) => ({
    ...item,
    valuePct: Math.round(item.value * 100),
    valueLabel: formatPercent(item.value * 100, 0),
  }))
}

export function buildDecisionStability(selectorOutput = null, scorecards = []) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const gap = getScoreGap(scorecards, selectorOutput)
  const penaltyLoad = Math.max(
    Number(best?.execution_penalty || 0),
    Number(best?.concavity_penalty || 0),
    Number(best?.crowding_penalty || 0),
    Number(best?.theta_drag_penalty || 0),
  )

  let tone = 'stable'
  let label = 'Stable'
  let detail = 'The top structure is clearly separated from the runner-up.'
  if (gap < 0.06) {
    tone = 'unstable'
    label = 'Unstable'
    detail = 'Small changes in the inputs could reshuffle the leading structure.'
  } else if (gap < 0.12 || penaltyLoad >= 0.07) {
    tone = 'moderate'
    label = 'Moderate'
    detail = 'The leader still stands out, but alternatives are close or the penalty load is meaningful.'
  }

  if (!best) {
    tone = 'unstable'
    label = 'Unstable'
    detail = 'No eligible structure remains after the minimum eligibility checks.'
  } else if (selectorOutput?.recommendation === 'No Trade') {
    detail = `${detail} The engine is still abstaining because the trade bar is higher than the ranking bar.`
  }

  return {
    tone,
    label,
    detail,
    gap,
    gapLabel: formatNumber(gap, 2),
  }
}

export function buildHistoricalAnalogSummary(selectorOutput = null, scorecards = []) {
  const best = getBestScorecard(scorecards, selectorOutput)
  if (!best) {
    return {
      label: 'Comparable Setups',
      value: 'Unavailable',
      detail: 'No walk-forward analog set is attached to the current structure view.',
    }
  }
  const count = Number(best.walk_forward_history_count || 0)
  const depthLabel = count >= 20 ? 'broad' : count >= 8 ? 'moderate' : 'thin'
  return {
    label: 'Comparable Setups',
    value: `${count} ${count === 1 ? 'setup' : 'setups'}`,
    detail: `Historical analog set is ${depthLabel}; based on the winning structure's walk-forward sample.`,
  }
}

export function buildNoTradeSummary(selectorOutput = null, scorecards = [], snapshot = null) {
  if (selectorOutput?.recommendation !== 'No Trade') return null

  const best = getBestScorecard(scorecards, selectorOutput)
  const reasons = []
  let action = 'A better decision needs cleaner execution, stronger evidence, or cheaper pricing.'

  if (!best || !best.eligible) {
    reasons.push('No supported structure clears the minimum eligibility requirements.')
    action = 'A trade becomes possible only after live chain data supports at least one valid structure.'
  } else {
    if ((best.expected_edge_pct ?? 0) <= 0) {
      reasons.push('Expected edge is not positive after costs and penalties.')
      action = 'The setup would need cheaper front-end pricing or stronger realized move support.'
    }
    if ((best.execution_penalty ?? 0) >= 0.10 || (snapshot?.near_term_spread_pct ?? 0) >= 8) {
      reasons.push('Execution friction is too high for the modeled edge to survive live fills.')
      action = 'Tighter spreads or more liquid options would improve the decision.'
    }
    if ((snapshot?.data_quality_score ?? selectorOutput?.data_quality_score ?? 1) < 0.45) {
      reasons.push('Snapshot quality is too weak to support a trade decision.')
      action = 'Fresher underlying and chain data would be needed before acting.'
    }
    if ((best.walk_forward_history_count ?? 0) < 8 && (best.sample_confidence ?? 0) < 0.55) {
      reasons.push('Comparable historical evidence is still too thin to support conviction.')
      action = 'A stronger score gap or deeper analog history would improve the case.'
    }
  }

  return {
    headline: 'Edge not sufficient after costs',
    reasons: reasons.length ? reasons : ['The leading structure still does not clear the engine\'s conservative abstention rules.'],
    action,
  }
}

function describeVolRegime(snapshot = null) {
  if (!snapshot) return 'Unknown volatility regime'
  const label = snapshot.vol_regime_label || 'unknown'
  const rvPct = snapshot.rv_percentile_rank != null ? `${Math.round(Number(snapshot.rv_percentile_rank))}th percentile` : 'percentile unavailable'
  if (label === 'Low') return `Low realized volatility (${rvPct})`
  if (label === 'High' || label === 'Elevated') return `${label} realized volatility (${rvPct})`
  if (label === 'Normal') return `Normal realized volatility (${rvPct})`
  return `${label} realized volatility (${rvPct})`
}

function describeIvPremium(snapshot = null) {
  const ratio = snapshot?.iv_rv_yz
  if (ratio == null || Number.isNaN(Number(ratio))) return 'IV premium unavailable'
  const n = Number(ratio)
  if (n < 0.95) return `IV is trading below realized baseline (${formatNumber(n, 2)}x IV/RV)`
  if (n <= 1.25) return `IV carries a moderate premium to realized baseline (${formatNumber(n, 2)}x IV/RV)`
  return `IV already carries an elevated premium to realized baseline (${formatNumber(n, 2)}x IV/RV)`
}

function describeStructureRegimeFit(snapshot = null, best = null) {
  if (!best || !snapshot) return 'Historical regime fit is unavailable.'
  const evidence = Number(best.walk_forward_history_count || 0)
  const quality = Number(best.sample_confidence || 0)
  const structure = best.structure
  const lowOrNormalRv = ['Low', 'Normal'].includes(snapshot.vol_regime_label)
  const moderatePremium = snapshot.iv_rv_yz != null && Number(snapshot.iv_rv_yz) <= 1.25
  const positiveSlope = snapshot.term_structure_slope != null && Number(snapshot.term_structure_slope) >= 0
  const richFront = snapshot.near_back_iv_ratio != null && Number(snapshot.near_back_iv_ratio) >= 1

  if (structure === 'atm_straddle' || structure === 'otm_strangle') {
    if (lowOrNormalRv && moderatePremium) return 'Historically supportive for long-vol expansion structures.'
    if (!moderatePremium) return 'Historically mixed because front-end IV is already rich relative to realized volatility.'
    return 'Historically mixed; event convexity is present but the regime is less clean than ideal.'
  }
  if (structure === 'call_calendar' || structure === 'put_calendar') {
    if (positiveSlope && !richFront) return 'Historically supportive for calendar carry because the front term is not yet fully inflated.'
    if (richFront) return 'Historically weaker for calendars because front-end event premium already looks elevated.'
    return 'Historically mixed; term structure support exists but is not dominant.'
  }
  if (evidence < 8 || quality < 0.55) return 'Historical regime fit is still thin.'
  return 'Historical regime fit is available but not strongly differentiated.'
}

export function buildVolRegimeBanner(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const regimeText = describeVolRegime(snapshot)
  const premiumText = describeIvPremium(snapshot)
  const fitText = describeStructureRegimeFit(snapshot, best)
  let tone = 'balanced'

  if (selectorOutput?.recommendation === 'No Trade') {
    tone = 'cautious'
  } else if ((best?.sample_confidence ?? 0) >= 0.65 && (snapshot?.data_quality_score ?? 0) >= 0.75) {
    tone = 'supportive'
  } else if ((best?.sample_confidence ?? 0) < 0.55 || (snapshot?.data_quality_score ?? 0) < 0.60) {
    tone = 'cautious'
  }

  return {
    tone,
    headline: `Current regime: ${regimeText} with ${premiumText.toLowerCase()}.`,
    detail: fitText,
    stats: {
      winRateLabel: best ? formatPercent((best.walk_forward_win_rate ?? 0) * 100, 0) : 'n/a',
      avgReturnLabel: best ? formatSignedPercent(best.walk_forward_avg_return_pct, 1) : 'n/a',
      sampleLabel: best ? `${best.walk_forward_history_count ?? 0} comparable setups` : 'No comparable setups',
      note: 'Uses the winning structure\'s closest available comparable-setup evidence for the current regime context.',
    },
  }
}

export function buildEdgeNarrative(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const marketMove = snapshot?.event_implied_move_pct
  const historicalMove = snapshot?.historical_move_anchor_pct ?? snapshot?.historical_median_move_pct
  const mismatch = snapshot?.historical_vs_implied_move_ratio
  const mismatchText = mismatch == null || Number.isNaN(Number(mismatch))
    ? 'Mismatch unavailable.'
    : Number(mismatch) > 1
      ? `This points to underpriced event risk by about ${formatNumber((Number(mismatch) - 1) * 100, 0)}% versus history.`
      : Number(mismatch) < 1
        ? `This points to fully priced or rich event risk, with history running about ${formatNumber((1 - Number(mismatch)) * 100, 0)}% below the current implied move.`
        : 'Current implied move is roughly aligned with historical behavior.'

  return {
    marketLine: `Market is pricing about ${formatPercent(marketMove, 2)} through the event.`,
    historyLine: historicalMove == null || Number.isNaN(Number(historicalMove))
      ? 'Historical event-move context is unavailable.'
      : `Historical earnings behavior points to about ${formatPercent(historicalMove, 2)} on the current anchor.`,
    mismatchLine: mismatchText,
    note: best
      ? `${formatStructureLabel(best.structure)} ranks first because that mismatch lines up better with its payoff shape than with the alternatives.`
      : 'No structure cleared the minimum bar, so the mismatch is informational rather than actionable.',
  }
}

export function buildOutcomeDecomposition(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  if (!best) return []

  const ivContribution = Number(best.expected_iv_contribution_pct || 0)
  const moveContribution = Number(best.expected_return_pct || 0) - ivContribution
  const executionRisk = Number(best.execution_penalty || 0) * 100

  return [
    {
      label: 'IV Expansion Contribution',
      value: `${formatSignedPercent(ivContribution, 1)} signal`,
      detail: 'Score-derived contribution from pre-event IV expansion if the surface behaves like comparable setups. Not an empirical P&L forecast.',
    },
    {
      label: 'Realized Move Contribution',
      value: `${formatSignedPercent(moveContribution, 1)} signal`,
      detail: 'Score-derived residual from realized move capture and carry after separating the IV-expansion component.',
    },
    {
      label: 'Execution Friction Risk',
      value: formatPercent(executionRisk, 0),
      detail: 'Indicative loss pressure from spread crossing and live fill slippage. Modeled execution is not a promise of live retail fills.',
    },
  ]
}

export function buildOutcomeScenarios(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  if (!best) return []

  const baseEdge = Number(best.expected_edge_pct || 0)
  const ivContribution = Number(best.expected_iv_contribution_pct || 0)
  const moveContribution = Number(best.expected_return_pct || 0) - ivContribution
  const moveDispersion = Math.max(
    Number(snapshot?.historical_move_std_pct || 0) * 0.35,
    Number(snapshot?.historical_move_uncertainty_pct || 0) * 0.25,
    0.5,
  )
  const frictionStress = Math.max(Number(best.execution_penalty || 0) * 100, 0.6)
  const ivShock = Math.max(Math.abs(ivContribution) * 0.35, 0.5)
  const moveShock = Math.max(Math.abs(moveContribution) * 0.30, moveDispersion)

  const bestCase = baseEdge + ivShock + moveShock - frictionStress * 0.20
  const baseCase = baseEdge
  const worstCase = baseEdge - ivShock - moveShock - frictionStress * 0.60

  return [
    {
      label: 'Best Case',
      value: `${formatSignedPercent(bestCase, 1)} signal`,
      detail: 'Score-derived upside case: IV expands cleanly, the realized move lands toward the favorable end of history, and live fills stay efficient.',
    },
    {
      label: 'Base Case',
      value: `${formatSignedPercent(baseCase, 1)} signal`,
      detail: 'Matches the current modeled edge after existing penalties and costs. This is not a calibrated return forecast.',
    },
    {
      label: 'Worst Case',
      value: `${formatSignedPercent(worstCase, 1)} signal`,
      detail: 'Score-derived downside case: IV expansion underdelivers, the realized move is muted, and execution gets worse than the quoted snapshot.',
    },
  ]
}

export function buildEdgeDurability(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  if (!best) {
    return {
      label: 'Fragile',
      tone: 'fragile',
      stressedEdgeLabel: 'n/a',
      detail: 'No winning structure remains, so the edge cannot be tested for durability.',
    }
  }

  const baseEdge = Number(best.expected_edge_pct || 0)
  const ivContribution = Number(best.expected_iv_contribution_pct || 0)
  const moveContribution = Number(best.expected_return_pct || 0) - ivContribution
  const extraExecution = Math.max(Number(best.execution_penalty || 0) * 100 * 0.75, 0.7)
  const reducedIv = Math.max(Math.abs(ivContribution) * 0.40, 0.5)
  const reducedMove = Math.max(Math.abs(moveContribution) * 0.35, Number(snapshot?.historical_move_uncertainty_pct || 0) * 0.20, 0.5)
  const stressedEdge = baseEdge - extraExecution - reducedIv - reducedMove

  let label = 'Fragile'
  let tone = 'fragile'
  let detail = 'Small adverse changes in IV expansion, realized move, or execution would likely erase the modeled edge.'
  if (stressedEdge > 1.0) {
    label = 'Robust'
    tone = 'robust'
    detail = 'The edge still survives a conservative stress on execution, IV expansion, and realized move assumptions.'
  } else if (stressedEdge > 0) {
    label = 'Moderate'
    tone = 'moderate'
    detail = 'The edge survives some stress, but not with much cushion.'
  }

  return {
    label,
    tone,
    stressedEdgeLabel: `${formatSignedPercent(stressedEdge, 1)} signal`,
    detail,
  }
}

export function buildStructureAdvantage(selectorOutput = null, scorecards = []) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const runnerUp = getRunnerUpScorecard(scorecards, selectorOutput)
  if (!best || !runnerUp) {
    return {
      headline: 'No runner-up comparison is available.',
      bullets: ['Only one eligible structure remains, so the selector does not have a close alternative to compare.'],
    }
  }

  const returnDiff = Number(best.expected_return_pct || 0) - Number(runnerUp.expected_return_pct || 0)
  const executionDiff = Number(runnerUp.execution_penalty || 0) - Number(best.execution_penalty || 0)
  const winDiff = (Number(best.walk_forward_win_rate || 0) - Number(runnerUp.walk_forward_win_rate || 0)) * 100

  return {
    headline: `${formatStructureLabel(best.structure)} leads ${formatStructureLabel(runnerUp.structure)} on return, execution, and evidence balance.`,
    bullets: [
      `Score-derived return signal is ${formatSignedPercent(returnDiff, 1)} higher than the runner-up.`,
      `${formatStructureLabel(best.structure)} carries ${formatSignedPercent(executionDiff, 1)} less execution penalty than the runner-up.`,
      `Historical positive-outcome rate is ${winDiff >= 0 ? '+' : ''}${formatNumber(winDiff, 0)} percentage points versus the runner-up.`,
    ],
  }
}

export function buildFailureModes(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  if (!best) {
    return [
      'No structure is currently robust enough to survive the minimum eligibility, execution, and evidence requirements.',
    ]
  }

  const failures = []
  const structure = best.structure

  if ((snapshot?.iv_rv_yz ?? 0) >= 1.30 || (best.crowding_penalty ?? 0) >= 0.05) {
    failures.push('IV is already elevated enough that further expansion may be limited.')
  }
  if ((snapshot?.historical_vs_implied_move_ratio ?? 1) <= 1.0) {
    failures.push('Realized earnings moves may not outrun what the market is already pricing.')
  }
  if ((best.execution_penalty ?? 0) >= 0.04 || (snapshot?.near_term_spread_pct ?? 0) >= 6) {
    failures.push('Spreads can widen enough to absorb much of the modeled edge.')
  }

  if (structure === 'atm_straddle' || structure === 'otm_strangle') {
    failures.push('A muted move or early IV stall can let theta dominate before the event.')
  }
  if (structure === 'call_calendar' || structure === 'put_calendar') {
    failures.push('Front-end event premium can inflate faster than expected, leaving less room for the calendar to improve.')
    if ((snapshot?.tail_vs_implied_move_ratio ?? 0) > 1.3) {
      failures.push('Large tail moves can still overwhelm a calendar structure even if the term structure looks attractive.')
    }
  }

  return failures.slice(0, 4)
}

export function buildSensitivityWarnings(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  if (!best) return []

  const warnings = []
  const historicalCount = Number(snapshot?.historical_event_count || 0)
  const uncertainty = Number(snapshot?.historical_move_uncertainty_pct || 0)
  const tailRatio = Number(snapshot?.tail_vs_implied_move_ratio || 0)
  const edge = Number(best.expected_edge_pct || 0)
  const execPenaltyPct = Number(best.execution_penalty || 0) * 100

  if (historicalCount > 0 && historicalCount < 8) {
    warnings.push('Edge is sensitive to assumptions because the historical sample is still small.')
  }
  if (historicalCount > 0 && historicalCount < 10 && tailRatio >= 1.45) {
    warnings.push('Edge may lean heavily on one or two large historical tail events rather than a broad distribution.')
  }
  if (edge > 0 && execPenaltyPct >= Math.max(edge * 0.55, 1.5)) {
    warnings.push('Edge is sensitive to tight spreads; modest execution slippage could materially reduce it.')
  }
  if (uncertainty >= 2.0) {
    warnings.push('Edge is sensitive to the move distribution because historical earnings behavior is unusually dispersed.')
  }

  return warnings.slice(0, 3)
}

export function buildRegimeConfidenceMessage(selectorOutput = null, scorecards = [], snapshot = null) {
  const best = getBestScorecard(scorecards, selectorOutput)
  const dataQuality = Number(snapshot?.data_quality_score ?? selectorOutput?.data_quality_score ?? 0)
  const sample = Number(best?.sample_confidence ?? 0)
  const regime = snapshot?.vol_regime_label || 'unknown'

  if (!best) {
    return {
      tone: 'cautious',
      label: 'Decision quality reduced',
      detail: 'No winning structure remains, so the regime context can only be descriptive.',
    }
  }

  if (sample < 0.55 || dataQuality < 0.60) {
    return {
      tone: 'cautious',
      label: 'Decision quality reduced',
      detail: `Comparable evidence in the current ${regime.toLowerCase()}-volatility regime is still limited or noisy.`,
    }
  }

  if (sample >= 0.70 && dataQuality >= 0.80) {
    return {
      tone: 'supportive',
      label: 'Decision quality strengthened',
      detail: `Comparable setups in the current ${regime.toLowerCase()}-volatility regime have been relatively consistent.`,
    }
  }

  return {
    tone: 'balanced',
    label: 'Decision quality balanced',
    detail: `The current ${regime.toLowerCase()}-volatility regime is represented, but not deeply enough to remove uncertainty.`,
  }
}

export function buildCalibrationModel(curveData = null, score = null) {
  if (!curveData || !Array.isArray(curveData.buckets) || !curveData.buckets.length) return null
  const numericScore = score == null || Number.isNaN(Number(score)) ? null : Number(score)
  const activeBucket = numericScore == null
    ? null
    : curveData.buckets.find((bucket) => numericScore >= bucket.score_lo && numericScore < bucket.score_hi) || curveData.buckets[curveData.buckets.length - 1]

  const phase = curveData.phase || 'bootstrap_prior'
  const isPrior = phase === 'bootstrap_prior'
  const isObservational = phase === 'observational'
  const isFitted = phase === 'fitted_moderate' || phase === 'fitted_high'
  const phaseLabel = {
    bootstrap_prior: 'Prior / Ordering Only',
    observational: 'Observational',
    fitted_moderate: 'Fitted (Moderate Sample)',
    fitted_high: 'Fitted (High Sample)',
  }[phase] || 'Phase Unknown'

  const phaseSummary = (() => {
    if (isPrior) {
      return `Research prior only · ${curveData.n_observations}/${curveData.min_for_observational} observations.`
    }
    if (isObservational) {
      return `Raw bucket observations · N=${curveData.n_observations}. No fitted curve is used yet.`
    }
    if (phase === 'fitted_moderate') {
      return `Moderate fitted phase · N=${curveData.n_observations}. Use as empirical context, not certainty.`
    }
    return `High-sample fitted phase · N=${curveData.n_observations}.`
  })()

  return {
    phase,
    phaseLabel,
    isPrior,
    isObservational,
    isFitted,
    nObservations: curveData.n_observations,
    minForObservational: curveData.min_for_observational,
    minForFit: curveData.min_for_fit,
    minForHighFit: curveData.min_for_high_fit,
    phaseSummary,
    activeBucket,
    activeSummary: activeBucket
      ? {
          rangeLabel: `${activeBucket.score_lo.toFixed(1)}-${activeBucket.score_hi.toFixed(1)}`,
          headline: `This setup score falls in bucket ${activeBucket.score_lo.toFixed(1)}-${activeBucket.score_hi.toFixed(1)}.`,
          detail: isPrior
            ? `Research prior for this range is ${formatPercent(activeBucket.expected_expansion_pct, 1)}. This is ordering-only guidance, not an empirical estimate.`
            : isObservational
              ? activeBucket.n > 0
                ? `Observed bucket average is ${formatPercent(activeBucket.expected_expansion_pct, 1)} from ${activeBucket.n} comparable observations.${activeBucket.std_pct != null ? ` Dispersion is ${formatPercent(activeBucket.std_pct, 1)}.` : ''}`
                : `This bucket has no direct observations yet, so it still falls back to the research prior of ${formatPercent(activeBucket.expected_expansion_pct, 1)}.`
              : `Empirical calibration for this range is ${formatPercent(activeBucket.expected_expansion_pct, 1)}${activeBucket.std_pct != null ? ` with ${formatPercent(activeBucket.std_pct, 1)} bucket dispersion` : ''}.`,
          evidenceLabel: activeBucket.n > 0 ? `Bucket N=${activeBucket.n}` : 'Research prior',
        }
      : null,
    buckets: curveData.buckets.map((bucket) => ({
      ...bucket,
      key: `${bucket.score_lo}-${bucket.score_hi}`,
      rangeLabel: `${bucket.score_lo.toFixed(1)}-${bucket.score_hi.toFixed(1)}`,
      expectedExpansionLabel: formatPercent(bucket.expected_expansion_pct, 1),
      bandLabel: bucket.std_pct != null ? formatPercent(bucket.std_pct, 1) : 'n/a',
      isActive: activeBucket ? bucket.score_lo === activeBucket.score_lo && bucket.score_hi === activeBucket.score_hi : false,
    })),
  }
}

export function getEdgeLayoutSections() {
  return [
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
  ]
}
