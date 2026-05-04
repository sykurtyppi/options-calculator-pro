export function formatLedgerScore(value) {
  const n = Number(value)
  return Number.isFinite(n) ? n.toFixed(2) : 'n/a'
}

export function formatLedgerTimestamp(value) {
  if (!value) return 'n/a'
  try {
    return new Date(value).toLocaleString()
  } catch {
    return String(value)
  }
}

export function formatStructureLabel(value) {
  return value ? String(value).replace(/_/g, ' ') : 'No structure'
}

export function buildLedgerEndpoint(apiBase, { symbol = '', limit = 25, offset = 0 } = {}) {
  const cleaned = String(symbol || '').trim().toUpperCase()
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(Math.max(0, Number(offset) || 0)),
  })
  if (cleaned) params.set('symbol', cleaned)
  return `${apiBase}/api/diagnostics/recommendations?${params.toString()}`
}

export function buildLedgerExportUrls(apiBase, { symbol = '', limit = 25, offset = 0 } = {}) {
  const cleaned = String(symbol || '').trim().toUpperCase()
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(Math.max(0, Number(offset) || 0)),
  })
  if (cleaned) params.set('symbol', cleaned)

  const recommendationCsv = new URLSearchParams(params)
  recommendationCsv.set('format', 'csv')
  const recommendationJson = new URLSearchParams(params)
  recommendationJson.set('format', 'json')
  const linkageCsv = new URLSearchParams(params)
  linkageCsv.set('format', 'csv')
  const linkageJson = new URLSearchParams(params)
  linkageJson.set('format', 'json')

  return {
    recommendationCsv: `${apiBase}/api/diagnostics/recommendations/export?${recommendationCsv.toString()}`,
    recommendationJson: `${apiBase}/api/diagnostics/recommendations/export?${recommendationJson.toString()}`,
    linkageCsv: `${apiBase}/api/diagnostics/recommendations/linkage/export?${linkageCsv.toString()}`,
    linkageJson: `${apiBase}/api/diagnostics/recommendations/linkage/export?${linkageJson.toString()}`,
  }
}

export function buildLedgerRows(rows = []) {
  return rows.map((row) => {
    const isNoTrade = row.recommendation === 'No Trade'
    return {
      ...row,
      displayTimestamp: formatLedgerTimestamp(row.created_at),
      displayStructureOrReason: isNoTrade
        ? row.no_trade_reason || 'selector abstained'
        : formatStructureLabel(row.selected_structure),
      displayQuality: formatLedgerScore(row.data_quality_score),
      displayEarningsSource: row.earnings_source || 'unknown',
      staleFlagVisible: Boolean(row.earnings_source_stale),
      outcomeLabel: row.outcome_status || 'unlinked',
    }
  })
}

export function getLedgerEmptyMessage(rows = [], loading = false) {
  if (loading || rows.length) return ''
  return 'No recommendation records found yet.'
}

export function buildLedgerSummaryPills(summary = {}) {
  const topStructure = Object.entries(summary.by_selected_structure || {})
    .filter(([key]) => key !== 'unknown')
    .sort((a, b) => b[1] - a[1])[0]
  return {
    total: summary.total ?? 0,
    topStructure: topStructure ? `${formatStructureLabel(topStructure[0])} · ${topStructure[1]}` : 'n/a',
    freshOrUnknown: summary.by_stale_source_flag?.fresh_or_unknown ?? 0,
    stale: summary.by_stale_source_flag?.stale ?? 0,
  }
}

export function buildLedgerDetailModel(detail = null, linkage = null) {
  if (!detail) return null
  const quote = detail.quote_provenance || {}
  const paperTrade = linkage?.paper_trade || null
  return {
    recommendationId: detail.recommendation?.recommendation_id || null,
    selectedStructure: formatStructureLabel(detail.recommendation?.selected_structure),
    paperTradeLabel: paperTrade ? `${paperTrade.status} · ${paperTrade.trade_id}` : 'No linked paper trade',
    quoteSource: quote.quote_source || 'unknown',
    quoteQuality: quote.quote_quality || 'not recorded',
    quoteTimestamp: quote.quote_timestamp || null,
    primaryThesis: detail.selector_explanation?.primary_thesis || 'No thesis stored.',
    primaryRisks: (detail.selector_explanation?.primary_risks || []).slice(0, 5),
    scorecardCount: (detail.structure_scorecards || []).length,
    bidAskMid: quote.bid_ask_mid || {},
  }
}
