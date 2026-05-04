import test from 'node:test'
import assert from 'node:assert/strict'
import {
  buildLedgerDetailModel,
  buildLedgerEndpoint,
  buildLedgerExportUrls,
  buildLedgerRows,
  getLedgerEmptyMessage,
} from './ledgerDiagnosticsViewModel.js'

const sampleRows = [
  {
    recommendation_id: 'rec_aapl',
    created_at: '2026-04-23T12:00:00Z',
    symbol: 'AAPL',
    recommendation: 'Candidate',
    selected_structure: 'atm_straddle',
    data_quality_score: 0.77,
    earnings_source: 'alpha_vantage',
    earnings_source_stale: true,
    outcome_status: 'open',
  },
  {
    recommendation_id: 'rec_msft',
    created_at: '2026-04-23T12:05:00Z',
    symbol: 'MSFT',
    recommendation: 'No Trade',
    selected_structure: null,
    no_trade_reason: 'Data quality too low.',
    data_quality_score: 0.31,
    earnings_source: 'fmp',
    earnings_source_stale: false,
    outcome_status: null,
  },
]

test('ledger empty state is explicit and quiet while loading', () => {
  assert.equal(getLedgerEmptyMessage([], false), 'No recommendation records found yet.')
  assert.equal(getLedgerEmptyMessage([], true), '')
  assert.equal(getLedgerEmptyMessage(sampleRows, false), '')
})

test('ledger rows render recent decisions with stale flags and outcome linkage', () => {
  const rows = buildLedgerRows(sampleRows)

  assert.equal(rows[0].symbol, 'AAPL')
  assert.equal(rows[0].displayStructureOrReason, 'atm straddle')
  assert.equal(rows[0].displayQuality, '0.77')
  assert.equal(rows[0].displayEarningsSource, 'alpha_vantage')
  assert.equal(rows[0].staleFlagVisible, true)
  assert.equal(rows[0].outcomeLabel, 'open')
})

test('ledger rows show no-trade abstain reason without hiding the row', () => {
  const rows = buildLedgerRows(sampleRows)

  assert.equal(rows[1].recommendation, 'No Trade')
  assert.equal(rows[1].displayStructureOrReason, 'Data quality too low.')
  assert.equal(rows[1].staleFlagVisible, false)
  assert.equal(rows[1].outcomeLabel, 'unlinked')
})

test('symbol filter and pagination are encoded into the ledger endpoint', () => {
  const endpoint = buildLedgerEndpoint('http://127.0.0.1:8000', {
    symbol: 'aapl',
    limit: 10,
    offset: 20,
  })

  assert.equal(endpoint, 'http://127.0.0.1:8000/api/diagnostics/recommendations?limit=10&offset=20&symbol=AAPL')
})

test('export URLs include recommendation and linkage downloads', () => {
  const urls = buildLedgerExportUrls('http://127.0.0.1:8000', {
    symbol: 'AAPL',
    limit: 25,
    offset: 0,
  })

  assert.match(urls.recommendationCsv, /recommendations\/export/)
  assert.match(urls.recommendationCsv, /format=csv/)
  assert.match(urls.recommendationJson, /format=json/)
  assert.match(urls.linkageCsv, /linkage\/export/)
  assert.match(urls.linkageCsv, /symbol=AAPL/)
})

test('drill-down view model exposes quote provenance and linked paper trade', () => {
  const detail = {
    recommendation: {
      recommendation_id: 'rec_aapl',
      selected_structure: 'atm_straddle',
    },
    selector_explanation: {
      primary_thesis: 'Structure fits this event-vol snapshot.',
      primary_risks: ['Execution risk.'],
    },
    structure_scorecards: [{ structure: 'atm_straddle' }],
    quote_provenance: {
      quote_source: 'yfinance',
      quote_quality: 'paper_research_mid_not_execution_grade',
      quote_timestamp: '2026-04-23T12:00:00Z',
      bid_ask_mid: { call: { bid: 4, ask: 4.2, mid: 4.1 } },
    },
  }
  const linkage = {
    paper_trade: {
      status: 'open',
      trade_id: 'AAPL|2026-04-23|atm_straddle',
    },
  }

  const model = buildLedgerDetailModel(detail, linkage)
  assert.equal(model.recommendationId, 'rec_aapl')
  assert.equal(model.selectedStructure, 'atm straddle')
  assert.equal(model.paperTradeLabel, 'open · AAPL|2026-04-23|atm_straddle')
  assert.equal(model.quoteSource, 'yfinance')
  assert.equal(model.quoteQuality, 'paper_research_mid_not_execution_grade')
  assert.equal(model.scorecardCount, 1)
  assert.equal(model.bidAskMid.call.mid, 4.1)
})
