import React from 'react'
import { Badge, SectionTitle } from '../common/DisplayAtoms'
import { fmtMoney, fmtNum, fmtPct, fmtVol } from '../../lib/formatters'

export default function HistoricalWarehousePanel({
  warehouse,
  setWarehouse,
  onLoadSymbols,
  onRunQuery,
}) {
  const symbols = warehouse.symbols || []
  const rows = warehouse.rows || []
  const coverage = warehouse.coverage?.[0]

  return (
    <section className="warehouse-block">
      <div className="oos-head">
        <div>
          <SectionTitle>Historical Warehouse</SectionTitle>
          <p className="oos-help">
            Local Parquet/ZSTD replay layer on the T9. Query decoded historical chains without spending API credits.
          </p>
        </div>
        <div className="warehouse-status">
          <Badge variant={warehouse.available ? 'mda' : 'default'}>
            {warehouse.available ? `${symbols.length} symbols` : 'offline'}
          </Badge>
          <button className="secondary-button" type="button" onClick={onLoadSymbols} disabled={warehouse.loadingSymbols}>
            {warehouse.loadingSymbols ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      <div className="warehouse-controls">
        <label className="oos-field">
          <span>Symbol</span>
          <input
            list="warehouse-symbols"
            value={warehouse.symbol}
            onChange={e => setWarehouse(p => ({ ...p, symbol: e.target.value.toUpperCase() }))}
            placeholder="SPY"
          />
          <datalist id="warehouse-symbols">
            {symbols.map(sym => <option key={sym} value={sym} />)}
          </datalist>
        </label>
        <label className="oos-field">
          <span>Trade Date</span>
          <input
            type="date"
            value={warehouse.tradeDate}
            onChange={e => setWarehouse(p => ({ ...p, tradeDate: e.target.value }))}
          />
        </label>
        <label className="oos-field">
          <span>Min DTE</span>
          <input
            type="number"
            value={warehouse.minDte}
            onChange={e => setWarehouse(p => ({ ...p, minDte: e.target.value }))}
          />
        </label>
        <label className="oos-field">
          <span>Max DTE</span>
          <input
            type="number"
            value={warehouse.maxDte}
            onChange={e => setWarehouse(p => ({ ...p, maxDte: e.target.value }))}
          />
        </label>
        <label className="oos-field">
          <span>Side</span>
          <select
            value={warehouse.callPut}
            onChange={e => setWarehouse(p => ({ ...p, callPut: e.target.value }))}
          >
            <option value="">Calls + Puts</option>
            <option value="C">Calls</option>
            <option value="P">Puts</option>
          </select>
        </label>
        <label className="oos-field">
          <span>Limit</span>
          <input
            type="number"
            value={warehouse.limit}
            onChange={e => setWarehouse(p => ({ ...p, limit: e.target.value }))}
          />
        </label>
      </div>

      <div className="warehouse-actions">
        <button type="button" onClick={onRunQuery} disabled={warehouse.loadingRows || !warehouse.symbol || !warehouse.tradeDate}>
          {warehouse.loadingRows ? 'Querying…' : 'Query Historical Chain'}
        </button>
        {coverage && (
          <span className="warehouse-coverage">
            {coverage.symbol}: {coverage.start_date} → {coverage.end_date} · {Number(coverage.rows).toLocaleString()} rows
          </span>
        )}
      </div>

      {warehouse.error && <div className="error-banner">{warehouse.error}</div>}

      {rows.length > 0 && (
        <div className="warehouse-table-wrap">
          <div className="quality-flag-legend">
            <span className="quality-legend-label">Quote Quality:</span>
            <span className="quality-flag quality-ok" title="Bid ≤ ask, bid > 0, mid > 0 — usable for pricing and IV analysis">ok</span>
            <span className="quality-flag quality-zero-bid" title="Bid is 0 but ask > 0 — no buy-side interest; mid is unreliable. Excluded from backtests.">zero_bid</span>
            <span className="quality-flag quality-inverted-quote" title="Bid > ask (crossed market) — stale or erroneous data. Excluded from all analysis.">inverted_quote</span>
            <span className="quality-flag quality-missing-quote" title="Bid or ask is NULL — no market maker quote; option may be illiquid or expired.">missing_quote</span>
            <span className="quality-flag quality-non-positive-mid" title="Mid is NULL or ≤ 0 — cannot price; excluded from IV calculations.">non_positive_mid</span>
          </div>
          <table className="warehouse-table">
            <thead>
              <tr>
                <th>Expiry</th>
                <th>Side</th>
                <th>DTE</th>
                <th>Strike</th>
                <th>Bid</th>
                <th>Ask</th>
                <th>Mid</th>
                <th>IV</th>
                <th>Delta</th>
                <th>Spread</th>
                <th title="Quote quality: ok=tradeable · zero_bid=no buyers · inverted_quote=crossed/stale · missing_quote=no quote · non_positive_mid=unpriced">Quality ⓘ</th>
                <th>Liquidity</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={`${row.option_symbol}-${i}`}>
                  <td>{row.expiry}</td>
                  <td>{row.call_put}</td>
                  <td>{row.dte}</td>
                  <td>{fmtNum(row.strike, 2)}</td>
                  <td>{fmtMoney(row.bid)}</td>
                  <td>{fmtMoney(row.ask)}</td>
                  <td>{fmtMoney(row.mid)}</td>
                  <td>{fmtVol(row.iv, 2)}</td>
                  <td>{fmtNum(row.delta, 3)}</td>
                  <td>{fmtPct(row.spread_pct)}</td>
                  <td>
                    <span className={`quality-flag quality-${(row.quote_quality_flag || 'unknown').replace(/_/g, '-')}`}>
                      {row.quote_quality_flag || 'unknown'}
                    </span>
                  </td>
                  <td>{fmtNum(row.liquidity_score, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
