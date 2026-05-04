import React from 'react'

import QualificationBadge from './QualificationBadge'
import SpreadChangeIndicator from './SpreadChangeIndicator'
import { expiryModeLabel, formatDate, formatMoney, formatNumber, formatPct, formatTimestamp } from './formatters'

function SignalMetric({ label, value, sub }) {
  return (
    <div className="detail-metric">
      <div className="detail-metric-label">{label}</div>
      <div className="detail-metric-value">{value}</div>
      {sub && <div className="detail-metric-sub">{sub}</div>}
    </div>
  )
}

export default function SetupDetailPanel({ row, analysisState, onAnalyzeSymbol }) {
  if (!row) {
    return (
      <aside className="setup-detail-panel">
        <div className="empty-state">Select a setup to inspect qualification logic.</div>
      </aside>
    )
  }

  const detail = row.detail_metrics || {}
  const analysisMetrics = analysisState?.result?.metrics || {}

  return (
    <aside className="setup-detail-panel">
      <div className="detail-header">
        <div>
          <div className="detail-symbol-row">
            <h3>{row.symbol}</h3>
            <QualificationBadge status={row.status} />
          </div>
          <p>{row.status_reason}</p>
        </div>
        <button type="button" className="secondary-btn" onClick={() => onAnalyzeSymbol(row.symbol)}>
          Open Single-Ticker Analysis
        </button>
      </div>

      <div className="detail-summary-grid">
        <SignalMetric label="Earnings" value={formatDate(row.earnings_date)} sub={row.release_timing} />
        <SignalMetric label="Entry" value={`${row.entry_label}`} sub={formatDate(row.entry_date)} />
        <SignalMetric label="Selected expiry" value={row.selected_expiry || 'n/a'} sub={expiryModeLabel(row.expiry_mode)} />
        <SignalMetric label="Alternative expiry" value={row.alternative_expiry || 'n/a'} />
        <SignalMetric label="Spread drift" value={<SpreadChangeIndicator state={row.spread_change_state} change={row.spread_change_pct} />} sub={`Last refresh ${formatTimestamp(row.last_updated)}`} />
        <SignalMetric label="Entry debit mid" value={formatMoney(row.entry_debit_mid)} sub="Current screener snapshot" />
      </div>

      <section className="detail-section">
        <div className="detail-section-title">Qualification Checks</div>
        <div className="check-list">
          {(row.checks || []).map((check) => (
            <div key={check.label} className={`check-card${check.passed ? ' passed' : ' failed'}`}>
              <div className="check-card-top">
                <span>{check.label}</span>
                <span>{check.passed ? 'PASS' : 'FAIL'}</span>
              </div>
              <div className="check-card-body">Threshold: {check.threshold}</div>
              <div className="check-card-body">Observed: {check.actual}</div>
              {check.note && <div className="check-card-note">{check.note}</div>}
            </div>
          ))}
        </div>
      </section>

      <section className="detail-section">
        <div className="detail-section-title">Selected Structure</div>
        <div className="detail-summary-grid">
          <SignalMetric label="Call strike" value={formatNumber(row.call_strike, 2)} sub={`OI ${row.call_oi ?? 'n/a'} · IV ${row.call_iv != null ? formatPct(row.call_iv * 100, 1) : 'n/a'}`} />
          <SignalMetric label="Put strike" value={formatNumber(row.put_strike, 2)} sub={`OI ${row.put_oi ?? 'n/a'} · IV ${row.put_iv != null ? formatPct(row.put_iv * 100, 1) : 'n/a'}`} />
          <SignalMetric label="Call spread" value={formatPct(detail.call_spread_pct, 2)} sub={`${formatMoney(detail.call_bid)} bid / ${formatMoney(detail.call_ask)} ask`} />
          <SignalMetric label="Put spread" value={formatPct(detail.put_spread_pct, 2)} sub={`${formatMoney(detail.put_bid)} bid / ${formatMoney(detail.put_ask)} ask`} />
          <SignalMetric label="Implied move" value={formatPct(row.implied_move_pct, 1)} sub={`Spot ${formatMoney(detail.spot)}`} />
          <SignalMetric label="Contract IDs" value="Exact live contracts" sub={`${detail.call_contract_id || 'n/a'} · ${detail.put_contract_id || 'n/a'}`} />
        </div>
      </section>

      <section className="detail-section">
        <div className="detail-section-title">Signal Context</div>
        {analysisState?.loading ? (
          <div className="detail-loading">Loading deeper signal context…</div>
        ) : analysisState?.error ? (
          <div className="detail-inline-error">{analysisState.error}</div>
        ) : (
          <div className="detail-summary-grid">
            <SignalMetric label="IV / RV30" value={formatNumber(analysisMetrics.iv_rv30, 2)} />
            <SignalMetric label="Term slope" value={formatNumber(analysisMetrics.term_structure_slope, 3)} />
            <SignalMetric label="NBR" value={formatNumber(analysisMetrics.near_back_iv_ratio, 2)} />
            <SignalMetric label="Vol regime" value={analysisMetrics.vol_regime || 'n/a'} sub={analysisMetrics.rv_percentile_rank != null ? `${formatNumber(analysisMetrics.rv_percentile_rank, 0)}th pct` : null} />
            <SignalMetric label="Near spread" value={formatPct(analysisMetrics.near_term_spread_pct, 2)} />
            <SignalMetric label="Release source" value={analysisMetrics.earnings_release_time || 'n/a'} />
          </div>
        )}
      </section>

      {(row.notes?.length || row.caveats?.length) ? (
        <section className="detail-section">
          <div className="detail-section-title">Notes & Caveats</div>
          <ul className="detail-note-list">
            {(row.notes || []).map((note) => <li key={note}>{note}</li>)}
            {(row.caveats || []).map((note) => <li key={note}>{note}</li>)}
          </ul>
        </section>
      ) : null}
    </aside>
  )
}
