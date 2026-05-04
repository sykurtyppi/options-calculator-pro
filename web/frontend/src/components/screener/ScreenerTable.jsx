import React from 'react'

import QualificationBadge from './QualificationBadge'
import SpreadChangeIndicator from './SpreadChangeIndicator'
import { formatDate, formatPct, formatTimestamp } from './formatters'

function signalSummary(summary) {
  if (!Array.isArray(summary) || summary.length === 0) return 'No signal summary'
  return summary.slice(0, 2).join(' · ')
}

function SortableHeader({ label, colKey, sortKey, sortDirection, onSortKeyChange, onSortDirectionChange }) {
  const isActive = sortKey === colKey
  const arrow = isActive ? (sortDirection === 'asc' ? ' ↑' : ' ↓') : ''

  function handleClick() {
    if (isActive) {
      onSortDirectionChange(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      onSortKeyChange(colKey)
      onSortDirectionChange('asc')
    }
  }

  return (
    <th
      className={`sortable-th${isActive ? ' sort-active' : ''}`}
      onClick={handleClick}
      title={`Sort by ${label}`}
    >
      {label}{arrow}
    </th>
  )
}

export default function ScreenerTable({ rows, selectedKey, onSelect, sortKey, sortDirection, onSortKeyChange, onSortDirectionChange }) {
  if (!rows.length) {
    return <div className="empty-state">No setups match the current filters.</div>
  }

  const sortProps = { sortKey, sortDirection, onSortKeyChange, onSortDirectionChange }

  return (
    <div className="screener-table-wrap">
      <table className="screener-table">
        <thead>
          <tr>
            <SortableHeader label="Symbol" colKey="symbol" {...sortProps} />
            <SortableHeader label="Earnings" colKey="earnings_date" {...sortProps} />
            <th>Timing</th>
            <th>Entry</th>
            <th>Expiry</th>
            <SortableHeader label="Avg Spread" colKey="avg_spread_pct" {...sortProps} />
            <th>Drift</th>
            <th>Call OI</th>
            <th>Put OI</th>
            <SortableHeader label="Impl Move" colKey="implied_move_pct" {...sortProps} />
            <th>Signal</th>
            <SortableHeader label="Status" colKey="status" {...sortProps} />
            <th>Updated</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const rowKey = `${row.symbol}-${row.earnings_date}-${row.expiry_mode}`
            return (
              <tr
                key={rowKey}
                className={selectedKey === rowKey ? 'selected' : ''}
                onClick={() => onSelect(row)}
              >
                <td className="symbol-cell">
                  <div className="symbol-primary">{row.symbol}</div>
                  <div className="symbol-secondary">{row.status_reason}</div>
                </td>
                <td>{formatDate(row.earnings_date)}</td>
                <td>{row.release_timing}</td>
                <td>{row.entry_label}<span className="muted-inline"> · {formatDate(row.entry_date)}</span></td>
                <td>{row.selected_expiry || 'n/a'}</td>
                <td>{formatPct(row.avg_spread_pct, 2)}</td>
                <td><SpreadChangeIndicator state={row.spread_change_state} change={row.spread_change_pct} /></td>
                <td>{row.call_oi ?? 'n/a'}</td>
                <td>{row.put_oi ?? 'n/a'}</td>
                <td>{formatPct(row.implied_move_pct, 1)}</td>
                <td className="signal-cell">{signalSummary(row.compact_signal_summary)}</td>
                <td><QualificationBadge status={row.status} /></td>
                <td>{formatTimestamp(row.last_updated)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
