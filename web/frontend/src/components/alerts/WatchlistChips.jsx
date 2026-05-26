import React from 'react'

/**
 * Horizontal strip of watchlist chips. Click a chip's symbol to re-run
 * analysis on that ticker; click × to remove it from the watchlist.
 * Returns null on empty list.
 */
export default function WatchlistChips({ watchlist, onSelect, onRemove }) {
  if (!watchlist.length) return null
  return (
    <div className="watchlist-strip">
      <span className="watchlist-label">Watchlist</span>
      {watchlist.map((e) => (
        <span
          key={e.symbol}
          className={`watchlist-chip rec-chip-${(e.recommendation || 'default').toLowerCase().replace(/[\s/]+/g, '-')}`}
        >
          <span className="chip-symbol" onClick={() => onSelect(e.symbol)} title={`Re-run ${e.symbol}`}>
            {e.symbol}
          </span>
          {e.confidence_pct != null && (
            <span className="chip-conf">{Math.round(e.confidence_pct)}%</span>
          )}
          <button className="watchlist-chip-remove" onClick={() => onRemove(e.symbol)} title="Remove">×</button>
        </span>
      ))}
    </div>
  )
}
