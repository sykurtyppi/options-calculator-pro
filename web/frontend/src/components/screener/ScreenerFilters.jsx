import React from 'react'

export default function ScreenerFilters({
  filters,
  onFilterChange,
  sortKey,
  sortDirection,
  onSortKeyChange,
  onSortDirectionChange,
}) {
  return (
    <div className="screener-filters">
      <div className="filter-grid">
        <label>
          <span>Symbol</span>
          <input
            type="text"
            value={filters.query}
            onChange={(event) => onFilterChange('query', event.target.value.toUpperCase())}
            placeholder="AAPL"
          />
        </label>

        <label>
          <span>Release</span>
          <select value={filters.releaseTiming} onChange={(event) => onFilterChange('releaseTiming', event.target.value)}>
            <option value="ALL">All</option>
            <option value="AMC">AMC</option>
            <option value="BMO">BMO</option>
            <option value="UNKNOWN">Unknown</option>
          </select>
        </label>

        <label>
          <span>Status</span>
          <select value={filters.status} onChange={(event) => onFilterChange('status', event.target.value)}>
            <option value="ALL">All</option>
            <option value="QUALIFIED">Qualified</option>
            <option value="MARGINAL">Marginal</option>
            <option value="EXCLUDED">Excluded</option>
          </select>
        </label>

        <label>
          <span>Start date</span>
          <input
            type="date"
            value={filters.startDate}
            onChange={(event) => onFilterChange('startDate', event.target.value)}
          />
        </label>

        <label>
          <span>End date</span>
          <input
            type="date"
            value={filters.endDate}
            onChange={(event) => onFilterChange('endDate', event.target.value)}
          />
        </label>

        <label>
          <span>Max spread %</span>
          <input
            type="number"
            min="0"
            step="0.1"
            value={filters.maxSpread}
            onChange={(event) => onFilterChange('maxSpread', event.target.value)}
            placeholder="5"
          />
        </label>

        <label>
          <span>Min OI</span>
          <input
            type="number"
            min="0"
            step="50"
            value={filters.minOi}
            onChange={(event) => onFilterChange('minOi', event.target.value)}
            placeholder="100"
          />
        </label>

        <label>
          <span>Sort by</span>
          <select value={sortKey} onChange={(event) => onSortKeyChange(event.target.value)}>
            <option value="status">Qualification</option>
            <option value="avg_spread_pct">Avg spread</option>
            <option value="implied_move_pct">Implied move</option>
            <option value="earnings_date">Earnings date</option>
            <option value="symbol">Symbol</option>
          </select>
        </label>

        <label>
          <span>Direction</span>
          <select value={sortDirection} onChange={(event) => onSortDirectionChange(event.target.value)}>
            <option value="asc">Ascending</option>
            <option value="desc">Descending</option>
          </select>
        </label>
      </div>
    </div>
  )
}
