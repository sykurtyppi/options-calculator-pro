import React from 'react'

const NA = '—'

function fmt(value, digits = 2, suffix = '') {
  if (value == null || value === '') return NA
  const num = Number(value)
  if (!Number.isFinite(num)) return NA
  return `${num.toFixed(digits)}${suffix}`
}

function scoreBar(score) {
  if (score == null) return null
  const pct = Math.round(Math.min(Math.max(Number(score), 0), 1) * 100)
  const color = pct >= 60 ? '#2ea043' : pct >= 35 ? '#d29922' : '#da3633'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div
        style={{
          width: 48,
          height: 6,
          background: '#21262d',
          borderRadius: 3,
          overflow: 'hidden',
        }}
      >
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span style={{ color, fontVariantNumeric: 'tabular-nums' }}>{(Number(score)).toFixed(2)}</span>
    </div>
  )
}

function releaseBadge(timing) {
  const style = {
    fontSize: '0.7rem',
    padding: '1px 5px',
    borderRadius: 3,
    fontWeight: 600,
    letterSpacing: '0.03em',
    background: timing === 'BMO' ? '#1c3a5e' : timing === 'AMC' ? '#2d2a1e' : '#21262d',
    color: timing === 'BMO' ? '#79c0ff' : timing === 'AMC' ? '#e3b341' : '#8b949e',
  }
  return <span style={style}>{timing || '?'}</span>
}

const COLS = [
  { key: 'rank', label: '#', width: 32, align: 'right', tip: 'Rank by setup score (upcoming rows sort after scored ones).' },
  { key: 'symbol', label: 'Symbol', width: 70, tip: 'Underlying ticker. Click a row to open the full analysis.' },
  { key: 'earnings_date', label: 'Earnings', width: 90, tip: 'Next earnings report date.' },
  { key: 'dte', label: 'DTE', width: 44, align: 'right', tip: 'Days to earnings — calendar days until the report.' },
  { key: 'release_timing', label: 'Rel', width: 48, align: 'center', tip: 'Release timing: BMO = before market open, AMC = after market close.' },
  { key: 'iv_rv_ratio', label: 'IV/RV', width: 60, align: 'right', tip: 'Implied vol ÷ realized vol. Below 1.0 = options look cheap vs the stock’s recent actual movement.' },
  { key: 'atm_iv', label: 'ATM IV', width: 66, align: 'right', tip: 'At-the-money implied volatility — the move priced into front-month options.' },
  { key: 'ts_ratio', label: 'TS', width: 56, align: 'right', tip: 'Term-structure ratio: front-month IV ÷ back-month IV. Steeper = more vol concentrated in the event.' },
  { key: 'median_earnings_move_pct', label: 'Med Move', width: 76, align: 'right', tip: 'Median absolute stock move on past earnings days.' },
  { key: 'sample_size', label: 'N', width: 36, align: 'right', tip: 'Sample size — number of past earnings events behind the stats. More = more reliable.' },
  { key: 'ranking_score', label: 'Setup Score', width: 130, tip: 'Composite setup-quality ranking (0–1). Ordering context only — NOT a calibrated win rate.' },
]

function upcomingPill() {
  return (
    <span
      style={{
        fontSize: '0.72rem',
        color: '#8b949e',
        fontStyle: 'italic',
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
      }}
    >
      <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#58a6ff', display: 'inline-block' }} />
      upcoming
    </span>
  )
}

export default function RankedSetupTable({ rows, selectedSymbol, onSelect }) {
  if (!rows || rows.length === 0) {
    return (
      <div style={{ padding: '20px 0', color: '#8b949e', fontSize: '0.85rem', textAlign: 'center', lineHeight: 1.5 }}>
        No upcoming earnings found in the look-ahead window.
        <br />
        <span style={{ fontSize: '0.78rem', color: '#6e7681' }}>
          Try increasing <strong>Weeks</strong> or widening the <strong>DTE</strong> range.
        </span>
      </div>
    )
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
        <thead>
          <tr>
            {COLS.map((col) => (
              <th
                key={col.key}
                title={col.tip}
                style={{
                  padding: '5px 8px',
                  textAlign: col.align || 'left',
                  color: '#8b949e',
                  fontWeight: 500,
                  borderBottom: '1px solid #21262d',
                  whiteSpace: 'nowrap',
                  minWidth: col.width,
                  cursor: col.tip ? 'help' : 'default',
                }}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const isSelected = row.symbol === selectedSymbol
            const isError = row.status === 'error'
            const isNoEvent = row.status === 'no_earnings'
            const isUpcoming = row.status === 'upcoming'
            const isInert = isError || isNoEvent
            return (
              <tr
                key={`${row.symbol}-${row.earnings_date}`}
                onClick={() => !isInert && onSelect && onSelect(row)}
                style={{
                  cursor: isInert ? 'default' : 'pointer',
                  background: isSelected ? '#1c3a5e22' : 'transparent',
                  opacity: isInert ? 0.45 : isUpcoming ? 0.72 : 1,
                }}
              >
                <td style={{ padding: '5px 8px', textAlign: 'right', color: '#8b949e' }}>{row.rank}</td>
                <td style={{ padding: '5px 8px', fontWeight: isSelected ? 600 : 400, color: '#e6edf3' }}>
                  {row.symbol}
                </td>
                <td style={{ padding: '5px 8px', color: '#c9d1d9', whiteSpace: 'nowrap' }}>
                  {row.earnings_date || NA}
                </td>
                <td style={{ padding: '5px 8px', textAlign: 'right', color: '#c9d1d9' }}>
                  {row.dte != null ? row.dte : NA}
                </td>
                <td style={{ padding: '5px 8px', textAlign: 'center' }}>
                  {releaseBadge(row.release_timing)}
                </td>
                <td style={{ padding: '5px 8px', textAlign: 'right', color: row.iv_rv_ratio != null && row.iv_rv_ratio < 1.0 ? '#2ea043' : '#c9d1d9' }}>
                  {fmt(row.iv_rv_ratio)}
                </td>
                <td style={{ padding: '5px 8px', textAlign: 'right', color: '#c9d1d9' }}>
                  {fmt(row.atm_iv, 1, '%')}
                </td>
                <td style={{ padding: '5px 8px', textAlign: 'right', color: row.ts_ratio != null && row.ts_ratio < 1.0 ? '#2ea043' : '#c9d1d9' }}>
                  {fmt(row.ts_ratio)}
                </td>
                <td style={{ padding: '5px 8px', textAlign: 'right', color: '#c9d1d9' }}>
                  {fmt(row.median_earnings_move_pct, 1, '%')}
                </td>
                <td style={{ padding: '5px 8px', textAlign: 'right', color: '#8b949e' }}>
                  {row.sample_size ?? NA}
                </td>
                <td style={{ padding: '5px 8px' }}>
                  {isError
                    ? <span style={{ color: '#da3633', fontSize: '0.75rem' }}>{row.error_note || 'error'}</span>
                    : isNoEvent
                    ? <span style={{ color: '#8b949e', fontSize: '0.75rem' }}>no event in window</span>
                    : isUpcoming
                    ? upcomingPill()
                    : scoreBar(row.ranking_score)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
