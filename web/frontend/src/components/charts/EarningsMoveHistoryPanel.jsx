import React from 'react'
import EarningsMoveHistoryChart from './EarningsMoveHistoryChart'
import { buildEarningsMoveHistory } from './earningsMoveHistory'

/**
 * Panel around the historical-earnings-move chart. Makes the core value prop
 * concrete: how this stock has ACTUALLY moved on past earnings versus what the
 * market is pricing for the upcoming one. Renders nothing when the provider had
 * no dated per-event history (e.g. the daily-fallback path) — this is an
 * additive evidence panel, not a guaranteed one, so it stays quiet rather than
 * showing an empty frame.
 */
export default function EarningsMoveHistoryPanel({ history, impliedMove, unknownTimingCount }) {
  const { data, impliedMove: implied, exceedRate, median, takeawaySufficient } = buildEarningsMoveHistory(history, impliedMove)
  if (data.length < 1) return null

  const n = data.length
  const excluded = Number(unknownTimingCount) || 0

  return (
    <div className="selector-panel selector-panel-earnings-history">
      <div className="selector-panel-header">
        <h3>Historical Earnings Moves</h3>
        <span>
          What the stock actually did on past earnings days versus the move the market is pricing
          now — the core “are these options cheap or rich?” read.
        </span>
      </div>
      <EarningsMoveHistoryChart data={data} impliedMove={implied} />
      <p className="earnings-history-caption">
        Actual close-to-close moves on the last {n} earnings
        {Number.isFinite(median) ? <> · median <strong>{median.toFixed(1)}%</strong></> : null}
        {implied != null ? <> · implied now <strong>{implied.toFixed(1)}%</strong></> : null}
        {exceedRate != null && (
          <>
            {' '}· moved ≥ implied in <strong>{Math.round(exceedRate * n)}/{n}</strong>
          </>
        )}
        .{' '}
        {exceedRate != null && (
          takeawaySufficient ? (
            <span className="earnings-history-takeaway">
              {exceedRate >= 0.5
                ? 'History suggests the priced move is on the low side.'
                : 'History suggests the priced move is on the high side.'}
            </span>
          ) : (
            <span className="earnings-history-takeaway-insufficient">
              Too few past earnings ({n}) to call the priced move cheap or rich.
            </span>
          )
        )}
        {excluded > 0 && (
          <>
            {' '}
            <span className="earnings-history-takeaway-insufficient">
              {excluded} past {excluded === 1 ? 'event' : 'events'} excluded — release timing
              unknown, so the reaction day can’t be measured reliably.
            </span>
          </>
        )}
      </p>
    </div>
  )
}
