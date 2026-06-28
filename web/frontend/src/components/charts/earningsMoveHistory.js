// Pure transform for the historical-earnings-move panel. Kept React-free so it
// can be unit-tested under node --test. Takes the backend's dated per-event
// history (chronological, oldest→newest) plus the current implied move, and
// returns recharts-ready rows + the implied-move reference value.

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// "2024-05-02" -> "May '24". Parsed manually (no Date) to avoid TZ/locale drift.
export function shortDateLabel(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(iso || ''))
  if (!m) return String(iso || '')
  return `${MONTHS[Number(m[2]) - 1] || m[2]} '${m[1].slice(2)}`
}

export function buildEarningsMoveHistory(history, impliedMovePct, limit = 12) {
  const rows = Array.isArray(history) ? history : []
  const data = rows
    .slice(-limit) // most-recent N, preserving chronological order
    .map((e) => {
      const raw = e == null ? null : e.move_pct
      if (raw == null) return null // guard: Number(null) === 0 would slip through
      const move = Number(raw)
      if (!Number.isFinite(move)) return null
      return {
        date: e.date,
        label: shortDateLabel(e.date),
        move: Number(move.toFixed(2)),
        timing: e.release_timing || 'unknown',
      }
    })
    .filter(Boolean)

  // Guard null explicitly — Number(null) === 0 would render a bogus 0% reference.
  const impliedMove =
    impliedMovePct == null || !Number.isFinite(Number(impliedMovePct))
      ? null
      : Number(Number(impliedMovePct).toFixed(2))

  // How often the stock actually moved at least as much as the market is now
  // pricing — the headline "is this priced cheap?" read. Null when no implied.
  let exceedRate = null
  if (impliedMove != null && data.length) {
    const exceed = data.filter((d) => d.move >= impliedMove).length
    exceedRate = exceed / data.length
  }

  return { data, impliedMove, exceedRate }
}
