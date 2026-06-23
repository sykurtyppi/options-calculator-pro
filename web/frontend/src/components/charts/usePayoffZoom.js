import { useState } from 'react'

// Pure core: given the chart rows, the y-series keys, and an optional committed
// {x1,x2} zoom window, return the x and y domains to feed recharts. When zoomed,
// Y rescales to the visible x-window — so dragging into the cramped near-0% peak
// of a payoff curve also expands it vertically, which is the whole point. Kept
// pure (no React) so it can be unit-tested under node --test.
export function computeZoomDomains(data, seriesKeys, bounds, xKey = 'move') {
  const rows = Array.isArray(data) ? data : []
  let x1 = null
  let x2 = null
  if (bounds && bounds.x1 != null && bounds.x2 != null) {
    x1 = Number(bounds.x1)
    x2 = Number(bounds.x2)
    if (x1 > x2) [x1, x2] = [x2, x1]
  }
  const zoomed = x1 != null && x2 != null && x1 !== x2
  const visible = zoomed
    ? rows.filter((d) => Number(d[xKey]) >= x1 && Number(d[xKey]) <= x2)
    : rows
  const xDomain = zoomed ? [x1, x2] : ['dataMin', 'dataMax']

  const ys = visible
    .flatMap((d) => seriesKeys.map((k) => Number(d[k])))
    .filter((v) => Number.isFinite(v))
  if (!ys.length) return { xDomain, yDomain: ['auto', 'auto'] }

  const lo = Math.min(...ys)
  const hi = Math.max(...ys)
  const pad = Math.max((hi - lo) * 0.14, 0.01)
  return {
    xDomain,
    yDomain: [Number((lo - pad).toFixed(3)), Number((hi + pad).toFixed(3))],
  }
}

// Drag-to-zoom interaction for the payoff charts. Returns the live domains, the
// in-progress selection rectangle (for a ReferenceArea), and the recharts mouse
// handlers. Handlers are redefined each render so they close over fresh `sel`
// (no stale-closure bug, negligible cost for a chart).
export function usePayoffZoom(data, seriesKeys, xKey = 'move') {
  const [bounds, setBounds] = useState(null) // committed zoom window
  const [sel, setSel] = useState(null) // in-progress drag selection {x1,x2}

  function onMouseDown(e) {
    if (e && e.activeLabel != null) setSel({ x1: e.activeLabel, x2: e.activeLabel })
  }
  function onMouseMove(e) {
    if (sel && e && e.activeLabel != null) setSel({ x1: sel.x1, x2: e.activeLabel })
  }
  function onMouseUp() {
    if (sel && sel.x1 != null && sel.x2 != null && Number(sel.x1) !== Number(sel.x2)) {
      setBounds({ x1: Number(sel.x1), x2: Number(sel.x2) })
    }
    setSel(null)
  }
  function onMouseLeave() {
    // Drag that ends off the chart never gets a mouseUp — drop the in-progress
    // selection so a dangling rectangle can't linger.
    if (sel) setSel(null)
  }
  function reset() {
    setBounds(null)
    setSel(null)
  }

  const { xDomain, yDomain } = computeZoomDomains(data, seriesKeys, bounds, xKey)
  return {
    xDomain,
    yDomain,
    sel,
    zoomed: !!bounds,
    reset,
    handlers: { onMouseDown, onMouseMove, onMouseUp, onMouseLeave, onDoubleClick: reset },
  }
}
