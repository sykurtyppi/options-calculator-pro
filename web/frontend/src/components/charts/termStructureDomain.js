// Pure: clean integer y-domain [min, max] for the term-structure chart.
//
// Integer bounds (via floor/ceil) — NOT `.toFixed()` strings. A string domain
// endpoint made recharts emit the raw padded value (e.g. "72.64%") as the top
// tick instead of a nice round number, and it under-sized the top so the peak
// and the top tick clipped. Always returns two finite numbers with the max
// strictly above the min (pad floor of 0.5 guards a flat term structure).
export function termStructureYDomain(ivMin, ivMax) {
  const pad = Math.max((ivMax - ivMin) * 0.25, 0.5)
  return [Math.max(0, Math.floor(ivMin - pad)), Math.ceil(ivMax + pad)]
}
