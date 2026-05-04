export function fmtPct(v) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return `${(Number(v) * 100).toFixed(2)}%`
}

export function fmtNum(v, d = 3) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return Number(v).toFixed(d)
}

export function fmtMoney(v) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return `$${Number(v).toFixed(2)}`
}

export function fmtPp(v, d = 2) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return `${Number(v).toFixed(d)}%`
}

export function fmtVol(v, d = 1) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  return `${(Number(v) * 100).toFixed(d)}%`
}

export function fmtSpp(v, d = 2) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  const n = Number(v)
  return `${n > 0 ? '+' : ''}${n.toFixed(d)}%`
}

export function fmtSn(v, d = 2) {
  if (v == null || Number.isNaN(Number(v))) return 'n/a'
  const n = Number(v)
  return `${n > 0 ? '+' : ''}${n.toFixed(d)}`
}

export function tonePos(v, good = 0, warn = 0) {
  if (v == null || Number.isNaN(Number(v))) return 'default'
  const n = Number(v)
  if (n >= good) return 'good'
  if (n > warn) return 'warn'
  return 'bad'
}

export function toneNeg(v, good = 1.25, warn = 2.0) {
  if (v == null || Number.isNaN(Number(v))) return 'default'
  const n = Number(v)
  if (n <= good) return 'good'
  if (n <= warn) return 'warn'
  return 'bad'
}

export function parseIntOr(v, fb) {
  const p = parseInt(v, 10)
  return Number.isFinite(p) ? p : fb
}

export function parseFloatOr(v, fb) {
  const p = parseFloat(v)
  return Number.isFinite(p) ? p : fb
}

export function formatTimestamp(ts) {
  if (!ts) return 'n/a'
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return String(ts)
  return d.toLocaleString()
}
