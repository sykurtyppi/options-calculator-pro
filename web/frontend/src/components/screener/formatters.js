export function formatDate(value) {
  if (!value) return 'n/a'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return String(value)
  return parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function formatTimestamp(value) {
  if (!value) return 'n/a'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return 'n/a'
  return parsed.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

export function formatPct(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  return `${Number(value).toFixed(digits)}%`
}

export function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  return Number(value).toFixed(digits)
}

export function formatMoney(value) {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  return `$${Number(value).toFixed(2)}`
}

export function formatOi(callOi, putOi) {
  const callValue = callOi ?? 'n/a'
  const putValue = putOi ?? 'n/a'
  return `${callValue}/${putValue}`
}

export function expiryModeLabel(mode) {
  if (mode === 'next_monthly_opex') return 'Next Monthly OPEX'
  return 'Front Expiry After Earnings'
}
