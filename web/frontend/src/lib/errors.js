// Plain-English messages for fetch failures.
//
// The backend already returns a meaningful `detail` for most 4xx/5xx (rate
// limits, sanitized internal errors), so we prefer that. These helpers cover
// the gaps that otherwise surfaced as raw strings to the user: a response with
// no detail (bare "HTTP 500") and — the common one — a network failure, where
// fetch rejects with a TypeError and there is no response at all.

const STATUS_MESSAGES = {
  429: 'Too many requests right now — the market-data provider is rate-limited. Wait a minute and try again.',
  500: "The server hit an error processing this request. It's usually transient — try again in a moment.",
  502: 'The server is temporarily unavailable (bad gateway). Try again shortly.',
  503: 'The service is temporarily unavailable. Try again shortly.',
  504: 'The request timed out upstream. Try again in a moment.',
}

// Message for a failed HTTP *response*. Prefer a meaningful server-provided
// detail; otherwise map the status code to plain English.
export function httpErrorMessage(status, detail) {
  const d = typeof detail === 'string' ? detail.trim() : ''
  if (d) return d
  return STATUS_MESSAGES[status] || `Something went wrong (HTTP ${status}). Please try again.`
}

// Message for any error thrown in a fetch flow. A network failure rejects with
// a TypeError and carries no HTTP status — give a connectivity hint rather than
// the browser's raw "Failed to fetch". Errors we already built with
// httpErrorMessage() pass through unchanged.
export function fetchErrorMessage(err) {
  if (err && err.name === 'TypeError') {
    return "Couldn't reach the server — check your connection and try again."
  }
  const msg = err && err.message ? String(err.message) : String(err || '')
  return msg.trim() || 'Something went wrong. Please try again.'
}
