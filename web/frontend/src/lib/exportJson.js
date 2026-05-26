import { TODAY_ISO } from '../config/oosConfig'

/**
 * Save an analysis result as a downloadable JSON blob.
 * Filename is suffixed with today's ISO date so repeated exports don't clobber.
 * No-op when result is null/undefined.
 */
export function exportJson(symbol, result) {
  if (!result) return
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${symbol}-edge-${TODAY_ISO}.json`
  a.click()
  URL.revokeObjectURL(url)
}
