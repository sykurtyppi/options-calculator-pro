import { useState } from 'react'

export const ALERT_DEFAULTS = {
  min_confidence: 70,
  min_edge: 0.5,
  vol_regime_filter: 'any',
  enabled: true,
}

const LS_KEY = 'alert_config_v1'

/**
 * Alert-threshold config hook backed by localStorage.
 *
 * `setConfig(patch)` merges into existing state — pass only the keys you
 * want to update. Persistence is best-effort; localStorage write errors
 * are swallowed so the in-memory state stays consistent.
 */
export function useAlertConfig() {
  const [config, setConfigState] = useState(() => {
    try {
      return { ...ALERT_DEFAULTS, ...JSON.parse(localStorage.getItem(LS_KEY) || '{}') }
    } catch {
      return ALERT_DEFAULTS
    }
  })

  function setConfig(patch) {
    setConfigState((prev) => {
      const next = { ...prev, ...patch }
      try {
        localStorage.setItem(LS_KEY, JSON.stringify(next))
      } catch {
        /* quota or disabled storage — keep the in-memory copy */
      }
      return next
    })
  }

  return { config, setConfig }
}
