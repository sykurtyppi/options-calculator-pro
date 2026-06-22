import React, { useEffect, useState } from 'react'
import { apiFetch } from '../../lib/api'
import { buildForwardPerformanceSummary } from '../diagnostics/forwardPerformanceViewModel'

/**
 * Compact "live forward evidence" strip for the Decision view.
 *
 * The forward paper-trade collector is the project's value story — the
 * out-of-sample proof that the backtested edge survives live. It was buried in
 * the diagnostics tail; this surfaces it next to the decision. Strategy-level
 * (not per-ticker): "here is how the validated pocket is doing live."
 *
 * Fails quiet (renders nothing) on error so it never clutters the decision.
 */
export default function ForwardEvidenceStrip({ apiBase }) {
  const [summary, setSummary] = useState(null)

  useEffect(() => {
    const controller = new AbortController()
    let ignore = false  // guard setState after unmount (abort only stops the fetch)
    apiFetch(`${apiBase}/api/diagnostics/forward-performance`, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('forward-performance unavailable'))))
      .then((payload) => { if (!ignore) setSummary(buildForwardPerformanceSummary(payload || {})) })
      .catch((e) => { if (!ignore && e?.name !== 'AbortError') setSummary(null) })
    return () => { ignore = true; controller.abort() }
  }, [apiBase])

  if (!summary) return null

  // Accruing if no resolved outcomes OR resolved outcomes exist without usable
  // summary stats (the counts and stats come from different backend paths).
  const accruing = summary.resolvedCount === 0 || summary.winRateLabel === 'n/a'

  return (
    <div
      className="forward-evidence-strip"
      data-accruing={accruing ? 'true' : 'false'}
      title="A paper trade is a simulated position we track but never actually place — it measures the signal’s real out-of-sample performance without risking money."
    >
      <div className="fe-label">
        <span className="fe-dot" aria-hidden="true" />
        Live forward evidence
      </div>
      {accruing ? (
        <div className="fe-body fe-accruing">
          We paper-trade every qualifying signal forward and publish the real out-of-sample
          results here — wins and losses alike. Nothing has resolved yet: qualifying setups are
          rare (~10–40/year){summary.openCount > 0 ? ` · ${summary.openCount} open now` : ''}.
        </div>
      ) : (
        <div className="fe-body">
          <span><strong>{summary.resolvedCount}</strong> resolved</span>
          <span><strong>{summary.winRateLabel}</strong> win</span>
          <span>realized <strong>{summary.avgRealizedLabel}</strong></span>
          {summary.openCount > 0 && <span>{summary.openCount} open</span>}
          <span className="fe-tag">paper · not execution-grade</span>
        </div>
      )}
    </div>
  )
}
