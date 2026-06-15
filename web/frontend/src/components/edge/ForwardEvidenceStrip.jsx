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
    apiFetch(`${apiBase}/api/diagnostics/forward-performance`, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('forward-performance unavailable'))))
      .then((payload) => setSummary(buildForwardPerformanceSummary(payload || {})))
      .catch((e) => { if (e?.name !== 'AbortError') setSummary(null) })
    return () => controller.abort()
  }, [apiBase])

  if (!summary) return null

  const accruing = summary.resolvedCount === 0

  return (
    <div className="forward-evidence-strip" data-accruing={accruing ? 'true' : 'false'}>
      <div className="fe-label">
        <span className="fe-dot" aria-hidden="true" />
        Live forward evidence
      </div>
      {accruing ? (
        <div className="fe-body fe-accruing">
          Collecting — no resolved paper trades yet
          {summary.openCount > 0 ? ` · ${summary.openCount} open` : ''}. The validated pocket
          qualifies only ~10–40×/yr; this is the out-of-sample proof as it lands.
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
