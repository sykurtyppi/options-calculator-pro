import React, { useEffect, useState } from 'react'
import { apiFetch } from '../../lib/api'
import { buildCalibrationModel } from './selectorViewModel'

export default function CalibrationInsightPanel({ apiBase, score, curveData = null }) {
  const [data, setData] = useState(curveData)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (curveData) {
      setData(curveData)
      return undefined
    }

    let cancelled = false
    setLoading(true)
    setError('')

    apiFetch(`${apiBase}/api/calibration/curve`)
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => ({}))
          throw new Error(body.detail || `HTTP ${response.status}`)
        }
        return response.json()
      })
      .then((payload) => {
        if (!cancelled) setData(payload)
      })
      .catch((err) => {
        if (!cancelled) setError(String(err.message || err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => { cancelled = true }
  }, [apiBase, curveData])

  const model = buildCalibrationModel(data, score)
  if (loading && !model) {
    return (
      <section className="selector-panel calibration-panel">
        <div className="selector-panel-header">
          <h3>Historical Calibration</h3>
          <span>Loading score-to-expansion evidence...</span>
        </div>
      </section>
    )
  }

  if (error && !model) {
    return (
      <section className="selector-panel calibration-panel">
        <div className="selector-panel-header">
          <h3>Historical Calibration</h3>
          <span>Calibration data unavailable: {error}</span>
        </div>
      </section>
    )
  }

  if (!model) return null

  const maxExpansion = Math.max(...model.buckets.map((bucket) => Number(bucket.expected_expansion_pct || 0)), 1)

  return (
    <section className="selector-panel calibration-panel">
      <div className="selector-panel-header">
        <h3>Historical Calibration</h3>
        <span>Shows how much of the calibration is prior, observational, or empirically fitted at the current sample depth.</span>
      </div>

      <div className="calibration-phase-strip">
        <strong>{model.phaseLabel}</strong>
        <span>{model.phaseSummary}</span>
      </div>

      {model.activeSummary && (
        <div className="calibration-callout">
          <div className="calibration-callout-label">Current bucket</div>
          <strong>{model.activeSummary.headline}</strong>
          <p>{model.activeSummary.detail}</p>
          <span>{model.activeSummary.evidenceLabel}</span>
        </div>
      )}

      {model.isFitted ? (
        <div className="calibration-bars">
          {model.buckets.map((bucket) => (
            <div className={`calibration-bar-item${bucket.isActive ? ' active' : ''}`} key={bucket.key}>
              <div
                className="calibration-bar-fill"
                style={{ height: `${Math.max((Number(bucket.expected_expansion_pct || 0) / maxExpansion) * 100, 6)}%` }}
                title={`${bucket.rangeLabel}: ${bucket.expectedExpansionLabel} avg IV expansion`}
              />
              <strong>{bucket.expectedExpansionLabel}</strong>
              <span>{bucket.rangeLabel}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="calibration-observational-list">
          {model.buckets.map((bucket) => (
            <div className={`calibration-observational-row${bucket.isActive ? ' active' : ''}`} key={bucket.key}>
              <strong>{bucket.rangeLabel}</strong>
              <span>{bucket.expectedExpansionLabel}</span>
              <em>{bucket.n > 0 ? `Bucket N=${bucket.n}` : 'Research prior'}</em>
            </div>
          ))}
        </div>
      )}

      <div className="calibration-footer">
        <span>
          {model.isPrior
            ? `Prior only · ${model.nObservations}/${model.minForObservational} observations before observational mode`
            : model.isObservational
              ? `Observational mode · N=${model.nObservations} · fitted curve held back until N=${model.minForFit}`
              : `${model.phaseLabel} · N=${model.nObservations}`}
        </span>
        <span>Use this as historical context, not as a promise of future IV expansion.</span>
      </div>
    </section>
  )
}
