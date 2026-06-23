import React, { useEffect, useState } from 'react'

import { apiFetch } from '../../lib/api'
const NA = '—'

function fmt(v, digits = 1) {
  if (v == null) return NA
  const n = Number(v)
  return Number.isFinite(n) ? `${n.toFixed(digits)}%` : NA
}

/**
 * CalibrationInsight
 *
 * Shows phase-aware calibration context for the setup_score → IV-expansion map.
 * Sparse phases are shown as prior / observational context rather than as a
 * smooth empirical-looking curve.
 *
 * Props
 * -----
 * apiBase  : string  — base URL of the API (e.g. "http://localhost:8000")
 * score    : number | null — the current symbol's ranking_score (highlighted in chart)
 */
export default function CalibrationInsight({ apiBase, score }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')

    apiFetch(`${apiBase}/api/calibration/curve`)
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}))
          throw new Error(body.detail || `HTTP ${r.status}`)
        }
        return r.json()
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
  }, [apiBase])

  if (loading) {
    return <div style={{ color: 'var(--muted)', fontSize: '0.8rem', padding: '8px 0' }}>Loading calibration…</div>
  }

  if (error) {
    return <div style={{ color: 'var(--neg)', fontSize: '0.8rem', padding: '8px 0' }}>Calibration unavailable: {error}</div>
  }

  if (!data) return null

  const {
    buckets = [],
    phase,
    n_observations,
    min_for_observational = 40,
    min_for_fit,
  } = data
  const isPrior = phase === 'bootstrap_prior'
  const isObservational = phase === 'observational'
  const isFitted = phase === 'fitted_moderate' || phase === 'fitted_high'

  // Find the bucket containing the current score
  const activeBucket = score != null
    ? buckets.find((b) => score >= b.score_lo && score < b.score_hi) || buckets[buckets.length - 1]
    : null

  // Max expansion for bar scaling
  const maxExp = Math.max(...buckets.map((b) => b.expected_expansion_pct), 1)

  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          IV Expansion Calibration
        </span>
        <span
          style={{
            fontSize: '0.68rem',
            padding: '1px 5px',
            borderRadius: 3,
            background: isPrior ? 'var(--warn-surface)' : 'var(--accent-2-surface)',
            color: isPrior ? 'var(--warn-bright)' : 'var(--accent-2-bright)',
            fontWeight: 600,
          }}
        >
          {isPrior
            ? `Prior · ${n_observations}/${min_for_observational} obs`
            : isObservational
              ? `Observational · N=${n_observations}`
              : `Fitted · N=${n_observations}`}
        </span>
      </div>

      {isPrior && (
        <p style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: 8, lineHeight: 1.4 }}>
          Research prior only. This is ordering context, not an empirical estimate.
        </p>
      )}

      {isObservational && (
        <p style={{ fontSize: '0.72rem', color: 'var(--muted)', marginBottom: 8, lineHeight: 1.4 }}>
          Raw bucket observations are being accumulated, but the fitted curve is held back until {min_for_fit} observations.
        </p>
      )}

      {isFitted ? (
        <>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 48, marginBottom: 4 }}>
            {buckets.map((b) => {
              const isActive = activeBucket && b.score_lo === activeBucket.score_lo
              const height = Math.round((b.expected_expansion_pct / maxExp) * 44)
              return (
                <div
                  key={b.score_lo}
                  title={`Score ${b.score_lo.toFixed(1)}–${b.score_hi.toFixed(1)}: ~${b.expected_expansion_pct.toFixed(1)}% IV expansion`}
                  style={{
                    flex: 1,
                    height: `${Math.max(height, 2)}px`,
                    background: isActive ? 'var(--accent-2)' : 'var(--line)',
                    borderRadius: '2px 2px 0 0',
                    transition: 'height 0.2s',
                    outline: isActive ? '1px solid var(--accent-2)' : 'none',
                  }}
                />
              )
            })}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--muted-dim)', marginBottom: 6 }}>
            <span>0.0</span>
            <span>Score →</span>
            <span>1.0</span>
          </div>
        </>
      ) : (
        <div style={{ display: 'grid', gap: 4, marginBottom: 8 }}>
          {buckets.map((b) => {
            const isActive = activeBucket && b.score_lo === activeBucket.score_lo
            return (
              <div
                key={b.score_lo}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr auto auto',
                  gap: 8,
                  padding: '4px 8px',
                  borderRadius: 6,
                  border: isActive ? '1px solid var(--accent-2)' : '1px solid var(--line)',
                  background: isActive ? 'var(--row-selected-bg)' : 'var(--surface-sunken)',
                }}
              >
                <strong>{b.score_lo.toFixed(1)}–{b.score_hi.toFixed(1)}</strong>
                <span>{fmt(b.expected_expansion_pct)}</span>
                <span style={{ color: 'var(--muted)' }}>{b.n > 0 ? `N=${b.n}` : 'Prior'}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Active bucket callout */}
      {activeBucket && (
        <div
          style={{
            padding: '6px 10px',
            background: 'var(--surface-sunken)',
            border: '1px solid var(--line)',
            borderRadius: 6,
            fontSize: '0.78rem',
            color: 'var(--text-secondary)',
          }}
        >
          <span style={{ color: 'var(--muted)' }}>Score {activeBucket.score_lo.toFixed(1)}–{activeBucket.score_hi.toFixed(1)}: </span>
          <strong style={{ color: 'var(--accent-2)' }}>
            ~{fmt(activeBucket.expected_expansion_pct)} {isFitted ? 'empirical IV expansion' : 'context estimate'}
          </strong>
          <span style={{ color: 'var(--muted)' }}>
            {' '}(±{fmt(activeBucket.std_pct)}{activeBucket.n > 0 ? `, N=${activeBucket.n}` : ', prior'})
          </span>
        </div>
      )}
    </div>
  )
}
