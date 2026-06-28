import React from 'react'

// Shimmer placeholder shown while an analysis is in flight. The analyze call
// hits a live data provider and can take a few seconds; without this the result
// area is blank (the empty-state hides on loading), which reads as "did it
// break?". The skeleton roughly mirrors the Decision card layout so the wait
// feels like loading, not nothing. Purely presentational, no props.
export default function AnalysisSkeleton() {
  return (
    <div className="analysis-skeleton" role="status" aria-live="polite" aria-busy="true">
      <span className="sr-only">Running analysis…</span>
      <div className="skel skel-tabs" />
      <div className="skel-card">
        <div className="skel skel-line skel-line-lg" />
        <div className="skel skel-line skel-line-md" />
        <div className="skel-stats">
          <div className="skel skel-stat" />
          <div className="skel skel-stat" />
          <div className="skel skel-stat" />
        </div>
        <div className="skel skel-block" />
      </div>
    </div>
  )
}
