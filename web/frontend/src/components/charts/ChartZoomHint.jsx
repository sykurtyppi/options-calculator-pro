import React from 'react'

// Small affordance under a zoomable chart: a hint while at full view, a reset
// button once zoomed. Shared by the payoff charts.
export default function ChartZoomHint({ zoomed, onReset }) {
  return (
    <div className="chart-zoom-hint">
      {zoomed ? (
        <button type="button" className="chart-zoom-reset" onClick={onReset}>
          Reset zoom
        </button>
      ) : (
        <span>Drag across the chart to zoom · double-click to reset</span>
      )}
    </div>
  )
}
