// Single source of truth for recharts chart colors.
//
// recharts has no theming API and passes colors as SVG fill/stroke props, where
// CSS var() can break color interpolation — so these are literal values that
// MIRROR the design tokens in src/design-system.css. Keep them in sync with that
// file (axis = --muted, grid/border = --line, series = --pos/--neg/--accent-2/--warn).
export const CHART = {
  axis: '#959ba2', // --muted (axis ticks + labels)
  axisDim: '#6c747c', // --muted-dim
  grid: '#2d3238', // --line (gridlines + tooltip border)
  text: '#eaecee', // --text (tooltip label, emphasis)
  tooltipBg: '#181b1f', // --panel (raised tooltip)
  series: {
    pos: '#46a07a', // --pos
    neg: '#d36450', // --neg
    accent: '#4f86e8', // --accent
    warn: '#e8943a', // --warn
    posFill: 'rgba(70, 160, 122, 0.55)', // translucent --pos for bars
    negFill: 'rgba(211, 100, 80, 0.55)', // translucent --neg for bars
  },
}

// Ready-made prop objects for the common recharts elements, so every chart
// renders identical axis/grid/tooltip chrome.
export const axisTick = { fill: CHART.axis, fontSize: 11 }

export const tooltipContentStyle = {
  background: CHART.tooltipBg,
  border: `1px solid ${CHART.grid}`,
  borderRadius: 8,
  color: CHART.text,
  fontSize: '0.8rem',
}
