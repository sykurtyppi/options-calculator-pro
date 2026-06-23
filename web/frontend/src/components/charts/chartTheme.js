// Single source of truth for recharts chart colors.
//
// recharts has no theming API and passes colors as SVG fill/stroke props, where
// CSS var() can break color interpolation — so these are literal values that
// MIRROR the design tokens in src/design-system.css. Keep them in sync with that
// file (axis = --muted, grid/border = --line, series = --pos/--neg/--accent-2/--warn).
export const CHART = {
  axis: '#8ea4b7', // --muted (axis ticks + labels)
  axisDim: '#6b7f93', // --muted-dim
  grid: '#27445c', // --line (gridlines + tooltip border)
  text: '#e6f0f8', // --text (tooltip label, emphasis)
  tooltipBg: '#0a1826', // --surface-sunken
  series: {
    pos: '#2ea043', // --pos
    neg: '#da3633', // --neg
    accent: '#39a0ff', // --accent-2
    warn: '#f0a020', // --warn
    posFill: 'rgba(46, 160, 67, 0.65)', // translucent --pos for bars
    negFill: 'rgba(218, 54, 51, 0.65)', // translucent --neg for bars
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
