// Pure-function tests for the payoff-chart zoom domain math (node --test).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { computeZoomDomains } from './usePayoffZoom.js'

const KEYS = ['expand', 'flat', 'crush25', 'crush45']
const DATA = [
  { move: -20, expand: -4.0, flat: -4.0, crush25: -4.0, crush45: -4.19 },
  { move: -1, expand: 0.2, flat: 0.1, crush25: 0.0, crush45: -0.1 },
  { move: 0, expand: 1.19, flat: 1.0, crush25: 0.8, crush45: 0.6 },
  { move: 1, expand: 0.3, flat: 0.2, crush25: 0.1, crush45: 0.0 },
  { move: 20, expand: -3.0, flat: -3.0, crush25: -3.0, crush45: -4.0 },
]

test('no bounds → full x domain and y spans all rows', () => {
  const { xDomain, yDomain } = computeZoomDomains(DATA, KEYS, null)
  assert.deepEqual(xDomain, ['dataMin', 'dataMax'])
  assert.ok(yDomain[0] < -4.19 && yDomain[1] > 1.19, 'y covers global min/max with padding')
})

test('zoom into the near-0% window rescales Y much tighter (the whole point)', () => {
  const full = computeZoomDomains(DATA, KEYS, null)
  const zoomed = computeZoomDomains(DATA, KEYS, { x1: -1, x2: 1 })
  assert.deepEqual(zoomed.xDomain, [-1, 1])
  const fullSpan = full.yDomain[1] - full.yDomain[0]
  const zoomSpan = zoomed.yDomain[1] - zoomed.yDomain[0]
  assert.ok(zoomSpan < fullSpan / 2, `zoomed y-span ${zoomSpan} should be far tighter than full ${fullSpan}`)
  // visible rows in [-1,1] are the 3 middle rows; min y = -0.1, max y = 1.19
  assert.ok(zoomed.yDomain[0] <= -0.1 && zoomed.yDomain[1] >= 1.19)
  assert.ok(zoomed.yDomain[0] > -4, 'the deep -4.19 tail is excluded when zoomed in')
})

test('reversed bounds are normalized (drag right-to-left)', () => {
  const a = computeZoomDomains(DATA, KEYS, { x1: 1, x2: -1 })
  const b = computeZoomDomains(DATA, KEYS, { x1: -1, x2: 1 })
  assert.deepEqual(a.xDomain, [-1, 1])
  assert.deepEqual(a.yDomain, b.yDomain)
})

test('degenerate (x1===x2) is treated as no zoom', () => {
  const { xDomain } = computeZoomDomains(DATA, KEYS, { x1: 0, x2: 0 })
  assert.deepEqual(xDomain, ['dataMin', 'dataMax'])
})

test('empty / no finite y values → auto y domain, no throw', () => {
  assert.deepEqual(computeZoomDomains([], KEYS, null).yDomain, ['auto', 'auto'])
  const window = computeZoomDomains(DATA, KEYS, { x1: 5, x2: 6 }) // no rows in window
  assert.deepEqual(window.yDomain, ['auto', 'auto'])
})
