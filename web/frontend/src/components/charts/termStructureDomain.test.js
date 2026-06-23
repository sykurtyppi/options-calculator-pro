// Pure-function tests for the term-structure y-domain (node --test).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { termStructureYDomain } from './termStructureDomain.js'

test('returns integers, never toFixed strings (the bug that emitted "72.64%")', () => {
  const [lo, hi] = termStructureYDomain(40.13, 72.64)
  assert.equal(typeof lo, 'number')
  assert.equal(typeof hi, 'number')
  assert.equal(lo, Math.trunc(lo), 'min is an integer')
  assert.equal(hi, Math.trunc(hi), 'max is an integer')
})

test('gives headroom above the peak and below the trough', () => {
  const ivMin = 40.13
  const ivMax = 72.64
  const [lo, hi] = termStructureYDomain(ivMin, ivMax)
  assert.ok(hi > ivMax, 'top has headroom so the peak is not clipped')
  assert.ok(lo < ivMin, 'bottom has headroom')
})

test('clamps the floor at 0 (IV % is never negative)', () => {
  const [lo] = termStructureYDomain(0.4, 1.2)
  assert.ok(lo >= 0)
})

test('flat term structure still yields a non-degenerate domain', () => {
  const [lo, hi] = termStructureYDomain(55, 55)
  assert.ok(hi > lo, 'max strictly above min even when ivMin === ivMax')
})
