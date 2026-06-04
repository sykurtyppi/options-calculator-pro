// Vitest + Testing Library setup for rendered-component tests.
// - jest-dom adds DOM matchers (toBeInTheDocument, toHaveTextContent, ...).
// - afterEach(cleanup) unmounts React trees between tests so they don't leak
//   into each other's queries.
import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

afterEach(() => {
  cleanup()
})

// jsdom gaps that full-page renders (and recharts) rely on. Stubbing them here
// keeps rendered tests from throwing on APIs jsdom doesn't implement.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}
if (typeof globalThis.matchMedia === 'undefined') {
  globalThis.matchMedia = (query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })
}
