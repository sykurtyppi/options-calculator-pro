// Vitest + Testing Library setup for rendered-component tests.
// - jest-dom adds DOM matchers (toBeInTheDocument, toHaveTextContent, ...).
// - afterEach(cleanup) unmounts React trees between tests so they don't leak
//   into each other's queries.
import '@testing-library/jest-dom/vitest'
import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

afterEach(() => {
  cleanup()
})
