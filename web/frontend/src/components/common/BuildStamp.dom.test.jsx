import React from 'react'
import { describe, test, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'

const apiFetch = vi.fn()
vi.mock('../../lib/api', () => ({ apiFetch: (...a) => apiFetch(...a) }))

import BuildStamp from './BuildStamp'

const A = 'aaaaaaa1111111111111111111111111111111111'
const B = 'bbbbbbb2222222222222222222222222222222222'
const _resp = (payload) => ({ ok: true, status: 200, json: async () => payload })

describe('BuildStamp', () => {
  beforeEach(() => apiFetch.mockReset())

  test('shows the build and warns when the API runs other code', async () => {
    apiFetch.mockResolvedValue(_resp({ api: { git_sha: B } }))
    render(<BuildStamp apiBase="" build={{ git_sha: A, dirty: false }} />)
    await waitFor(() => expect(screen.getByText(/API runs bbbbbbb/)).toBeInTheDocument())
    expect(apiFetch.mock.calls[0][0]).toBe('/api/version')
    expect(document.body.textContent).toMatch(/Build aaaaaaa/)
  })

  test('stays quiet when the version endpoint is unavailable', async () => {
    apiFetch.mockResolvedValue({ ok: false, status: 500, json: async () => ({}) })
    render(<BuildStamp apiBase="" build={{ git_sha: A, dirty: false }} />)
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    expect(document.body.textContent).toBe('Build aaaaaaa')
  })
})
