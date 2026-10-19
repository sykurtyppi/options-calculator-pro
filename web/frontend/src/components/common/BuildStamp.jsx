import React, { useEffect, useState } from 'react'
import { apiFetch } from '../../lib/api'
import { describeBuild } from './buildStamp'

/* global __OCP_BUILD__ */
const FRONTEND_BUILD = typeof __OCP_BUILD__ !== 'undefined' ? __OCP_BUILD__ : null

/**
 * Footer line naming the commit this page was built from and, when the API
 * reports a different one, saying so: a stale dist/ served by newer code
 * (or the reverse) is otherwise invisible.
 */
export default function BuildStamp({ apiBase, build = FRONTEND_BUILD }) {
  const [version, setVersion] = useState(null)

  useEffect(() => {
    const controller = new AbortController()
    let ignore = false
    apiFetch(`${apiBase}/api/version`, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((payload) => { if (!ignore) setVersion(payload) })
      .catch(() => { if (!ignore) setVersion(null) })
    return () => { ignore = true; controller.abort() }
  }, [apiBase])

  const { label, mismatch } = describeBuild(build, version)
  return (
    <span className="build-stamp" data-mismatch={mismatch ? 'true' : 'false'}>
      {label}
      {mismatch && <strong className="build-stamp-warning"> · {mismatch}</strong>}
    </span>
  )
}
