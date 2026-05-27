// Frontend → backend API client.
//
// This module is the SINGLE place the frontend builds API URLs and
// dispatches network requests. Two security-relevant behaviours live
// here, both surfaced in the PR #59 follow-up audit:
//
// 1. Frontend-audit P0 — every fetch sends ``credentials: 'include'``
//    so the session cookie is attached on cross-origin XHR (Vite dev
//    server on :5173 → backend on :8000 are different origins). The
//    backend's CORS middleware sets ``Access-Control-Allow-Credentials:
//    true``, so the contract works end-to-end as long as the fetch
//    opts in. Without this opt-in the cookie is silently dropped and
//    every authed call returns 401 with no recovery path.
//
// 2. Frontend-audit P1 — ``VITE_API_BASE`` is build-time-substituted
//    into the JS bundle. If it ever points at an attacker-controlled
//    host (env-injection at CI build time, deploy misconfig), every
//    fetch — now with ``credentials: 'include'`` — would ship the
//    session cookie there. The validator below refuses any base that
//    would leak cookies over plaintext to a non-loopback host. The
//    allowed shapes are:
//
//      - ``''``                  → same-origin (production default)
//      - ``http://localhost:N``  → dev cross-origin
//      - ``http://127.0.0.1:N``  → dev cross-origin
//      - ``https://<anything>``  → TLS-protected deployment
//
//    Anything else throws at module load (i.e. on first import). The
//    fail-loud-at-import behaviour means a misconfigured build never
//    silently ships cookies to the wrong host — the app simply
//    refuses to mount.

const ALLOWED_LOCALHOST_HOSTNAMES = new Set(['localhost', '127.0.0.1'])

/**
 * Validate and normalize a raw ``VITE_API_BASE`` value.
 *
 * Pure function so it can be unit-tested under plain node `--test`
 * (which has no DOM and no ``import.meta.env``). Throws synchronously
 * on any unsafe input rather than warn-and-fallback — a misconfigured
 * VITE_API_BASE should be detected at first paint, not at first
 * authenticated request.
 *
 * @param {string | undefined | null} raw
 * @returns {string} validated base URL (without trailing slash), or ``''``
 *   for same-origin
 */
export function validateApiBase(raw) {
  if (raw == null || raw === '') return ''
  const trimmed = String(raw).trim()
  if (trimmed === '') return ''
  let url
  try {
    url = new URL(trimmed)
  } catch {
    throw new Error(
      `Invalid VITE_API_BASE: "${raw}" is not a valid URL. ` +
      `Use either an empty value (same-origin), http://localhost:PORT, ` +
      `http://127.0.0.1:PORT, or an https:// URL.`,
    )
  }
  // Strip trailing slash so callers always concatenate ``${API_BASE}/api/...``
  // cleanly without double slashes.
  const normalized = trimmed.replace(/\/+$/, '')
  if (url.protocol === 'https:') return normalized
  if (
    url.protocol === 'http:' &&
    ALLOWED_LOCALHOST_HOSTNAMES.has(url.hostname)
  ) {
    return normalized
  }
  throw new Error(
    `Refusing to use VITE_API_BASE="${raw}": cookies (credentials: 'include') ` +
    `would be sent over plaintext to a non-loopback host. Use https:// in production, ` +
    `or serve the frontend same-origin from the backend (leave VITE_API_BASE unset).`,
  )
}

/**
 * Resolve the raw (pre-validation) API base from a Vite env object.
 *
 * Codex follow-up P1: the previous implementation defaulted to ``''``
 * (same-origin) when ``VITE_API_BASE`` was unset. That's correct for
 * production (frontend served from backend), but broke local dev,
 * where Vite runs on ``:5173`` and the backend on ``:8000`` with no
 * proxy — fetches were going to ``:5173/api/...`` and 404'ing.
 *
 * Contract:
 *   - ``VITE_API_BASE`` set → use it verbatim (will be validated next)
 *   - ``DEV=true`` and no explicit env → ``http://127.0.0.1:8000``
 *     (matches the pre-PR-#60 dev default; backend's documented port)
 *   - otherwise → ``''`` (same-origin; production default)
 *
 * Pure function so it can be unit-tested under node ``--test`` (which
 * has no ``import.meta.env``). Pass a plain object to exercise each
 * branch.
 *
 * @param {Record<string, any>} env
 * @returns {string}
 */
export function resolveRawApiBase(env) {
  const e = env || {}
  if (e.VITE_API_BASE) return String(e.VITE_API_BASE)
  if (e.DEV) return 'http://127.0.0.1:8000'
  return ''
}

// ``import.meta.env`` is a Vite-only API. The optional-chaining +
// fallback keeps this importable from node --test scripts that don't
// go through Vite — the test imports the pure helpers and never has
// to evaluate this module-level constant.
const _env =
  typeof import.meta !== 'undefined' && import.meta.env ? import.meta.env : {}
const RAW_API_BASE = resolveRawApiBase(_env)

export const API_BASE = validateApiBase(RAW_API_BASE)

/**
 * Fetch wrapper that injects ``credentials: 'include'`` so the
 * session cookie is sent on every authenticated request.
 *
 * Callers can still override any other fetch option (method, headers,
 * body, signal, etc.) — but if they explicitly pass
 * ``credentials: 'omit'`` we honour that, since they're presumably
 * fetching a public resource.
 *
 * @param {string | URL | Request} input
 * @param {RequestInit} [options]
 */
export function apiFetch(input, options = {}) {
  const opts = { credentials: 'include', ...options }
  return fetch(input, opts)
}
