const short = (sha) => (typeof sha === 'string' && sha ? sha.slice(0, 7) : null)

/**
 * Footer text for the build stamp. `build` is the bundle's own stamp, and
 * `version` the /api/version payload (null while unknown). A known
 * difference, or one side built from modified sources, is flagged; an
 * unknown side is labelled unknown.
 */
export function describeBuild(build, version) {
  const ui = short(build?.git_sha)
  const label = `Build ${ui ?? 'unknown'}${build?.dirty ? ' (modified)' : ''}`
  const api = short(version?.api?.git_sha)
  if (ui && api && build.git_sha !== version.api.git_sha) {
    return { label, mismatch: `API runs ${api}: rebuild the frontend or restart the server` }
  }
  if (ui && api && (build.dirty !== false || version.api.dirty !== false)) {
    // Same commit, but modified sources on either side can still differ.
    return { label, mismatch: 'same commit as the API, but one side has modified or unknown sources: cannot confirm they match' }
  }
  return { label, mismatch: null }
}
