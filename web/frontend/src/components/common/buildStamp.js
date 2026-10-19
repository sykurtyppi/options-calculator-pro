const short = (sha) => (typeof sha === 'string' && sha ? sha.slice(0, 7) : null)

/**
 * Footer text for the build stamp. `build` is the bundle's own stamp, and
 * `version` the /api/version payload (null while unknown). Only a known
 * difference is called a mismatch; an unknown side is labelled unknown.
 */
export function describeBuild(build, version) {
  const ui = short(build?.git_sha)
  const label = `Build ${ui ?? 'unknown'}${build?.dirty ? ' (modified)' : ''}`
  const api = short(version?.api?.git_sha)
  if (ui && api && build.git_sha !== version.api.git_sha) {
    return { label, mismatch: `API runs ${api}: rebuild the frontend or restart the server` }
  }
  return { label, mismatch: null }
}
