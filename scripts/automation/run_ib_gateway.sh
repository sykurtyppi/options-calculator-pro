#!/usr/bin/env bash
set -euo pipefail

# IB Gateway wrapper, kept alive by launchd (com.optionscalculator.ib-gateway).
# Like run_api_server.sh it takes no lock and no timeout: launchd runs a single
# instance and restarts it whenever it exits. `exec` hands the process
# to the gateway's own launcher so launchd supervises the gateway itself.
#
# What this does and does not do:
#   - Starts the gateway at login and brings it back after a crash or IB's
#     daily auto-logoff, so the ib_gateway data provider does not depend on
#     someone having double-clicked the app since the last reboot.
#   - Does NOT log in. The gateway comes back to its login window; the API
#     port (IB_GATEWAY_PORT) opens only after a person signs in. Unattended
#     sign-in needs a third-party controller (IBC) holding the credentials.
#   - Never starts a second gateway. If one is already running (started by
#     hand, or relaunched by IB's own daily auto-restart and still signed
#     in), this adopts it: waits for it to exit, then exits so launchd runs
#     the wrapper again.
#
# Which app: IB_GATEWAY_APP (process env, else the project .env) names the
# .app bundle; otherwise the newest "IB Gateway <version>.app" under
# ~/Applications or /Applications is used, so an in-place IB upgrade that
# bumps the version folder keeps working without an edit here.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${HOME}/.options_calculator_pro/logs"
mkdir -p "${LOG_DIR}"

# launchd starts this wrapper with a bare environment, so read the one knob it
# needs from the .env file WITHOUT sourcing it (sourcing would execute
# arbitrary shell); the process environment still wins.
ENV_FILE="${OPTIONS_CALCULATOR_ENV_FILE:-${PROJECT_ROOT}/.env}"
_env_file_value() {
  # $1 = KEY. Prints the value of the LAST "KEY=VALUE" line in ENV_FILE with
  # surrounding quotes stripped; prints nothing when the file or key is absent.
  [ -f "${ENV_FILE}" ] || return 0
  { grep -E "^[[:space:]]*${1}=" "${ENV_FILE}" 2>/dev/null || true; } | tail -1 \
    | sed -E "s/^[[:space:]]*${1}=//; s/^[\"']//; s/[\"']\$//"
}

APP="${IB_GATEWAY_APP:-$(_env_file_value IB_GATEWAY_APP)}"
APP="${APP/#\~/${HOME}}"
if [ -z "${APP}" ]; then
  # IB's installer puts the bundle in a version folder ("IB Gateway 10.51/IB
  # Gateway 10.51.app") next to "IB Gateway 10.51 Uninstaller.app"; a
  # drag-install puts it directly in Applications. Look in both layouts under
  # ~/Applications and /Applications, never pick an uninstaller, and let the
  # newest version win: `sort -V` orders "IB Gateway 10.51" after "IB Gateway 10.9".
  APP="$( { ls -d \
      "${HOME}/Applications/IB Gateway"*.app \
      "${HOME}/Applications/IB Gateway"*/"IB Gateway"*.app \
      "/Applications/IB Gateway"*.app \
      "/Applications/IB Gateway"*/"IB Gateway"*.app 2>/dev/null || true; } \
    | grep -viE 'uninstall' | sort -V | tail -1 || true)"
fi
STUB="${APP%/}/Contents/MacOS/JavaApplicationStub"

if [ -z "${APP}" ] || [ ! -x "${STUB}" ]; then
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ib gateway not found =====" >&2
  echo "No IB Gateway app at '${APP:-<none>}'. Install IB Gateway or set IB_GATEWAY_APP in ${ENV_FILE} to the .app bundle." >&2
  exit 78  # EX_CONFIG; launchd retries after ThrottleInterval
fi

# Never start a second gateway: it would fight the first over the API port and
# the IB session. Two ways one can already be running when we get here:
#   - someone started it by hand;
#   - IB's own daily auto-restart (Configure -> Settings -> Lock and Exit ->
#     Auto restart). The gateway then relaunches ITSELF, still signed in, and
#     the old process exits. launchd sees its job exit and runs this wrapper
#     again, while the relaunched gateway is alive (the plist sets
#     AbandonProcessGroup so launchd does not kill it). Starting a fresh one
#     here would replace a signed-in gateway with a login window, which is
#     exactly what auto-restart exists to avoid.
# So: adopt whatever is running and wait for it. The relaunch can trail the
# old process's exit by a moment, so look for it for a short grace period
# before concluding nothing is running.
_gateway_running() { pgrep -f "${STUB}" >/dev/null 2>&1; }

_whole_seconds() {
  # $1 = candidate, $2 = default. Prints $1 when it is a whole number, else $2.
  case "${1}" in ''|*[!0-9]*) printf '%s' "${2}" ;; *) printf '%s' "${1}" ;; esac
}
ADOPT_GRACE="$(_whole_seconds "${IB_GATEWAY_ADOPT_GRACE_SECONDS:-$(_env_file_value IB_GATEWAY_ADOPT_GRACE_SECONDS)}" 20)"
POLL_SECONDS="$(_whole_seconds "${IB_GATEWAY_POLL_SECONDS:-}" 30)"

waited=0
while ! _gateway_running && [ "${waited}" -lt "${ADOPT_GRACE}" ]; do
  sleep 1
  waited=$((waited + 1))
done

if _gateway_running; then
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ib gateway already running (started by hand or relaunched by IB's auto-restart); adopting it ====="
  while _gateway_running; do
    sleep "${POLL_SECONDS}"
  done
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') adopted ib gateway exited; launchd will run this wrapper again ====="
  exit 75  # EX_TEMPFAIL; the next run adopts a relaunched gateway or starts one
fi

cd "${PROJECT_ROOT}"
echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ib gateway start app=${APP} ====="
exec "${STUB}"
