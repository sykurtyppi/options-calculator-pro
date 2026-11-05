#!/usr/bin/env bash
set -euo pipefail

# IB Gateway wrapper, kept alive by launchd (com.optionscalculator.ib-gateway).
# Like run_api_server.sh it takes no lock and no timeout: launchd runs a single
# instance and restarts it if it exits with a failure. `exec` hands the process
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
#     hand), this waits for it to exit and then exits non-zero so launchd
#     starts a supervised one in its place.
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
  # Newest version wins: `sort -V` orders "IB Gateway 10.51" after "IB Gateway 10.9".
  APP="$( { ls -d "${HOME}/Applications/IB Gateway"*.app /Applications/IB\ Gateway*.app 2>/dev/null || true; } | sort -V | tail -1)"
fi
STUB="${APP%/}/Contents/MacOS/JavaApplicationStub"

if [ -z "${APP}" ] || [ ! -x "${STUB}" ]; then
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ib gateway not found =====" >&2
  echo "No IB Gateway app at '${APP:-<none>}'. Install IB Gateway or set IB_GATEWAY_APP in ${ENV_FILE} to the .app bundle." >&2
  exit 78  # EX_CONFIG; launchd retries after ThrottleInterval
fi

# A gateway started by hand is left alone; a second instance would fight over
# the API port and the IB session. Wait for it, then hand over to launchd.
if pgrep -f "${STUB}" >/dev/null 2>&1; then
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ib gateway already running (not started by launchd); waiting for it to exit ====="
  while pgrep -f "${STUB}" >/dev/null 2>&1; do
    sleep 30
  done
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') hand-started ib gateway exited; launchd will start a supervised one ====="
  exit 75  # EX_TEMPFAIL; a failure exit, so KeepAlive restarts us
fi

cd "${PROJECT_ROOT}"
echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ib gateway start app=${APP} ====="
exec "${STUB}"
