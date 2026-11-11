#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
TEMPLATE_DIR="${PROJECT_ROOT}/scripts/automation"

STATE_DIR="${HOME}/.options_calculator_pro/state"
ACTIVATION_PATH="${STATE_DIR}/watchdog_activated_at"
INSTALL_PENDING_PATH="${STATE_DIR}/watchdog_install_pending"
PENDING_ACTIVATION_PATH="${STATE_DIR}/watchdog_activation_pending_at"
INSTALL_LOCK_PATH="${STATE_DIR}/watchdog_install_lock"
EXISTING_CYCLE_PLIST="${LAUNCH_AGENTS_DIR}/com.optionscalculator.evidence-cycle.plist"
EXISTING_CYCLE_LOG="${HOME}/.options_calculator_pro/logs/daily_evidence_cycle_launchd.log"
EXISTING_REPORT_DIR="${HOME}/.options_calculator_pro/reports/evidence"

mkdir -p "${LAUNCH_AGENTS_DIR}" "${HOME}/.options_calculator_pro/logs" "${STATE_DIR}"

# Serialize the unload/load/activation transaction. Concurrent installers can
# otherwise interleave their launchctl operations and leave the watchdog loaded
# while the evidence-cycle job is unloaded. Never reclaim an existing lock in
# this process: deleting a seemingly stale lock is racy with another reclaimer.
# After an interrupted install, an operator must first establish that no
# installer is active and then remove the stale directory explicitly.
acquire_install_lock() {
  if mkdir "${INSTALL_LOCK_PATH}" 2>/dev/null; then
    printf '%s\n' "$$" > "${INSTALL_LOCK_PATH}/pid"
    return 0
  fi
  echo "watchdog LaunchAgent installation is already in progress: ${INSTALL_LOCK_PATH}" >&2
  echo "if this is stale, verify no installer is running before removing that directory" >&2
  return 75
}
acquire_install_lock || exit $?
cleanup_install_lock() {
  if [ -f "${INSTALL_LOCK_PATH}/pid" ] && \
     [ "$(cat "${INSTALL_LOCK_PATH}/pid")" = "$$" ]; then
    rm -rf "${INSTALL_LOCK_PATH}"
  fi
}
handle_install_interrupt() {
  exit 130
}
handle_install_terminate() {
  exit 143
}
trap cleanup_install_lock EXIT
trap handle_install_interrupt INT
trap handle_install_terminate TERM

create_activation_marker() {
  local value="$1"
  local message="$2"
  if [ -e "${ACTIVATION_PATH}" ]; then
    return 0
  fi
  umask 077
  local activation_tmp="${ACTIVATION_PATH}.tmp.$$"
  printf '%s\n' "${value}" > "${activation_tmp}"
  if ln "${activation_tmp}" "${ACTIVATION_PATH}" 2>/dev/null; then
    echo "${message}: ${ACTIVATION_PATH}"
  fi
  rm -f "${activation_tmp}"
  if [ ! -e "${ACTIVATION_PATH}" ]; then
    echo "failed to publish watchdog activation marker: ${ACTIVATION_PATH}" >&2
    return 1
  fi
  rm -rf "${INSTALL_PENDING_PATH}"
  rm -f "${PENDING_ACTIVATION_PATH}"
}

# Existing installations predate the activation marker. They must retain
# ownership of all prior cycles rather than looking like a fresh install on
# the first upgrade that includes this code.
if [ ! -e "${ACTIVATION_PATH}" ] && [ ! -d "${INSTALL_PENDING_PATH}" ] && {
  [ -e "${EXISTING_CYCLE_PLIST}" ] ||
  [ -s "${EXISTING_CYCLE_LOG}" ] ||
  find "${EXISTING_REPORT_DIR}" -type f -name 'evidence_report_*.json' -print -quit 2>/dev/null | grep -q .
}; then
  create_activation_marker \
    "1970-01-01T00:00:00Z" \
    "migrated established watchdog activation"
fi

# Mark a genuinely fresh installation before rendering any plist. If a later
# load fails, this durable marker keeps a retry from mistaking its own partial
# artifacts for evidence of an established deployment.
if [ ! -e "${ACTIVATION_PATH}" ] && [ ! -d "${INSTALL_PENDING_PATH}" ]; then
  mkdir "${INSTALL_PENDING_PATH}"
fi

for plist in \
  com.optionscalculator.candidate-exit-resolver.plist \
  com.optionscalculator.evidence-cycle.plist \
  com.optionscalculator.evidence-watchdog.plist \
  com.optionscalculator.weekly-evidence-report.plist \
  com.optionscalculator.log-rotation.plist \
  com.optionscalculator.forward-paper-collector.plist \
  com.optionscalculator.premarket-screener-alert.plist \
  com.optionscalculator.state-backup.plist \
  com.optionscalculator.api.plist \
  com.optionscalculator.ib-gateway.plist
do
  src="${TEMPLATE_DIR}/${plist}"
  dst="${LAUNCH_AGENTS_DIR}/${plist}"

  # Render the plist template: substitute __PROJECT_ROOT__ / __HOME__ with the
  # absolute paths resolved above. launchd does not expand env vars in plists,
  # so the substitution must happen at install time. Pipe used as the sed
  # delimiter because the replacement strings contain slashes.
  sed \
    -e "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${src}" > "${dst}"

  launchctl unload "${dst}" >/dev/null 2>&1 || true

  if [ "${plist}" = "com.optionscalculator.evidence-cycle.plist" ] && \
     [ ! -e "${ACTIVATION_PATH}" ] && \
     [ ! -e "${PENDING_ACTIVATION_PATH}" ]; then
    # Persist a conservative ownership timestamp immediately before the load.
    # If the process dies after launchctl succeeds but before marker publication,
    # a retry must preserve this time rather than suppress an intervening cycle.
    umask 077
    pending_tmp="${PENDING_ACTIVATION_PATH}.tmp.$$"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "${pending_tmp}"
    mv "${pending_tmp}" "${PENDING_ACTIVATION_PATH}"
  fi

  if launchctl load "${dst}"; then
    :
  else
    load_status=$?
    if [ "${plist}" = "com.optionscalculator.evidence-cycle.plist" ] && \
       [ ! -e "${ACTIVATION_PATH}" ]; then
      # A confirmed load failure did not activate the job. Discard the attempt
      # time so a later successful retry starts ownership at the retry.
      rm -f "${PENDING_ACTIVATION_PATH}"
    fi
    exit "${load_status}"
  fi
  echo "rendered and loaded ${plist}"

  if [ "${plist}" = "com.optionscalculator.evidence-cycle.plist" ]; then
    # A fresh installation starts owning cycles only after the evidence job
    # is successfully loaded. Create this before loading the watchdog so it
    # can never observe a half-activated fresh installation.
    if [ ! -e "${ACTIVATION_PATH}" ]; then
      if [ ! -s "${PENDING_ACTIVATION_PATH}" ]; then
        echo "missing pending watchdog activation timestamp: ${PENDING_ACTIVATION_PATH}" >&2
        exit 1
      fi
      create_activation_marker \
        "$(cat "${PENDING_ACTIVATION_PATH}")" \
        "recorded watchdog activation"
    fi
  fi
done

echo "LaunchAgents installed. Check with: launchctl list | grep optionscalculator"
