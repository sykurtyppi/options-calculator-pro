#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
TEMPLATE_DIR="${PROJECT_ROOT}/scripts/automation"

mkdir -p "${LAUNCH_AGENTS_DIR}" "${HOME}/.options_calculator_pro/logs"

for plist in \
  com.optionscalculator.evidence-cycle.plist \
  com.optionscalculator.evidence-watchdog.plist \
  com.optionscalculator.weekly-evidence-report.plist
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
  launchctl load "${dst}"
  echo "rendered and loaded ${plist}"
done

echo "LaunchAgents installed. Check with: launchctl list | grep optionscalculator"
