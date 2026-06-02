#!/usr/bin/env bash
set -euo pipefail

LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
LEGACY_RUNTIME_DIR="${HOME}/.options_calculator_pro/automation"

for plist in \
  com.optionscalculator.candidate-exit-resolver.plist \
  com.optionscalculator.evidence-cycle.plist \
  com.optionscalculator.evidence-watchdog.plist \
  com.optionscalculator.weekly-evidence-report.plist \
  com.optionscalculator.log-rotation.plist \
  com.optionscalculator.forward-paper-collector.plist
do
  dst="${LAUNCH_AGENTS_DIR}/${plist}"
  if [ -e "${dst}" ]; then
    launchctl unload "${dst}" >/dev/null 2>&1 || true
    rm -f "${dst}"
    echo "removed ${plist}"
  else
    echo "skipped ${plist} (not installed)"
  fi
done

# Legacy: pre-template installs copied wrappers into ${LEGACY_RUNTIME_DIR}.
# Current installs don't write there; clean up only what actually exists.
for script in \
  run_candidate_exit_resolver.sh \
  run_daily_evidence_cycle.sh \
  run_daily_evidence_watchdog.sh \
  run_weekly_evidence_report.sh
do
  path="${LEGACY_RUNTIME_DIR}/${script}"
  if [ -e "${path}" ]; then
    rm -f "${path}"
    echo "removed legacy runtime wrapper ${script}"
  fi
done

if [ -d "${LEGACY_RUNTIME_DIR}" ]; then
  rmdir "${LEGACY_RUNTIME_DIR}" 2>/dev/null || true
fi

echo "LaunchAgents removed."
