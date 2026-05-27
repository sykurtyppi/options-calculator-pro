#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv311/bin/python"
LOG_DIR="${HOME}/.options_calculator_pro/logs"
LOG_FILE="${LOG_DIR}/daily_evidence_cycle_launchd.log"
STATE_DIR="${HOME}/.options_calculator_pro/state"
LOCK_DIR="${STATE_DIR}/daily_evidence_cycle.lock"
TIMEOUT_SECONDS="${EVIDENCE_CYCLE_TIMEOUT_SECONDS:-7200}"
LOCK_MAX_AGE_SECONDS="${EVIDENCE_CYCLE_LOCK_MAX_AGE_SECONDS:-10800}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  NOW_EPOCH="$(date +%s)"
  LOCK_EPOCH="$(stat -f %m "${LOCK_DIR}" 2>/dev/null || echo "${NOW_EPOCH}")"
  LOCK_AGE_SECONDS="$((NOW_EPOCH - LOCK_EPOCH))"
  if [ "${LOCK_AGE_SECONDS}" -gt "${LOCK_MAX_AGE_SECONDS}" ]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') daily evidence cycle stale lock replaced ====="
        echo "lock_age_seconds=${LOCK_AGE_SECONDS}"
      } >> "${LOG_FILE}" 2>&1
    else
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') daily evidence cycle skipped ====="
        echo "reason=already_running lock_dir=${LOCK_DIR}"
      } >> "${LOG_FILE}" 2>&1
      exit 0
    fi
  else
  {
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') daily evidence cycle skipped ====="
    echo "reason=already_running lock_dir=${LOCK_DIR}"
  } >> "${LOG_FILE}" 2>&1
  exit 0
  fi
fi

cleanup() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"

{
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') daily evidence cycle start ====="
  # Hardening P1-4: capture and propagate the Python heredoc's exit code
  # the same way scripts/automation/run_candidate_exit_resolver.sh does
  # (set +e around the call, then EXIT_CODE=$?, then a branch on the
  # code, then explicit `exit ${EXIT_CODE}` at the bottom of the script).
  # The previous version ran the heredoc inside a brace block that
  # didn't capture $? — launchd saw exit 0 on a hard failure (e.g.
  # SystemExit(124) on timeout), masking the real outcome from anything
  # that greps launchd's tracking later. The watchdog still detects the
  # missing completion marker, so this isn't a silent failure, but
  # operators expect exit codes to tell the truth.
  set +e
  "${PYTHON_BIN}" - "${PYTHON_BIN}" "scripts/run_evidence_cycle.py" "${TIMEOUT_SECONDS}" <<'PY'
import subprocess
import sys

python_bin, script_path, timeout_raw = sys.argv[1:4]
timeout_seconds = int(float(timeout_raw))
try:
    result = subprocess.run([python_bin, script_path], timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(f"daily evidence cycle timed out after {timeout_seconds}s", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(result.returncode)
PY
  EXIT_CODE=$?
  set -e
  if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') daily evidence cycle complete ====="
  else
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') daily evidence cycle failed exit_code=${EXIT_CODE} ====="
  fi
} >> "${LOG_FILE}" 2>&1

exit "${EXIT_CODE}"
