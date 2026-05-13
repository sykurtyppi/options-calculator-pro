#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv311/bin/python"
LOG_DIR="${HOME}/.options_calculator_pro/logs"
LOG_FILE="${LOG_DIR}/weekly_evidence_report_launchd.log"
STATE_DIR="${HOME}/.options_calculator_pro/state"
LOCK_DIR="${STATE_DIR}/weekly_evidence_report.lock"
TIMEOUT_SECONDS="${WEEKLY_EVIDENCE_REPORT_TIMEOUT_SECONDS:-900}"
LOCK_MAX_AGE_SECONDS="${WEEKLY_EVIDENCE_REPORT_LOCK_MAX_AGE_SECONDS:-1800}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  NOW_EPOCH="$(date +%s)"
  LOCK_EPOCH="$(stat -f %m "${LOCK_DIR}" 2>/dev/null || echo "${NOW_EPOCH}")"
  LOCK_AGE_SECONDS="$((NOW_EPOCH - LOCK_EPOCH))"
  if [ "${LOCK_AGE_SECONDS}" -gt "${LOCK_MAX_AGE_SECONDS}" ]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    mkdir "${LOCK_DIR}" 2>/dev/null || exit 0
  else
    {
      echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') weekly evidence report skipped ====="
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
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') weekly evidence report start ====="
  "${PYTHON_BIN}" - "${PYTHON_BIN}" "scripts/export_weekly_evidence_report.py" "${TIMEOUT_SECONDS}" <<'PY'
import subprocess
import sys

python_bin, script_path, timeout_raw = sys.argv[1:4]
timeout_seconds = int(float(timeout_raw))
try:
    result = subprocess.run([python_bin, script_path], timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(f"weekly evidence report timed out after {timeout_seconds}s", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(result.returncode)
PY
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') weekly evidence report complete ====="
} >> "${LOG_FILE}" 2>&1
