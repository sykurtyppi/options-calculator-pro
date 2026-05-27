#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv311/bin/python"
LOG_DIR="${HOME}/.options_calculator_pro/logs"
LOG_FILE="${LOG_DIR}/candidate_exit_resolver_launchd.log"
STATE_DIR="${HOME}/.options_calculator_pro/state"
LOCK_DIR="${STATE_DIR}/candidate_exit_resolver.lock"
TIMEOUT_SECONDS="${CANDIDATE_EXIT_RESOLVER_TIMEOUT_SECONDS:-1800}"
LOCK_MAX_AGE_SECONDS="${CANDIDATE_EXIT_RESOLVER_LOCK_MAX_AGE_SECONDS:-3600}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  NOW_EPOCH="$(date +%s)"
  LOCK_EPOCH="$(stat -f %m "${LOCK_DIR}" 2>/dev/null || echo "${NOW_EPOCH}")"
  LOCK_AGE_SECONDS="$((NOW_EPOCH - LOCK_EPOCH))"
  if [ "${LOCK_AGE_SECONDS}" -gt "${LOCK_MAX_AGE_SECONDS}" ]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') candidate exit resolver stale lock replaced ====="
        echo "lock_age_seconds=${LOCK_AGE_SECONDS}"
      } >> "${LOG_FILE}" 2>&1
    else
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') candidate exit resolver skipped ====="
        echo "reason=already_running lock_dir=${LOCK_DIR}"
      } >> "${LOG_FILE}" 2>&1
      exit 0
    fi
  else
    {
      echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') candidate exit resolver skipped ====="
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
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') candidate exit resolver start ====="
  set +e
  "${PYTHON_BIN}" - "${PYTHON_BIN}" "scripts/resolve_candidate_exits.py" "${TIMEOUT_SECONDS}" <<'PY'
import subprocess
import sys

python_bin, script_path, timeout_raw = sys.argv[1:4]
timeout_seconds = int(float(timeout_raw))
try:
    result = subprocess.run([python_bin, script_path], timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(f"candidate exit resolver timed out after {timeout_seconds}s", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(result.returncode)
PY
  EXIT_CODE=$?
  set -e
  if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') candidate exit resolver complete ====="
  else
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') candidate exit resolver failed exit_code=${EXIT_CODE} ====="
  fi
} >> "${LOG_FILE}" 2>&1

exit "${EXIT_CODE}"
