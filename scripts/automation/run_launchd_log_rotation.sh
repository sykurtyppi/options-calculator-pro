#!/usr/bin/env bash
set -euo pipefail

# Daily launchd-log rotation wrapper. Mirrors the shape of
# run_candidate_exit_resolver.sh:
#   - mkdir-as-lock with stale-lock recovery
#   - timeout-bounded Python invocation
#   - explicit EXIT_CODE capture + propagation (so launchd sees real failures)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv311/bin/python"
LOG_DIR="${HOME}/.options_calculator_pro/logs"
LOG_FILE="${LOG_DIR}/log_rotation_launchd.log"
STATE_DIR="${HOME}/.options_calculator_pro/state"
LOCK_DIR="${STATE_DIR}/log_rotation.lock"
# Rotation walks a handful of files and gzips them; 5 minutes is generous.
TIMEOUT_SECONDS="${LOG_ROTATION_TIMEOUT_SECONDS:-300}"
# Stale-lock recovery threshold: if the lock dir is older than this and
# no process is holding it, blow it away (matches resolver pattern).
LOCK_MAX_AGE_SECONDS="${LOG_ROTATION_LOCK_MAX_AGE_SECONDS:-1800}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  NOW_EPOCH="$(date +%s)"
  LOCK_EPOCH="$(stat -f %m "${LOCK_DIR}" 2>/dev/null || echo "${NOW_EPOCH}")"
  LOCK_AGE_SECONDS="$((NOW_EPOCH - LOCK_EPOCH))"
  if [ "${LOCK_AGE_SECONDS}" -gt "${LOCK_MAX_AGE_SECONDS}" ]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') log rotation stale lock replaced ====="
        echo "lock_age_seconds=${LOCK_AGE_SECONDS}"
      } >> "${LOG_FILE}" 2>&1
    else
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') log rotation skipped ====="
        echo "reason=already_running lock_dir=${LOCK_DIR}"
      } >> "${LOG_FILE}" 2>&1
      exit 0
    fi
  else
    {
      echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') log rotation skipped ====="
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
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') log rotation start ====="
  set +e
  "${PYTHON_BIN}" - "${PYTHON_BIN}" "scripts/rotate_launchd_logs.py" "${TIMEOUT_SECONDS}" <<'PY'
import subprocess
import sys

python_bin, script_path, timeout_raw = sys.argv[1:4]
timeout_seconds = int(float(timeout_raw))
try:
    result = subprocess.run([python_bin, script_path], timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(f"log rotation timed out after {timeout_seconds}s", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(result.returncode)
PY
  EXIT_CODE=$?
  set -e
  if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') log rotation complete ====="
  else
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') log rotation failed exit_code=${EXIT_CODE} ====="
  fi
} >> "${LOG_FILE}" 2>&1

exit "${EXIT_CODE}"
