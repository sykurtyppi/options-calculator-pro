#!/usr/bin/env bash
# Launchd wrapper for scripts/forward_paper_collector.py.
#
# Invoked by ~/Library/LaunchAgents/com.optionscalculator.forward-paper-collector.plist
# on its daily schedule. Mirrors the established evidence-cycle wrapper pattern:
#   - self-locates PROJECT_ROOT from BASH_SOURCE (no install-time path injection)
#   - dir-based anti-overlap lock with stale-lock recovery via mtime
#   - timeout enforced via a tiny Python subprocess.run heredoc (hard 30-min cap)
#   - honest exit-code propagation back to launchd (P1-4 pattern)
#   - structured log markers (start / complete / failed / skipped) for grep + watchdog
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv311/bin/python"
LOG_DIR="${HOME}/.options_calculator_pro/logs"
LOG_FILE="${LOG_DIR}/forward_paper_collector_launchd.log"
STATE_DIR="${HOME}/.options_calculator_pro/state"
LOCK_DIR="${STATE_DIR}/forward_paper_collector.lock"
TIMEOUT_SECONDS="${FORWARD_PAPER_COLLECTOR_TIMEOUT_SECONDS:-1800}"
LOCK_MAX_AGE_SECONDS="${FORWARD_PAPER_COLLECTOR_LOCK_MAX_AGE_SECONDS:-7200}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

# ── Anti-overlap lock with stale-lock recovery ───────────────────────────────
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  NOW_EPOCH="$(date +%s)"
  LOCK_EPOCH="$(stat -f %m "${LOCK_DIR}" 2>/dev/null || echo "${NOW_EPOCH}")"
  LOCK_AGE_SECONDS="$((NOW_EPOCH - LOCK_EPOCH))"
  if [ "${LOCK_AGE_SECONDS}" -gt "${LOCK_MAX_AGE_SECONDS}" ]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') forward paper collector stale lock replaced ====="
        echo "lock_age_seconds=${LOCK_AGE_SECONDS}"
      } >> "${LOG_FILE}" 2>&1
    else
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') forward paper collector skipped ====="
        echo "reason=already_running lock_dir=${LOCK_DIR}"
      } >> "${LOG_FILE}" 2>&1
      exit 0
    fi
  else
    {
      echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') forward paper collector skipped ====="
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

# ── Run with hard timeout; capture exit code honestly so launchd sees the truth ─
{
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') forward paper collector start ====="
  set +e
  "${PYTHON_BIN}" - "${PYTHON_BIN}" "scripts/forward_paper_collector.py" "${TIMEOUT_SECONDS}" <<'PY'
import subprocess
import sys

python_bin, script_path, timeout_raw = sys.argv[1:4]
timeout_seconds = int(float(timeout_raw))
try:
    result = subprocess.run([python_bin, script_path], timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(f"forward paper collector timed out after {timeout_seconds}s", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(result.returncode)
PY
  EXIT_CODE=$?
  set -e
  if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') forward paper collector complete ====="
  else
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') forward paper collector failed exit_code=${EXIT_CODE} ====="
  fi
} >> "${LOG_FILE}" 2>&1

exit "${EXIT_CODE}"
