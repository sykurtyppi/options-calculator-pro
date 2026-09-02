#!/usr/bin/env bash
set -euo pipefail

# Pre-market ranked-screener alert wrapper. Mirrors run_state_backup.sh:
#   - mkdir-as-lock with stale-lock recovery
#   - timeout-bounded Python invocation
#   - explicit EXIT_CODE capture + propagation (so launchd sees real failures)
#
# Runs scripts/premarket_screener_alert.py, which screens the universe and
# sends an iMessage only when a setup clears the score threshold. The script
# is idempotent per (date, qualifying-symbol-set), so a manual re-run on the
# same day will not double-send.
#
# Configuration (env, all optional):
#   PREMARKET_ALERT_MIN_SCORE    minimum ranking_score to alert (default 0.65)
#   PREMARKET_ALERT_TOP          max setups per message (default 5)
#   PREMARKET_ALERT_EXTRA_ARGS   extra flags passed through verbatim
#   WATCHDOG_IMESSAGE_TO         recipient (shared with the evidence watchdog)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv311/bin/python"
LOG_DIR="${HOME}/.options_calculator_pro/logs"
LOG_FILE="${LOG_DIR}/premarket_screener_alert_launchd.log"
STATE_DIR="${HOME}/.options_calculator_pro/state"
LOCK_DIR="${STATE_DIR}/premarket_screener_alert.lock"
# The screener fans out across the universe with per-symbol timeouts; 10
# minutes is generous even when several providers are slow.
TIMEOUT_SECONDS="${PREMARKET_ALERT_TIMEOUT_SECONDS:-600}"
LOCK_MAX_AGE_SECONDS="${PREMARKET_ALERT_LOCK_MAX_AGE_SECONDS:-3600}"
MIN_SCORE="${PREMARKET_ALERT_MIN_SCORE:-0.65}"
TOP_N="${PREMARKET_ALERT_TOP:-5}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  NOW_EPOCH="$(date +%s)"
  LOCK_EPOCH="$(stat -f %m "${LOCK_DIR}" 2>/dev/null || echo "${NOW_EPOCH}")"
  LOCK_AGE_SECONDS="$((NOW_EPOCH - LOCK_EPOCH))"
  if [ "${LOCK_AGE_SECONDS}" -gt "${LOCK_MAX_AGE_SECONDS}" ]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') premarket alert stale lock replaced ====="
        echo "lock_age_seconds=${LOCK_AGE_SECONDS}"
      } >> "${LOG_FILE}" 2>&1
    else
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') premarket alert skipped ====="
        echo "reason=already_running lock_dir=${LOCK_DIR}"
      } >> "${LOG_FILE}" 2>&1
      exit 0
    fi
  else
    {
      echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') premarket alert skipped ====="
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

ALERT_ARGS=("scripts/premarket_screener_alert.py" "--min-score" "${MIN_SCORE}" "--top" "${TOP_N}")
# Word-splitting is intentional here: EXTRA_ARGS carries multiple flags.
# shellcheck disable=SC2206
if [ -n "${PREMARKET_ALERT_EXTRA_ARGS:-}" ]; then
  ALERT_ARGS+=(${PREMARKET_ALERT_EXTRA_ARGS})
fi

{
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') premarket alert start ====="
  echo "min_score=${MIN_SCORE} top=${TOP_N}"
  set +e
  "${PYTHON_BIN}" - "${PYTHON_BIN}" "${TIMEOUT_SECONDS}" "${ALERT_ARGS[@]}" <<'PY'
import subprocess
import sys

python_bin = sys.argv[1]
timeout_seconds = int(float(sys.argv[2]))
alert_args = sys.argv[3:]
try:
    result = subprocess.run([python_bin, *alert_args], timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(f"premarket alert timed out after {timeout_seconds}s", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(result.returncode)
PY
  EXIT_CODE=$?
  set -e
  if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') premarket alert complete ====="
  else
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') premarket alert failed exit_code=${EXIT_CODE} ====="
  fi
} >> "${LOG_FILE}" 2>&1

exit "${EXIT_CODE}"
