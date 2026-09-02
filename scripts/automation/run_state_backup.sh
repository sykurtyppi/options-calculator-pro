#!/usr/bin/env bash
set -euo pipefail

# Daily state-backup wrapper. Mirrors the shape of
# run_launchd_log_rotation.sh:
#   - mkdir-as-lock with stale-lock recovery
#   - timeout-bounded Python invocation
#   - explicit EXIT_CODE capture + propagation (so launchd sees real failures)
#
# Produces a hot, consistent snapshot of ~/.options_calculator_pro via
# scripts/backup_state.py (SQLite Online Backup API + byte copy of the rest),
# with newest-N retention.
#
# Output location:
#   By default the archive lands in ~/.options_calculator_pro/backups/, which
#   protects against `rm -rf` typos and SQLite corruption but NOT disk loss (the
#   backups live on the same disk as the source). For real disaster recovery,
#   set OPTIONS_CALCULATOR_BACKUP_DIR to an external sync folder (iCloud Drive /
#   Dropbox / external mount) — the wrapper passes it through as --output-dir.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv311/bin/python"
LOG_DIR="${HOME}/.options_calculator_pro/logs"
LOG_FILE="${LOG_DIR}/state_backup_launchd.log"
STATE_DIR="${HOME}/.options_calculator_pro/state"
LOCK_DIR="${STATE_DIR}/state_backup.lock"
# SQLite online-backup of a few small DBs + a byte copy of the rest; 15 minutes
# is generous even on a large log tree.
TIMEOUT_SECONDS="${STATE_BACKUP_TIMEOUT_SECONDS:-900}"
# Stale-lock recovery threshold: if the lock dir is older than this and no
# process is holding it, blow it away (matches log-rotation pattern).
LOCK_MAX_AGE_SECONDS="${STATE_BACKUP_LOCK_MAX_AGE_SECONDS:-3600}"
# Optional off-host output dir. When unset, backup_state.py uses its own default
# (~/.options_calculator_pro/backups/).
BACKUP_OUTPUT_DIR="${OPTIONS_CALCULATOR_BACKUP_DIR:-}"
# Newest-N retention passed through to backup_state.py.
BACKUP_RETENTION="${OPTIONS_CALCULATOR_BACKUP_RETENTION:-14}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  NOW_EPOCH="$(date +%s)"
  LOCK_EPOCH="$(stat -f %m "${LOCK_DIR}" 2>/dev/null || echo "${NOW_EPOCH}")"
  LOCK_AGE_SECONDS="$((NOW_EPOCH - LOCK_EPOCH))"
  if [ "${LOCK_AGE_SECONDS}" -gt "${LOCK_MAX_AGE_SECONDS}" ]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') state backup stale lock replaced ====="
        echo "lock_age_seconds=${LOCK_AGE_SECONDS}"
      } >> "${LOG_FILE}" 2>&1
    else
      {
        echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') state backup skipped ====="
        echo "reason=already_running lock_dir=${LOCK_DIR}"
      } >> "${LOG_FILE}" 2>&1
      exit 0
    fi
  else
    {
      echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') state backup skipped ====="
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

# Assemble backup_state.py args. --output-dir only when an external dir is set.
BACKUP_ARGS=("scripts/backup_state.py" "--retention" "${BACKUP_RETENTION}")
if [ -n "${BACKUP_OUTPUT_DIR}" ]; then
  BACKUP_ARGS+=("--output-dir" "${BACKUP_OUTPUT_DIR}")
fi

{
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') state backup start ====="
  if [ -n "${BACKUP_OUTPUT_DIR}" ]; then
    echo "output_dir=${BACKUP_OUTPUT_DIR} retention=${BACKUP_RETENTION}"
  else
    echo "output_dir=<default ~/.options_calculator_pro/backups> retention=${BACKUP_RETENTION}"
    echo "note: same-disk backups do not protect against disk loss; set OPTIONS_CALCULATOR_BACKUP_DIR to an external folder for disaster recovery."
  fi
  set +e
  "${PYTHON_BIN}" - "${PYTHON_BIN}" "${TIMEOUT_SECONDS}" "${BACKUP_ARGS[@]}" <<'PY'
import subprocess
import sys

python_bin = sys.argv[1]
timeout_seconds = int(float(sys.argv[2]))
backup_args = sys.argv[3:]
try:
    result = subprocess.run([python_bin, *backup_args], timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(f"state backup timed out after {timeout_seconds}s", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(result.returncode)
PY
  EXIT_CODE=$?
  set -e
  if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') state backup complete ====="
  else
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') state backup failed exit_code=${EXIT_CODE} ====="
  fi
} >> "${LOG_FILE}" 2>&1

exit "${EXIT_CODE}"
