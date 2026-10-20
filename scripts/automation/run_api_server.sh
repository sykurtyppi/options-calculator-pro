#!/usr/bin/env bash
set -euo pipefail

# Long-running API server wrapper, kept alive by launchd
# (com.optionscalculator.api). Unlike the scheduled jobs it takes no lock and
# no timeout: launchd runs a single instance and restarts it if it exits.
# `exec` hands the process to uvicorn so launchd supervises the server itself.
#
# Bound to 127.0.0.1 by default. web/api/app.py loads the project .env itself.
# Access logging is off: launchd holds this job's stdout/stderr open for the
# server's whole life, so the daily log rotation cannot rename them safely,
# and per-request lines would grow them without bound. Errors still land in
# api_server_stderr.log.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv311/bin/python"
HOST="${OPTIONS_CALCULATOR_API_HOST:-127.0.0.1}"
PORT="${OPTIONS_CALCULATOR_API_PORT:-8000}"

cd "${PROJECT_ROOT}"
echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') api server start host=${HOST} port=${PORT} ====="
exec "${PYTHON_BIN}" -m uvicorn web.api.app:app --host "${HOST}" --port "${PORT}" --no-access-log
