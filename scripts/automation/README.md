# Automation: launchd jobs

Local launchd jobs (macOS) that run the daily/weekly evidence cycles and the PR-AE candidate exit resolver for this repo.

## Contents

- `*.plist` — launchd job templates with `__HOME__` and `__PROJECT_ROOT__` placeholders. Rendered at install time; not loaded directly.
- `run_*.sh` — wrappers invoked by launchd. Each self-locates the repo root from `${BASH_SOURCE[0]}` and uses `${PROJECT_ROOT}/.venv311/bin/python`.
- `install_launchd_jobs.sh` / `uninstall_launchd_jobs.sh` — install/remove the jobs.

## Install

```sh
./scripts/automation/install_launchd_jobs.sh
```

The installer:
1. Auto-detects the project root from the script's own location.
2. Renders each plist template into `~/Library/LaunchAgents/` with `${PROJECT_ROOT}` and `${HOME}` substituted.
3. Loads the rendered plists with `launchctl`.

Verify:

```sh
launchctl list | grep optionscalculator
```

## Uninstall

```sh
./scripts/automation/uninstall_launchd_jobs.sh
```

## Schedule

| Job | When |
|---|---|
| `com.optionscalculator.candidate-exit-resolver` | Daily at 12:30 |
| `com.optionscalculator.evidence-cycle` | Daily at 21:30 |
| `com.optionscalculator.evidence-watchdog` | Daily at 22:15 |
| `com.optionscalculator.weekly-evidence-report` | Mondays at 22:45 |

All jobs use `RunAtLoad=false`; they fire only on the calendar schedule, never on `launchctl load`.

The candidate exit resolver is scheduled at 12:30 local time so prior-day post-event chains have time to settle before the resolver scans pending forward observations. It is operational infrastructure only: it records whether candidate shadow outcomes could be resolved, and it never alerts on positive/negative PnL or candidate-vs-legacy performance.

## Requirements

- macOS (uses `launchctl` and `~/Library/LaunchAgents`).
- A `.venv311` virtualenv at the project root (`${PROJECT_ROOT}/.venv311/bin/python`) with the project's Python dependencies installed.
- `~/.options_calculator_pro/{logs,state}` are auto-created by the wrappers on first run.

## Logs and state

- Wrapper logs: `~/.options_calculator_pro/logs/*.log`
- Launchd stdout/stderr: same directory, suffixed `_launchd_stdout.log` / `_launchd_stderr.log`
- Candidate resolver row telemetry: `~/.options_calculator_pro/logs/candidate_exit_resolutions.jsonl` when candidate rows are processed. A clean resolver run with zero pending rows is recorded in `candidate_exit_resolver_launchd.log`, not as a JSONL row.
- Lock files (anti-overlap): `~/.options_calculator_pro/state/*.lock` directories — auto-removed on script exit, with stale-lock recovery via mtime.

## Health checks

Manual health check:

```sh
./.venv311/bin/python scripts/check_evidence_health.py
```

Resolver-specific operational failures are:

- launchd wrapper log missing or stale after the first scheduled run
- resolver wrapper failure exit code
- `count_balance_holds: false` in the resolver stdout summary
- any row stuck with `days_in_awaiting_state > 10`
- any `permanently_failed:simulator_error` row

These are deliberately operational alerts, not trading signals. Do not alert on `mid_realized_return_pct`, candidate-vs-legacy performance, or promotion-threshold progress.
