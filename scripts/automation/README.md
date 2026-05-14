# Automation: launchd jobs

Local launchd jobs (macOS) that run the daily and weekly evidence cycles for this repo.

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
| `com.optionscalculator.evidence-cycle` | Daily at 21:30 |
| `com.optionscalculator.evidence-watchdog` | Daily at 22:15 |
| `com.optionscalculator.weekly-evidence-report` | Mondays at 22:45 |

All jobs use `RunAtLoad=false`; they fire only on the calendar schedule, never on `launchctl load`.

## Requirements

- macOS (uses `launchctl` and `~/Library/LaunchAgents`).
- A `.venv311` virtualenv at the project root (`${PROJECT_ROOT}/.venv311/bin/python`) with the project's Python dependencies installed.
- `~/.options_calculator_pro/{logs,state}` are auto-created by the wrappers on first run.

## Logs and state

- Wrapper logs: `~/.options_calculator_pro/logs/*.log`
- Launchd stdout/stderr: same directory, suffixed `_launchd_stdout.log` / `_launchd_stderr.log`
- Lock files (anti-overlap): `~/.options_calculator_pro/state/*.lock` directories — auto-removed on script exit, with stale-lock recovery via mtime.
