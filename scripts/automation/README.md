# Automation: launchd jobs

Local launchd jobs (macOS) that run the daily/weekly evidence cycles and the PR-AE candidate exit resolver for this repo.

For the full operational story (env-var contracts, verification, alert semantics, troubleshooting matrix) see **[../../docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md)**. This README focuses on the mechanics of the launchd jobs themselves.

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
| `com.optionscalculator.log-rotation` | Daily at 03:00 |
| `com.optionscalculator.forward-paper-collector` | Daily at 19:30 |

All jobs use `RunAtLoad=false`; they fire only on the calendar schedule, never on `launchctl load`.

The log-rotation job runs at 03:00 local — chosen to be safely away from every other launchd job so no other job has an active handle on the `.log` files we rotate. Rotation is size-based (default 5 MB threshold) with gzip + 7-archive retention per file; see `scripts/rotate_launchd_logs.py --help` for the exact contract and tunables. Only `*_launchd*.log` shapes are touched — the Python-logger files (`__main__.log`, `services.*.log`) manage their own rotation via `RotatingFileHandler`.

The candidate exit resolver is scheduled at 12:30 local time so prior-day post-event chains have time to settle before the resolver scans pending forward observations. It is operational infrastructure only: it records whether candidate shadow outcomes could be resolved, and it never alerts on positive/negative PnL or candidate-vs-legacy performance.

The forward paper collector runs at 19:30 local — on an Atlantic/Reykjavik (GMT) machine this maps to **15:30 ET in EDT / 14:30 ET in EST**, i.e. comfortably inside the US options session and well before any AMC earnings print (after 16:00 ET). The job is the daily entry/exit pass for the AMC T-3/T-0 OTM-strangle paper-trade pocket; it appends to `exports/reports/forward_paper_trades.csv` and is idempotent. It is research infrastructure only — it accrues forward samples on the validated config so the edge can be confirmed or refuted over time. Tunables: `FORWARD_PAPER_COLLECTOR_TIMEOUT_SECONDS` (default 1800), `FORWARD_PAPER_COLLECTOR_LOCK_MAX_AGE_SECONDS` (default 7200). Operators in other timezones: change `Hour=19 Minute=30` in `com.optionscalculator.forward-paper-collector.plist` to your local-time equivalent of "US market open, well before 16:00 ET", then reinstall.

### Resolver due-window check — timezone mapping

The health/watchdog scripts include a "first-run not due yet" check so a fresh install before today's 12:30 fire doesn't false-alarm at the same-day 22:15 watchdog. The check compares `now` against the resolver's expected fire time in **UTC**. The launchd plist's `StartCalendarInterval Hour=12 Minute=30` is in **local** time. The defaults in `EvidenceHealthConfig` (`resolver_due_hour_utc=12`, `resolver_due_minute_utc=30`) assume the system local timezone is UTC.

If your Mac runs in a non-UTC timezone, override via the CLI flags on `scripts/check_evidence_health.py` and `scripts/watch_daily_evidence_cycle.py`:

| System local timezone | `--resolver-due-hour-utc` | `--resolver-due-minute-utc` |
|---|---|---|
| UTC (e.g. Iceland) | 12 (default) | 30 (default) |
| Europe/London BST (UTC+1) | 11 | 30 |
| Europe/Berlin CEST (UTC+2) | 10 | 30 |
| US/Eastern EST (UTC-5) | 17 | 30 |
| US/Eastern EDT (UTC-4) | 16 | 30 |
| US/Pacific PST (UTC-8) | 20 | 30 |
| US/Pacific PDT (UTC-7) | 19 | 30 |

The mapping changes with DST. If the values become wrong (e.g. after a DST transition), the symptom is either:
- False alert on install day: the override is *later* than reality (the resolver fires before the configured "due window," so the watchdog thinks it's still pending when it's actually already fired and missing).
- False non-alert on a missed run: the override is *earlier* than reality (the watchdog still thinks it's not due yet when it actually was due).

Both modes are operationally noisy but never escalate to PnL/trading-signal alerts — they're pipeline-health observations only.

For users who want a deterministic UTC schedule, change `Hour=12 Minute=30` in `com.optionscalculator.candidate-exit-resolver.plist` to a value that maps cleanly to UTC for your tz, reinstall via `install_launchd_jobs.sh`, and update the override flags accordingly.

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
