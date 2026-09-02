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
| `com.optionscalculator.premarket-screener-alert` | Weekdays at 14:30 |
| `com.optionscalculator.state-backup` | Daily at 04:00 |

All jobs use `RunAtLoad=false`; they fire only on the calendar schedule, never on `launchctl load`.

The state-backup job runs at 04:00 local — offset from the 03:00 log rotation so the two don't contend for the SQLite files. It calls `scripts/backup_state.py` to write a hot, consistent `.tar.gz` snapshot of `~/.options_calculator_pro` (SQLite files via the Online Backup API, everything else byte-copied) with newest-N retention. By default the archive lands in `~/.options_calculator_pro/backups/`, which guards against `rm -rf` typos and SQLite corruption but **not disk loss** (same disk as the source). For real disaster recovery, set `OPTIONS_CALCULATOR_BACKUP_DIR` to an external sync folder (iCloud Drive / Dropbox / external mount) — the wrapper passes it through as `--output-dir`. Tunables: `OPTIONS_CALCULATOR_BACKUP_DIR` (default in-tree `backups/`), `OPTIONS_CALCULATOR_BACKUP_RETENTION` (default 14), `STATE_BACKUP_TIMEOUT_SECONDS` (default 900), `STATE_BACKUP_LOCK_MAX_AGE_SECONDS` (default 3600). Restore via `scripts/restore_state.py <archive.tar.gz>`; see [../../docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md#backup-and-restore).

The screener alert runs weekdays at 14:30 local — on this Atlantic/Reykjavik (GMT) machine that is **10:30 ET in EDT / 09:30 ET in EST**, i.e. inside the US options session. It is deliberately NOT pre-open: the ranked screener reads yfinance, which returns bid=ask=0 and a placeholder IV (~1e-5) before the open, so a pre-open run scores absent quotes as maximally cheap vol. It runs `scripts/premarket_screener_alert.py`, which calls the canonical ranked screener (`services.screener_service.build_ranked_screener`, the same path behind `/api/screener/ranked`) in-process — **the backend does not need to be running** — and sends an iMessage only when a setup inside the entry window clears the score threshold. Silence is the expected default: qualifying setups are rare (~10-40/year), and a daily "nothing today" message would train the operator to ignore the channel. The job is idempotent per `(date, qualifying-symbol-set)` via `~/.options_calculator_pro/state/premarket_alert_state.json`, so a manual re-run the same day will not double-send (override with `--force`). The message carries setup-quality ordering only — `ranking_score` is **not** a calibrated win probability and the payload says so. Recipient comes from `WATCHDOG_IMESSAGE_TO` (shared with the evidence watchdog); with no recipient set the job exits non-zero rather than failing silently. Tunables: `PREMARKET_ALERT_MIN_SCORE` (default 0.65), `PREMARKET_ALERT_TOP` (default 5), `PREMARKET_ALERT_EXTRA_ARGS`, `PREMARKET_ALERT_TIMEOUT_SECONDS` (default 600), `PREMARKET_ALERT_LOCK_MAX_AGE_SECONDS` (default 3600). Preview without sending:

```sh
.venv311/bin/python scripts/premarket_screener_alert.py --dry-run
```

Operators in other timezones: change `Hour=13` in `com.optionscalculator.premarket-screener-alert.plist` to your local equivalent of "~30-90 minutes before the 09:30 ET open", then reinstall.

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
