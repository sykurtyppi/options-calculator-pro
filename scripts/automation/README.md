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
| `com.optionscalculator.premarket-screener-alert` | Weekdays at 15:00 |
| `com.optionscalculator.state-backup` | Daily at 04:00 |

All jobs use `RunAtLoad=false`; they fire only on the calendar schedule, never on `launchctl load`.

The state-backup job runs at 04:00 local — offset from the 03:00 log rotation so the two don't contend for the SQLite files. It calls `scripts/backup_state.py` to write a hot, consistent `.tar.gz` snapshot of `~/.options_calculator_pro` (SQLite files via the Online Backup API, everything else byte-copied) with newest-N retention. By default the archive lands in `~/.options_calculator_pro/backups/`, which guards against `rm -rf` typos and SQLite corruption but **not disk loss** (same disk as the source). For real disaster recovery, set `OPTIONS_CALCULATOR_BACKUP_DIR` to an external sync folder (iCloud Drive / Dropbox / external mount) — the wrapper passes it through as `--output-dir`. The wrapper reads `OPTIONS_CALCULATOR_BACKUP_DIR` / `OPTIONS_CALCULATOR_BACKUP_RETENTION` from the project `.env` (parsed, never sourced) because launchd provides a bare environment; a value in the process environment still overrides the file. Tunables: `OPTIONS_CALCULATOR_BACKUP_DIR` (default in-tree `backups/`), `OPTIONS_CALCULATOR_BACKUP_RETENTION` (default 14), `STATE_BACKUP_TIMEOUT_SECONDS` (default 900), `STATE_BACKUP_LOCK_MAX_AGE_SECONDS` (default 3600). Restore via `scripts/restore_state.py <archive.tar.gz>`; see [../../docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md#backup-and-restore).

The screener alert runs weekdays at 15:00 local — on this Atlantic/Reykjavik (GMT) machine that is **11:00 ET in EDT / 10:00 ET in EST**, i.e. at least 30 minutes after the open in both DST regimes (14:30 would have been the opening bell during EST). It is deliberately NOT pre-open: the ranked screener reads yfinance, which returns bid=ask=0 and a placeholder IV (~1e-5) before the open, so a pre-open run scores absent quotes as maximally cheap vol. It runs `scripts/premarket_screener_alert.py`, which calls the canonical ranked screener (`services.screener_service.build_ranked_screener`, the same path behind `/api/screener/ranked`) in-process — **the backend does not need to be running** — and sends an iMessage only when a setup inside the entry window clears the score threshold. Silence is the expected default: qualifying setups are rare (~10-40/year), and a daily "nothing today" message would train the operator to ignore the channel. Each `(date, qualifying-symbol-set)` alert is sent at most once:
- **Durable outbox.** It is reserved in a SQLite outbox at `~/.options_calculator_pro/state/premarket_alert_outbox.sqlite` before anything is handed to Messages, so a manual re-run or a concurrent run will not double-send. Override with `--force`.
- **Send states.** Each alert moves through `prepared → provider_submitted → accepted / delivery_unknown / failed`.
- **Unknown delivery.** Once `osascript` has been launched, any error is `delivery_unknown`: a timeout, a non-zero exit, or an error reading its output may all come after Messages took the alert. Only an error that proves `osascript` never ran is a retryable `failed`: the external-IO gate refusing, or the OS failing to execute it. A `delivery_unknown` alert is never retried automatically, and later runs exit 1 until someone checks.
- **Unreadable state.** An unreadable outbox or `premarket_alert_state.json` is moved aside as `*.corrupt-<timestamp>`, and nothing is sent.
- **Quarantined outbox.** Quarantining the outbox first writes `premarket_alert_outbox.sqlite.quarantined.json` next to it, because the moved file was the only record of what was already sent. While that marker exists every run sends nothing and exits 1. Check the quarantined file against what was actually delivered, then re-run once with `--outbox-reconciled` to clear the marker. If the marker cannot be written, the damaged outbox is left in place, which blocks sending the same way. The message lists pre-earnings long-vol **candidates**: `ranking_score` is a heuristic ordering (weights and the 0.65 cutoff are not empirically calibrated), **not** a win probability, and a candidate still needs the forward loop's selector to approve it — the payload says so and states the New York time it was built. A row must carry real evidence, not neutral defaults: term structure, median historical move, an ATM spread no wider than 15%, an ATM IV between 1% and 500%, and a non-stale earnings date. An unconfirmed earnings date is labelled `[date unconfirmed]`, not dropped. Every alert appends its full rows, score components (including the IV/RV actually scored) and drop counts to `~/.options_calculator_pro/logs/premarket_alert_audit.jsonl` as `prepared` before the send, then a line with its final state. Recipient comes from `WATCHDOG_IMESSAGE_TO` (shared with the evidence watchdog); with no recipient set the job exits non-zero rather than failing silently. Tunables: `PREMARKET_ALERT_MIN_SCORE` (default 0.65), `PREMARKET_ALERT_TOP` (default 5), `PREMARKET_ALERT_EXTRA_ARGS`, `PREMARKET_ALERT_TIMEOUT_SECONDS` (default 600), `PREMARKET_ALERT_LOCK_MAX_AGE_SECONDS` (default 3600). Preview without sending:

```sh
.venv311/bin/python scripts/premarket_screener_alert.py --dry-run
```

Operators in other timezones: change `Hour=15` in `com.optionscalculator.premarket-screener-alert.plist` to your local equivalent of "at least 30 minutes AFTER the 09:30 ET open in both DST regimes" (never pre-open — yfinance has no usable quotes before the bell), then reinstall.

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
