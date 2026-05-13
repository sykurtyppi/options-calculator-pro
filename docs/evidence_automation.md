# Evidence Automation

This project should collect forward evidence before any paid beta claims are made.
The daily automation entrypoint is:

```bash
cd /Users/tristanalejandro/Downloads/options_calculator_pro
./.venv311/bin/python scripts/run_evidence_cycle.py
```

What it does:

- runs the forward screener and selector
- records actionable paper entries
- records shadow baseline quotes for `always_atm_straddle` and `always_otm_strangle`
- finalizes due paper outcomes at T-1 when quotes are available
- finalizes due shadow baselines without updating calibration or priors
- writes a daily evidence snapshot to `~/.options_calculator_pro/reports/evidence/`

Safe dry run:

```bash
./.venv311/bin/python scripts/run_evidence_cycle.py --dry-run
```

Recommended collection window:

- minimum: 60 calendar days
- target: 90 calendar days
- do not market the tool as edge-proven until the Evidence Report has enough resolved paper outcomes and baseline comparisons

## macOS launchd Scheduling

Use `launchd` on the Mac mini. The daily evidence cycle already runs the
forward loop, records paper recommendations, resolves due outcomes, and writes
the daily evidence report. Do not schedule a second standalone forward loop
unless you intentionally want a separate research run.

Install jobs:

```bash
cd /Users/tristanalejandro/Downloads/options_calculator_pro
scripts/automation/install_launchd_jobs.sh
```

Installed jobs:

- `com.optionscalculator.evidence-cycle`: daily at 21:30 local time.
- `com.optionscalculator.evidence-watchdog`: daily at 22:15 local time.
- `com.optionscalculator.weekly-evidence-report`: weekly Monday at 22:45 local time.

Check loaded jobs:

```bash
launchctl list | grep optionscalculator
```

Disable jobs:

```bash
cd /Users/tristanalejandro/Downloads/options_calculator_pro
scripts/automation/uninstall_launchd_jobs.sh
```

Manual launchd-style commands:

```bash
scripts/automation/run_daily_evidence_cycle.sh
scripts/automation/run_daily_evidence_watchdog.sh
scripts/automation/run_weekly_evidence_report.sh
```

The weekly report is also produced automatically by the Monday daily evidence
cycle, but the separate weekly job is intentionally present as a clean export
cadence that does not rerun the selector.

The launchd wrappers include operational guardrails:

- daily evidence cycle timeout: `EVIDENCE_CYCLE_TIMEOUT_SECONDS`, default `7200`
- daily evidence cycle stale-lock expiry: `EVIDENCE_CYCLE_LOCK_MAX_AGE_SECONDS`, default `10800`
- watchdog timeout: `EVIDENCE_WATCHDOG_TIMEOUT_SECONDS`, default `180`
- watchdog stale-lock expiry: `EVIDENCE_WATCHDOG_LOCK_MAX_AGE_SECONDS`, default `600`
- weekly report timeout: `WEEKLY_EVIDENCE_REPORT_TIMEOUT_SECONDS`, default `900`
- weekly report stale-lock expiry: `WEEKLY_EVIDENCE_REPORT_LOCK_MAX_AGE_SECONDS`, default `1800`

## Logs And Reports

Human-readable launchd logs:

```text
~/.options_calculator_pro/logs/daily_evidence_cycle_launchd.log
~/.options_calculator_pro/logs/daily_evidence_watchdog_launchd.log
~/.options_calculator_pro/logs/weekly_evidence_report_launchd.log
```

Structured operational JSONL logs:

```text
~/.options_calculator_pro/logs/evidence_cycle_runs.jsonl
~/.options_calculator_pro/logs/weekly_evidence_exports.jsonl
```

Evidence reports:

```text
~/.options_calculator_pro/reports/evidence/latest.json
~/.options_calculator_pro/reports/evidence/evidence_report_YYYY-MM-DD.json
~/.options_calculator_pro/reports/evidence/weekly/weekly_latest.json
~/.options_calculator_pro/reports/evidence/weekly/weekly_evidence_report_YYYY-MM-DD.json
```

Structured run logs include:

- start/end time and duration
- recommendations recorded
- paper entries and resolved outcomes
- provider failures, fallback count, and stale count
- evidence maturity status
- weekly report path when generated
- warnings and errors

## Watchdog / iMessage Alerts

The daily watchdog checks that the evidence cycle produced today's report and
that the launchd log contains a completion marker. It does not re-run models or
change decisions.

Manual health check:

```bash
./.venv311/bin/python scripts/watch_daily_evidence_cycle.py
```

Dry-run alert flow:

```bash
./.venv311/bin/python scripts/watch_daily_evidence_cycle.py --dry-run-alert --force-alert
```

iMessage alerts use the local macOS Messages app if this `.env` value is
configured:

```bash
WATCHDOG_IMESSAGE_TO=+15551234567
```

The value can be a phone number or Apple ID that Messages can reach from this
Mac. The first live send may trigger a macOS automation permission prompt for
Terminal/launchd to control Messages.

Alert state is stored at:

```text
~/.options_calculator_pro/state/daily_evidence_watchdog_alerts.json
```

The watchdog suppresses repeated iMessages for the same failure signature so a
provider outage does not spam your phone.

## Health Check

Run this any time:

```bash
cd /Users/tristanalejandro/Downloads/options_calculator_pro
./.venv311/bin/python scripts/check_evidence_health.py
```

The health check returns:

- `OK`: reports, telemetry, logs, and SQLite stores look healthy.
- `WARN`: evidence is still usable, but something needs attention soon.
- `FAIL`: evidence automation or persistence is broken enough to require action.

Checks performed:

- last successful evidence-cycle run
- latest daily evidence report freshness
- latest weekly report freshness
- provider telemetry freshness
- recommendation ledger SQLite readability/writability
- outcome store SQLite readability/writability
- baseline store SQLite readability/writability
- provider telemetry SQLite readability/writability
- launchd watchdog status

Each issue includes a fix suggestion. The health check does not run the selector
and does not change strategy state.

Useful flags:

```bash
./.venv311/bin/python scripts/check_evidence_health.py --no-completion-log-required
./.venv311/bin/python scripts/check_evidence_health.py --date 2026-04-27
```

## Troubleshooting

If the daily report is missing:

```bash
./.venv311/bin/python scripts/run_evidence_cycle.py
tail -n 80 ~/.options_calculator_pro/logs/daily_evidence_cycle_launchd.log
```

If launchd did not run:

```bash
launchctl list | grep optionscalculator
scripts/automation/install_launchd_jobs.sh
```

If provider telemetry is stale, check credentials and network first, then run:

```bash
./.venv311/bin/python scripts/run_evidence_cycle.py --dry-run
./.venv311/bin/python scripts/check_evidence_health.py
```

If SQLite health fails, do not delete databases casually. Back them up first,
then inspect with `sqlite3` and `PRAGMA quick_check`.
