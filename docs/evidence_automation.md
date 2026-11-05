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
- records shadow baseline quotes for `always_atm_straddle`, `always_otm_strangle` and `always_iron_condor` in two cohorts:
  - `paired`: beside each selector paper entry (the `baseline_comparison` in the Evidence Report)
  - `universe`: once per eligible earnings event on its first day in the DTE window, whatever the selector recommended, so No Trade calls have a counterfactual (`universe_shadow`, split by selector decision)
- finalizes due paper outcomes at T-1 when quotes are available
- finalizes due shadow baselines by repricing the contracts booked at entry, without updating calibration or priors. Baselines resolved before booked-strike exits existed re-discovered strikes at exit; they are reported only under `legacy_repriced_baselines` and excluded from comparisons
- because exits are at T-1, the iron condor baseline measures shorting the pre-earnings IV run-up, not the post-earnings crush
- an exit is valued only on T-1 and only on the booked contracts (every leg strike, the expiry, and the contract symbol where both sides recorded one). Running the loop again the same day retries a failed exit; once T-1 has passed, a trade or baseline that was never priced moves to the terminal status `exit_missing` (never re-priced on a later date or a nearby contract). The Evidence Report counts this under `exit_attrition` by reason and structure
- baselines whose entry context lacks a leg strike or the expiry resolve as `unverifiable_entry_context` and are excluded from comparisons
- the calibration and structure-prior stores serialize every update behind a cross-process file lock (sidecar `.<name>.lock` files next to each JSON store), re-read the file before adding an observation, and write atomically; a failed write raises instead of being reported as learned, and the trade is left with a retryable `learning_update_status`
- finalizing a trade takes an exclusive, owner-token claim (15-minute lease); a second worker can neither claim, overwrite the exit of, nor finalize a trade another worker is finalizing
- a recorded exit is final: it is never re-quoted, not even by a same-day retry. A finalize that fails or loses its claim after the exit was written hands the claim back, and the next loop run (from the exit day on) re-finalizes the trade from the stored exit, keeping every exit field, and reports it as `refinalized`. An exit recorded without realized values (including a legacy NaN, which SQLite stores as NULL) becomes `exit_missing` with `exit_recorded_without_value` once the exit day has passed
- a finalized trade whose calibration or prior write failed is retried by every loop run (`learning_retried`); the health check warns if it still fails after 24h
- exits and entries are valued only at a point in time: the live quote fetcher is refused for a run date other than today (`live_quote_refused_for_past_as_of`), and a quote whose timestamp falls on another local day is not used (`quote_not_from_as_of_date`). A missed T-1 exit therefore becomes `exit_missing`, never a later price
- seeded replay/backtest rows (`scripts/seed_outcomes_from_replay.py`) are never forward evidence: the report, maturity, gate, attrition and health figures count forward paper trades only, and replay rows are counted separately under `excluded_replay_outcomes`. The loop never exits or expires a replay row; a seed run completes rows an interrupted run left open
- outcomes invalidated as evidence (`scripts/invalidate_outcome.py`, or a `notes.evidence_invalidated` flag) are refused by every exit/finalize/learning write and excluded from reports before the row limit; invalidating a trade while its learning updates run is refused, so retry once it is finalized
- realized values that are NaN, infinite or boolean are refused when an outcome or baseline exit is written. An exit that would produce one is skipped with reason `non_finite_exit_value` (selector trades retry that day, then become `exit_missing`; baselines become `exit_skipped`), and replay rows with one are counted as bad data by the seed script. Older rows holding one are excluded from every report figure (including maturity, attrition, universe and forward diagnostics), counted under `non_finite_outcomes`, and moved to `exit_missing` if they were never finalized
- an option surface classified `degraded_surface` holds the selector at Watch: the event is recorded and shadow-priced, never actionable
- writes a daily evidence snapshot to `~/.options_calculator_pro/reports/evidence/`

Safe dry run:

```bash
./.venv311/bin/python scripts/run_evidence_cycle.py --dry-run
```

Recommended collection window:

- minimum: 60 calendar days
- target: 90 calendar days
- do not market the tool as edge-proven until the Evidence Report has enough resolved paper outcomes and baseline comparisons
- `commercialization_gate.ready_for_paid_beta` is true only when every condition holds (60+ evidence days, 30+ resolved selector outcomes, 30+ of them claimable, meaningful baseline comparison, interpretable calibration buckets, maturity past early observation); `blocking_reasons` names each unmet one

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

- `com.optionscalculator.evidence-cycle`: daily at 16:45 local time.

  The cycle prices entries and T-1 exits from live option quotes, so it runs
  only while the NYSE regular session is open. 16:45 on a GMT Mac is 11:45 ET
  in EST and 12:45 ET in EDT, before the 13:00 early closes; on a Mac in
  another time zone, set the plist to the local time of about 12:00 ET. A
  scheduled run on a day NYSE is closed (weekends, exchange holidays) logs
  `skipped` and writes no report; the watchdog and health check expect none
  that day, and allow for the gap afterwards. A scheduled run on a trading
  day outside the session (the Mac was asleep at 16:45 and launchd fired it
  on wake) refuses to price, exits 3, and logs a failure, so the missed day
  is visible rather than filled from stale quotes. Manual runs with `--date`
  or `--dry-run` are not held to the session; `--allow-outside-session`
  forces a live run.

  Before this schedule (until early November 2026) the cycle ran at 21:30
  local, after the close and on weekends, so entries and exits up to then
  were priced from after-hours yfinance quotes. Treat that as a break in the
  evidence: compare cohorts on the same side of it.
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

It also checks which code is running. Each evidence report records the commit
the cycle ran (`code_identity`). The watchdog fetches `origin/main` and alerts
when the checkout the jobs run from is behind it, or when today's report came
from a different commit than that checkout (two checkouts writing the same
state). A failed fetch, an unidentifiable checkout or uncommitted changes are
reported as warnings without paging. Pass `--no-fetch` to compare against the
last fetched `origin/main` instead.

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
- evidence integrity (`evidence_integrity`, WARN), read-only against the outcome and baseline stores:
  - trades stuck in `finalizing` for more than 24h (a finalize crashed mid-way)
  - trades still `open` after their T-1 exit day (exit detection is not running)
  - trades with an exit recorded but never finalized (`exited_not_finalized`; the loop re-finalizes them)
  - selector or baseline exit attrition above 20% over the last 90 days, with at least 5 exits
  - universe shadow entries that never entered, for more than 25% of closed events
  - invalidated outcomes whose observation is in the calibration or prior store (read from the store files, so it clears once the backfill repair has run; run `scripts/backfill_prior_store_timestamps.py`)
  - finalized trades whose learning update has kept failing for over 24h
  - dates are local: a trade is overdue only once the day after its earnings date has begun

Each issue includes a fix suggestion. The health check does not run the selector
and does not change strategy state.

Useful flags:

```bash
./.venv311/bin/python scripts/check_evidence_health.py --no-completion-log-required
./.venv311/bin/python scripts/check_evidence_health.py --date 2026-04-27
```

## Pre-registration

What the forward evidence is meant to show, which events count, the fixed
analysis, and the kill criteria are set in [PREREGISTRATION.md](PREREGISTRATION.md)
before the cohort starts. Note its first precondition: the claim contract only
admits MarketData.app quotes or live IB Gateway quotes, so with yfinance alone
(or a delayed IB feed) nothing becomes claimable.

## Quote Provenance

Every analysis records where and when its option quotes came from
(`services/quote_provenance.py`). The recommendation ledger's `quote_source`,
`quote_timestamp` and `quote_quality` columns are filled from it, and the full
record is kept in the row's `metadata_json.quote_provenance`:

- `quote_source`: the provider that supplied the chain (`marketdata`,
  `yfinance`, ...);
- `fetched_at`: when the chain was fetched (UTC);
- `quote_timestamp`: the provider's own quote time when it has one (the median
  of MarketData's per-contract `updated`), otherwise the fetch time;
  `quote_quality` says which (`provider_quote_time` or `fetch_time_only`).
  yfinance has no quote time, so for it only the fetch time is known, and no
  quote time is invented;
- `in_regular_session` and `quote_session`: whether NYSE was open at the fetch,
  and which session's quotes were current.

The forward loop refuses to price forward evidence from an analysis whose
quotes were fetched outside the regular session, whose provider quotes were
more than 30 minutes old at the fetch, or that has no provenance at all. Such
a candidate stops at the `eligible` stage with
`untrusted_quotes:<outside_session|stale_quotes:<n>m|missing_quote_provenance>`
before any trade or universe-shadow baseline is priced. The analysis itself is
still written to the ledger, so the stop is auditable. The funnel diagnosis
lists these stops with the other eligibility stops.

## Funnel Diagnosis

The daily report's funnel section counts candidate passes over at most 30
days. To see where forward evidence actually stops, run:

```bash
./.venv311/bin/python scripts/diagnose_evidence_funnel.py          # text summary
./.venv311/bin/python scripts/diagnose_evidence_funnel.py --json   # full detail
./.venv311/bin/python scripts/diagnose_evidence_funnel.py --since 2026-09-01
./.venv311/bin/python scripts/diagnose_evidence_funnel.py \
  --recommendations /path/to/recommendation_ledger.sqlite
```

It reads both the funnel store and the optional recommendation ledger read-only
and reports, over all history:

- every earnings event's furthest stage and how it ended there: claimable,
  in flight, stopped (with the reason), deferred, or deferred and never
  picked up again before the event passed;
- the binding constraint: the stage that ends the most events, and its
  leading reasons (`no_trade:<gate>` names the selector gate that said no);
- underlying scorecard eligibility causes for `no_eligible_structure`, such as
  spread/liquidity, surface quality, and missing legs;
- the selector-ranked eligible structure and its modeled edge for each
  `non_positive_edge` stop, using the persisted selection when available and
  never substituting an ineligible or merely higher-edge scorecard;
- event-level match status plus aggregate usable/matched counts, so missing,
  malformed, or unclassified recommendation evidence is visible rather than
  overstated. A stop is matched to the ledger row by the `recommendation_id`
  the funnel recorded with it (`match_method: recommendation_id`), so a UI
  analysis of the same symbol that day never stands in for the forward loop's
  own; only stops recorded without an id fall back to the newest row for the
  event and gate (`match_method: event_key`);
- event identities and last-recorded dates for deferred events that passed
  without being revisited and for events with no downstream record;
- claimable trades per week over the last 8 weeks, and at that rate how many
  weeks until the paid-beta gate's 30-outcome sample.

If the recommendation ledger does not exist or cannot be read with the expected
schema, the core funnel diagnosis still runs, reports the enrichment status in
JSON/text, and warns on stderr. When recommendation revisions exist, the newest
complete revision is used rather than the immutable first-write payload. The
default ledger is
`~/.options_calculator_pro/recommendations/recommendation_ledger.sqlite`.

An event whose earnings date moved is counted once per date.

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
