# Deployment runbook

Operational guide for installing, verifying, and running Options Calculator
Pro on a single host with autonomous evidence collection. Audience: future
you, at 2am, who knows this project existed but doesn't remember the
details.

This is a **single-user / small-team deployment** behind a proxy or tunnel
(Cloudflare, Tailscale Funnel, nginx). Not designed for public internet
exposure without additional review.

> **What this is not.** Nothing this system outputs is a trading
> recommendation. Watchdog alerts page on **pipeline health**, never on
> profit/loss or trading-signal quality. See [Alerts and what they
> mean](#alerts-and-what-they-mean) before silencing anything.

---

## Contents

- [Deployment shape](#deployment-shape)
- [Prerequisites](#prerequisites)
- [Environment configuration](#environment-configuration)
- [First-time install](#first-time-install)
- [First-run verification](#first-run-verification)
- [Day-to-day operations](#day-to-day-operations)
- [Alerts and what they mean](#alerts-and-what-they-mean)
- [Troubleshooting matrix](#troubleshooting-matrix)
- [Uninstall](#uninstall)

---

## Deployment shape

```
┌─────────────┐   HTTPS / tunnel    ┌──────────────────────────┐
│  Browser    │ ─────────────────▶  │  Reverse proxy / tunnel  │
│  (operator) │                     │  (Cloudflare / Tailscale)│
└─────────────┘                     └──────────┬───────────────┘
                                               │  X-Forwarded-For
                                               ▼
                                    ┌─────────────────────────┐
                                    │  uvicorn :8000          │
                                    │  FastAPI + React build  │
                                    │  (web/api/app.py)       │
                                    └──────────┬──────────────┘
                                               │
                          ┌────────────────────┼──────────────────────┐
                          ▼                    ▼                      ▼
                ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
                │  SQLite stores   │  │  Evidence reports│  │  Telemetry JSONL │
                │  ~/.options_     │  │  ~/.options_     │  │  ~/.options_     │
                │  calculator_pro/ │  │  calculator_pro/ │  │  calculator_pro/ │
                │  *.sqlite        │  │  reports/        │  │  logs/*.jsonl    │
                └──────────────────┘  └──────────────────┘  └──────────────────┘
                          ▲
                          │  daily writes
                          │
            ┌─────────────┴──────────────┐
            │  5 launchd jobs (macOS)    │
            │  resolver, cycle, watchdog,│
            │  weekly, log-rotation      │
            └────────────────────────────┘
```

The backend serves the API and (in production) the built React frontend
from the same origin. The five launchd jobs run on schedule to produce
evidence reports, resolve candidate exits, and rotate logs.

---

## Prerequisites

- **macOS** (the launchd jobs are macOS-specific; the FastAPI backend
  itself runs on Linux too, but the operational automation does not).
- **Python 3.11** with a virtualenv at `${PROJECT_ROOT}/.venv311/` —
  every launchd wrapper hardcodes `${PROJECT_ROOT}/.venv311/bin/python`.
- **Node 20+ / npm** for the frontend build.
- The repo cloned somewhere stable; the install scripts use
  `dirname "${BASH_SOURCE[0]}"` so the path can be anything, but moving
  the repo after install requires reinstalling the launchd jobs (they
  capture absolute paths at install time).
- A **non-empty `.env`** in the repo root with at minimum the
  data-provider keys; see [README.md](../README.md#marketdata-app-setup-recommended).

```sh
# One-time Python env setup
[ -d .venv311 ] || uv venv .venv311 --python 3.11
source .venv311/bin/activate
uv pip install -r requirements.lock
deactivate

# One-time frontend build (production, same-origin)
cd web/frontend
npm install
npm run build
cd ../..
```

---

## Environment configuration

Three categories of env var. **Read all three before flipping
`ENABLE_SHARE_AUTH=true`** — the backend's startup validators refuse to
boot if the combination is unsafe.

### A. Required when share-auth is enabled

Set in `.env` at the repo root. The backend runs `_validate_auth_config()`
at module load; missing or weak values cause a clear `RuntimeError` on
import.

| Variable | Required when | Notes |
|---|---|---|
| `ENABLE_SHARE_AUTH` | always (set to `true` to enable) | Without this the auth middleware is permissive. |
| `SHARE_PASSWORD` | `ENABLE_SHARE_AUTH=true` | Single shared access code. `hmac.compare_digest` is used on compare. |
| `SESSION_SECRET` | `ENABLE_SHARE_AUTH=true` | ≥24 chars, NOT `change-me-in-env`. Used as HMAC-SHA256 key over `v1.<issued_at>.<nonce>` cookie payload. |
| `OPTIONS_CALCULATOR_ALLOWED_ORIGINS` | `ENABLE_SHARE_AUTH=true` | Comma-separated. Must NOT include `*` or `null` — the startup validator refuses both because they combine unsafely with `allow_credentials=True`. |
| `OPTIONS_CALCULATOR_SECURE_COOKIES` or `OPTIONS_CALCULATOR_HOSTED_MODE` or `OPTIONS_CALCULATOR_ALLOW_INSECURE_SESSION_COOKIE` | **at least one** when `ENABLE_SHARE_AUTH=true` | Hosted mode implies Secure cookies (so setting both is harmless and common). The "allow insecure" escape hatch is for plaintext-HTTP local dev; production should always use Secure. Multiple flags can coexist — the validator only requires that at least one safe posture is selected. |

### B. Recommended for production-behind-tunnel

| Variable | Notes |
|---|---|
| `OPTIONS_CALCULATOR_HOSTED_MODE=true` | Implies `OPTIONS_CALCULATOR_SECURE_COOKIES=true`, `OPTIONS_CALCULATOR_PROTECT_API_DOCS=true`, and `OPTIONS_CALCULATOR_TRUST_PROXY_HEADERS=true`. |
| `OPTIONS_CALCULATOR_TRUST_PROXY_HEADERS=true` | Honour leftmost `X-Forwarded-For` for per-IP rate-limiting on `/login`, `/api/ml/train`, `/api/oos/submit`. Without this, behind a tunnel ALL login attempts share one global bucket. Spoof-safe only when a real proxy is enforced in front. |
| `WATCHDOG_IMESSAGE_TO` | Address (phone or email) that receives daily watchdog iMessage alerts. If unset, the watchdog logs but does not page. |

### C. Optional tuning

| Variable | Default | Notes |
|---|---|---|
| `OPTIONS_LOGIN_RATE_LIMIT_PER_MIN` / `_PER_HOUR` | 5 / 30 | Per-IP cap on `/login`. |
| `OPTIONS_ML_TRAIN_RATE_LIMIT_PER_MIN` / `_PER_HOUR` | 2 / 10 | Per-IP cap on `/api/ml/train`. |
| `OPTIONS_OOS_SUBMIT_RATE_LIMIT_PER_MIN` / `_PER_HOUR` | 3 / 20 | Per-IP cap on both `/api/oos/submit` and `/api/oos/report-card` (shared bucket). |
| `OPTIONS_CALCULATOR_MAX_OOS_RUNNING_JOBS` | 1 | Concurrency cap for OOS jobs. |
| `OPTIONS_CALCULATOR_MAX_ML_RUNNING_JOBS` | 1 | Concurrency cap for ML training. |
| `EVIDENCE_CYCLE_TIMEOUT_SECONDS` | 7200 | Hard cap for daily evidence cycle. |
| `CANDIDATE_EXIT_RESOLVER_TIMEOUT_SECONDS` | 1800 | Hard cap for resolver. |
| `LOG_ROTATION_TIMEOUT_SECONDS` | 300 | Hard cap for log rotation. |

### D. Frontend build-time

`VITE_API_BASE` is substituted at `npm run build` time and validated at
module load in [`web/frontend/src/lib/api.js`](../web/frontend/src/lib/api.js).
The validator refuses unsafe values; reasoning is in PR #60's commit.

| Build context | `VITE_API_BASE` | What the frontend will do |
|---|---|---|
| Production (frontend served by backend) | **unset** | Same-origin — every API call goes to the backend that served the page. |
| Local dev (Vite `:5173` + backend `:8000`) | **unset** | `import.meta.env.DEV` is true → defaults to `http://127.0.0.1:8000`. |
| Production cross-origin (frontend on its own host) | `https://api.example.com` | HTTPS required. |

The validator **throws at module load** on:
- `http://anything-other-than-localhost-or-127.0.0.1` (cookies would
  ship over plaintext)
- Malformed URLs
- `file://`, `javascript:`, or anything else non-http(s)

### E. Sample `.env` for a TLS-tunnel deployment

```sh
# Data providers (see README.md for the full list)
MARKETDATA_TOKEN=...
ALPHA_VANTAGE_API_KEY=...
SEC_API_USER_AGENT=your-email@example.com options-calculator-pro/1.0

# Auth
ENABLE_SHARE_AUTH=true
SHARE_PASSWORD=long-random-shared-code
SESSION_SECRET=at-least-24-random-chars-totally-not-the-default

# Production posture. HOSTED_MODE is a convenience flag that implies
# all three of:
#   - OPTIONS_CALCULATOR_SECURE_COOKIES=true   (cookie marked Secure)
#   - OPTIONS_CALCULATOR_PROTECT_API_DOCS=true (/docs and /redoc behind auth)
#   - OPTIONS_CALCULATOR_TRUST_PROXY_HEADERS=true (XFF honoured for rate-limit)
# Set the individual flags instead if you want to opt into only some of
# these behaviours.
OPTIONS_CALCULATOR_HOSTED_MODE=true

# CORS (NO wildcards allowed under share-auth)
OPTIONS_CALCULATOR_ALLOWED_ORIGINS=https://your-tunnel-domain.example.com

# iMessage alert recipient (optional but recommended)
WATCHDOG_IMESSAGE_TO=+15551234567
```

---

## First-time install

```sh
cd /path/to/options_calculator_pro

# 1. Verify the Python env (re-run setup from "Prerequisites" if missing)
ls .venv311/bin/python || echo "MISSING — set up the venv first"

# 2. Build the frontend if you haven't (production same-origin assumes
#    a fresh dist/ exists; backend serves it via StaticFiles).
cd web/frontend && npm install && npm run build && cd ../..

# 3. Confirm .env contains the required share-auth combination (see §C).
#    The backend will refuse to start if anything is missing — that's
#    by design. Smoke-test the validator before installing launchd jobs:
.venv311/bin/python -c "from web.api import app; print('startup ok')"

# 4. Install all five launchd jobs. The script template-renders each
#    plist (substituting __PROJECT_ROOT__ and __HOME__), drops the
#    rendered version into ~/Library/LaunchAgents/, and launchctl-loads
#    it.
scripts/automation/install_launchd_jobs.sh

# 5. Confirm the jobs are loaded.
launchctl list | grep optionscalculator
# Expected: 5 lines, each ending with "0" (last exit status).
```

Expected `launchctl list` output:

```
-	0	com.optionscalculator.candidate-exit-resolver
-	0	com.optionscalculator.evidence-cycle
-	0	com.optionscalculator.evidence-watchdog
-	0	com.optionscalculator.log-rotation
-	0	com.optionscalculator.weekly-evidence-report
```

The first column is PID (`-` = not currently running). Second is the last
exit status. `RunAtLoad=false` is set on every plist, so all five sit
quiet until their `StartCalendarInterval` fires.

---

## First-run verification

After install, the jobs won't fire until their scheduled time. The
verification sequence below is what to check **after each first-fire**
or **immediately if you want to force one**.

### Force-fire a single job (debugging only)

```sh
launchctl kickstart -k gui/$(id -u)/com.optionscalculator.candidate-exit-resolver
# Wait 5–10s, then:
tail -50 ~/.options_calculator_pro/logs/candidate_exit_resolver_launchd.log
```

Expected tail of a clean run:

```
===== 2026-05-27T15:57:32Z candidate exit resolver start =====
PR-AE resolver run @ 2026-05-27T15:57:33+00:00 (now=2026-05-27)
  scanned:                  0 candidate rows from ledger
  ...
  count_balance_holds:    true
===== 2026-05-27T15:57:33Z candidate exit resolver complete =====
```

The line `count_balance_holds: true` is the integrity invariant — if it
ever says `false`, the resolver bucketing is wrong and the launchd job
exits non-zero. Watchdog will alert.

### Check overall evidence health

```sh
.venv311/bin/python scripts/check_evidence_health.py
```

Outputs JSON. The top-level `status` is one of `OK`, `WARN`, or `FAIL`.
`WARN` covers expected first-day states ("evidence report not generated
yet — daily cycle fires at 22:15 local"). `FAIL` is what the watchdog
escalates to iMessage.

### Watchdog dry-run

```sh
.venv311/bin/python scripts/watch_daily_evidence_cycle.py --dry-run-alert
```

Validates the alert flow without sending iMessage. On a fresh install
before 22:15 local, expect:

```json
{
  "alert": {"attempted": false, "reason": "not_due_yet"},
  ...
}
```

After the daily cycle has fired successfully, expect `"reason":
"watchdog_ok"`.

### Log rotation dry-run

```sh
.venv311/bin/python scripts/rotate_launchd_logs.py --dry-run
```

Lists every launchd log under `~/.options_calculator_pro/logs/` and
whether it would rotate (`WOULD ROTATE` if ≥5MB, `ok` otherwise,
`skip(self)` for the rotator's own logs). On a fresh install everything
should be tiny and `ok`.

### Backend smoke

```sh
.venv311/bin/python -m uvicorn web.api.app:app --host 127.0.0.1 --port 8000 &
sleep 2

# 1. Public health check (unauthenticated).
curl -i http://127.0.0.1:8000/api/health
# Expected: HTTP 200 with {"status": "ok", "timestamp": "..."}.

# 2. Authenticated endpoint without cookie → 401.
curl -i http://127.0.0.1:8000/api/diagnostics/learning
# Expected: HTTP 401 with detail "Unauthorized — please log in at /login".

# 3. Login flow.
curl -i -c /tmp/cookies.txt -X POST \
  -d "password=${SHARE_PASSWORD}" \
  http://127.0.0.1:8000/login
# Expected: HTTP 303 with Set-Cookie: ops_session=v1.... Secure; HttpOnly; SameSite=Lax

# 4. Authenticated call with cookie.
curl -i -b /tmp/cookies.txt http://127.0.0.1:8000/api/diagnostics/learning
# Expected: HTTP 200.

# 5. Logout.
curl -i -b /tmp/cookies.txt -X POST http://127.0.0.1:8000/logout
# Expected: HTTP 303 with the cookie cleared. The same cookie replayed
# next would now fail because its nonce was added to _revoked_nonces.

kill %1
rm /tmp/cookies.txt
```

---

## Day-to-day operations

### Schedule (local time)

| Job | When | Purpose |
|---|---|---|
| `candidate-exit-resolver` | Daily 12:30 | Resolve pending candidate-shadow-outcome rows (PR-AE). |
| `evidence-cycle` | Daily 21:30 | Run the daily forward loop + write evidence-report snapshot. |
| `evidence-watchdog` | Daily 22:15 | Check that today's evidence report is present + fresh; iMessage on failure. |
| `weekly-evidence-report` | Mondays 22:45 | Write the weekly evidence export. |
| `log-rotation` | Daily 03:00 | Size-based gzip + 7-archive retention for launchd logs (skips self-logs). |

All times are **local**. If your machine is not in UTC, see
[`scripts/automation/README.md`](../scripts/automation/README.md#resolver-due-window-check--timezone-mapping)
for the resolver's TZ override flags. Without overrides the watchdog
assumes UTC and may false-alarm or miss alerts after DST transitions.

### File layout under `~/.options_calculator_pro/`

```
logs/
  ├─ candidate_exit_resolver_launchd.log    ← wrapper output
  ├─ daily_evidence_cycle_launchd.log
  ├─ daily_evidence_watchdog_launchd.log
  ├─ weekly_evidence_report_launchd.log
  ├─ log_rotation_launchd.log               ← NOT rotated by default
  ├─ *_launchd_stderr.log / _stdout.log     ← plist StandardOut/ErrPath
  ├─ candidate_exit_resolutions.jsonl       ← per-row resolver telemetry
  ├─ evidence_cycle_runs.jsonl              ← structured run log
  └─ __main__.log / services.*.log          ← Python-logger files (own rotation)
recommendations/
  └─ recommendation_ledger.sqlite           ← PR-K immutable revisions
reports/
  ├─ evidence/evidence_report_<DATE>.json   ← daily reports (atomic-write)
  ├─ evidence/latest.json                   ← symlink-equivalent
  └─ evidence/weekly/weekly_evidence_report_<DATE>.json
state/
  ├─ daily_evidence_cycle.lock              ← mkdir-as-lock dirs
  ├─ candidate_exit_resolver.lock
  ├─ log_rotation.lock
  └─ daily_evidence_watchdog_alerts.json    ← dedup state (atomic-write)
models/
  └─ crush_model_meta.json                  ← post-event vol classifier
```

### Manual health check

```sh
.venv311/bin/python scripts/check_evidence_health.py | jq .summary
```

A healthy long-running deployment shows `"summary": "OK"`. If it shows
`WARN` or `FAIL`, inspect the full JSON for the `issues` array.

### Inspecting recent runs

```sh
# Tail the daily cycle log (no rotation race — 03:00 rotation is far
# from 21:30 cycle fire).
tail -200 ~/.options_calculator_pro/logs/daily_evidence_cycle_launchd.log

# Last 5 resolver outcomes from JSONL telemetry.
tail -5 ~/.options_calculator_pro/logs/candidate_exit_resolutions.jsonl | jq .

# Ledger row count (should grow ~1/day during the evidence-collection
# window).
.venv311/bin/python -c "
from services.recommendation_ledger import get_recommendation_ledger
print('rows:', get_recommendation_ledger().count())
"
```

---

## Alerts and what they mean

### iMessage alert anatomy

The watchdog sends iMessage to `WATCHDOG_IMESSAGE_TO` when the
combined daily-cycle + candidate-resolver health has an alertable
`FAIL` *or* `WARN` after 22:15 local. (Ops-AE C1c escalated specific
resolver `WARN` severities — stuck-awaiting >10 days, stale completion
— into the alert path. Most `WARN` issues stay observation-only; the
ones that page have `alertable=True` in the resolver health payload.
See [What alerts ARE](#what-alerts-are) below for the full list.)
Format:

```
[Options Calculator Watchdog] Daily evidence cycle failed.
Date=2026-05-27. Errors=Missing evidence report for 2026-05-27;
Daily evidence cycle completion marker is missing from launchd log.
Report=/Users/.../evidence_report_2026-05-27.json
```

Alerts are deduplicated: the same failure signature won't re-alert
until the underlying state changes (state file at
`~/.options_calculator_pro/state/daily_evidence_watchdog_alerts.json`).

### What alerts ARE

- **Pipeline health failures**: missing daily report, stale report
  (>30h old by default), completion marker missing from launchd log,
  resolver `count_balance_holds=false`, candidate row stuck >10 days
  in awaiting state.
- **Operational integrity**: schema mismatches, malformed JSON in
  state files, SQLite quick-check failures.

### What alerts are NOT

This is the most important section in this doc. The watchdog **never**
alerts on:

- A losing day, week, or month — there is no P&L threshold anywhere.
- A trading-signal degradation — strategy quality is observed via
  forward-paper diagnostics, not the alert path.
- A model confidence drop — the ML model can recalibrate itself; that
  is design, not a bug.
- A market-data provider returning empty results — the provider chain
  has a fallback (yfinance) and the watchdog observes only "did
  *something* land," not "was that something good."

If you find yourself silencing an alert because "the strategy was
wrong today," stop. The alert was about pipeline health, not the
strategy. Diagnose the pipeline.

### Suppressing a duplicate alert during ongoing remediation

```sh
# Force re-alert after fixing root cause (clears the dedup state).
rm ~/.options_calculator_pro/state/daily_evidence_watchdog_alerts.json
```

Or: use `--force-alert` on the watchdog CLI if you want to test the
iMessage flow without clearing dedup state.

---

## Troubleshooting matrix

Indexed by **symptom**, not by root cause. Look up what you're seeing,
not what you think is broken.

### Symptom: frontend gets 401 on every API call

| Likely cause | Check |
|---|---|
| Session cookie not being sent across origins | Open DevTools → Network → any failing request → confirm `Cookie:` header is present. If missing, the request was issued without `credentials: 'include'` — verify the call goes through `apiFetch` from `web/frontend/src/lib/api.js`. |
| Cookie was cleared by `/logout` and not yet refreshed | Hit `/login` and submit the share password; new cookie issued. |
| Cookie's nonce was revoked (you logged out previously) | Same fix: re-login. |
| Cookie's age > `_SESSION_MAX_AGE` (7 days) | Same fix: re-login. |
| Wrong `VITE_API_BASE` at build time | `npm run build` again with the correct env var; verify by inspecting `web/frontend/dist/assets/index-*.js` for the resolved URL. |

### Symptom: CORS error in browser console

```
Access to fetch at '<...>' has been blocked by CORS policy
```

| Likely cause | Check |
|---|---|
| `OPTIONS_CALCULATOR_ALLOWED_ORIGINS` doesn't list the frontend's origin | `echo $OPTIONS_CALCULATOR_ALLOWED_ORIGINS` → must include the exact scheme+host+port the browser is loading from. |
| `OPTIONS_CALCULATOR_ALLOWED_ORIGINS` contains `*` or `null` under share-auth | Backend refuses to start — check uvicorn stderr for `RuntimeError: OPTIONS_CALCULATOR_ALLOWED_ORIGINS cannot include …`. |
| Frontend on `:5173` (dev) hitting backend on `:8000` without `apiFetch` | Should not happen post-PR-#60, but if it does the cookie wasn't sent. |

### Symptom: launchd job didn't run / log file empty

| Likely cause | Check |
|---|---|
| Job not installed | `launchctl list \| grep optionscalculator` — must show all five. Re-run `scripts/automation/install_launchd_jobs.sh`. |
| Plist points at a venv path that no longer exists | Wrappers hardcode `${PROJECT_ROOT}/.venv311/bin/python`. If you renamed or moved `.venv311`, reinstall the jobs. |
| Repo moved after install | Same: plists capture absolute paths at install time. Reinstall. |
| launchd exit non-zero last run | `launchctl list \| grep optionscalculator` — second column is last exit code. Inspect the wrapper log (`*_launchd.log`) for the matching `===== … failed exit_code=N =====` line. |
| Lock dir is stale | `ls ~/.options_calculator_pro/state/*.lock` — wrappers GC stale locks at install-defined thresholds, but a wedged process can leave one. `rmdir` the lock dir; the next run will succeed. |

### Symptom: watchdog says "Daily evidence cycle completion marker is missing"

| Likely cause | Check |
|---|---|
| Cycle is in progress | Check `tail ~/.options_calculator_pro/logs/daily_evidence_cycle_launchd.log` — if you see `start` without `complete`, wait. Default timeout 7200s. |
| Cycle hit a Python exception | Look for stack trace in the launchd log. |
| Cycle timed out | Look for `daily evidence cycle timed out after Ns`. Bump `EVIDENCE_CYCLE_TIMEOUT_SECONDS` if the workload genuinely needs more time. |
| First install before 22:15 local | The watchdog should self-defer with `"reason": "not_due_yet"`. If it's still alerting, your TZ overrides are wrong — see [resolver TZ table](../scripts/automation/README.md#resolver-due-window-check--timezone-mapping). |

### Symptom: resolver stuck — `days_in_awaiting_state` keeps growing

| Likely cause | Check |
|---|---|
| Post-event chain genuinely missing in market data | Look at the row's `recommendation_id` in `candidate_exit_resolutions.jsonl`; if the same row's been awaiting for >10 days the watchdog WARNs (this is the design). Manual review needed — the data provider may have never published the chain. |
| Provider rate limit / outage | Check `~/.options_calculator_pro/logs/__main__.log` for retry-loop noise. |

### Symptom: SQLite "readonly database" / "unable to open database file"

| Likely cause | Check |
|---|---|
| `pytest-home` permission issue (common in CI sandboxes) | This is the `.pytest_home/.options_calculator_pro/...` directory leaking permission bits. Affects tests, not production. Reset with `rm -rf .pytest_home`. |
| Real production DB perms | `ls -l ~/.options_calculator_pro/recommendations/recommendation_ledger.sqlite` — must be writable by the user running uvicorn. |
| WAL file lock from a crashed process | If `*.sqlite-wal` exists and is large after a crash, `sqlite3 <db> "PRAGMA wal_checkpoint(FULL);"` cleans it. |

### Symptom: backend refuses to start with `RuntimeError: …`

The PR #59 startup validators are deliberately fail-loud. Read the
error message — it names the specific env var that's misconfigured.
Common ones:

- `ENABLE_SHARE_AUTH=true requires SESSION_SECRET to be non-default and
  at least 24 characters.` → fix `SESSION_SECRET`.
- `ENABLE_SHARE_AUTH=true requires either OPTIONS_CALCULATOR_SECURE_COOKIES=true …
  OR an explicit OPTIONS_CALCULATOR_ALLOW_INSECURE_SESSION_COOKIE=true escape hatch.`
  → set one of those env vars.
- `ENABLE_SHARE_AUTH=true requires OPTIONS_CALCULATOR_ALLOWED_ORIGINS to
  be set explicitly.` → set the env var (no `*`, no `null`).

### Symptom: log files growing without bound

| Likely cause | Check |
|---|---|
| Log-rotation job not installed | `launchctl list \| grep log-rotation` — should be present after `install_launchd_jobs.sh`. |
| Files don't match the rotator's glob | `scripts/rotate_launchd_logs.py --dry-run` lists what it sees. If your problem file isn't there, it's a non-launchd log (probably `__main__.log`, which Python's `RotatingFileHandler` manages itself). |
| Self-logs growing | Expected — they're skipped by default. Manual cleanup: `scripts/rotate_launchd_logs.py --include-self`. |

### Symptom: false-positive iMessage alert

```sh
# 1. Confirm health is now OK.
.venv311/bin/python scripts/check_evidence_health.py | jq .summary

# 2. Clear dedup state so the next real failure can alert.
rm ~/.options_calculator_pro/state/daily_evidence_watchdog_alerts.json

# 3. Dry-run the watchdog to make sure it agrees nothing is wrong.
.venv311/bin/python scripts/watch_daily_evidence_cycle.py --dry-run-alert
```

---

## Uninstall

```sh
# Stop and remove all five launchd jobs.
scripts/automation/uninstall_launchd_jobs.sh

# Confirm none remain.
launchctl list | grep optionscalculator || echo "all gone"

# Optionally remove operational state and data (DESTRUCTIVE — 233+ rows
# of ledger history live here on a long-running deployment).
# rm -rf ~/.options_calculator_pro/
```

The uninstall script template-renders nothing and does not touch the
`~/.options_calculator_pro/` data directory. Leaving the data in place
means a future reinstall picks up from where you left off.
