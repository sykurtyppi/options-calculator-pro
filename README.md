# Options Calculator Pro (Web Edition)

Selector-first event-volatility decision platform for earnings setups.

> **Deploying or redeploying?** See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** —
> single artifact covering env-var contracts, launchd install, first-run
> verification, alert semantics, and a troubleshooting matrix. This
> README covers what the project IS; the runbook covers how to operate
> it.

## Scope
- Single-ticker event-volatility analysis and structure selection.
- Ranked earnings screener and walk-forward OOS report card generation.
- Canonical volatility snapshot, structure scorecards, selector output, calibration diagnostics, and learning-loop instrumentation.

Legacy desktop UI (PySide/Tk) and scanner stack were removed from this repository.

## Project Layout
- `web/api`: FastAPI backend
- `web/frontend`: React + Vite frontend
- `scripts/institutional_backfill.py`: data/backtest pipeline entrypoint
- `services/institutional_ml_db.py`: institutional analytics engine

## Run Backend
```bash
cd /path/to/options_calculator_pro
[ -d .venv311 ] || uv venv .venv311 --python 3.11
source .venv311/bin/activate
uv pip install -r requirements.lock
python -m uvicorn web.api.app:app --reload --host 0.0.0.0 --port 8000
```

## MarketData.app Setup (Recommended)
Create `.env` from `.env.example` and set:

```bash
MARKETDATA_TOKEN=your_real_token
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FMP_API_KEY=your_fmp_key_optional
SEC_API_USER_AGENT=your-email@example.com options-calculator-pro/1.0
```

Token resolution order:
1. `--marketdata-token` CLI flag
2. `MARKETDATA_TOKEN` environment variable

If no token is configured, the stack automatically falls back to yfinance.

Multi-source earnings discovery notes:
- `Alpha Vantage` is used as a cached bulk earnings-calendar feed. Their official free-tier support currently documents up to `25 requests/day`, so this project treats it as daily discovery infrastructure rather than a live per-symbol dependency.
- `FMP` is optional and can provide a second calendar feed plus confirmed earnings events.
- `SEC EDGAR` is used as a confirmation layer, not a future-calendar source. The API requires no key, but a descriptive `User-Agent` is strongly recommended for compliant automated access.
- `yfinance` remains the last-resort research fallback when stronger sources are unavailable.

## Optional Share-Code Auth
Auth is disabled by default. To enable login protection for shared deployments, set:

```bash
ENABLE_SHARE_AUTH=true
SHARE_PASSWORD=your_access_code
SESSION_SECRET=long_random_secret_at_least_24_chars
```

When enabled, unauthenticated API requests return `401` and browser routes redirect to `/login`.
`/api/health` remains public for health checks.

Hosted mode is stricter and fails fast if auth is incomplete:

```bash
OPTIONS_CALCULATOR_HOSTED_MODE=true
ENABLE_SHARE_AUTH=true
SHARE_PASSWORD=your_access_code
SESSION_SECRET=long_random_secret_at_least_24_chars
```

In hosted mode, session cookies are marked `Secure` and API docs are protected by the same login middleware.
For non-hosted deployments, set `OPTIONS_CALCULATOR_PROTECT_API_DOCS=true` or
`OPTIONS_CALCULATOR_SECURE_COOKIES=true` independently if needed.

## Run Frontend
```bash
cd /path/to/options_calculator_pro/web/frontend
npm install
npm run dev
```

Frontend default URL: `http://127.0.0.1:5173`  
Backend health check: `http://127.0.0.1:8000/api/health`

Frontend trust-layer tests:

```bash
cd /path/to/options_calculator_pro/web/frontend
npm test
```

## Run Tests (Parity Setup)
Use both root and web dependency sets so API/runtime tests match the deployed stack:

```bash
cd /path/to/options_calculator_pro
source .venv/bin/activate
pip install -r requirements.txt
pip install -r web/api/requirements-web.txt -c web/api/constraints-web.txt
pytest -q
```

## Notes
- OOS artifacts are written to `exports/reports` at runtime.
- Free data feeds can cause sparse earnings/timing coverage; hard no-trade gating is enforced in the API.
- OOS defaults are evidence-first (`lookback=1095`, `max_backtest_symbols=50`, `train/test/step=189/42/42`).
- If OOS sample gates fail, the API can run one adaptive retry to improve split/trade coverage.
- Share-code auth is optional for local use and disabled by default. Hosted mode requires
  `ENABLE_SHARE_AUTH=true`, `SHARE_PASSWORD`, and a non-default `SESSION_SECRET`.
- Allowed browser origins default to `http://localhost:5173` and `http://127.0.0.1:5173`.
  Override with `OPTIONS_CALCULATOR_ALLOWED_ORIGINS` as a comma-separated list.
- Background OOS and legacy ML training endpoints are bounded by conservative in-process
  concurrency caps. Tune with `OPTIONS_CALCULATOR_MAX_OOS_RUNNING_JOBS`,
  `OPTIONS_CALCULATOR_MAX_ML_RUNNING_JOBS`, and `OPTIONS_CALCULATOR_MAX_RETAINED_JOBS`.
- Historical post-event volatility-label backfill from MarketData.app is available for legacy research workflows via:
  - `python scripts/institutional_backfill.py --full-universe --capture-historical-mda-snapshots --mda-lookback-years 2`
  - Optional token override: `--marketdata-token <TOKEN>`
- MarketData budget controls (daily, UTC reset):
  - `MARKETDATA_DAILY_CREDIT_LIMIT` (default: `100000`)
  - `MARKETDATA_DAILY_CREDIT_RESERVE` (default: `20000`)
  - When reserve mode is active, research-tier requests (e.g., historical chain backfills)
    are paused while critical/production endpoints continue until hard cap.
  - A per-endpoint HTTP-402 circuit breaker blocks repeated paid-plan failures for the
    rest of the UTC day to avoid burning calls on unavailable endpoints.
- Readiness report promotion gates can be tuned from CLI:
  - `--promotion-min-oos-grade`
  - `--promotion-min-live-trades`
  - `--promotion-evidence-mode` (`and` default for stricter promotion, or `or`)
  - `--promotion-live-lookback-days`
  - `--promotion-live-session-id`
  - `--promotion-live-min-confidence`
  - `--promotion-max-forward-drawdown`
  - `--promotion-max-forward-execution-drag-bps`
  - `--promotion-min-forward-directional-accuracy`
  - `--disable-promotion-forward-status-gate`
  - `--promotion-require-fill-log`
  - `--forward-trade-log-path` (for real forward paper/live trade outcomes)
  - `--forward-fill-log-path` (for real fill/slippage diagnostics)
  - `--forward-paper-tracker` (standalone forward diagnostics report)
  - `--forward-tracker-session-id`
  - `--forward-tracker-lookback-days`
  - `--forward-tracker-min-confidence`
  - `--forward-tracker-output-dir`
  - Without `--forward-trade-log-path`, forward tracker status is `simulated_backtest`
    and live-trade promotion counts stay at 0 by design.
- Walk-forward post-event volatility profiles are now loaded with an as-of cutoff (session start date)
  to reduce future-label leakage in backtests.
