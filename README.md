# Options Calculator Pro (Web Edition)

Web-first IV crush and earnings calendar spread research platform.

## Scope
- Single-ticker edge analysis (no scanner).
- Walk-forward OOS report card generation.
- Institutional backfill and diagnostics pipeline.

Legacy desktop UI (PySide/Tk) and scanner stack were removed from this repository.

## Project Layout
- `web/api`: FastAPI backend
- `web/frontend`: React + Vite frontend
- `scripts/institutional_backfill.py`: data/backtest pipeline entrypoint
- `services/institutional_ml_db.py`: institutional analytics engine

## Run Backend
```bash
cd /path/to/options_calculator_pro
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r web/api/requirements-web.txt -c web/api/constraints-web.txt
python -m uvicorn web.api.app:app --reload --host 0.0.0.0 --port 8000
```

## MarketData.app Setup (Recommended)
Create `.env` from `.env.example` and set:

```bash
MARKETDATA_TOKEN=your_real_token
```

Token resolution order:
1. `--marketdata-token` CLI flag
2. `MARKETDATA_TOKEN` environment variable

If no token is configured, the stack automatically falls back to yfinance.

## Optional Share-Code Auth
Auth is disabled by default. To enable login protection for shared deployments, set:

```bash
ENABLE_SHARE_AUTH=true
SHARE_PASSWORD=your_access_code
SESSION_SECRET=long_random_secret
```

When enabled, unauthenticated API requests return `401` and browser routes redirect to `/login`.
`/api/health` remains public for health checks.

## Run Frontend
```bash
cd /path/to/options_calculator_pro/web/frontend
npm install
npm run dev
```

Frontend default URL: `http://127.0.0.1:5173`  
Backend health check: `http://127.0.0.1:8000/api/health`

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
- Share-code auth is optional and disabled by default; enable it for shared deployments.
  Auth is enforced only when `ENABLE_SHARE_AUTH=true` and `SHARE_PASSWORD` is set.
- Allowed browser origins default to `http://localhost:5173` and `http://127.0.0.1:5173`.
  Override with `OPTIONS_CALCULATOR_ALLOWED_ORIGINS` as a comma-separated list.
- Historical IV-crush label backfill from MarketData.app is available via:
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
- Walk-forward crush profiles are now loaded with an as-of cutoff (session start date)
  to reduce future-label leakage in backtests.
