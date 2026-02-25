# Web Platform

This folder contains the production web surface:

- `web/api`: FastAPI backend
- `web/frontend`: React (Vite) frontend

## Backend (FastAPI)

Run from repo root:

```bash
cd /path/to/options_calculator_pro
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r web/api/requirements-web.txt -c web/api/constraints-web.txt
python -m uvicorn web.api.app:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /api/health`
- `POST /api/edge/analyze`
- `POST /api/oos/report-card`

## Frontend (React)

```bash
cd web/frontend
npm install
npm run dev
```

By default the frontend calls `http://127.0.0.1:8000`.
Override with:

```bash
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

## Notes

- Workflow is intentionally single-ticker and scanner-free.
- OOS endpoint writes artifacts under `exports/reports` using existing project backtesting logic.
- OOS defaults are evidence-first (`lookback=1095`, `max_backtest_symbols=50`, `train/test/step=189/42/42`).
- When sample gates fail, `/api/oos/report-card` may run one adaptive retry with broader coverage.
- For free data feeds, strict no-trade gating is expected on sparse or low-confidence setups.
