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

## Run Frontend
```bash
cd /path/to/options_calculator_pro/web/frontend
npm install
npm run dev
```

Frontend default URL: `http://127.0.0.1:5173`  
Backend health check: `http://127.0.0.1:8000/api/health`

## Notes
- OOS artifacts are written to `exports/reports` at runtime.
- Free data feeds can cause sparse earnings/timing coverage; hard no-trade gating is enforced in the API.
- OOS defaults are evidence-first (`lookback=1095`, `max_backtest_symbols=50`, `train/test/step=189/42/42`).
- If OOS sample gates fail, the API can run one adaptive retry to improve split/trade coverage.
- API auth is not implemented yet; run this service on localhost/private networks only.
- Allowed browser origins default to `http://localhost:5173` and `http://127.0.0.1:5173`.
  Override with `OPTIONS_CALCULATOR_ALLOWED_ORIGINS` as a comma-separated list.
