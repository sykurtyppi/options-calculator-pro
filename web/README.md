# Web Migration Skeleton

This folder contains the first web migration slice:

- `web/api`: FastAPI backend
- `web/frontend`: React (Vite) frontend

## Backend (FastAPI)

Run from repo root:

```bash
pip install fastapi uvicorn pydantic yfinance pandas numpy
uvicorn web.api.app:app --reload --host 0.0.0.0 --port 8000
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
- This is a skeleton foundation; next iteration should add auth, persistence, and structured async job handling.
