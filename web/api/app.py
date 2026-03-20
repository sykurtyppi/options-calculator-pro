from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import math
import os
from pathlib import Path
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple
import sys
import sqlite3

_OOS_VALIDATION_TIMEOUT_SECONDS: float = 300.0  # 5 min hard wall-clock limit per call

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env before importing any service that reads env vars
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

# ── Access-control config (loaded from .env) ─────────────────────────────────
_SHARE_AUTH_ENABLED = os.getenv("ENABLE_SHARE_AUTH", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SHARE_PASSWORD = os.getenv("SHARE_PASSWORD", "")
_SESSION_SECRET = os.getenv("SESSION_SECRET", "change-me-in-env")
_SESSION_COOKIE = "ops_session"
_SESSION_MAX_AGE = 7 * 24 * 3600  # 1 week

# Paths that don't require authentication
_PUBLIC_PATHS = {"/login", "/favicon.ico", "/api/health"}

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Options Calculator Pro</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117;
      color: #c9d1d9;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }
    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 40px;
      width: 100%;
      max-width: 380px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    h1 { font-size: 1.25rem; font-weight: 600; margin-bottom: 6px; color: #e6edf3; }
    p.sub { font-size: 0.82rem; color: #8b949e; margin-bottom: 28px; }
    label { display: block; font-size: 0.8rem; color: #8b949e; margin-bottom: 6px; letter-spacing: 0.02em; }
    input[type="password"] {
      width: 100%;
      padding: 10px 14px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #e6edf3;
      font-size: 1.05rem;
      outline: none;
      transition: border-color 0.2s;
      letter-spacing: 0.18em;
    }
    input[type="password"]:focus { border-color: #388bfd; }
    .error { color: #f85149; font-size: 0.82rem; margin-top: 8px; min-height: 1.2em; }
    button {
      margin-top: 20px;
      width: 100%;
      padding: 10px;
      background: #1f6feb;
      border: none;
      border-radius: 6px;
      color: #fff;
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover { background: #388bfd; }
    .footer { text-align: center; margin-top: 28px; font-size: 0.72rem; color: #484f58; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Options Calculator Pro</h1>
    <p class="sub">Enter your access code to continue.</p>
    <form method="POST" action="/login">
      <label for="pw">Access Code</label>
      <input type="password" id="pw" name="password" autofocus autocomplete="current-password" />
      <div class="error">{{ERROR}}</div>
      <button type="submit">Continue &rarr;</button>
    </form>
    <div class="footer">Created by Tristan Alejandro &middot; Not financial advice.</div>
  </div>
</body>
</html>"""


def _session_token() -> str:
    """Derive a stable HMAC token from the session secret."""
    return hmac.new(
        _SESSION_SECRET.encode(),
        b"ops-authenticated-v1",
        hashlib.sha256,
    ).hexdigest()


def _valid_session(cookie_val: str) -> bool:
    if not _SHARE_AUTH_ENABLED:
        return True
    if not cookie_val or not _SHARE_PASSWORD:
        return False
    try:
        return hmac.compare_digest(cookie_val, _session_token())
    except Exception:
        return False


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Let public paths and API docs through unauthenticated
        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith(("/docs", "/openapi", "/redoc")):
            return await call_next(request)

        session = request.cookies.get(_SESSION_COOKIE, "")
        if not _valid_session(session):
            # API requests get a 401 (so the frontend can handle it gracefully)
            if path.startswith("/api/"):
                return HTMLResponse(
                    content='{"detail":"Unauthorized — please log in at /login"}',
                    status_code=401,
                    media_type="application/json",
                )
            # Everything else → redirect to login
            return RedirectResponse(url="/login", status_code=302)

        return await call_next(request)


from scripts.institutional_backfill import InstitutionalDataCollector
from services.market_data_client import MarketDataClient
from web.api.edge_engine import analyze_single_ticker
from web.api.schemas import (
    EdgeAnalyzeRequest,
    EdgeAnalyzeResponse,
    OOSReportRequest,
    OOSReportResponse,
)

# Module-level MarketDataClient singleton — shared across requests so the
# in-process cache persists between calls (saves credits on repeated symbols).
_mda_client: Optional[MarketDataClient] = None


def _get_mda_client() -> MarketDataClient:
    global _mda_client
    if _mda_client is None:
        _mda_client = MarketDataClient()
    return _mda_client


app = FastAPI(
    title="Options Calculator Pro API",
    version="0.1.0",
    description="Single-ticker IV crush edge analysis and OOS robustness reports.",
)

origins_env = os.getenv("OPTIONS_CALCULATOR_ALLOWED_ORIGINS", "")
if origins_env.strip():
    allowed_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth middleware — must be added AFTER CORS so CORS headers are present on 401s
app.add_middleware(_AuthMiddleware)


# ── Login routes ──────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    return HTMLResponse(_LOGIN_HTML.replace("{{ERROR}}", ""))


@app.post("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_submit(request: Request):
    form = await request.form()
    password = form.get("password", "")
    if _SHARE_AUTH_ENABLED and _SHARE_PASSWORD and hmac.compare_digest(str(password), _SHARE_PASSWORD):
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            _SESSION_COOKIE,
            _session_token(),
            httponly=True,
            samesite="lax",
            max_age=_SESSION_MAX_AGE,
        )
        return response
    # Wrong password — re-render with error
    return HTMLResponse(
        _LOGIN_HTML.replace("{{ERROR}}", "Incorrect access code."),
        status_code=401,
    )

_OOS_STABILITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "evidence_balanced": {
        "execution_profiles": ["institutional"],
        "hold_days_grid": [1, 3],
        "trades_per_day_grid": [2, 3],
        "entry_days_grid": [3, 5, 7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.48,
            "min_crush_confidence": 0.28,
            "min_crush_magnitude": 0.06,
            "min_crush_edge": 0.025,
            "target_entry_dte": 6,
            "entry_dte_band": 5,
            "min_daily_share_volume": 1_500_000,
            "max_abs_momentum_5d": 0.11,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
    "sample_expansion": {
        "execution_profiles": ["institutional"],
        "hold_days_grid": [1, 3],
        "trades_per_day_grid": [2, 3],
        "entry_days_grid": [3, 5, 7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.45,
            "min_crush_confidence": 0.25,
            "min_crush_magnitude": 0.05,
            "min_crush_edge": 0.015,
            "target_entry_dte": 6,
            "entry_dte_band": 6,
            "min_daily_share_volume": 1_000_000,
            "max_abs_momentum_5d": 0.11,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
    "variance_control": {
        "execution_profiles": ["institutional_tight", "institutional"],
        "hold_days_grid": [1],
        "trades_per_day_grid": [2],
        "entry_days_grid": [5, 7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.65,
            "min_crush_confidence": 0.50,
            "min_crush_magnitude": 0.09,
            "min_crush_edge": 0.025,
            "target_entry_dte": 6,
            "entry_dte_band": 4,
            "min_daily_share_volume": 10_000_000,
            "max_abs_momentum_5d": 0.09,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
    "alpha_focus": {
        "execution_profiles": ["institutional_tight"],
        "hold_days_grid": [1],
        "trades_per_day_grid": [1, 2],
        "entry_days_grid": [5, 7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.65,
            "min_crush_confidence": 0.50,
            "min_crush_magnitude": 0.09,
            "min_crush_edge": 0.03,
            "target_entry_dte": 6,
            "entry_dte_band": 3,
            "min_daily_share_volume": 5_000_000,
            "max_abs_momentum_5d": 0.08,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
}

_OOS_AUTO_PROFILE_ORDER: Tuple[str, ...] = (
    "evidence_balanced",
    "sample_expansion",
    "variance_control",
    "alpha_focus",
)


def _resolve_stability_profile(profile: Optional[str]) -> str:
    normalized = (profile or "stability_auto").strip().lower()
    if normalized == "stability_auto":
        return normalized
    if normalized in _OOS_STABILITY_PROFILES:
        return normalized
    return "stability_auto"


def _build_profiled_run_kwargs(
    request: OOSReportRequest,
    profile_name: str,
    train_days: int,
    test_days: int,
    step_days: int,
    use_profile_defaults: bool,
) -> Dict[str, Any]:
    profile = _OOS_STABILITY_PROFILES[profile_name]
    defaults = profile["defaults"]

    if use_profile_defaults:
        min_signal_score = float(defaults["min_signal_score"])
        min_crush_confidence = float(defaults["min_crush_confidence"])
        min_crush_magnitude = float(defaults["min_crush_magnitude"])
        min_crush_edge = float(defaults["min_crush_edge"])
        target_entry_dte = int(defaults["target_entry_dte"])
        entry_dte_band = int(defaults["entry_dte_band"])
        min_daily_share_volume = int(defaults["min_daily_share_volume"])
        max_abs_momentum_5d = float(defaults["max_abs_momentum_5d"])
    else:
        min_signal_score = max(float(request.min_signal_score), float(defaults["min_signal_score"]))
        min_crush_confidence = max(float(request.min_crush_confidence), float(defaults["min_crush_confidence"]))
        min_crush_magnitude = max(float(request.min_crush_magnitude), float(defaults["min_crush_magnitude"]))
        min_crush_edge = max(float(request.min_crush_edge), float(defaults["min_crush_edge"]))
        target_entry_dte = int(request.target_entry_dte)
        entry_dte_band = min(int(request.entry_dte_band), int(defaults["entry_dte_band"]))
        min_daily_share_volume = max(int(request.min_daily_share_volume), int(defaults["min_daily_share_volume"]))
        max_abs_momentum_5d = min(float(request.max_abs_momentum_5d), float(defaults["max_abs_momentum_5d"]))

    return {
        "execution_profiles": list(profile["execution_profiles"]),
        "hold_days_grid": list(profile["hold_days_grid"]),
        "signal_threshold_grid": [min_signal_score],
        "trades_per_day_grid": list(profile["trades_per_day_grid"]),
        "entry_days_grid": list(profile["entry_days_grid"]),
        "exit_days_grid": list(profile["exit_days_grid"]),
        "target_entry_dte": target_entry_dte,
        "entry_dte_band": max(1, entry_dte_band),
        "min_daily_share_volume": min_daily_share_volume,
        "max_abs_momentum_5d": max_abs_momentum_5d,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "top_n_train": request.oos_top_n_train,
        "lookback_days": max(int(request.lookback_days), int(defaults["lookback_days"])),
        "max_backtest_symbols": max(int(request.max_backtest_symbols), int(defaults["max_backtest_symbols"])),
        "use_crush_confidence_gate": True,
        "allow_global_crush_profile": True,
        "min_crush_confidence": min_crush_confidence,
        "min_crush_magnitude": min_crush_magnitude,
        "min_crush_edge": min_crush_edge,
        "allow_proxy_earnings": True,
        "min_splits": request.oos_min_splits,
        "min_total_test_trades": request.oos_min_total_test_trades,
        "min_trades_per_split": request.oos_min_trades_per_split,
        "output_dir": "exports/reports",
        "start_date": request.backtest_start_date,
        "end_date": request.backtest_end_date,
    }


def _compute_cross_split_sharpe(
    splits_detail: List[Dict[str, Any]],
    test_days: int,
) -> Optional[float]:
    """
    Fix 3: Portfolio-level Sharpe computed across OOS splits.

    Per-trade Sharpe (mean_trade_pnl / std_trade_pnl * sqrt(252/hold_days)) is
    inflated by low within-trade variance. The cross-split Sharpe treats each split
    as one 'period' return and annualises using the test window length — a more
    realistic portfolio-level measure.

    Formula: (mean_split_pnl / std_split_pnl) * sqrt(252 / test_days)
    """
    import math as _math
    if not splits_detail or test_days <= 0:
        return None
    pnls = [float(s["pnl"]) for s in splits_detail if s.get("pnl") is not None]
    if len(pnls) < 3:
        return None
    mean_p = sum(pnls) / len(pnls)
    var_p  = sum((x - mean_p) ** 2 for x in pnls) / (len(pnls) - 1)
    std_p  = _math.sqrt(var_p) if var_p > 0 else None
    if std_p is None or std_p < 1e-9:
        return None
    annualisation = _math.sqrt(max(252 / test_days, 1.0))
    sharpe = (mean_p / std_p) * annualisation
    return round(float(sharpe), 3) if _math.isfinite(sharpe) else None


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas/path-like objects to JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    # numpy/pandas scalar support (np.int64, np.float64, pd.Int64Dtype scalars, etc.)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    # numpy arrays / pandas containers
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass

    return str(value)


def _oos_report_card_from_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    report_card = result.get("report_card", {})
    return report_card if isinstance(report_card, dict) else {}


def _oos_result_score(result: Optional[Dict[str, Any]]) -> Tuple[int, int, int, int, float, float, float]:
    report_card = _oos_report_card_from_result(result)
    verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
    sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
    metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
    alpha = metrics.get("alpha", {}) if isinstance(metrics, dict) else {}
    sharpe = metrics.get("sharpe", {}) if isinstance(metrics, dict) else {}
    pnl = metrics.get("pnl", {}) if isinstance(metrics, dict) else {}

    overall_pass = 1 if bool(verdict.get("overall_pass")) else 0
    alpha_low = float(alpha.get("low", -999.0) or -999.0)
    sharpe_low = float(sharpe.get("low", -999.0) or -999.0)
    pnl_low = float(pnl.get("low", -999.0) or -999.0)
    ci_positive_count = int(alpha_low > 0.0) + int(sharpe_low > 0.0) + int(pnl_low > 0.0)
    total_trades = int(sample.get("total_test_trades", 0) or 0)
    splits = int(sample.get("splits", 0) or 0)
    return overall_pass, ci_positive_count, total_trades, splits, alpha_low, sharpe_low, pnl_low


def _oos_profile_summary(profile_name: str, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    report_card = _oos_report_card_from_result(result)
    verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
    sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
    metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
    return {
        "profile": profile_name,
        "grade": verdict.get("grade"),
        "overall_pass": verdict.get("overall_pass"),
        "total_test_trades": sample.get("total_test_trades"),
        "avg_trades_per_split": sample.get("avg_trades_per_split"),
        "alpha_low": (metrics.get("alpha", {}) if isinstance(metrics, dict) else {}).get("low"),
        "sharpe_low": (metrics.get("sharpe", {}) if isinstance(metrics, dict) else {}).get("low"),
        "pnl_low": (metrics.get("pnl", {}) if isinstance(metrics, dict) else {}).get("low"),
    }


def _oos_sample_insufficient(result: Optional[Dict[str, Any]]) -> bool:
    report_card = _oos_report_card_from_result(result)
    if not report_card:
        return True
    if not bool(report_card.get("ready", False)):
        return True
    verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
    if bool(verdict.get("overall_pass", False)):
        return False
    gates = report_card.get("gates", {}) if isinstance(report_card, dict) else {}
    for gate_name in ("min_splits", "min_total_test_trades", "min_trades_per_split"):
        gate = gates.get(gate_name, {}) if isinstance(gates, dict) else {}
        if isinstance(gate, dict) and gate.get("passed") is False:
            return True
    return False


def _run_oos_with_timeout(
    collector: Any,
    kwargs: Dict[str, Any],
    timeout_seconds: float = _OOS_VALIDATION_TIMEOUT_SECONDS,
) -> Optional[Dict[str, Any]]:
    """Run run_oos_validation in a daemon thread with a hard wall-clock timeout.

    Returns None on timeout (the daemon thread keeps running until it finishes
    naturally or the process exits). Raises on uncaught exceptions from the
    worker thread so callers can surface them normally.
    """
    result_box: List[Any] = []
    exc_box: List[BaseException] = []

    def _target() -> None:
        try:
            result_box.append(collector.run_oos_validation(**kwargs))
        except Exception as exc:  # noqa: BLE001
            exc_box.append(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        return None  # timed out
    if exc_box:
        raise exc_box[0]
    return result_box[0] if result_box else None


def _oos_sample_bottleneck_note(result: Optional[Dict[str, Any]]) -> Optional[str]:
    report_card = _oos_report_card_from_result(result)
    if not report_card:
        return None
    gates = report_card.get("gates", {}) if isinstance(report_card, dict) else {}
    if not isinstance(gates, dict):
        return None

    alpha_gate = gates.get("alpha_ci_positive", {}) if isinstance(gates.get("alpha_ci_positive"), dict) else {}
    sharpe_gate = gates.get("sharpe_ci_positive", {}) if isinstance(gates.get("sharpe_ci_positive"), dict) else {}
    pnl_gate = gates.get("pnl_ci_positive", {}) if isinstance(gates.get("pnl_ci_positive"), dict) else {}
    split_gate = gates.get("min_splits", {}) if isinstance(gates.get("min_splits"), dict) else {}
    trades_gate = gates.get("min_total_test_trades", {}) if isinstance(gates.get("min_total_test_trades"), dict) else {}
    per_split_gate = gates.get("min_trades_per_split", {}) if isinstance(gates.get("min_trades_per_split"), dict) else {}

    ci_positive_count = sum([
        bool(alpha_gate.get("passed")),
        bool(sharpe_gate.get("passed")),
        bool(pnl_gate.get("passed")),
    ])
    sample_failed = (
        split_gate.get("passed") is False
        or trades_gate.get("passed") is False
        or per_split_gate.get("passed") is False
    )
    if ci_positive_count >= 2 and sample_failed:
        return (
            "Signal quality passed CI gates but OOS sample size is insufficient. "
            "Increase history (e.g. run backfill with >=4 years) or use `sample_expansion` "
            "to build evidence before enforcing strict sample floors."
        )
    return None


def _oos_sparse_alpha_note(result: Optional[Dict[str, Any]]) -> Optional[str]:
    """Warn when profitable splits are fewer than half — returns are event-concentrated."""
    report_card = _oos_report_card_from_result(result)
    if not report_card:
        return None
    metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
    sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
    if not isinstance(metrics, dict) or not isinstance(sample, dict):
        return None
    rate = metrics.get("positive_alpha_split_rate")
    splits = sample.get("splits", 0)
    if isinstance(rate, (int, float)) and int(splits or 0) >= 8 and rate < 0.5:
        return (
            f"Alpha is positive in only {rate:.0%} of OOS splits. "
            "Returns are concentrated in a small number of earnings events; "
            "CI bounds may be unreliable."
        )
    return None


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/edge/analyze", response_model=EdgeAnalyzeResponse)
def analyze_edge(request: EdgeAnalyzeRequest) -> EdgeAnalyzeResponse:
    try:
        snapshot = analyze_single_ticker(request.symbol, mda_client=_get_mda_client())
        return EdgeAnalyzeResponse(
            generated_at=datetime.now(timezone.utc),
            symbol=snapshot.symbol,
            recommendation=snapshot.recommendation,
            confidence_pct=snapshot.confidence_pct,
            setup_score=snapshot.setup_score,
            metrics=snapshot.metrics,
            rationale=snapshot.rationale,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}")


def _execute_oos_logic(request: OOSReportRequest) -> Dict[str, Any]:
    """Core OOS computation — returns a plain dict with 'summary', 'output_files',
    and 'generated_at' keys.  Raises raw exceptions; callers wrap as needed.
    Safe to call from a background thread."""
    collector = InstitutionalDataCollector()
    train_days = int(request.oos_train_days)
    test_days = int(request.oos_test_days)
    step_days = int(request.oos_step_days)
    adjustment_note = None
    notes: List[str] = []
    warnings: List[str] = []

    # Auto-fit OOS windows to available feature history so first-run users
    # get a diagnostic result instead of hard 422 failures.
    try:
        with sqlite3.connect(collector.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(date), MAX(date) FROM ml_features")
            row = cursor.fetchone() or (None, None)
        min_date, max_date = row
        if min_date and max_date:
            start_dt = datetime.fromisoformat(str(min_date))
            end_dt = datetime.fromisoformat(str(max_date))
            available_days = max(0, (end_dt - start_dt).days)
            if available_days > 0 and (train_days + test_days) > available_days:
                fitted_train = max(63, int(available_days * 0.60))
                fitted_test = max(21, int(available_days * 0.25))
                fitted_step = max(21, min(fitted_test, int(available_days * 0.20)))
                if (fitted_train + fitted_test) <= available_days:
                    adjustment_note = (
                        f"Adjusted OOS windows to dataset: "
                        f"train={fitted_train}, test={fitted_test}, step={fitted_step} "
                        f"(available_days={available_days})."
                    )
                    train_days, test_days, step_days = fitted_train, fitted_test, fitted_step
    except Exception:
        pass
    if adjustment_note:
        notes.append(adjustment_note)

    stability_profile_requested = _resolve_stability_profile(request.oos_stability_profile)
    stability_profile_used: str = stability_profile_requested
    profile_summaries: List[Dict[str, Any]] = []

    run_kwargs: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    if stability_profile_requested == "stability_auto":
        auto_candidates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for profile_name in _OOS_AUTO_PROFILE_ORDER:
            candidate_kwargs = _build_profiled_run_kwargs(
                request=request,
                profile_name=profile_name,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                use_profile_defaults=True,
            )
            candidate_result = _run_oos_with_timeout(collector, candidate_kwargs)
            if candidate_result:
                auto_candidates.append((profile_name, candidate_result, candidate_kwargs))
                profile_summaries.append(_oos_profile_summary(profile_name, candidate_result))
            elif candidate_result is None:
                notes.append(f"Auto profile `{profile_name}` produced no OOS rows.")

        if auto_candidates:
            stability_profile_used, result, run_kwargs = max(
                auto_candidates,
                key=lambda row: _oos_result_score(row[1]),
            )
            notes.append(f"Stability auto selected `{stability_profile_used}` profile.")
        else:
            stability_profile_used = "evidence_balanced"
            run_kwargs = _build_profiled_run_kwargs(
                request=request,
                profile_name=stability_profile_used,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                use_profile_defaults=True,
            )
            result = None
    else:
        run_kwargs = _build_profiled_run_kwargs(
            request=request,
            profile_name=stability_profile_requested,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            use_profile_defaults=False,
        )
        result = _run_oos_with_timeout(collector, run_kwargs)
        profile_summaries.append(_oos_profile_summary(stability_profile_requested, result))

    # Evidence-first fallback: if sample coverage is weak, rerun once with
    # larger universe and denser rolling windows to increase OOS evidence.
    adaptive_kwargs = dict(run_kwargs)
    adaptive_kwargs["lookback_days"] = max(int(run_kwargs["lookback_days"]), 1095)
    adaptive_kwargs["max_backtest_symbols"] = max(int(run_kwargs["max_backtest_symbols"]), 50)
    adaptive_kwargs["train_days"] = max(63, min(int(run_kwargs["train_days"]), 189))
    adaptive_kwargs["test_days"] = max(21, min(int(run_kwargs["test_days"]), 42))
    adaptive_kwargs["step_days"] = max(21, min(int(run_kwargs["step_days"]), 42))
    adaptive_changed = any(
        adaptive_kwargs[key] != run_kwargs[key]
        for key in ("lookback_days", "max_backtest_symbols", "train_days", "test_days", "step_days")
    )
    adaptive_used = False
    if adaptive_changed and (result is None or _oos_sample_insufficient(result)):
        retry_result = _run_oos_with_timeout(collector, adaptive_kwargs)
        if retry_result:
            baseline_score = _oos_result_score(result)
            retry_score = _oos_result_score(retry_result)
            if result is None or retry_score > baseline_score:
                result = retry_result
                train_days = int(adaptive_kwargs["train_days"])
                test_days = int(adaptive_kwargs["test_days"])
                step_days = int(adaptive_kwargs["step_days"])
                adaptive_used = True
                stability_profile_used = (
                    f"{stability_profile_used}+adaptive"
                    if not stability_profile_used.endswith("+adaptive")
                    else stability_profile_used
                )
                notes.append(
                    "Adaptive OOS retry applied (evidence-first): "
                    f"symbols={adaptive_kwargs['max_backtest_symbols']}, "
                    f"lookback={adaptive_kwargs['lookback_days']}, "
                    f"train/test/step={train_days}/{test_days}/{step_days}."
                )
            else:
                notes.append("Adaptive OOS retry completed but baseline run retained.")
        else:
            notes.append("Adaptive OOS retry produced no rows.")

    if not result:
        timeout_secs = int(_OOS_VALIDATION_TIMEOUT_SECONDS)
        warnings.append(
            f"OOS validation did not complete within {timeout_secs}s. "
            "Reduce lookback_days or max_backtest_symbols and retry."
        )
        summary = _json_safe({
            "status": "no_oos_rows",
            "grade": "N/A",
            "overall_pass": False,
            "message": "No OOS rows produced for current filters/history.",
            "windows_used": {
                "train_days": train_days,
                "test_days": test_days,
                "step_days": step_days,
                "lookback_days": int(run_kwargs["lookback_days"]),
                "max_backtest_symbols": int(run_kwargs["max_backtest_symbols"]),
            },
            "stability_profile_requested": stability_profile_requested,
            "stability_profile_used": stability_profile_used,
            "stability_profiles_evaluated": profile_summaries,
            "adaptive_retry_used": adaptive_used,
            "notes": notes,
            "warnings": warnings,
        })
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "output_files": {},
        }

    report_card = result.get("report_card", {}) if isinstance(result, dict) else {}
    verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
    sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
    metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
    gates = report_card.get("gates", {}) if isinstance(report_card, dict) else {}
    sample_bottleneck_note = _oos_sample_bottleneck_note(result)
    if sample_bottleneck_note:
        warnings.append(sample_bottleneck_note)
    sparse_alpha_note = _oos_sparse_alpha_note(result)
    if sparse_alpha_note:
        warnings.append(sparse_alpha_note)
    summary = {
        "splits": int(result.get("splits", 0)),
        "best_params": result.get("best_params", {}),
        "grade": verdict.get("grade"),
        "overall_pass": verdict.get("overall_pass"),
        "verdict": verdict,
        "sample": sample,
        "metrics": metrics,
        "gates": gates,
        "splits_detail": result.get("splits_detail", []),
        "cross_split_sharpe": _compute_cross_split_sharpe(
            result.get("splits_detail", []), test_days
        ),
        "status": "ok",
        "windows_used": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "lookback_days": int(adaptive_kwargs["lookback_days"] if adaptive_used else run_kwargs["lookback_days"]),
            "max_backtest_symbols": int(adaptive_kwargs["max_backtest_symbols"] if adaptive_used else run_kwargs["max_backtest_symbols"]),
        },
        "stability_profile_requested": stability_profile_requested,
        "stability_profile_used": stability_profile_used,
        "stability_profiles_evaluated": profile_summaries,
        "adaptive_retry_used": adaptive_used,
        "notes": notes,
        "warnings": warnings,
    }
    output_files = {
        "csv": result.get("csv_path"),
        "summary_markdown": result.get("markdown_path"),
        "summary_json": result.get("json_path"),
        "report_card_markdown": result.get("report_card_markdown_path"),
        "report_card_json": result.get("report_card_json_path"),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": _json_safe(summary),
        "output_files": _json_safe(output_files),
    }


@app.post("/api/oos/report-card", response_model=OOSReportResponse)
def run_oos_report_card(request: OOSReportRequest) -> OOSReportResponse:
    """Synchronous OOS endpoint (legacy). Prefer /api/oos/submit for long runs."""
    try:
        data = _execute_oos_logic(request)
        return OOSReportResponse(
            generated_at=datetime.fromisoformat(data["generated_at"]),
            summary=data["summary"],
            output_files=data["output_files"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OOS report generation failed: {exc}")


# ── Async OOS job store ───────────────────────────────────────────────────────
# Simple in-process dict; keys are UUID strings, values track job state.
# Sufficient for single-instance deployment; replaced by a task queue (Celery,
# ARQ) when moving to multi-process / multi-worker.

_oos_jobs: Dict[str, Dict[str, Any]] = {}


def _oos_job_worker(job_id: str, request: OOSReportRequest) -> None:
    """Background thread target — runs OOS logic and writes result to _oos_jobs."""
    _oos_jobs[job_id]["status"] = "running"
    try:
        data = _execute_oos_logic(request)
        _oos_jobs[job_id].update({"status": "complete", "data": data, "error": None})
    except Exception as exc:
        _oos_jobs[job_id].update({"status": "error", "data": None, "error": str(exc)})


@app.post("/api/oos/submit")
def submit_oos_job(request: OOSReportRequest) -> Dict[str, str]:
    """Submit an OOS job and return immediately with a job_id.
    Poll GET /api/oos/status/{job_id} every 2–3 s for completion.
    """
    job_id = str(uuid.uuid4())
    _oos_jobs[job_id] = {
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data": None,
        "error": None,
    }
    t = threading.Thread(target=_oos_job_worker, args=(job_id, request), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/oos/status/{job_id}")
def get_oos_job_status(job_id: str) -> Dict[str, Any]:
    """Poll job status. Returns status, elapsed_sec, and (when complete) data."""
    job = _oos_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"OOS job {job_id!r} not found")
    elapsed_sec: Optional[int] = None
    try:
        start = datetime.fromisoformat(job["started_at"])
        elapsed_sec = int((datetime.now(timezone.utc) - start).total_seconds())
    except Exception:
        pass
    return {
        "job_id": job_id,
        "status": job["status"],
        "elapsed_sec": elapsed_sec,
        "data": job.get("data"),
        "error": job.get("error"),
    }


# ── ML training job store ─────────────────────────────────────────────────────

_ml_train_jobs: Dict[str, Dict[str, Any]] = {}


def _ml_train_worker(job_id: str) -> None:
    """Background thread: calibrates labels, trains the crush classifier, reloads it."""
    _ml_train_jobs[job_id]["status"] = "running"
    try:
        collector = InstitutionalDataCollector()
        # Always calibrate labels first — this pairs any new pre/post snapshots from
        # a recent backfill into earnings_iv_decay_labels before training reads from it.
        collector.db.calibrate_earnings_iv_decay_labels()
        result = collector.db.train_ml_model_on_historical_spreads()
        # Reload the model in the edge engine so subsequent analyses pick it up
        from web.api.edge_engine import reload_crush_model
        reload_crush_model()
        _ml_train_jobs[job_id].update({"status": "complete", "data": result, "error": None})
    except Exception as exc:
        _ml_train_jobs[job_id].update({"status": "error", "data": None, "error": str(exc)})


@app.post("/api/ml/train")
def submit_ml_train_job() -> Dict[str, str]:
    """Train the ML IV crush classifier in the background.
    Returns job_id — poll GET /api/ml/train-status/{job_id} for completion.
    Training requires earnings_iv_decay_labels rows (run backfill first).
    """
    job_id = str(uuid.uuid4())
    _ml_train_jobs[job_id] = {
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data": None,
        "error": None,
    }
    t = threading.Thread(target=_ml_train_worker, args=(job_id,), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/ml/train-status/{job_id}")
def get_ml_train_status(job_id: str) -> Dict[str, Any]:
    """Poll ML training job. Returns status, elapsed_sec, metrics when complete."""
    job = _ml_train_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"ML training job {job_id!r} not found")
    elapsed_sec: Optional[int] = None
    try:
        start = datetime.fromisoformat(job["started_at"])
        elapsed_sec = int((datetime.now(timezone.utc) - start).total_seconds())
    except Exception:
        pass
    return {
        "job_id": job_id,
        "status": job["status"],
        "elapsed_sec": elapsed_sec,
        "data": job.get("data"),
        "error": job.get("error"),
    }


# ── Serve the built React frontend (must be LAST — catches all remaining paths) ─
_FRONTEND_DIST = project_root / "web" / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
