from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any, Dict
import sys
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.institutional_backfill import InstitutionalDataCollector
from web.api.edge_engine import analyze_single_ticker
from web.api.schemas import (
    EdgeAnalyzeRequest,
    EdgeAnalyzeResponse,
    OOSReportRequest,
    OOSReportResponse,
)


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


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/edge/analyze", response_model=EdgeAnalyzeResponse)
def analyze_edge(request: EdgeAnalyzeRequest) -> EdgeAnalyzeResponse:
    try:
        snapshot = analyze_single_ticker(request.symbol)
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


@app.post("/api/oos/report-card", response_model=OOSReportResponse)
def run_oos_report_card(request: OOSReportRequest) -> OOSReportResponse:
    try:
        collector = InstitutionalDataCollector()
        train_days = int(request.oos_train_days)
        test_days = int(request.oos_test_days)
        step_days = int(request.oos_step_days)
        adjustment_note = None

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
            # Non-fatal: keep user-supplied windows.
            pass

        result = collector.run_oos_validation(
            execution_profiles=["institutional"],
            hold_days_grid=[7],
            signal_threshold_grid=[request.min_signal_score],
            trades_per_day_grid=[4],
            entry_days_grid=[7],
            exit_days_grid=[1],
            target_entry_dte=request.target_entry_dte,
            entry_dte_band=request.entry_dte_band,
            min_daily_share_volume=request.min_daily_share_volume,
            max_abs_momentum_5d=request.max_abs_momentum_5d,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            top_n_train=request.oos_top_n_train,
            lookback_days=request.lookback_days,
            max_backtest_symbols=request.max_backtest_symbols,
            use_crush_confidence_gate=True,
            allow_global_crush_profile=True,
            min_crush_confidence=request.min_crush_confidence,
            min_crush_magnitude=request.min_crush_magnitude,
            min_crush_edge=request.min_crush_edge,
            allow_proxy_earnings=True,
            min_splits=request.oos_min_splits,
            min_total_test_trades=request.oos_min_total_test_trades,
            min_trades_per_split=request.oos_min_trades_per_split,
            output_dir="exports/reports",
            start_date=request.backtest_start_date,
            end_date=request.backtest_end_date,
        )
        if not result:
            return OOSReportResponse(
                generated_at=datetime.now(timezone.utc),
                summary=_json_safe({
                    "status": "no_oos_rows",
                    "grade": "N/A",
                    "overall_pass": False,
                    "message": "No OOS rows produced for current filters/history.",
                    "windows_used": {
                        "train_days": train_days,
                        "test_days": test_days,
                        "step_days": step_days,
                    },
                    "notes": [adjustment_note] if adjustment_note else [],
                }),
                output_files={},
            )

        report_card = result.get("report_card", {}) if isinstance(result, dict) else {}
        verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
        sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
        metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
        gates = report_card.get("gates", {}) if isinstance(report_card, dict) else {}
        summary = {
            "splits": int(result.get("splits", 0)),
            "best_params": result.get("best_params", {}),
            "grade": verdict.get("grade"),
            "overall_pass": verdict.get("overall_pass"),
            "verdict": verdict,
            "sample": sample,
            "metrics": metrics,
            "gates": gates,
            "status": "ok",
            "windows_used": {
                "train_days": train_days,
                "test_days": test_days,
                "step_days": step_days,
            },
            "notes": [adjustment_note] if adjustment_note else [],
        }
        output_files = {
            "csv": result.get("csv_path"),
            "summary_markdown": result.get("markdown_path"),
            "summary_json": result.get("json_path"),
            "report_card_markdown": result.get("report_card_markdown_path"),
            "report_card_json": result.get("report_card_json_path"),
        }
        return OOSReportResponse(
            generated_at=datetime.now(timezone.utc),
            summary=_json_safe(summary),
            output_files=_json_safe(output_files),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OOS report generation failed: {exc}")
