from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

_OOS_STABILITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "evidence_balanced": {
        "execution_profiles": ["institutional"],
        "hold_days_grid": [7],
        "trades_per_day_grid": [4],
        "entry_days_grid": [7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.50,
            "min_crush_confidence": 0.30,
            "min_crush_magnitude": 0.06,
            "min_crush_edge": 0.02,
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
        "hold_days_grid": [1, 3],
        "trades_per_day_grid": [1, 2],
        "entry_days_grid": [3, 5],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.55,
            "min_crush_confidence": 0.40,
            "min_crush_magnitude": 0.07,
            "min_crush_edge": 0.03,
            "target_entry_dte": 6,
            "entry_dte_band": 4,
            "min_daily_share_volume": 2_500_000,
            "max_abs_momentum_5d": 0.08,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
    "alpha_focus": {
        "execution_profiles": ["institutional_tight"],
        "hold_days_grid": [1],
        "trades_per_day_grid": [1],
        "entry_days_grid": [2, 3],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.60,
            "min_crush_confidence": 0.45,
            "min_crush_magnitude": 0.08,
            "min_crush_edge": 0.035,
            "target_entry_dte": 6,
            "entry_dte_band": 3,
            "min_daily_share_volume": 5_000_000,
            "max_abs_momentum_5d": 0.07,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
}

_OOS_AUTO_PROFILE_ORDER: Tuple[str, ...] = (
    "evidence_balanced",
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


def _oos_result_score(result: Optional[Dict[str, Any]]) -> Tuple[int, int, float, float, float, int, int]:
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
    return overall_pass, ci_positive_count, alpha_low, sharpe_low, pnl_low, total_trades, splits


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
        notes: List[str] = []

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
                candidate_result = collector.run_oos_validation(**candidate_kwargs)
                if candidate_result:
                    auto_candidates.append((profile_name, candidate_result, candidate_kwargs))
                    profile_summaries.append(_oos_profile_summary(profile_name, candidate_result))
                else:
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
            result = collector.run_oos_validation(**run_kwargs)
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
            retry_result = collector.run_oos_validation(**adaptive_kwargs)
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
                        "lookback_days": int(run_kwargs["lookback_days"]),
                        "max_backtest_symbols": int(run_kwargs["max_backtest_symbols"]),
                    },
                    "stability_profile_requested": stability_profile_requested,
                    "stability_profile_used": stability_profile_used,
                    "stability_profiles_evaluated": profile_summaries,
                    "adaptive_retry_used": adaptive_used,
                    "notes": notes,
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
                "lookback_days": int(adaptive_kwargs["lookback_days"] if adaptive_used else run_kwargs["lookback_days"]),
                "max_backtest_symbols": int(adaptive_kwargs["max_backtest_symbols"] if adaptive_used else run_kwargs["max_backtest_symbols"]),
            },
            "stability_profile_requested": stability_profile_requested,
            "stability_profile_used": stability_profile_used,
            "stability_profiles_evaluated": profile_summaries,
            "adaptive_retry_used": adaptive_used,
            "notes": notes,
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
