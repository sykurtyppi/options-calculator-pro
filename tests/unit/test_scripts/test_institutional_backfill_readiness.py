import numpy as np
import pandas as pd

from scripts.institutional_backfill import InstitutionalDataCollector


def test_forward_quality_gate_passes_with_healthy_metrics():
    summary = {
        "status": "ok",
        "max_drawdown": -1200.0,
        "mean_execution_drag_bps": 95.0,
        "crush_directional_accuracy": 0.72,
        "fill_metrics": {"fill_events": 12},
    }
    passed, details, advisories = InstitutionalDataCollector._evaluate_forward_quality_gate(summary)
    assert passed is True
    assert details["status_ok"] is True
    assert details["drawdown_ok"] is True
    assert details["execution_drag_ok"] is True
    assert details["directional_accuracy_ok"] is True
    assert details["fill_log_ok"] is True
    assert advisories == []


def test_forward_quality_gate_surfaces_all_failures():
    summary = {
        "status": "insufficient_data",
        "max_drawdown": -100_000.0,
        "mean_execution_drag_bps": 320.0,
        "crush_directional_accuracy": 0.41,
        "fill_metrics": {"fill_events": 0},
    }
    passed, details, advisories = InstitutionalDataCollector._evaluate_forward_quality_gate(
        summary,
        require_status_ok=True,
        max_drawdown=-5_000.0,
        max_execution_drag_bps=200.0,
        min_directional_accuracy=0.55,
        require_fill_log=True,
    )
    assert passed is False
    assert details["status_ok"] is False
    assert details["drawdown_ok"] is False
    assert details["execution_drag_ok"] is False
    assert details["directional_accuracy_ok"] is False
    assert details["fill_log_ok"] is False
    assert len(advisories) >= 5


def test_forward_quality_gate_respects_disabled_thresholds():
    summary = {
        "status": "insufficient_data",
        "max_drawdown": np.nan,
        "mean_execution_drag_bps": np.nan,
        "crush_directional_accuracy": np.nan,
        "fill_metrics": {"fill_events": 0},
    }
    passed, details, advisories = InstitutionalDataCollector._evaluate_forward_quality_gate(
        summary,
        require_status_ok=False,
        max_drawdown=None,
        max_execution_drag_bps=None,
        min_directional_accuracy=None,
        require_fill_log=False,
    )
    assert passed is True
    assert details["status_ok"] is True
    assert details["drawdown_ok"] is True
    assert details["execution_drag_ok"] is True
    assert details["directional_accuracy_ok"] is True
    assert details["fill_log_ok"] is True
    assert advisories == []


def test_resolve_promotion_live_session_prefers_explicit_value():
    session_id = InstitutionalDataCollector._resolve_promotion_live_session(
        promotion_live_session_id="session_manual_123",
        oos_result={"best_test_session_id": "session_oos_1"},
        sweep_result={"best_session_id": "session_sweep_1"},
    )
    assert session_id == "session_manual_123"


def test_resolve_promotion_live_session_falls_back_to_oos_then_sweep():
    oos_session = InstitutionalDataCollector._resolve_promotion_live_session(
        promotion_live_session_id=None,
        oos_result={"best_test_session_id": "session_oos_2"},
        sweep_result={"best_session_id": "session_sweep_2"},
    )
    assert oos_session == "session_oos_2"

    sweep_session = InstitutionalDataCollector._resolve_promotion_live_session(
        promotion_live_session_id=None,
        oos_result={"best_test_session_id": ""},
        sweep_result={"best_session_id": "session_sweep_2"},
    )
    assert sweep_session == "session_sweep_2"

    all_sessions = InstitutionalDataCollector._resolve_promotion_live_session(
        promotion_live_session_id=None,
        oos_result={},
        sweep_result={},
    )
    assert all_sessions == "all_sessions"


def test_forward_tracker_marks_backtest_source_as_simulated():
    collector = InstitutionalDataCollector.__new__(InstitutionalDataCollector)
    trades = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02"] * 25),
            "symbol": ["AAPL"] * 25,
            "contracts": [1] * 25,
            "debit_per_contract": [2.0] * 25,
            "transaction_cost_per_contract": [0.15] * 25,
            "gross_return_pct": [0.05] * 25,
            "net_return_pct": [0.04] * 25,
            "pnl_per_contract": [0.08] * 25,
            "crush_confidence": [0.80] * 25,
            "crush_edge_score": [0.03] * 25,
            "predicted_front_iv_crush_pct": [-0.20] * 25,
            "realized_front_iv_crush_pct": [-0.18] * 25,
        }
    )
    summary, _ = collector._compute_forward_tracker_metrics(
        trades_df=trades,
        scope_label="all_sessions",
        lookback_days=120,
        min_confidence=0.35,
        data_source="backtest_trades",
    )
    assert summary["status"] == "simulated_backtest"
    assert summary["data_source"] == "backtest_trades"


def test_forward_tracker_marks_trade_log_source_as_ok_when_sample_is_large():
    collector = InstitutionalDataCollector.__new__(InstitutionalDataCollector)
    trades = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02"] * 25),
            "symbol": ["MSFT"] * 25,
            "contracts": [1] * 25,
            "debit_per_contract": [2.0] * 25,
            "transaction_cost_per_contract": [0.10] * 25,
            "gross_return_pct": [0.06] * 25,
            "net_return_pct": [0.05] * 25,
            "pnl_per_contract": [0.10] * 25,
            "crush_confidence": [0.85] * 25,
            "crush_edge_score": [0.04] * 25,
            "predicted_front_iv_crush_pct": [-0.22] * 25,
            "realized_front_iv_crush_pct": [-0.19] * 25,
        }
    )
    summary, _ = collector._compute_forward_tracker_metrics(
        trades_df=trades,
        scope_label="trade_log:paper.csv",
        lookback_days=120,
        min_confidence=0.35,
        data_source="trade_log",
    )
    assert summary["status"] == "ok"
    assert summary["data_source"] == "trade_log"


def test_promotion_evidence_gate_supports_and_or_modes():
    gate_or, mode_or = InstitutionalDataCollector._evaluate_promotion_evidence_gate(
        grade_gate_pass=True,
        live_trade_gate_pass=False,
        evidence_mode="or",
    )
    assert mode_or == "or"
    assert gate_or is True

    gate_and, mode_and = InstitutionalDataCollector._evaluate_promotion_evidence_gate(
        grade_gate_pass=True,
        live_trade_gate_pass=False,
        evidence_mode="and",
    )
    assert mode_and == "and"
    assert gate_and is False


def test_promotion_evidence_gate_falls_back_to_and_for_invalid_mode():
    gate, mode = InstitutionalDataCollector._evaluate_promotion_evidence_gate(
        grade_gate_pass=True,
        live_trade_gate_pass=False,
        evidence_mode="invalid",
    )
    assert mode == "and"
    assert gate is False


def test_fill_log_preserves_bps_columns_without_rescaling(tmp_path):
    collector = InstitutionalDataCollector.__new__(InstitutionalDataCollector)
    today = pd.Timestamp.now().normalize()
    fill_path = tmp_path / "fills_bps.csv"
    pd.DataFrame(
        {
            "trade_date": [today, today],
            "slippage_bps": [0.5, 1.2],
            "fill_latency_seconds": [1.0, 2.0],
        }
    ).to_csv(fill_path, index=False)

    _, summary, note = collector._load_forward_fill_log(str(fill_path), lookback_days=30)
    assert note is None
    assert summary["fill_events"] == 2
    assert abs(float(summary["mean_slippage_bps"]) - 0.85) < 1e-9


def test_fill_log_converts_decimal_slippage_to_bps(tmp_path):
    collector = InstitutionalDataCollector.__new__(InstitutionalDataCollector)
    today = pd.Timestamp.now().normalize()
    fill_path = tmp_path / "fills_decimal.csv"
    pd.DataFrame(
        {
            "trade_date": [today, today],
            "slippage": [0.001, 0.002],
            "fill_latency_seconds": [1.0, 1.5],
        }
    ).to_csv(fill_path, index=False)

    _, summary, note = collector._load_forward_fill_log(str(fill_path), lookback_days=30)
    assert note is None
    assert summary["fill_events"] == 2
    assert abs(float(summary["mean_slippage_bps"]) - 15.0) < 1e-9
